"""Pictures on the Mac with mflux: text to image (reference stills) and image edit (keyframes).

Which model runs is config, not code: mac.engines.refs and mac.engines.keyframe are mflux
command lines (Z-Image Turbo and FLUX.2 [klein] 4B by default). Trained LoRAs listed in
mac.loras are added to the commands of the roles they name.
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from ..runtime.downloads import hf_cache_dir, is_hf_repo, prefetch_model, resolve_cmd
from ..runtime.shared import MODELS, log, seed_for
from ..runtime.tools import run_streaming
from .base import check_image


def lora_args(cfg: dict, role: str) -> list[str]:
    """--lora-paths/--lora-scales for the LoRAs that apply to `role` and exist on this Mac."""
    paths, scales = [], []
    for lo in cfg.get("loras", []):
        if role not in lo.get("roles", ["refs", "keyframe"]):
            continue
        path = Path(str(lo["path"])).expanduser()
        if not path.exists() and not is_hf_repo(str(lo["path"])):
            log(f"LoRA {lo['path']} is not on this Mac; rendering without it", event="lora.missing", level="warn",
                path=str(path))
            continue
        paths.append(str(path) if path.exists() else str(lo["path"]))
        scales.append(str(lo.get("scale", 1.0)))
    return ["--lora-paths", *paths, "--lora-scales", *scales] if paths else []


_SAVED = re.compile(r"Image saved successfully at:\s*(\S.*?\.(?:png|jpe?g|webp))", re.I)


def saved_path(output: str) -> Path | None:
    """Where mflux says it saved the picture (the last such line), if it says so."""
    found = _SAVED.findall(output or "")
    return Path(found[-1].strip()) if found else None


def lora_triggers(cfg: dict, role: str) -> str:
    """The trigger words of the LoRAs applied to `role`, to put in front of the prompt."""
    words = [lo["trigger"] for lo in cfg.get("loras", []) if lo.get("trigger")
             and role in lo.get("roles", ["refs", "keyframe"]) and Path(str(lo["path"])).expanduser().exists()]
    return ", ".join(words)


class MfluxImageEngine:
    """Runs one mflux command per picture. Jobs with input images use the keyframe (edit) engine,
    jobs without use the refs (text-to-image) engine."""

    def __init__(self, cfg: dict, models: Path | None = None):
        self.cfg = cfg
        self.models = models or MODELS
        tool = cfg["engines"]["refs"]["cmd"][0]
        if not shutil.which(tool):
            raise RuntimeError(f"{tool} is not installed: pip install -U mflux (see docs/mac-worker.md)")
        self.cmds: dict[str, list[str]] = {}
        self.loras: dict[str, int] = {}

    def _cmd(self, role: str) -> list[str]:
        if role not in self.cmds:
            cmd = resolve_cmd(self.cfg["engines"][role], self.models)
            if "--model" in cmd and is_hf_repo(cmd[cmd.index("--model") + 1]):
                prefetch_model(cmd[cmd.index("--model") + 1])
            extra = lora_args(self.cfg, role)
            self.loras[role] = (extra.index("--lora-scales") - 1) if extra else 0
            self.cmds[role] = cmd + extra
        return self.cmds[role]

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        images = [job["image"]] if job.get("image") else job.get("refs", [])
        role = "keyframe" if images else "refs"
        e = self.cfg["engines"][role]
        w, h = job.get("width", 1664), job.get("height", 928)
        triggers = lora_triggers(self.cfg, role)
        prompt = f"{triggers}, {job['prompt']}" if triggers else job["prompt"]
        cmd = [*self._cmd(role), "--prompt", prompt, "--seed", str(seed_for(job["key"])),
               "--width", str(w), "--height", str(h), "--output", str(out), *e.get("extra_args", [])]
        if images:
            paths = [str(inputs / x) for x in images][: e.get("max_refs", 3)]
            missing = [x for x in paths if not Path(x).exists()]
            if missing:
                raise FileNotFoundError(f"inputs missing from the batch: {missing}")
            cmd += [e.get("image_flag", "--image-paths"), *paths]
        steps = cmd[cmd.index("--steps") + 1] if "--steps" in cmd else "?"
        model = cmd[cmd.index("--model") + 1] if "--model" in cmd else cmd[0]
        log(f"rendering {job['key']}: {'edit of ' + str(len(images)) + ' image(s)' if images else 'text to image'}, "
            f"{w}x{h}, {steps} steps, {Path(model).name}", event="render.start", key=job["key"], role=role,
            steps=steps, model=Path(model).name, loras=self.loras.get(role, 0))
        cache = hf_cache_dir(e["hf_repo"]) if e.get("hf_repo") else None
        first_use = cache is not None and not cache.exists()
        if first_use:
            log(f"{job['key']}: {e['hf_repo']} is not on this Mac yet; mflux downloads it first (progress every "
                "minute)", event="model.download", model=e["hf_repo"])
        out.unlink(missing_ok=True)          # mflux never overwrites: with a file there it saves <name>_2.png
        t0 = time.time()
        code, tail = run_streaming(cmd, float(self.cfg["seconds_per_job"]) * 6 + (3600 if first_use else 0),
                                   label=job["key"] + ":", cache=cache)
        saved = saved_path(tail)
        if code == 0 and saved and saved != out and saved.exists():
            log(f"mflux saved {saved.name}; moving it to {out.name}", event="render.renamed", level="warn",
                saved=str(saved), file=str(out))
            saved.replace(out)
        if code != 0 or not out.exists() or out.stat().st_mtime < t0:
            raise RuntimeError(f"mflux exit {code}, no new {out.name}: {tail.strip()[-600:]}")
        check_image(out)
        took = time.time() - t0
        log(f"{job['key']}: saved {out} ({took:.0f} s)", event="render.done", key=job["key"], seconds=round(took),
            file=str(out))
        return {"seconds": round(took, 1), "seed": seed_for(job["key"]), "engine": cmd[0], "model": model}
