"""Training a LoRA on the Mac with mflux-train: a style (the film's 3D cel-shaded look) or a character.

    train_lora(cfg, "style3d", images_dir, trigger="mchz3d")

  1. dataset   the images are copied to ~/mochizuki-mac/train/<name>/data as 01.png + 01.txt pairs;
               a caption is the image's own .txt beside it if there is one, else the trigger plus
               --caption; preview_1.txt is the key-art prompt, rendered every few hundred steps
  2. config    train.json from config.json → mac.train (Z-Image Base, 8-bit, gradient checkpointing,
               low RAM: what fits a 32 GB Mac), checked with mflux-train --dry-run first
  3. training  mflux-train, logged as structured train.* lines; checkpoints let a stopped run resume
  4. result    the last checkpoint's adapter is saved as ~/mochizuki-mac/loras/<name>.safetensors and
               registered in ~/mochizuki-mac/loras.json, which the engines read: from then on every
               still uses it, with its trigger word put in front of the prompt
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

from ..runtime.shared import LORAS, WORK, log
from ..runtime.tools import run_streaming

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
LOCAL_LORAS = WORK / "loras.json"            # this Mac's LoRAs; kept out of git, merged into mac.loras


def find_images(folder: Path) -> list[Path]:
    return sorted(f for f in Path(folder).expanduser().iterdir() if f.suffix.lower() in IMAGE_TYPES
                  and not f.stem.startswith("preview"))


def build_dataset(images: list[Path], data: Path, trigger: str, caption: str, preview: str) -> int:
    """01.png + 01.txt pairs and preview_1.txt in `data`. The number of pairs."""
    shutil.rmtree(data, ignore_errors=True)
    data.mkdir(parents=True)
    for i, img in enumerate(images, 1):
        stem = f"{i:02d}"
        shutil.copyfile(img, data / f"{stem}{img.suffix.lower()}")
        own = img.with_suffix(".txt")
        text = own.read_text(encoding="utf-8").strip() if own.exists() else caption
        (data / f"{stem}.txt").write_text(f"{trigger}, {text}".strip(", "), encoding="utf-8")
    (data / "preview_1.txt").write_text(f"{trigger}, {preview}", encoding="utf-8")
    return len(images)


def train_config(t: dict, n_images: int, steps_total: int | None = None) -> dict:
    """mflux-train's config for `n_images` pairs from config.json → mac.train."""
    total = int(steps_total or t["steps_total"])
    epochs = max(1, math.ceil(total / max(1, n_images)))
    every = max(50, min(250, total // 6))
    base = t["model"] == "z-image"
    blocks = {"start": 0, "end": 30}
    targets = [{"module_path": f"layers.{{block}}.{m}", "blocks": blocks, "rank": int(t["rank"])}
               for m in ("attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0",
                         "feed_forward.w1", "feed_forward.w2", "feed_forward.w3")]
    return {
        "model": t["model"], "data": "data/", "seed": 4,
        "steps": 30 if base else 9, "guidance": 4.0 if base else 0.0,
        "quantize": t.get("quantize"), "max_resolution": int(t["max_resolution"]),
        "low_ram": bool(t.get("low_ram")), "gradient_checkpointing": bool(t.get("gradient_checkpointing")),
        "training_loop": {"num_epochs": epochs, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": float(t["learning_rate"])},
        "checkpoint": {"save_frequency": every, "output_path": "training"},
        "monitoring": {"preview_width": 832, "preview_height": 464, "plot_frequency": every,
                       "generate_image_frequency": every},
        "lora_layers": {"targets": targets},
    }


def latest_adapter(run: Path, dest: Path) -> Path:
    """The LoRA adapter from the newest checkpoint zip of a run, copied to `dest`."""
    zips = sorted((run / "training").rglob("*_checkpoint.zip"))
    if not zips:
        raise RuntimeError(f"no checkpoint in {run / 'training'}: training saved nothing")
    with zipfile.ZipFile(zips[-1]) as z:
        names = [n for n in z.namelist() if n.endswith("_adapter.safetensors")]
        if not names:
            raise RuntimeError(f"{zips[-1].name} holds no adapter")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with z.open(names[-1]) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
    return dest


def register(name: str, path: Path, trigger: str, scale: float, roles: list[str], registry: Path | None = None) -> dict:
    """Add (or replace) this LoRA in the Mac's own registry, which the engines read."""
    registry = registry or LOCAL_LORAS
    entries = json.loads(registry.read_text()) if registry.exists() else []
    entries = [e for e in entries if e.get("name") != name]
    entry = {"name": name, "path": str(path), "trigger": trigger, "scale": scale, "roles": roles,
             "trained": time.strftime("%Y-%m-%d %H:%M")}
    entries.append(entry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(entries, indent=2))
    return entry


def train_lora(cfg: dict, name: str, images_dir: Path, trigger: str | None = None, caption: str = "",
               preview: str = "", steps_total: int | None = None, scale: float = 0.8,
               roles: list[str] | None = None, dry_run: bool = False, tool: str = "mflux-train") -> dict:
    """Train one LoRA end to end; returns what was made (or why it stopped)."""
    t = cfg["train"]
    trigger = trigger or f"mchz_{name}"
    run = WORK / "train" / name
    images = find_images(images_dir)
    log(f"LoRA {name}: {len(images)} image(s) from {images_dir}, trigger '{trigger}', base model {t['model']}",
        event="train.start", name=name, images=len(images), trigger=trigger, model=t["model"])
    if len(images) < 8:
        raise ValueError(f"{len(images)} image(s) in {images_dir}: a LoRA needs at least 8, ideally 15-30")
    if len(images) < 15:
        log("fewer than 15 images: the LoRA may copy them instead of learning the look", event="train.few_images",
            level="warn", images=len(images))
    n = build_dataset(images, run / "data", trigger, caption, preview)
    conf = train_config(t, n, steps_total)
    conf["data"] = str(run / "data")                         # absolute: the run does not depend on the folder
    conf["checkpoint"]["output_path"] = str(run / "training")  # you start it from
    (run / "train.json").write_text(json.dumps(conf, indent=2))
    total = conf["training_loop"]["num_epochs"] * n
    log(f"LoRA {name}: {n} pairs x {conf['training_loop']['num_epochs']} epochs = {total} steps, rank {t['rank']}, "
        f"previews every {conf['checkpoint']['save_frequency']} steps in {run / 'training'}", event="train.config",
        steps=total, rank=t["rank"], config=str(run / "train.json"))
    check = subprocess.run([tool, "--config", str(run / "train.json"), "--dry-run"], capture_output=True, text=True,
                           cwd=run)
    if check.returncode != 0:
        raise RuntimeError(f"mflux-train refused the config: {(check.stderr or check.stdout).strip()[-600:]}")
    log("training config checked by mflux-train --dry-run", event="train.checked")
    if dry_run:
        return {"ok": True, "dry_run": True, "config": str(run / "train.json"), "steps": total}
    t0 = time.time()
    code, tail = run_streaming([tool, "--config", str(run / "train.json")], timeout=48 * 3600, cwd=run,
                               label=f"LoRA {name}:", every=120, kind="train", tool="mflux-train")
    if code != 0:
        raise RuntimeError(f"mflux-train exit {code}: {tail.strip()[-600:]}")
    path = latest_adapter(run, LORAS / f"{name}.safetensors")
    entry = register(name, path, trigger, scale, roles or ["refs", "keyframe"])
    log(f"LoRA {name} ready: {path} ({path.stat().st_size / 1e6:.0f} MB) after {(time.time() - t0) / 3600:.1f} h; "
        f"every still on this Mac now uses it (scale {scale}, trigger '{trigger}')", event="train.done", name=name,
        path=str(path), hours=round((time.time() - t0) / 3600, 1))
    return {"ok": True, **entry}


def local_loras(registry: Path | None = None) -> list[dict]:
    registry = registry or LOCAL_LORAS
    try:
        return json.loads(registry.read_text()) if registry.exists() else []
    except ValueError:
        log(f"{registry} is not valid JSON; ignoring it", event="lora.registry_bad", level="warn")
        return []
