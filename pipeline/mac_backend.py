"""Your own Apple Silicon Mac as a free, offline backend for reference stills and keyframes.

Two halves, talking only through the `batches` GitHub release (the repo is public):

  producer side (GitHub Actions)   MacBackend: packs a batch as `mac-<id>.tar.gz` and waits.
  Mac side (you run it)            `python -m pipeline mac-worker`: claims the batch, renders every
                                   job with mflux (MLX, native on Apple Silicon, bf16 so no fp16
                                   overflow), and uploads `mac-<id>.results.tar.gz`.

Release files of one batch:
  mac-<id>.tar.gz            inputs (manifest.json and any input images), written by the producer
  mac-<id>.claim.json        a Mac has taken it (host, time)
  mac-<id>.progress.json     heartbeat every few minutes: jobs done / total
  mac-<id>.results.tar.gz    out/results.json and the images, written by the Mac
All four are deleted once the producer has collected the results.

If no Mac takes a batch within mac.max_wait_hours, or its heartbeat stops for mac.stale_minutes,
the producer gives up on it and the jobs are free again. Nothing here costs money.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import tarfile
import threading
import time
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path

RELEASE = "batches"
DEFAULTS = {
    "enabled": False,
    "stages": ["refs", "keyframes"],
    "batch_hours": 3,
    "setup_minutes": 5,
    "max_wait_hours": 72,
    "stale_minutes": 45,
    "heartbeat_minutes": 5,
    "poll_seconds": 60,
    "seconds_per_job": 300,
    "engines": {
        "refs": {"cmd": ["mflux-generate-qwen", "--model", "mlx-community/Qwen-Image-2512-4bit", "--steps", "20",
                         "--low-ram"]},
        "keyframe": {"cmd": ["mflux-generate-qwen-edit", "--model", "{model_dir}", "--base-model", "qwen",
                             "--steps", "20", "--low-ram"],
                     "download": {"repo": "fcreait/Qwen-Image-Edit-mflux", "folder": "Qwen-Image-Edit-2511-q4"},
                     "image_flag": "--image-paths", "max_refs": 3},
    },
}


def settings(p) -> dict:
    cfg = {**DEFAULTS, **(p.config.get("mac", {}) if p is not None else {})}
    cfg["engines"] = {**DEFAULTS["engines"], **cfg.get("engines", {})}
    return cfg


def _iso(ts: str | None) -> float:
    if not ts:
        return 0.0
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


class Release:
    """The `batches` release, through the gh CLI (uploads, deletes) and plain HTTPS (downloads)."""

    def __init__(self, repo: str, gh=None, fetch=None):
        self.repo = repo
        self._gh = gh or self._gh_cli
        self._fetch = fetch or self._urllib

    @staticmethod
    def _gh_cli(args: list[str]) -> tuple[int, str]:
        r = subprocess.run(["gh", *args], capture_output=True, text=True)
        return r.returncode, (r.stdout or "") + (r.stderr or "")

    @staticmethod
    def _urllib(url: str, dest: Path) -> None:
        tmp = dest.with_suffix(dest.suffix + ".part")
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.replace(dest)

    def assets(self) -> dict[str, dict]:
        code, out = self._gh(["api", f"repos/{self.repo}/releases/tags/{RELEASE}"])
        if code != 0:
            return {}
        try:
            return {a["name"]: a for a in json.loads(out).get("assets", [])}
        except ValueError:
            return {}

    def ensure(self) -> None:
        code, _ = self._gh(["release", "view", RELEASE, "-R", self.repo, "--json", "tagName"])
        if code != 0:
            code, out = self._gh(["release", "create", RELEASE, "-R", self.repo, "--prerelease", "--title", RELEASE,
                                  "--notes", "Inputs and results of off-Kaggle batches, deleted after each batch."])
            if code != 0 and "already exists" not in out:
                raise RuntimeError(f"could not create release {RELEASE}: {out.strip()[-300:]}")

    def upload(self, path: Path) -> None:
        code, out = self._gh(["release", "upload", RELEASE, str(path), "-R", self.repo, "--clobber"])
        if code != 0:
            raise RuntimeError(f"upload of {Path(path).name} failed: {out.strip()[-300:]}")

    def delete(self, name: str) -> None:
        self._gh(["release", "delete-asset", RELEASE, name, "-R", self.repo, "-y"])

    def download(self, asset: dict, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._fetch(asset["browser_download_url"], dest)


def _names(batch_id: str) -> dict[str, str]:
    b = f"mac-{batch_id}"
    return {"inputs": f"{b}.tar.gz", "claim": f"{b}.claim.json", "progress": f"{b}.progress.json",
            "results": f"{b}.results.tar.gz"}


# ==================================================================== producer side
class MacBackend:
    name = "mac"
    accelerator = ""
    take_over = False

    def __init__(self, p, repo: str, gh=None, fetch=None, now=time.time):
        self.cfg = settings(p)
        self.rel = Release(repo, gh, fetch)
        self.now = now
        self.stages = tuple(self.cfg["stages"]) if self.cfg.get("stages") else None
        self.log: list[str] = []

    def budget(self, p, led) -> tuple[float, float]:
        return float(self.cfg["batch_hours"]) * 3600, float(self.cfg["setup_minutes"]) * 60

    def gpu_hours_left(self) -> float | None:
        return None

    def publish_dataset(self, folder: Path, ref: str, message: str) -> None:
        batch_id = message
        tgz = Path(folder).parent / _names(batch_id)["inputs"]
        with tarfile.open(tgz, "w:gz") as t:
            for f in sorted(Path(folder).iterdir()):
                if f.name not in ("dataset-metadata.json", "worker.py"):
                    t.add(f, arcname=f.name)
        self.rel.ensure()
        self.rel.upload(tgz)
        self.log.append(f"{tgz.name} ({tgz.stat().st_size // 1024} KB) waits for a Mac")

    def wait_dataset_ready(self, ref: str, **_) -> None:
        return None

    def push(self, folder: Path, accelerator: str, timeout_s: int) -> str:
        return Path(folder).parent.name                  # the handle is the batch id

    def status(self, ref: str, handle: str | None = None) -> str:
        n, a = _names(handle or ""), self.rel.assets()
        if not a:
            return "unknown"                             # GitHub unreachable: ask again next run
        if n["results"] in a:
            return "complete"
        if n["inputs"] not in a:
            self.log.append(f"{n['inputs']} is gone")
            return "error"
        if n["claim"] in a:
            beat = max(_iso(a[n["claim"]].get("updated_at")),
                       _iso((a.get(n["progress"]) or {}).get("updated_at")))
            if self.now() - beat > float(self.cfg["stale_minutes"]) * 60:
                self.log.append(f"the Mac working on {handle} went quiet "
                                f"{(self.now() - beat) / 60:.0f} min ago; giving up on it")
                return "error"
            return "running"
        waited = self.now() - _iso(a[n["inputs"]].get("created_at"))
        if waited > float(self.cfg["max_wait_hours"]) * 3600:
            self.log.append(f"no Mac took {handle} in {waited / 3600:.0f} h; its jobs are freed")
            return "error"
        return "queued"

    def output(self, ref: str, dest: Path, handle: str | None = None) -> None:
        n, a = _names(handle or ""), self.rel.assets()
        try:
            if n["results"] in a:
                tgz = Path(dest) / n["results"]
                self.rel.download(a[n["results"]], tgz)
                with tarfile.open(tgz) as t:
                    try:
                        t.extractall(dest, filter="data")
                    except TypeError:
                        t.extractall(dest)
        finally:
            if a:
                for name in n.values():
                    if name in a:
                        self.rel.delete(name)


def from_env(p):
    """The Mac backend when config.json switches it on (it needs no secret)."""
    if not settings(p).get("enabled"):
        return None
    return MacBackend(p, os.environ.get("GITHUB_REPOSITORY", ""))


# ==================================================================== Mac side
def _log(msg: str) -> None:
    print(f"[mac {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _png(path: Path, w: int, h: int) -> None:
    raw = b"".join(b"\x00" + bytes([60, 80 + y * 100 // h, 140]) * w for y in range(h))
    ck = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))  # noqa: E731
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + ck(b"IDAT", zlib.compress(raw)) + ck(b"IEND", b""))


def _seed(key: str) -> int:
    import hashlib  # noqa: PLC0415
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


class StubEngine:
    """For tests and dry runs: a flat image, no model."""

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        w, h = job.get("width", 64) // 16 or 4, job.get("height", 36) // 16 or 4
        _png(out, w, h)
        return {"width": w, "height": h, "engine": "stub"}


def resolve_cmd(e: dict, models: Path, download=None) -> list[str]:
    """The engine's command, with {model_dir} replaced by a local copy of e["download"]
    (a Hugging Face repo and a folder in it), fetched once into `models`."""
    cmd = list(e["cmd"])
    d = e.get("download")
    if not d:
        return cmd
    local = models / d["repo"].replace("/", "--")
    target = local / d.get("folder", "")
    if not (target / ".complete").exists():
        if download is None:
            from huggingface_hub import snapshot_download  # noqa: PLC0415 - comes with mflux
            download = snapshot_download
        _log(f"downloading {d['repo']}/{d.get('folder', '')} (once)")
        download(d["repo"], local_dir=str(local),
                 allow_patterns=[f"{d['folder']}/*"] if d.get("folder") else None)
        target.mkdir(parents=True, exist_ok=True)
        (target / ".complete").write_text("ok")
    return [str(target) if x == "{model_dir}" else x for x in cmd]


class MfluxEngine:
    """Runs mflux's command line once per image. Text-to-image for reference stills; image edit
    (1 to 3 input images) for derived stills and keyframes. Commands come from config mac.engines."""

    def __init__(self, cfg: dict, models: Path | None = None):
        self.cfg = cfg
        self.models = models or Path.home() / "mochizuki-mac" / "models"
        if not shutil.which(cfg["engines"]["refs"]["cmd"][0]):
            raise RuntimeError("mflux is not installed: pip install -U mflux (see docs/mac-worker.md)")
        self.cmds: dict[str, list[str]] = {}

    def _cmd(self, name: str) -> list[str]:
        if name not in self.cmds:
            self.cmds[name] = resolve_cmd(self.cfg["engines"][name], self.models)
        return self.cmds[name]

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        images = ([job["image"]] if job.get("image") else job.get("refs", []))
        edit = bool(images)
        e = self.cfg["engines"]["keyframe" if edit else "refs"]
        cmd = [*self._cmd("keyframe" if edit else "refs"), "--prompt", job["prompt"], "--seed", str(_seed(job["key"])),
               "--width", str(job.get("width", 1664)), "--height", str(job.get("height", 928)),
               "--output", str(out), *e.get("extra_args", [])]
        if edit:
            paths = [str(inputs / x) for x in images][: e.get("max_refs", 3)]
            missing = [x for x in paths if not Path(x).exists()]
            if missing:
                raise FileNotFoundError(f"inputs missing from the batch: {missing}")
            cmd += [e.get("image_flag", "--image-paths"), *paths]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=float(self.cfg["seconds_per_job"]) * 6)
        if r.returncode != 0 or not out.exists():
            raise RuntimeError(f"mflux exit {r.returncode}: {(r.stderr or r.stdout).strip()[-600:]}")
        self._check(out)
        return {"seconds": round(time.time() - t0, 1), "seed": _seed(job["key"]), "engine": cmd[0],
                "model": cmd[cmd.index("--model") + 1] if "--model" in cmd else None}

    @staticmethod
    def _check(out: Path) -> None:
        try:
            from PIL import Image, ImageStat  # noqa: PLC0415
        except ImportError:
            return
        st = ImageStat.Stat(Image.open(out).convert("L"))
        if st.extrema[0][1] - st.extrema[0][0] < 5:
            raise RuntimeError("image came out blank")


def run_batch(manifest: dict, inputs: Path, out: Path, engine, beat=None) -> dict:
    """Render every job; results.json in the same shape as the Kaggle worker writes."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    deadline = t0 + float(manifest.get("deadline_seconds", 3 * 3600))
    res = {"batch_id": manifest["batch_id"], "jobs": {}, "groups": {"mac": {"host": socket.gethostname()}},
           "error": None}
    jobs = manifest["jobs"]
    for i, job in enumerate(jobs):
        if time.time() + float(job.get("est_seconds", 0)) > deadline:
            break                                        # the rest go to the next batch
        target = out / f"{job['key']}.png"
        j0 = time.time()
        try:
            info = engine.run(job, inputs, target)
            res["jobs"][job["key"]] = {"ok": True, "file": target.name, "gpu_seconds": time.time() - j0,
                                       **(info or {})}
            _log(f"{job['key']} ok in {time.time() - j0:.0f} s [{i + 1}/{len(jobs)}]")
        except Exception as e:  # noqa: BLE001 - one bad job must not stop the batch
            res["jobs"][job["key"]] = {"ok": False, "error": repr(e)[:800], "gpu_seconds": time.time() - j0}
            _log(f"{job['key']} failed: {e!r}"[:400])
        if beat:
            beat(i + 1, len(jobs))
    res["elapsed"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def take_one(rel: Release, work: Path, engine, cfg: dict) -> str | None:
    """Claim the oldest unclaimed batch, render it, upload the results. The batch id, or None."""
    a = rel.assets()
    waiting = sorted((x for x in a.values() if x["name"].startswith("mac-") and x["name"].endswith(".tar.gz")
                      and not x["name"].endswith(".results.tar.gz")),
                     key=lambda x: x.get("created_at", ""))
    for asset in waiting:
        batch_id = asset["name"][4:-7]
        n = _names(batch_id)
        if n["claim"] in a or n["results"] in a:
            continue
        d = work / batch_id
        shutil.rmtree(d, ignore_errors=True)
        (d / "in").mkdir(parents=True)
        claim = d / n["claim"]
        claim.write_text(json.dumps({"host": socket.gethostname(), "at": time.time()}))
        rel.upload(claim)
        _log(f"claimed {batch_id}")
        rel.download(asset, d / n["inputs"])
        with tarfile.open(d / n["inputs"]) as t:
            try:
                t.extractall(d / "in", filter="data")
            except TypeError:
                t.extractall(d / "in")
        manifest = json.loads((d / "in" / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("batch_id") != batch_id:
            raise RuntimeError(f"{n['inputs']} holds batch {manifest.get('batch_id')}")
        progress = {"done": 0, "total": len(manifest["jobs"])}
        stop = threading.Event()

        def send():
            f = d / n["progress"]
            f.write_text(json.dumps({**progress, "host": socket.gethostname(), "at": time.time()}))
            try:
                rel.upload(f)
            except Exception as e:  # noqa: BLE001 - a missed heartbeat is not fatal
                _log(f"heartbeat failed: {e}")

        def beat_loop():
            while not stop.wait(float(cfg["heartbeat_minutes"]) * 60):
                send()

        def beat(done, total):
            progress.update(done=done, total=total)
            send()

        send()
        threading.Thread(target=beat_loop, daemon=True).start()
        try:
            res = run_batch(manifest, d / "in", d / "out", engine, beat)
        finally:
            stop.set()
        tgz = d / n["results"]
        with tarfile.open(tgz, "w:gz") as t:
            t.add(d / "out", arcname="out")
        rel.upload(tgz)
        ok = sum(1 for j in res["jobs"].values() if j.get("ok"))
        _log(f"{batch_id}: {ok}/{len(manifest['jobs'])} made, results uploaded")
        return batch_id
    return None


def worker_loop(rel: Release, work: Path, engine, cfg: dict, once: bool = False, sleep=time.sleep) -> int:
    work.mkdir(parents=True, exist_ok=True)
    rel.ensure()
    done = 0
    while True:
        b = take_one(rel, work, engine, cfg)
        if b:
            done += 1
            continue
        if once:
            return done
        sleep(float(cfg["poll_seconds"]))


def self_test(p, engine, out: Path) -> dict:
    """One reference still ("style") with the real prompt, nothing sent to GitHub."""
    style = (p.root / "style_bible" / "style.txt")
    r = next(x for x in p.ref_plan() if x["name"] == "style")
    prompt = "\n".join(x for x in ((style.read_text(encoding="utf-8").strip() if style.exists() else ""),
                                   r["prompt"]) if x)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    info = engine.run({"key": "selftest-style", "prompt": prompt, "width": r.get("width", 1664),
                       "height": r.get("height", 928)}, p.root, out)
    still = {"file": str(out), "seconds": round(time.time() - t0, 1), **(info or {})}
    edit_out = out.with_name(out.stem + "-edit.png")             # the keyframe engine, from that still
    t0 = time.time()
    info = engine.run({"key": "selftest-edit", "prompt": "Same place and light, seen from a lower angle at dusk.",
                       "refs": [out.name], "width": 1664, "height": 928}, out.parent, edit_out)
    return {"still": still, "edit": {"file": str(edit_out), "seconds": round(time.time() - t0, 1), **(info or {})}}
