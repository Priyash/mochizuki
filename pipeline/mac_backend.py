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
import sys
import tarfile
import threading
import time
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path

RELEASE = "batches"
NAME = "mac"
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


def _names(batch_id: str, prefix: str = "mac") -> dict[str, str]:
    b = f"{prefix}-{batch_id}"
    return {"inputs": f"{b}.tar.gz", "claim": f"{b}.claim.json", "progress": f"{b}.progress.json",
            "results": f"{b}.results.tar.gz"}


# ==================================================================== producer side
class ReleaseBackend:
    """A batch waits in the `batches` release as <name>-<id>.tar.gz until a worker elsewhere (the
    Mac, or the Colab runner) claims it, sends heartbeats, and uploads <name>-<id>.results.tar.gz."""
    name = "mac"
    accelerator = ""
    take_over = False

    def __init__(self, p, repo: str, gh=None, fetch=None, now=time.time, cfg: dict | None = None):
        self.cfg = cfg if cfg is not None else settings(p)
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
        tgz = Path(folder).parent / _names(batch_id, self.name)["inputs"]
        with tarfile.open(tgz, "w:gz") as t:
            for f in sorted(Path(folder).iterdir()):
                if f.name not in ("dataset-metadata.json", "worker.py"):
                    t.add(f, arcname=f.name)
        self.rel.ensure()
        self.rel.upload(tgz)
        self.log.append(f"{tgz.name} ({tgz.stat().st_size // 1024} KB) waits for the {self.name} worker")

    def wait_dataset_ready(self, ref: str, **_) -> None:
        return None

    def push(self, folder: Path, accelerator: str, timeout_s: int) -> str:
        return Path(folder).parent.name                  # the handle is the batch id

    def status(self, ref: str, handle: str | None = None) -> str:
        n, a = _names(handle or "", self.name), self.rel.assets()
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
                self.log.append(f"the {self.name} worker on {handle} went quiet "
                                f"{(self.now() - beat) / 60:.0f} min ago; giving up on it")
                return "error"
            return "running"
        waited = self.now() - _iso(a[n["inputs"]].get("created_at"))
        if waited > float(self.cfg["max_wait_hours"]) * 3600:
            self.log.append(f"no {self.name} worker took {handle} in {waited / 3600:.0f} h; its jobs are freed")
            return "error"
        return "queued"

    def output(self, ref: str, dest: Path, handle: str | None = None) -> None:
        n, a = _names(handle or "", self.name), self.rel.assets()
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


class MacBackend(ReleaseBackend):
    name = "mac"


def from_env(p):
    """The Mac backend when config.json switches it on (it needs no secret)."""
    if not settings(p).get("enabled"):
        return None
    return MacBackend(p, os.environ.get("GITHUB_REPOSITORY", ""))


# ==================================================================== Mac side
def _log(msg: str, event: str = "note", level: str = "info", comp: str = "mac", **fields) -> None:
    """A structured log line (pipeline/log.py): level, component, event, message, key=value fields."""
    from .log import log  # noqa: PLC0415
    log(comp, event, msg, level, **fields)


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


def _weights(folder: Path) -> list[Path]:
    return [f for f in folder.rglob("*") if f.is_file() and f.suffix in (".safetensors", ".npz", ".gguf")]


NEEDED = ("transformer", "text_encoder", "vae")       # what mflux loads from a local Qwen model folder


def _model_ok(target: Path) -> str | None:
    """None if the folder holds a loadable model, else what is missing."""
    missing = [d for d in NEEDED if not _weights(target / d)]
    return f"no weights in {', '.join(missing)}" if missing else None


def resolve_cmd(e: dict, models: Path, download=None, list_files=None) -> list[str]:
    """The engine's command, with {model_dir} replaced by a local copy of e["download"]
    (a Hugging Face repo and a folder in it), fetched once into `models` and checked."""
    cmd = list(e["cmd"])
    d = e.get("download")
    if not d:
        return cmd
    folder = d.get("folder", "")
    local = models / d["repo"].replace("/", "--")
    target = local / folder
    done = target / ".complete"
    if done.exists() and _model_ok(target):          # marked done by an older version, but empty: fetch again
        _log(f"{d['repo']}/{folder} was marked downloaded but {_model_ok(target)}; downloading again",
             event="model.redownload", level="warn", model=d["repo"], folder=folder)
        done.unlink()
    if not done.exists():
        if list_files is None or download is None:
            from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415 - comes with mflux
            list_files = list_files or (lambda repo: [(f.path, f.size or 0) for f in HfApi().list_repo_tree(
                repo, recursive=True) if getattr(f, "size", None) is not None])
            download = download or snapshot_download
        files = list_files(d["repo"])
        mine = [(n, sz) for n, sz in files if not folder or n.startswith(folder.rstrip("/") + "/")]
        if not mine:
            tops = sorted({n.split("/")[0] for n, _ in files})
            raise RuntimeError(f"{d['repo']} has no folder '{folder}'; it has: {', '.join(tops)[:400]}")
        _log(f"downloading {d['repo']}/{folder} once: {len(mine)} files, about {sum(s for _, s in mine) / 1e9:.0f} GB; "
             "the progress bars below are Hugging Face's", event="model.download", model=d["repo"], folder=folder,
             files=len(mine), gb=round(sum(s for _, s in mine) / 1e9, 1))
        t0 = time.time()
        download(d["repo"], local_dir=str(local), allow_patterns=[n for n, _ in mine])
        bad = _model_ok(target)
        if bad:
            raise RuntimeError(f"{d['repo']}/{folder}: downloaded, but {bad} in {target}; not marking it complete")
        gb = sum(f.stat().st_size for f in _weights(target)) / 1e9
        _log(f"downloaded {d['repo']}/{folder}: {gb:.1f} GB in {(time.time() - t0) / 60:.1f} min",
             event="model.downloaded", model=d["repo"], gb=round(gb, 1), minutes=round((time.time() - t0) / 60, 1))
        done.write_text("ok")
    return [str(target) if x == "{model_dir}" else x for x in cmd]


def _is_hf_repo(x: str) -> bool:
    return x.count("/") == 1 and not x.startswith((".", "/", "~")) and not Path(x).exists()


def prefetch_model(repo: str, download=None) -> None:
    """Download a Hugging Face model once, with progress, so a render never waits on it silently."""
    try:
        from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415 - comes with mflux
    except ImportError:
        return
    download = download or snapshot_download
    try:
        download(repo, local_files_only=True)
        _log(f"model {repo}: already on this Mac", event="model.cached", model=repo)
        return
    except Exception:  # noqa: BLE001 - not cached yet
        pass
    size = ""
    try:
        gb = sum(f.size or 0 for f in HfApi().model_info(repo, files_metadata=True).siblings) / 1e9
        size = f", about {gb:.0f} GB"
    except Exception:  # noqa: BLE001
        pass
    _log(f"model {repo}: downloading once{size}; the progress bars below are Hugging Face's", event="model.download", model=repo)
    t0 = time.time()
    download(repo)
    _log(f"model {repo}: downloaded in {(time.time() - t0) / 60:.1f} min", event="model.downloaded", model=repo, minutes=round((time.time() - t0) / 60, 1))


def run_streaming(cmd: list[str], timeout: float, prefix: str = "  | ") -> tuple[int, str]:
    """Run a command and show its output live (mflux's step progress included); returns
    (exit code, last few KB of output) for error messages. Killed after `timeout` seconds."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    killer = threading.Timer(timeout, p.kill)
    killer.start()
    tail = b""
    start_of_line = True
    try:
        while True:
            chunk = p.stdout.read1(4096) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
            if not chunk:
                break
            tail = (tail + chunk)[-6000:]
            out = b""
            for ch in chunk.splitlines(keepends=True):
                out += (prefix.encode() if start_of_line else b"") + ch
                start_of_line = ch.endswith((b"\n", b"\r"))
            sys.stdout.buffer.write(out)
            sys.stdout.flush()
        code = p.wait()
    finally:
        killer.cancel()
    if code < 0:
        _log(f"stopped after {timeout / 60:.0f} min (time limit)", event="cmd.timeout", level="error", limit_min=round(timeout / 60))
    return code, tail.decode("utf-8", "replace")


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
            cmd = resolve_cmd(self.cfg["engines"][name], self.models)
            if "--model" in cmd and _is_hf_repo(cmd[cmd.index("--model") + 1]):
                prefetch_model(cmd[cmd.index("--model") + 1])
            self.cmds[name] = cmd
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
        steps = cmd[cmd.index("--steps") + 1] if "--steps" in cmd else "?"
        _log(f"rendering {job['key']}: {'edit of ' + str(len(images)) + ' image(s)' if edit else 'text to image'}, "
             f"{job.get('width', 1664)}x{job.get('height', 928)}, {steps} steps", event="render.start", key=job["key"], mode="edit" if edit else "t2i", steps=steps)
        t0 = time.time()
        code, tail = run_streaming(cmd, float(self.cfg["seconds_per_job"]) * 6)
        if code != 0 or not out.exists():
            raise RuntimeError(f"mflux exit {code}: {tail.strip()[-600:]}")
        self._check(out)
        _log(f"{job['key']}: saved {out} ({time.time() - t0:.0f} s)", event="render.done", key=job["key"], seconds=round(time.time() - t0), file=str(out))
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
    _log(f"batch {manifest['batch_id']}: {len(jobs)} job(s), stage {manifest.get('stage', '?')}, "
         f"time limit {(deadline - t0) / 3600:.1f} h", event="batch.start", batch=manifest["batch_id"], jobs=len(jobs), stage=manifest.get("stage"))
    for i, job in enumerate(jobs):
        if time.time() + float(job.get("est_seconds", 0)) > deadline:
            _log(f"time limit reached; {len(jobs) - i} job(s) go to the next batch", event="batch.time_limit", level="warn", left=len(jobs) - i)
            break                                        # the rest go to the next batch
        if i:
            per = (time.time() - t0) / i
            _log(f"[{i + 1}/{len(jobs)}] about {per * (len(jobs) - i) / 60:.0f} min left for this batch", event="batch.progress", step=f"{i + 1}/{len(jobs)}", eta_min=round(per * (len(jobs) - i) / 60))
        target = out / f"{job['key']}.png"
        j0 = time.time()
        try:
            info = engine.run(job, inputs, target)
            res["jobs"][job["key"]] = {"ok": True, "file": target.name, "gpu_seconds": time.time() - j0,
                                       **(info or {})}
            _log(f"{job['key']} ok in {time.time() - j0:.0f} s [{i + 1}/{len(jobs)}]", event="job.ok", key=job["key"], seconds=round(time.time() - j0), step=f"{i + 1}/{len(jobs)}")
        except Exception as e:  # noqa: BLE001 - one bad job must not stop the batch
            res["jobs"][job["key"]] = {"ok": False, "error": repr(e)[:800], "gpu_seconds": time.time() - j0}
            _log(f"{job['key']} failed: {e!r}"[:400], event="job.failed", level="error", key=job["key"], seconds=round(time.time() - j0))
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
        worker_loop._idle = 0
        _log(f"claimed {batch_id}; downloading its inputs", event="batch.claimed", batch=batch_id)
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
                _log(f"heartbeat failed: {e}", event="heartbeat.failed", level="warn")

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
        _log(f"{batch_id}: uploading results ({tgz.stat().st_size / 1e6:.0f} MB)", event="results.upload", batch=batch_id, mb=round(tgz.stat().st_size / 1e6))
        rel.upload(tgz)
        ok = sum(1 for j in res["jobs"].values() if j.get("ok"))
        _log(f"{batch_id}: {ok}/{len(manifest['jobs'])} made, results uploaded", event="batch.done", batch=batch_id, made=ok, jobs=len(manifest["jobs"]))
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
        idle = getattr(worker_loop, "_idle", 0) + 1
        worker_loop._idle = idle
        if idle == 1 or idle % 10 == 0:
            _log(f"no batch for this Mac yet; checking every {float(cfg['poll_seconds']):.0f} s "
                 f"({done} batch(es) done so far)", event="worker.idle", batches_done=done)
        sleep(float(cfg["poll_seconds"]))


def self_test(p, engine, out: Path, edit_only: bool = False) -> dict:
    """One reference still ("style") with the real prompt, nothing sent to GitHub."""
    style = (p.root / "style_bible" / "style.txt")
    r = next(x for x in p.ref_plan() if x["name"] == "style")
    prompt = "\n".join(x for x in ((style.read_text(encoding="utf-8").strip() if style.exists() else ""),
                                   r["prompt"]) if x)
    out.parent.mkdir(parents=True, exist_ok=True)
    _about_this_mac(out.parent)
    if edit_only and not out.exists():
        _log(f"--edit-only: no still at {out} (moved or renamed?), so making one first", event="selftest.no_still",
             level="warn", file=str(out))
    if edit_only and out.exists():
        _log(f"self-test: reusing {out}; testing only the keyframe engine", event="selftest.edit_only", file=str(out))
        return _self_test_edit(engine, out, {"file": str(out), "seconds": 0, "reused": True})
    _log("self-test 1/2: one reference still (text to image). The first run also downloads the models.", event="selftest.start", step="1/2")
    t0 = time.time()
    try:
        info = engine.run({"key": "selftest-style", "prompt": prompt, "width": r.get("width", 1664),
                           "height": r.get("height", 928)}, p.root, out)
    except Exception as e:  # noqa: BLE001
        _log(f"self-test 1/2 failed after {(time.time() - t0) / 60:.1f} min. Send me this line and the lines above "
             f"it. Error: {e}", event="selftest.failed", level="error", step="1/2", error=str(e)[-800:])
        return {"still": {"ok": False, "error": str(e)[-1500:]}}
    still = {"file": str(out), "seconds": round(time.time() - t0, 1), **(info or {})}
    _log(f"self-test 1/2 done in {still['seconds'] / 60:.1f} min: {out}", event="selftest.step_done", step="1/2", seconds=still["seconds"], file=str(out))
    return _self_test_edit(engine, out, still)


def _self_test_edit(engine, out: Path, still: dict) -> dict:
    edit_out = out.with_name(out.stem + "-edit.png")             # the keyframe engine, from that still
    _log("self-test 2/2: one keyframe edited from that still (image edit)", event="selftest.start", step="2/2")
    t0 = time.time()
    try:
        info = engine.run({"key": "selftest-edit", "prompt": "Same place and light, seen from a lower angle at dusk.",
                           "refs": [out.name], "width": 1664, "height": 928}, out.parent, edit_out)
    except Exception as e:  # noqa: BLE001 - say clearly which half failed, and why
        _log(f"self-test 2/2 failed after {(time.time() - t0) / 60:.1f} min. Stills (1/2) work; keyframes do not "
             f"yet. Send me this line and the lines above it. Error: {e}", event="selftest.failed", level="error",
             step="2/2", error=str(e)[-800:])
        return {"still": still, "edit": {"ok": False, "error": str(e)[-1500:]}}
    edit = {"file": str(edit_out), "seconds": round(time.time() - t0, 1), **(info or {})}
    _log(f"self-test 2/2 done in {edit['seconds'] / 60:.1f} min: {edit_out}", event="selftest.step_done", step="2/2", seconds=edit["seconds"], file=str(edit_out))
    _log(f"self-test passed. Look at both pictures, then send the lines above. At this speed 24 stills take "
         f"about {24 * (still['seconds'] or edit['seconds']) / 3600:.1f} h (the models are downloaded now, so later stills are faster "
         "than the first).", event="selftest.passed", est_hours_24=round(24 * (still["seconds"] or edit["seconds"]) / 3600, 1))
    return {"still": still, "edit": edit}


def _about_this_mac(folder: Path) -> None:
    def sh(*a):
        try:
            return subprocess.run(a, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    chip = sh("sysctl", "-n", "machdep.cpu.brand_string") or "?"
    mem = sh("sysctl", "-n", "hw.memsize")
    ram = f"{int(mem) / 2**30:.0f} GB" if mem.isdigit() else "?"
    free = shutil.disk_usage(Path.home()).free / 1e9
    ver = ""
    try:
        from importlib.metadata import version  # noqa: PLC0415
        ver = version("mflux")
    except Exception:  # noqa: BLE001
        ver = "not installed"
    _log(f"this Mac: {chip}, {ram} memory, macOS {sh('sw_vers', '-productVersion') or '?'}, "
         f"{free:.0f} GB free disk, mflux {ver}; pictures go to {folder}", event="mac.info", chip=chip, ram=ram, free_gb=round(free), mflux=ver)
    if free < 45:
        _log(f"warning: {free:.0f} GB free; the models need about 40 GB", event="disk.low", level="warn", free_gb=round(free))
