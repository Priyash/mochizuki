"""Model files on the Mac: downloading them once with progress, checking them, finding them.

Two kinds of model are used:
  - a model the tool fetches by name (mflux's `--model z-image-turbo`): `hf_repo` in the engine
    config names its Hugging Face repo, so its first download shows up in the logs;
  - a model kept in a folder of our own (`download` in the engine config: repo, folder, the files
    that must be there): fetched file by file into ~/mochizuki-mac/models, checked, and put into
    the command as {model_dir}.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from .common import log
from .sysinfo import dir_gb

WEIGHTS = (".safetensors", ".npz", ".gguf")
QWEN_LAYOUT = ("transformer/", "text_encoder/", "vae/", "tokenizer/")   # what mflux loads from a local folder


def hf_cache_dir(repo: str) -> Path:
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return Path(os.environ.get("HF_HUB_CACHE", home / "hub")) / f"models--{repo.replace('/', '--')}"


def is_hf_repo(x: str) -> bool:
    return x.count("/") == 1 and not x.startswith((".", "/", "~")) and not Path(x).exists()


def weights(folder: Path) -> list[Path]:
    return [f for f in Path(folder).rglob("*") if f.is_file() and f.suffix in WEIGHTS]


def missing_parts(target: Path, require=QWEN_LAYOUT) -> list[str]:
    """What is missing from a model folder. `require` lists folders ('vae/': must hold weights,
    'tokenizer/': must not be empty) and files ('model.safetensors')."""
    gone = []
    for r in require:
        p = target / r.rstrip("/")
        if r.endswith("/"):
            ok = (bool(weights(p)) if r.rstrip("/") != "tokenizer" else p.is_dir() and any(p.iterdir())) \
                if p.is_dir() else False
        else:
            ok = p.is_file() and p.stat().st_size > 0
        if not ok:
            gone.append(r.rstrip("/"))
    return gone


class watch_download:
    """While a download runs: Hugging Face's own bars off, and one structured line every
    `every` seconds with GB done, speed and time left (measured from the folder's size)."""

    def __init__(self, folder: Path, total_gb: float | None, label: str, every: float = 30.0):
        self.folder, self.total, self.label, self.every = Path(folder), total_gb, label, every
        self.stop = threading.Event()

    def __enter__(self):
        self.old = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        try:
            from huggingface_hub.utils import disable_progress_bars  # noqa: PLC0415
            disable_progress_bars()
        except Exception:  # noqa: BLE001
            pass
        self.start_gb, self.t0 = dir_gb(self.folder), time.time()
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        last_gb, last_t = self.start_gb, self.t0
        while not self.stop.wait(self.every):
            gb, now = dir_gb(self.folder), time.time()
            rate = max(0.0, gb - last_gb) * 1000 / max(1.0, now - last_t)          # MB/s over the last period
            last_gb, last_t = gb, now
            done = gb - self.start_gb
            left = (f", about {(self.total - done) * 1000 / rate / 60:.0f} min left"
                    if self.total and rate > 0.1 and self.total > done else "")
            of = f" of {self.total:.1f}" if self.total else ""
            log(f"{self.label}: {done:.1f}{of} GB, {rate:.0f} MB/s{left}", event="model.download.progress",
                gb_done=round(done, 1), gb_total=self.total, mb_s=round(rate))

    def __exit__(self, *exc):
        self.stop.set()
        if self.old is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = self.old
        return False


def _hub(list_files, download):
    if list_files is None or download is None:
        from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415 - comes with mflux
        list_files = list_files or (lambda repo: [(f.path, f.size or 0) for f in HfApi().list_repo_tree(
            repo, recursive=True) if getattr(f, "size", None) is not None])
        download = download or snapshot_download
    return list_files, download


def ensure_local(d: dict, models: Path, download=None, list_files=None) -> Path:
    """A model folder from a Hugging Face repo (and a folder in it), downloaded once into `models`,
    checked against d["require"] (default: mflux's Qwen layout) before it is marked complete."""
    folder = d.get("folder", "")
    require = tuple(d.get("require", QWEN_LAYOUT))
    local = models / d["repo"].replace("/", "--")
    target = local / folder if folder else local
    done = target / ".complete"
    if done.exists() and missing_parts(target, require):     # marked done by an older version, but incomplete
        log(f"{d['repo']}/{folder} was marked downloaded but has no {', '.join(missing_parts(target, require))}; "
            "downloading again", event="model.redownload", level="warn", model=d["repo"], folder=folder)
        done.unlink()
    if done.exists():
        return target
    list_files, download = _hub(list_files, download)
    files = list_files(d["repo"])
    mine = [(n, sz) for n, sz in files if not folder or n.startswith(folder.rstrip("/") + "/")]
    if not mine:
        tops = sorted({n.split("/")[0] for n, _ in files})
        raise RuntimeError(f"{d['repo']} has no folder '{folder}'; it has: {', '.join(tops)[:400]}")
    gb = round(sum(s for _, s in mine) / 1e9, 1)
    log(f"downloading {d['repo']}{'/' + folder if folder else ''} once: {len(mine)} files, about {gb:.0f} GB; "
        "progress every 30 s", event="model.download", model=d["repo"], folder=folder, files=len(mine), gb=gb)
    t0 = time.time()
    with watch_download(local, gb, f"{d['repo']}/{folder}"):
        download(d["repo"], local_dir=str(local), allow_patterns=[n for n, _ in mine])
    tok = d.get("tokenizer_from")                 # pre-quantized folders may leave the tokenizer out
    if tok and "tokenizer" in missing_parts(target, require):
        log(f"{d['repo']}/{folder} has no tokenizer; taking it from {tok}", event="model.tokenizer", source=tok)
        download(tok, local_dir=str(target), allow_patterns=["tokenizer/*", "added_tokens.json", "chat_template.jinja"])
    gone = missing_parts(target, require)
    if gone:
        raise RuntimeError(f"{d['repo']}/{folder}: downloaded, but {target} has no {', '.join(gone)}; "
                           "not marking it complete")
    gb = sum(f.stat().st_size for f in weights(target)) / 1e9
    log(f"downloaded {d['repo']}/{folder}: {gb:.1f} GB in {(time.time() - t0) / 60:.1f} min", event="model.downloaded",
        model=d["repo"], gb=round(gb, 1), minutes=round((time.time() - t0) / 60, 1))
    done.write_text("ok")
    return target


def resolve_cmd(e: dict, models: Path, download=None, list_files=None) -> list[str]:
    """The engine's command with its placeholders filled: {model_dir} (a local copy of e["download"],
    fetched and checked once) and {python} (this Python, for `python -m` tools)."""
    model_dir = ensure_local(e["download"], models, download, list_files) if e.get("download") else None
    out = []
    for x in e["cmd"]:
        if x == "{model_dir}":
            if model_dir is None:
                raise ValueError("the command uses {model_dir} but the engine has no 'download'")
            out.append(str(model_dir))
        elif x == "{python}":
            out.append(sys.executable)
        else:
            out.append(x)
    return out


def prefetch_model(repo: str, download=None) -> None:
    """Download a whole Hugging Face model once, with progress, so a render never waits on it silently."""
    try:
        from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415 - comes with mflux
    except ImportError:
        return
    download = download or snapshot_download
    try:
        download(repo, local_files_only=True)
        log(f"model {repo}: already on this Mac", event="model.cached", model=repo)
        return
    except Exception:  # noqa: BLE001 - not cached yet
        pass
    gb = None
    try:
        gb = round(sum(f.size or 0 for f in HfApi().model_info(repo, files_metadata=True).siblings) / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    log(f"model {repo}: downloading once{f', about {gb:.0f} GB' if gb else ''}; progress every 30 s",
        event="model.download", model=repo)
    t0 = time.time()
    with watch_download(hf_cache_dir(repo), gb, repo):
        download(repo)
    log(f"model {repo}: downloaded in {(time.time() - t0) / 60:.1f} min", event="model.downloaded", model=repo,
        minutes=round((time.time() - t0) / 60, 1))
