"""The `batches` GitHub release: how off-Kaggle workers (the Mac, the Colab runner) get work.

A batch waits as <name>-<id>.tar.gz until a worker claims it, sends heartbeats and uploads
<name>-<id>.results.tar.gz. The producer collects the results and deletes all four files:

  <name>-<id>.tar.gz            inputs (manifest.json and input images), written by the producer
  <name>-<id>.claim.json        a worker has taken it (host, time)
  <name>-<id>.progress.json     heartbeat every few minutes: jobs done / total
  <name>-<id>.results.tar.gz    out/results.json and the outputs, written by the worker

Shared by the Colab provider (engine/providers/colab) and the Mac's (engine/providers/mac.py).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from .base import Provider, Status

RELEASE = "batches"


def iso_time(ts: str | None) -> float:
    """GitHub's ISO time stamps as unix time; 0 when missing."""
    if not ts:
        return 0.0
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def names(batch_id: str, prefix: str = "mac") -> dict[str, str]:
    """The four release file names of one batch."""
    b = f"{prefix}-{batch_id}"
    return {"inputs": f"{b}.tar.gz", "claim": f"{b}.claim.json", "progress": f"{b}.progress.json",
            "results": f"{b}.results.tar.gz"}


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


class ReleaseProvider(Provider):
    """A provider whose batches wait in the `batches` release until a worker elsewhere claims them
    (the Colab runner, the Mac). Subclasses set `name` and pass their settings as `cfg`."""

    def __init__(self, p, repo: str, gh=None, fetch=None, now=time.time, *, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.rel = Release(repo, gh, fetch)
        self.now = now
        self.stages = tuple(self.cfg["stages"]) if self.cfg.get("stages") else None

    def budget(self, p, led) -> tuple[float, float]:
        return float(self.cfg["batch_hours"]) * 3600, float(self.cfg["setup_minutes"]) * 60

    def gpu_hours_left(self) -> float | None:
        return None

    def publish_dataset(self, folder: Path, ref: str, message: str) -> None:
        batch_id = message
        tgz = Path(folder).parent / names(batch_id, self.name)["inputs"]
        with tarfile.open(tgz, "w:gz") as t:
            for f in sorted(Path(folder).iterdir()):
                if f.name not in ("dataset-metadata.json", "worker.py"):
                    t.add(f, arcname=f.name)
        self.rel.ensure()
        self.rel.upload(tgz)
        self.log.append(f"{tgz.name} ({tgz.stat().st_size // 1024} KB) waits for the {self.name} worker")


    def push(self, folder: Path, accelerator: str, timeout_s: int) -> str:
        return Path(folder).parent.name                  # the handle is the batch id

    def status(self, ref: str, handle: str | None = None) -> str:
        n, a = names(handle or "", self.name), self.rel.assets()
        if not a:
            return Status.UNKNOWN                        # GitHub unreachable: ask again next run
        if n["results"] in a:
            return Status.COMPLETE
        if n["inputs"] not in a:
            self.log.append(f"{n['inputs']} is gone")
            return Status.ERROR
        if n["claim"] in a:
            beat = max(iso_time(a[n["claim"]].get("updated_at")),
                       iso_time((a.get(n["progress"]) or {}).get("updated_at")))
            if self.now() - beat > float(self.cfg["stale_minutes"]) * 60:
                self.log.append(f"the {self.name} worker on {handle} went quiet "
                                f"{(self.now() - beat) / 60:.0f} min ago; giving up on it")
                return Status.ERROR
            return Status.RUNNING
        waited = self.now() - iso_time(a[n["inputs"]].get("created_at"))
        if waited > float(self.cfg["max_wait_hours"]) * 3600:
            self.log.append(f"no {self.name} worker took {handle} in {waited / 3600:.0f} h; its jobs are freed")
            return Status.ERROR
        return Status.QUEUED

    def output(self, ref: str, dest: Path, handle: str | None = None) -> None:
        """Download and unpack the results, then delete the batch's files from the release. If the
        release can't be read or the download fails, nothing is deleted: the next run tries again."""
        n, a = names(handle or "", self.name), self.rel.assets()
        if not a:
            raise RuntimeError(f"could not read the {RELEASE} release")
        if n["results"] in a:
            tgz = Path(dest) / n["results"]
            self.rel.download(a[n["results"]], tgz)
            with tarfile.open(tgz) as t:
                try:
                    t.extractall(dest, filter="data")
                except TypeError:
                    t.extractall(dest)
        for name in n.values():
            if name in a:
                self.rel.delete(name)
