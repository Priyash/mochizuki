"""The Mac worker loop: claim a batch from the `batches` release, render it, upload the results.

    worker_loop  wait for batches meant for the Mac, one at a time
    take_one     claim the oldest unclaimed batch, render it with heartbeats, upload results
    run_batch    render every job of one batch into out/ with results.json (same shape as Kaggle's)
"""
from __future__ import annotations

import json
import shutil
import socket
import tarfile
import threading
import time
from pathlib import Path

from ..release import Release, names
from .common import log
from .config import NAME


def _suffix(engine, job: dict) -> str:
    s = getattr(engine, "suffix", None)
    return s(job) if s else ".png"


def run_batch(manifest: dict, inputs: Path, out: Path, engine, beat=None) -> dict:
    """Render every job; a failed job is recorded and the batch goes on. Stops starting new jobs
    at the batch's deadline (the rest go to the next batch)."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    deadline = t0 + float(manifest.get("deadline_seconds", 3 * 3600))
    res = {"batch_id": manifest["batch_id"], "jobs": {}, "groups": {NAME: {"host": socket.gethostname()}},
           "error": None}
    jobs = manifest["jobs"]
    log(f"batch {manifest['batch_id']}: {len(jobs)} job(s), stage {manifest.get('stage', '?')}, "
        f"time limit {(deadline - t0) / 3600:.1f} h", event="batch.start", batch=manifest["batch_id"], jobs=len(jobs),
        stage=manifest.get("stage"))
    for i, job in enumerate(jobs):
        if time.time() + float(job.get("est_seconds", 0)) > deadline:
            log(f"time limit reached; {len(jobs) - i} job(s) go to the next batch", event="batch.time_limit",
                level="warn", left=len(jobs) - i)
            break
        if i:
            per = (time.time() - t0) / i
            log(f"[{i + 1}/{len(jobs)}] about {per * (len(jobs) - i) / 60:.0f} min left for this batch",
                event="batch.progress", step=f"{i + 1}/{len(jobs)}", eta_min=round(per * (len(jobs) - i) / 60))
        target = out / f"{job['key']}{_suffix(engine, job)}"
        j0 = time.time()
        try:
            info = engine.run(job, inputs, target)
            res["jobs"][job["key"]] = {"ok": True, "file": target.name, "gpu_seconds": time.time() - j0, **(info or {})}
            log(f"{job['key']} ok in {time.time() - j0:.0f} s [{i + 1}/{len(jobs)}]", event="job.ok", key=job["key"],
                seconds=round(time.time() - j0), step=f"{i + 1}/{len(jobs)}")
        except Exception as e:  # noqa: BLE001 - one bad job must not stop the batch
            res["jobs"][job["key"]] = {"ok": False, "error": repr(e)[:800], "gpu_seconds": time.time() - j0}
            log(f"{job['key']} failed: {e!r}"[:400], event="job.failed", level="error", key=job["key"],
                seconds=round(time.time() - j0))
        if beat:
            beat(i + 1, len(jobs))
    res["elapsed"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


class _Heartbeat:
    """Uploads <name>-<id>.progress.json now, on every job, and every few minutes in between."""

    def __init__(self, rel: Release, folder: Path, name: str, total: int, minutes: float):
        self.rel, self.file, self.minutes = rel, folder / name, minutes
        self.progress = {"done": 0, "total": total}
        self.stop = threading.Event()

    def send(self) -> None:
        self.file.write_text(json.dumps({**self.progress, "host": socket.gethostname(), "at": time.time()}))
        try:
            self.rel.upload(self.file)
        except Exception as e:  # noqa: BLE001 - a missed heartbeat is not fatal
            log(f"heartbeat failed: {e}", event="heartbeat.failed", level="warn")

    def __call__(self, done: int, total: int) -> None:
        self.progress.update(done=done, total=total)
        self.send()

    def __enter__(self):
        self.send()
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self.stop.wait(self.minutes * 60):
            self.send()

    def __exit__(self, *exc):
        self.stop.set()
        return False


def _waiting(assets: dict) -> list[dict]:
    return sorted((x for x in assets.values() if x["name"].startswith(f"{NAME}-") and x["name"].endswith(".tar.gz")
                   and not x["name"].endswith(".results.tar.gz")), key=lambda x: x.get("created_at", ""))


def take_one(rel: Release, work: Path, engine, cfg: dict) -> str | None:
    """Claim the oldest unclaimed batch, render it, upload the results. The batch id, or None."""
    a = rel.assets()
    for asset in _waiting(a):
        batch_id = asset["name"][len(NAME) + 1:-7]
        n = names(batch_id, NAME)
        if n["claim"] in a or n["results"] in a:
            continue
        d = work / batch_id
        shutil.rmtree(d, ignore_errors=True)
        (d / "in").mkdir(parents=True)
        claim = d / n["claim"]
        claim.write_text(json.dumps({"host": socket.gethostname(), "at": time.time()}))
        rel.upload(claim)
        log(f"claimed {batch_id}; downloading its inputs", event="batch.claimed", batch=batch_id)
        rel.download(asset, d / n["inputs"])
        with tarfile.open(d / n["inputs"]) as t:
            try:
                t.extractall(d / "in", filter="data")
            except TypeError:
                t.extractall(d / "in")
        manifest = json.loads((d / "in" / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("batch_id") != batch_id:
            raise RuntimeError(f"{n['inputs']} holds batch {manifest.get('batch_id')}")
        with _Heartbeat(rel, d, n["progress"], len(manifest["jobs"]), float(cfg["heartbeat_minutes"])) as beat:
            res = run_batch(manifest, d / "in", d / "out", engine, beat)
        tgz = d / n["results"]
        with tarfile.open(tgz, "w:gz") as t:
            t.add(d / "out", arcname="out")
        log(f"{batch_id}: uploading results ({tgz.stat().st_size / 1e6:.0f} MB)", event="results.upload",
            batch=batch_id, mb=round(tgz.stat().st_size / 1e6))
        rel.upload(tgz)
        ok = sum(1 for j in res["jobs"].values() if j.get("ok"))
        log(f"{batch_id}: {ok}/{len(manifest['jobs'])} made, results uploaded", event="batch.done", batch=batch_id,
            made=ok, jobs=len(manifest["jobs"]))
        return batch_id
    return None


def worker_loop(rel: Release, work: Path, engine, cfg: dict, once: bool = False, sleep=time.sleep) -> int:
    """Take batches until stopped (Ctrl-C), or until nothing waits when `once`. Batches done."""
    work.mkdir(parents=True, exist_ok=True)
    rel.ensure()
    done = idle = 0
    while True:
        if take_one(rel, work, engine, cfg):
            done, idle = done + 1, 0
            continue
        if once:
            return done
        idle += 1
        if idle == 1 or idle % 10 == 0:
            log(f"no batch for this Mac yet; checking every {float(cfg['poll_seconds']):.0f} s "
                f"({done} batch(es) done so far)", event="worker.idle", batches_done=done)
        sleep(float(cfg["poll_seconds"]))
