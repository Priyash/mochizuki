"""One batch's life: reserve its jobs and pack them (build), start it on a provider (submit), bring back
what it made (collect), or write it off (discard, cancel_run).

Provider rules (budgets, names, timeouts, notebook files) live in the provider (engine/providers);
this module holds none, so every provider goes through the same steps.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from .. import steps
from ..errors import BatchError, ProviderError
from ..ledger import MISSING, Ledger
from ..project import Project
from ..providers.base import WORKER, Provider, Status

STAGES = ("refs", "keyframes", "video", "upscale")
PLANNERS = {"refs": steps.refs, "keyframes": steps.keyframes, "video": steps.video, "upscale": steps.upscale}
COLLECT_TRIES = 3          # downloads of a finished run before its batch is closed without the output
GONE_WORDS = ("404", "not found")   # a download error that means the run's output no longer exists
QUOTA_SANITY = 0.5         # submit only if the provider says at least this share of the batch's time is left

# The layout of one batch's folder, media/batches/<id>/
BATCHES_DIR = "batches"
DATASET_DIR = "dataset"    # the inputs, the manifest and the worker
KERNEL_DIR = "kernel"      # what starts the worker (Kaggle's notebook)
DOWNLOAD_DIR = "download"  # what the run made
MANIFEST_FILE = "manifest.json"
RESULTS_FILE = "results.json"


def batch_dir(p: Project, batch_id: str) -> Path:
    return p.media / BATCHES_DIR / batch_id


def packed_here(p: Project, batch_id: str) -> bool:
    """The batch was packed on this machine (a GitHub run starts with an empty media folder)."""
    return (batch_dir(p, batch_id) / DATASET_DIR / MANIFEST_FILE).exists()


def build(p: Project, led: Ledger, stage: str, chapter: int, only: str | None = None, *,
          provider: Provider, engine_set: str = "gpu") -> dict:
    """Reserve the next batch of jobs for `provider` and pack everything one GPU session needs."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if led.active_batch(provider.name):
        raise BatchError(f"batch {led.active_batch(provider.name)['id']} is still open; run `collect` first")
    provider.refs(p)                       # fails now (no Kaggle username, say), before any job is reserved
    budget, setup = provider.budget(p, led)
    batch_id = time.strftime("b%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    ex = steps.BatchExecutor(batch_id, budget)
    work = batch_dir(p, batch_id)
    try:
        summary = PLANNERS[stage](p, led, ex, chapter, only)
        if not ex.specs:
            return {"batch": None, "stage": stage, "summary": summary}
        jobs = pack(p, ex.specs, work, {
            "batch_id": batch_id, "stage": stage, "chapter": chapter, "backend": provider.name,
            "deadline_seconds": budget + setup - provider.deadline_margin_s})
        provider.package(p, work, batch_id, engine_set)
        led.create_batch(batch_id, stage, chapter, ex.est + setup, backend=provider.name)
    except BaseException:
        discard(led, batch_id, "could not be packed")   # jobs reserved for a batch that was never made are freed
        shutil.rmtree(work, ignore_errors=True)
        raise
    return {"batch": batch_id, "backend": provider.name, "stage": stage, "jobs": len(jobs), "deferred": ex.deferred,
            "est_hours": round((ex.est + setup) / 3600, 2), "summary": summary}


def pack(p: Project, specs: list[dict], work: Path, head: dict) -> list[dict]:
    """Everything one GPU session needs, in work/dataset: every input file flattened into one folder
    (renamed so names cannot clash), the manifest and the worker. Providers add their own files."""
    data = work / DATASET_DIR
    data.mkdir(parents=True)
    (work / KERNEL_DIR).mkdir(parents=True)
    names: dict[str, str] = {}

    def ship(path: str) -> str:
        src = p.fetch(path)
        if str(src) not in names:
            if not src.exists():
                raise BatchError(f"input {src} is missing")
            name = f"{len(names):04d}_{src.name}"
            shutil.copyfile(src, data / name)
            names[str(src)] = name
        return names[str(src)]

    jobs = []
    for spec in specs:
        job = dict(spec)
        if spec.get("refs"):
            job["refs"] = [ship(r) for r in spec["refs"]]
        for field in ("image", "source"):
            if spec.get(field):
                job[field] = ship(spec[field])
        jobs.append(job)
    manifest = {**head, "engines": p.config["engines"], "jobs": jobs}
    (data / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.copyfile(WORKER, data / WORKER.name)
    return jobs


def discard(led: Ledger, batch_id: str, why: str) -> None:
    """Drop a batch that was never run, or is written off. Its jobs are freed and do not count as failures."""
    for job in led.batch_jobs(batch_id):
        if job["status"] == "pending":
            led.finish(job["key"], None, ok=False, note="not reached")
    led.set_batch(batch_id, status="failed", note=why, finished_at=time.time())


def submit(p: Project, led: Ledger, provider: Provider, batch_id: str) -> dict:
    """Upload a packed batch's inputs to its provider and start the run."""
    b = led.batch(batch_id)
    if not b or b["status"] != "built":
        raise BatchError(f"batch {batch_id} is not waiting to be submitted")
    if b["stage"] == "keyframes" and not p.references():
        discard(led, batch_id, "no reference stills")
        raise BatchError(f"batch dropped: no reference stills. Add at least {p.rel(p.ref_path('style'))}")
    dataset_ref, _ = provider.refs(p)
    work = batch_dir(p, batch_id)
    left = provider.gpu_hours_left()
    if left is not None and left * 3600 < b["est_seconds"] * QUOTA_SANITY:
        raise ProviderError(f"{provider.name.title()} says {left:.1f} GPU h left this week; "
                            f"batch needs ~{b['est_seconds'] / 3600:.1f} h")
    provider.publish_dataset(work / DATASET_DIR, dataset_ref, batch_id)
    provider.wait_dataset_ready(dataset_ref)
    handle = provider.push(work / KERNEL_DIR, provider.accelerator_for(p), provider.run_timeout(p, b["est_seconds"]))
    led.set_batch(batch_id, status="submitted", submitted_at=time.time(),
                  handle=handle if isinstance(handle, str) else None)
    return {"batch": batch_id, "backend": b["backend"], "submitted": True}


def collect(p: Project, led: Ledger, provider: Provider, batch_id: str) -> dict:
    """If the run has finished, download it and record every job. Safe to call repeatedly: a failed
    download keeps the batch open and is tried again, up to COLLECT_TRIES times."""
    b = led.batch(batch_id)
    if not b or b["status"] != "submitted":
        raise BatchError(f"batch {batch_id} is not running")
    run_ref = provider.refs(p)[1]
    st = provider.status(run_ref, handle=b["handle"])
    if st in (Status.QUEUED, Status.RUNNING, Status.UNKNOWN):
        return {"batch": batch_id, "backend": b["backend"], "status": st}
    work = batch_dir(p, batch_id)
    try:
        provider.output(run_ref, work / DOWNLOAD_DIR, handle=b["handle"])
        found = list((work / DOWNLOAD_DIR).rglob(RESULTS_FILE))
        results = json.loads(found[0].read_text(encoding="utf-8")) if found else None
    except Exception as e:  # noqa: BLE001 - the run's output may still be there: try again next run
        tries = (b["collect_tries"] or 0) + 1
        gone = any(w in str(e).lower() for w in GONE_WORDS)
        if not gone and tries < COLLECT_TRIES:
            led.set_batch(batch_id, collect_tries=tries)
            return {"batch": batch_id, "backend": b["backend"], "status": Status.COLLECTING,
                    "error": f"download failed (try {tries} of {COLLECT_TRIES}); trying again next run: {e}"[:400]}
        found, results = [], {"jobs": {}, "error": f"download failed {tries} times: {e}"[:300]}
    if results is None:
        results = {"jobs": {}, "error": f"no {RESULTS_FILE}"}
    stale = results.get("batch_id") not in (None, batch_id) or "stale" in str(results.get("error") or "")
    outdir = found[0].parent if found else work / DOWNLOAD_DIR
    made = failed = left = 0
    for job in led.batch_jobs(batch_id):
        if job["status"] != "pending":
            continue
        r = results.get("jobs", {}).get(job["key"]) if not stale else None
        if r and r.get("ok") and (outdir / r["file"]).exists():
            dest = steps.out_path(p, job["shot_id"], job["kind"], job["key"], Path(r["file"]).suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(outdir / r["file"]), dest)
            led.finish(job["key"], p.rel(dest), gpu_seconds=r.get("gpu_seconds"))
            made += 1
        elif r and r.get("ok"):                 # made, but its file did not come back: not the shot's fault
            led.finish(job["key"], None, ok=False, gpu_seconds=r.get("gpu_seconds"), note=MISSING)
            left += 1
        elif r:
            led.finish(job["key"], None, ok=False, gpu_seconds=r.get("gpu_seconds"),
                       note=("oom: " if r.get("oom") else "") + str(r.get("error"))[:180])
            failed += 1
        else:
            led.finish(job["key"], None, ok=False, note="stale inputs" if stale else "not reached")
            left += 1
    elapsed = results.get("elapsed")
    led.set_batch(batch_id, status="done" if st == Status.COMPLETE else "failed", finished_at=time.time(),
                  gpu_seconds=results.get("vm_seconds") if results.get("vm_seconds") is not None
                  else (elapsed + 300) if elapsed else None,           # + pip install before the worker
                  note=(results.get("error") or st)[:200] if (results.get("error") or st != Status.COMPLETE) else None)
    return {"batch": batch_id, "backend": b["backend"], "status": st, "made": made, "failed": failed,
            "not_reached": left, "stale": stale, "error": results.get("error")}


def cancel_run(p: Project, provider: Provider, batch) -> tuple[bool, str]:
    """Stop a submitted run that is being written off, so it cannot start later unwatched and spend
    GPU time nobody collects. (True, "") if it was stopped, else (False, why)."""
    try:
        provider.cancel(provider.refs(p)[1], handle=batch["handle"])
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)[:200]
