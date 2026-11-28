"""The producer loop: what a chapter needs next from the GPU, and one step on every provider. Each step
collects what finished, hands over what is stuck, and starts what is next. Cron or a GitHub
workflow calls it; it stops whenever a person is needed.
"""
from __future__ import annotations

import subprocess
import time
from enum import StrEnum

from .. import steps
from ..errors import BudgetUsedUp
from ..generators import needs_for
from ..ledger import NOT_A_FAILURE, Ledger, not_a_failure
from ..project import REPO, Project
from ..providers.base import IN_FLIGHT, Provider, Status
from ..stages import next_gpu_stage
from .batches import build, cancel_run, collect, discard, packed_here, submit

STEPS_PER_RUN = 4          # collect, then start the next, in the same run
BROKEN_AFTER_RUNS = 2      # batches in a row that made nothing (machine errors) before a stage rests
STUCK_PREFIX = "stuck in "


class Action(StrEnum):
    """What one step did. A StrEnum: it equals its value: Action.IDLE == "idle"."""
    IDLE = "idle"                         # nothing to run, or waiting for a person
    WAITING = "waiting"                   # a run is in flight
    COLLECTED = "collected"
    SUBMITTED = "submitted"
    BUILT_AND_SUBMITTED = "built_and_submitted"
    HANDED_OVER = "handed_over"           # stuck in a provider's queue: written off for another provider
    RESTING = "resting"                   # budget used up, or the stage's engine is broken
    CLEANED_UP = "cleaned_up"
    ERROR = "error"
    SKIPPED = "skipped"                   # no provider can run at all



def stuck_note(name: str) -> str:
    return f"{STUCK_PREFIX}{name.title()}'s queue"


def next_stage(p: Project, led: Ledger, chapter: int) -> str | None:
    """What the chapter needs next from the GPU, or None if it is waiting on a person (engine/stages)."""
    return next_gpu_stage(p, led, chapter)


def tick(p: Project, led: Ledger, provider: Provider, chapter: int, engine_set: str = "gpu",
         can_hand_over=False) -> dict:
    """One step on one provider: collect its finished run, or start the next one."""
    name = provider.name
    active = led.active_batch(name)
    if active and active["status"] == "built" and not packed_here(p, active["id"]):
        discard(led, active["id"], "packed on another machine")     # e.g. an earlier GitHub run; repack
        active = None
    if active:
        if active["status"] == "built":
            try:
                return {"action": Action.SUBMITTED, **submit(p, led, provider, active["id"])}
            except Exception:
                discard(led, active["id"], f"could not be submitted to {name}")
                raise
        res = collect(p, led, provider, active["id"])
        patience = provider.queue_patience_s(p)
        if can_hand_over and provider.hands_over_when_queued and patience is not None and \
                res["status"] == Status.QUEUED and \
                time.time() - (active["submitted_at"] or time.time()) > patience and \
                (can_hand_over(active["stage"]) if callable(can_hand_over) else True):
            cancelled, why = cancel_run(p, provider, active)
            discard(
                led,
                active["id"],
                f"{stuck_note(name)}; handed to the other backend; "
                + (
                    "the queued run was cancelled"
                    if cancelled
                    else f"the queued run could not be cancelled and may still start: {why}"
                ),
            )
            return {"action": Action.HANDED_OVER, "backend": name, "batch": active["id"], "cancelled": cancelled,
                    "reason": f"queued on {name.title()} for over {patience / 3600:.0f} h; "
                              + ("cancelled there" if cancelled else f"could not cancel it there ({why})")}
        if res["status"] in IN_FLIGHT:
            return {"action": Action.WAITING, **res}
        return {"action": Action.COLLECTED, **res}
    stage = next_stage(p, led, chapter)
    allowed = provider.stages
    if stage and allowed and stage not in allowed and not (provider.take_over and queue_stuck(led)):
        return {"action": Action.IDLE, "backend": name,
                "reason": f"{stage} is not run on {name}: config {name}.stages only takes {', '.join(allowed)}"}
    missing = needs_for(stage) - provider.capabilities if stage else frozenset()
    if missing:
        return {"action": Action.IDLE, "backend": name,
                "reason": f"{stage} needs {', '.join(sorted(missing))}, which {name}'s GPUs do not have"}
    if stage is None:
        queue = steps.review_queue(p, led, chapter)
        return {"action": Action.IDLE, "reason": "waiting for a person", "to_review": len(queue),
                "states": steps.status(p, led, chapter)["states"]}
    broken = engine_broken(led, name, stage)
    if broken:
        return {"action": Action.RESTING, "backend": name, "reason": broken}
    try:
        built = build(p, led, stage, chapter, provider=provider, engine_set=engine_set)
    except BudgetUsedUp as e:
        return {"action": Action.RESTING, "backend": name, "reason": str(e)}
    if not built["batch"]:
        return {"action": Action.IDLE, "reason": f"nothing to run for {stage}", **built}
    try:
        return {"action": Action.BUILT_AND_SUBMITTED, **built, **submit(p, led, provider, built["batch"])}
    except Exception:
        discard(led, built["batch"], f"could not be submitted to {name}")   # free the jobs for the other provider
        raise


def code_time() -> float:
    """When the checked-out code was committed (0 if unknown)."""
    try:
        r = subprocess.run(["git", "-C", str(REPO), "log", "-1", "--format=%ct"], capture_output=True, text=True)
        return float(r.stdout.strip() or 0)
    except (OSError, ValueError):
        return 0.0


def engine_broken(led: Ledger, backend: str, stage: str, runs: int = BROKEN_AFTER_RUNS) -> str | None:
    """If the last `runs` batches of this stage on this provider made nothing at all because of
    machine errors (engine would not load, out of memory, CUDA), say so: running again would only
    spend GPU time. A code or config change is needed; any new commit earns a fresh try."""
    rows = led.db.execute("SELECT id FROM batches WHERE backend=? AND stage=? AND status IN ('done','failed')"
                          " AND created_at > ? ORDER BY created_at DESC LIMIT ?",
                          (backend, stage, code_time(), runs)).fetchall()
    if len(rows) < runs:
        return None
    first = None
    for r in rows:
        jobs = led.batch_jobs(r["id"])
        tried = [j for j in jobs if (j["note"] or "") not in NOT_A_FAILURE]
        if not tried or any(j["status"] == "done" for j in jobs) or \
                not all(not_a_failure(j["note"]) for j in tried):
            return None
        first = first or tried[0]["note"]
    return (f"the {stage} engine made nothing in the last {runs} {backend} batches (machine errors); "
            f"stopped until the next code change: {(first or '')[:300]}")


def someone_else_can_take(p: Project, led: Ledger, providers: list[Provider], me: Provider, stage: str) -> bool:
    """Another provider would take this stage now: allowed to (its stages, or it takes over from a
    stuck queue) and within its budget (Colab: the agreed unit cap). Otherwise a batch stuck in a
    queue keeps its place instead of being dropped for nothing."""
    for other in providers:
        if other is me:
            continue
        if other.stages and stage not in other.stages and not other.take_over:
            continue
        if needs_for(stage) - other.capabilities:
            continue
        try:
            other.budget(p, led)
            return True
        except Exception:  # noqa: BLE001 - resting, no credit, no stock: it can't take it
            continue
    return False


def queue_stuck(led: Ledger) -> bool:
    """A provider's latest batch was handed over because it sat in that provider's queue too long."""
    rows = led.db.execute("SELECT b.status, b.note FROM batches b WHERE b.created_at = "
                          "(SELECT MAX(created_at) FROM batches WHERE backend = b.backend)").fetchall()
    return any(r["status"] == "failed" and (r["note"] or "").startswith(STUCK_PREFIX) for r in rows)


def tick_all(p: Project, led: Ledger, providers: list[Provider], chapter: int, engine_set: str = "gpu") -> list[dict]:
    """One step on every provider: each collects its finished run, then starts the next.

    Each provider holds at most one open batch. Jobs are reserved when a batch is packed, so two
    providers never make the same job; the second one takes what the first left.
    """
    out = []
    for b in led.active_batches():                      # repack what an earlier run packed
        if b and b["status"] == "built" and not packed_here(p, b["id"]):
            discard(led, b["id"], "packed on another machine")
    for prov in providers:                              # paid machines nobody tracks are stopped first
        try:
            n = prov.cleanup(led)
            if n:
                out.append({"action": Action.CLEANED_UP, "backend": prov.name, "pods_deleted": n})
        except Exception as e:  # noqa: BLE001
            out.append({"action": Action.ERROR, "backend": prov.name, "error": f"cleanup: {e}"[:300]})
    for prov in providers:
        for _ in range(STEPS_PER_RUN):
            try:
                r = tick(p, led, prov, chapter, engine_set=engine_set,
                         can_hand_over=lambda stage, me=prov: someone_else_can_take(p, led, providers, me, stage))
            except Exception as e:  # noqa: BLE001 - one provider failing must not stop the other
                r = {"action": Action.ERROR, "error": f"{type(e).__name__}: {e}"[:500]}
            r.setdefault("backend", prov.name)
            out.append(r)
            if r["action"] != Action.COLLECTED:
                break                       # after a handover Kaggle waits a run, so the other provider takes it
    return out
