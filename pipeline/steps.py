"""The production steps. Each reads its state from the ledger and can be rerun safely.

A step decides WHAT should be made next and hands each job to an executor:
SyncExecutor runs it now (mock or paid API); BatchExecutor queues it for one Kaggle run.
"""
from __future__ import annotations

from pathlib import Path

from .ledger import BudgetExceeded, Ledger, not_a_failure
from .project import Project, Shot, job_key, ref_shot

OUT = {"ref": ("refs", ".png"), "keyframe": ("keyframes", ".png"), "variant": ("keyframes", ".png"),
       "take": ("takes", ".mp4"), "upscale": ("4k", None)}


def open_ledger(p: Project) -> Ledger:
    b = p.config["budget"]
    return Ledger(p.root / "ledger.sqlite", b["total_usd"], b["per_chapter_usd"])


def out_path(p: Project, shot_id: str, kind: str, key: str, ext: str | None) -> Path:
    folder, default_ext = OUT[kind]
    return p.media / folder / shot_id / f"{key}{ext or default_ext}"


# ---------------------------------------------------------------- executors
class SyncExecutor:
    """Run each job immediately with a provider that has run(spec, out_path)."""

    def __init__(self, provider, only_provider: str | None = None):
        self.provider = provider
        self.only_provider = only_provider     # e.g. "fal": leave Kaggle-routed takes alone

    def submit(self, p: Project, led: Ledger, s: Shot, spec: dict, cost: float) -> str:
        if self.only_provider and spec["kind"] == "take" and p.provider_of(s) != self.only_provider:
            return "elsewhere"
        if not led.begin(spec["key"], s.id, s.chapter, spec["kind"], spec["idx"], s.model, cost):
            return "skipped"
        path = out_path(p, s.id, spec["kind"], spec["key"], spec.get("ext"))
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.provider.run(spec, path)
        except Exception as e:  # noqa: BLE001 - any provider failure is recorded, not fatal
            led.finish(spec["key"], None, ok=False, note=str(e)[:200])
            print(f"  {spec['key']} failed: {e}")
            return "failed"
        led.finish(spec["key"], p.rel(path))
        return "made"


class BatchExecutor:
    """Reserve jobs for one Kaggle session, up to a GPU-time budget. Nothing runs here."""

    def __init__(self, batch_id: str, budget_seconds: float):
        self.batch_id = batch_id
        self.budget = budget_seconds
        self.est = 0.0
        self.specs: list[dict] = []
        self.deferred = 0

    def submit(self, p: Project, led: Ledger, s: Shot, spec: dict, cost: float) -> str:
        if spec["kind"] == "take" and p.provider_of(s) != "kaggle":
            return "elsewhere"          # a paid-routed shot
        if led.get(spec["key"]):
            return "skipped"
        est = p.gpu_estimate(spec["kind"], s)
        if self.specs and self.est + est > self.budget:
            self.deferred += 1
            return "deferred"           # next session
        led.begin(spec["key"], s.id, s.chapter, spec["kind"], spec["idx"], s.model, cost,
                  batch=self.batch_id)
        spec["est_seconds"] = est
        self.specs.append(spec)
        self.est += est
        return "queued"


# ---------------------------------------------------------------- state
def state(p: Project, led: Ledger, s: Shot) -> str:
    """Where a shot stands, worked out from the ledger alone."""
    pending = [j for j in led.jobs(s.id) if j["status"] == "pending"]
    if any(not j["batch"] for j in pending):
        return "pending_unknown"      # a call was cut off: resolve it before buying more
    if pending:
        return "in_batch"
    max_fail = p.config.get("kaggle", {}).get("max_failures_per_shot", 2)
    keyframes = [j for j in led.jobs(s.id, "keyframe") if j["status"] == "done"]
    if not keyframes:
        if led.failures(s.id, "keyframe") >= max_fail:
            return "needs_human"
        src = p.continuity_source(s)
        if src and not led.selected(src):
            return "waiting_anchor"    # drawn from another shot's keyframe, which is not chosen yet
        return "todo"
    if not led.selected(s.id):
        if all(j["verdict"] == "fail" for j in keyframes):
            cap = p.config["keyframes"].get("max_candidates", 0)
            return "needs_human" if len(keyframes) >= cap else "needs_keyframes"
        return "keyframes_review"
    if not led.approved(s.chapter, "board"):
        return "board_review"
    if not p.model(s.model).get("generates_video"):
        missing = []
        for slot, v in enumerate(p.variants(s)):
            found = _attempt_key(p, led, s, "variant", slot, variant_prompt(p, s, v))
            if found is None:
                return "needs_human"
            if found[2] is None or found[2]["status"] != "done":
                missing.append(slot)
        return "needs_variants" if missing else "ready_for_editor"
    takes = [j for j in led.jobs(s.id, "take") if j["status"] == "done"]
    if any(j["verdict"] == "pass" for j in takes):
        return "take_passed"
    if any(j["verdict"] is None for j in takes):
        return "take_review"
    if len(takes) >= s.takes or led.failures(s.id, "take") >= max_fail:
        return "needs_human"
    return "needs_take"


def ref_state(p: Project, led: Ledger, r: dict) -> str:
    """Where one planned reference still stands (gate 1)."""
    if p.ref_path(r["name"]).exists():
        return "in_refs"
    s = ref_shot(r["name"])
    jobs = led.jobs(s.id, "ref")
    if any(j["status"] == "pending" for j in jobs):
        return "in_batch"
    if r.get("from") and not p.ref_path(r["from"]).exists():
        return "waiting_source"        # painted as an edit of a still that is not approved yet
    if led.selected(s.id):
        return "selected"
    done = [j for j in jobs if j["status"] == "done"]
    if not done:
        max_fail = p.config.get("kaggle", {}).get("max_failures_per_shot", 2)
        return "needs_human" if led.failures(s.id, "ref") >= max_fail else "todo"
    if all(j["verdict"] == "fail" for j in done):
        cap = p.config.get("refs", {}).get("max_candidates", 8)
        return "needs_human" if len(done) >= cap else "needs_candidates"
    return "review"


def refs(p: Project, led: Ledger, ex, chapter: int | None = None, only: str | None = None) -> dict:
    """Paint candidate reference stills for every planned ref that style_bible/refs lacks."""
    n = p.config.get("refs", {}).get("candidates", 4)
    cap = p.config.get("refs", {}).get("max_candidates", 2 * n)
    style = p.root / "style_bible" / "style.txt"
    style_text = style.read_text(encoding="utf-8").strip() if style.exists() else ""
    made, deferred = 0, 0
    for r in p.refs_missing(chapter):
        if only and r["name"] != only:
            continue
        st = ref_state(p, led, r)
        if st not in ("todo", "needs_candidates"):
            continue
        s = ref_shot(r["name"])
        done = [j for j in led.jobs(s.id, "ref") if j["status"] == "done"]
        want = n if st == "todo" else min(n, cap - len(done))
        start = len(led.jobs(s.id, "ref"))
        prompt = "\n".join(x for x in (style_text, r["prompt"]) if x)
        for i in range(start, start + want):
            extra = {"refs": [p.rel(p.ref_path(r["from"]))]} if r.get("from") else {}
            spec = _spec(p, s, "ref", i, prompt, name=r["name"],
                         width=r.get("width", 1664), height=r.get("height", 928), **extra)
            res = ex.submit(p, led, s, spec, 0.0)
            made += res in ("made", "queued")
            deferred += res == "deferred"
    return {"made": made, "deferred": deferred}


def upscale_sources(p: Project, led: Ledger, s: Shot) -> list:
    """What goes to 4K for a shot: its passing take, or its keyframe and mouth shapes."""
    if p.model(s.model).get("generates_video"):
        passed = [j for j in led.jobs(s.id, "take") if j["status"] == "done" and j["verdict"] == "pass"]
        return passed[-1:]
    sel = led.selected(s.id)
    variants = [j for j in led.jobs(s.id, "variant") if j["status"] == "done"]
    return ([sel] if sel else []) + variants


def variant_prompt(p: Project, s: Shot, v: str) -> str:
    return p.prompt(s, extra=f"Same frame and same character, {v}. Change nothing else.")


def _attempt_key(p: Project, led: Ledger, s: Shot, kind: str, slot: int, prompt: str):
    """Key for the next try at one slot (a mouth shape, a 4K source): retries get a new key.

    Returns (key, idx, job), where job is the existing done or pending job if there is one,
    or None once the slot has really failed max_failures_per_shot times.
    Tries cut short by the session clock or stale inputs do not count.
    """
    max_fail = p.config.get("kaggle", {}).get("max_failures_per_shot", 2)
    real = 0
    for attempt in range(10):
        idx = slot * 10 + attempt
        key = job_key(s.id, kind, idx, prompt)
        job = led.get(key)
        if job is None or job["status"] in ("done", "pending"):
            return (key, idx, job) if real < max_fail else None
        real += not not_a_failure(job["note"])
    return None


def upscale_state(p: Project, led: Ledger, s: Shot) -> str:
    st = state(p, led, s)
    if st not in ("take_passed", "ready_for_editor"):
        return "not_ready"
    result = "done"
    for slot, src in enumerate(upscale_sources(p, led, s)):
        found = _attempt_key(p, led, s, "upscale", slot, src["key"])
        if found is None:
            return "needs_human"
        _, _, job = found
        if job is None:
            result = "needed"
        elif job["status"] == "pending" and result == "done":
            result = "in_batch"
    return result


# ---------------------------------------------------------------- the steps
def _spec(p: Project, s: Shot, kind: str, idx: int, prompt: str, **extra) -> dict:
    return {"key": job_key(s.id, kind, idx, prompt), "shot_id": s.id, "chapter": s.chapter,
            "kind": kind, "idx": idx, "model": s.model, "prompt": prompt,
            "seconds": s.seconds, **extra}


def continuity_refs(p: Project, led: Ledger, s: Shot, limit: int = 3) -> list[str]:
    """Reference images for a keyframe: the continuity source's chosen keyframe first (it
    already carries the look, so style.png gives way), then the shot's characters and props."""
    refs = [str(r) for r in p.refs_for(s, limit)]
    src = p.continuity_source(s)
    sel = led.selected(src) if src else None
    if not sel:
        return refs
    subjects = [r for r in refs if not Path(r).stem.lower().startswith("style")]
    return [sel["path"]] + subjects[: limit - 1]


def keyframes(p: Project, led: Ledger, ex, chapter: int, only: str | None = None) -> dict:
    """Make keyframe candidates for shots that have none selected.

    A second batch is made only after every candidate so far has been judged and
    failed, up to max_candidates. Good first batches never cost a second.
    """
    if not p.references():
        raise ValueError("no reference stills: add at least style_bible/refs/style.png (gate 1)")
    n = p.config["keyframes"]["candidates"]
    cap = p.config["keyframes"].get("max_candidates", n)
    limit = p.engine("keyframe").get("max_refs", 3)
    made, waiting, human, deferred = 0, [], [], 0
    for s in p.chapter(chapter):
        if only and s.id != only:
            continue
        st = state(p, led, s)
        if led.selected(s.id) or st in ("pending_unknown", "in_batch", "needs_human", "waiting_anchor"):
            if st == "needs_human":
                human.append(s.id)
            continue
        done = [j for j in led.jobs(s.id, "keyframe") if j["status"] == "done"]
        unfinished_batch = len(done) % n != 0
        if done and not unfinished_batch:
            if not all(j["verdict"] == "fail" for j in done):
                waiting.append(s.id)          # judge or select these first
                continue
            if len(done) >= cap:
                human.append(s.id)            # every candidate failed and the cap is used
                continue
        want = (n - len(done) % n) if unfinished_batch else min(n, cap - len(done))
        start = len(led.jobs(s.id, "keyframe"))
        prompt = p.prompt(s)
        refs = continuity_refs(p, led, s, limit)
        for i in range(start, start + want):
            r = ex.submit(p, led, s, _spec(p, s, "keyframe", i, prompt, refs=refs), p.keyframe_cost())
            made += r in ("made", "queued")
            deferred += r == "deferred"
    return {"made": made, "waiting_for_verdict": waiting, "needs_human": human, "deferred": deferred}


def video(p: Project, led: Ledger, ex, chapter: int, only: str | None = None) -> dict:
    """One step forward per shot: ONE more take, or mouth variants for a lip-flap still.

    A second take is only made after the first has been judged and failed,
    so a shot whose first take passes never uses its whole cap.
    """
    if not led.approved(chapter, "board"):
        raise PermissionError(f"chapter {chapter} storyboard is not approved (gate 2)")
    made, waiting, human, cut_off, deferred = 0, [], [], [], 0
    for s in p.chapter(chapter):
        if only and s.id != only:
            continue
        st = state(p, led, s)
        sel = led.selected(s.id)
        if st == "needs_variants":
            for slot, v in enumerate(p.variants(s)):
                prompt = variant_prompt(p, s, v)
                found = _attempt_key(p, led, s, "variant", slot, prompt)
                if found is None or found[2] is not None:
                    continue                    # done, in a batch, or out of tries
                key, idx, _ = found
                spec = _spec(p, s, "variant", idx, prompt, image=sel["path"], variant=v)
                spec["key"] = key
                r = ex.submit(p, led, s, spec, p.keyframe_cost())
                made += r in ("made", "queued")
                deferred += r == "deferred"
        elif st == "needs_take":
            idx = len(led.jobs(s.id, "take"))
            prompt = p.prompt(s)
            r = ex.submit(p, led, s, _spec(p, s, "take", idx, prompt, image=sel["path"]), p.take_cost(s))
            made += r in ("made", "queued")
            deferred += r == "deferred"
        elif st == "take_review":
            waiting.append(s.id)
        elif st == "needs_human":
            human.append(s.id)
        elif st == "pending_unknown":
            cut_off.append(s.id)
    return {"made": made, "waiting_for_verdict": waiting, "needs_human": human,
            "cut_off": cut_off, "deferred": deferred}


def upscale(p: Project, led: Ledger, ex, chapter: int, only: str | None = None) -> dict:
    """Send approved takes, keyframes and mouth shapes to 4K. Needs the animatic approved (gate 3)."""
    if not led.approved(chapter, "animatic"):
        raise PermissionError(f"chapter {chapter} animatic is not approved (gate 3)")
    made, deferred = 0, 0
    for s in p.chapter(chapter):
        if only and s.id != only:
            continue
        if upscale_state(p, led, s) != "needed":
            continue
        for slot, src in enumerate(upscale_sources(p, led, s)):
            found = _attempt_key(p, led, s, "upscale", slot, src["key"])
            if found is None or found[2] is not None:
                continue
            key, idx, _ = found
            media = "video" if src["kind"] == "take" else "image"
            spec = _spec(p, s, "upscale", idx, src["key"], source=src["path"], media=media,
                         source_key=src["key"], ext=".mp4" if media == "video" else ".png")
            spec["key"] = key
            r = ex.submit(p, led, s, spec, 0.0)
            made += r in ("made", "queued")
            deferred += r == "deferred"
    return {"made": made, "deferred": deferred}


# ---------------------------------------------------------------- reporting
def review_queue(p: Project, led: Ledger, chapter: int) -> list[dict]:
    """Everything a critic still has to look at."""
    out = []
    for r in p.refs_missing(chapter):
        if ref_state(p, led, r) == "review":
            for j in led.jobs(ref_shot(r["name"]).id, "ref"):
                if j["status"] == "done" and j["verdict"] is None:
                    out.append({"key": j["key"], "what": "ref", "shot": r["name"], "type": "Ref", "path": j["path"]})
    for s in p.chapter(chapter):
        st = state(p, led, s)
        kind = {"keyframes_review": "keyframe", "take_review": "take"}.get(st)
        if not kind:
            continue
        for j in led.jobs(s.id, kind):
            if j["status"] == "done" and j["verdict"] is None:
                out.append({"key": j["key"], "what": kind, "shot": s.shot,
                            "type": s.type, "path": j["path"]})
    return out


def status(p: Project, led: Ledger, chapter: int | None) -> dict:
    counts: dict[str, int] = {}
    four_k: dict[str, int] = {}
    for s in p.chapter(chapter):
        st = state(p, led, s)
        counts[st] = counts.get(st, 0) + 1
        u = upscale_state(p, led, s)
        four_k[u] = four_k.get(u, 0) + 1
    return {"states": counts, "4k": four_k, "spent": led.spent(chapter),
            "spent_total": led.spent(), "total_cap": led.total_cap,
            "chapter_cap": led.chapter_caps.get(chapter) if chapter else None,
            "pending_jobs": [j["key"] for j in led.pending()]}


__all__ = ["BudgetExceeded", "SyncExecutor", "BatchExecutor", "keyframes", "video", "upscale",
           "state", "upscale_state", "status", "review_queue", "open_ledger", "out_path"]
