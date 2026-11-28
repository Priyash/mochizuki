"""One run of the GitHub Actions producer, and the repository owner's commands from the issue.

A producer run: apply the reviewer's verdicts, take Kaggle steps until it has to wait,
upload new media, export what needs review, and say what is needed next. The workflow
posts that notice to the production issue when it changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

from . import assemble as asm
from . import review, scheduler, steps
from .ledger import Ledger
from .project import Project, ref_shot
from .providers.base import Status
from .scheduler import Action

DONE = ("take_passed", "ready_for_editor")


def chapter_of(state: Path) -> int:
    f = state / "chapter.txt"
    return int(f.read_text().strip()) if f.exists() else 1


def step(p: Project, led: Ledger, chapter: int, state: Path, backend, engine_set: str = "gpu") -> dict:
    """`backend` is one backend or a list of backends side by side, or None."""
    backends = [b for b in (backend if isinstance(backend, list) else [backend]) if b is not None]
    out: dict = {"chapter": chapter, "actions": []}
    out["reviews"] = review.apply(led, state / "reviews")
    for b in backends:                             # can we sign in, and how much GPU time is left?
        check = b.account_check()
        if check is not None:
            out[f"{b.name}_check"] = check
    if not backends:
        out["actions"].append({"action": Action.SKIPPED, "reason": "no Kaggle credentials"})
    else:
        out["actions"] = scheduler.tick_all(p, led, backends, chapter, engine_set=engine_set)
    for b in backends:
        if getattr(b, "log", None):
            out[f"{b.name}_log"] = b.log[-10:]
    if p.store is not None:
        out["uploaded"] = p.store.push(p)
    out["review"] = review.export(p, led, chapter, state / "review" / f"ch{chapter}")
    out["notice"] = notice(p, led, chapter, state, out)
    out["status"] = steps.status(p, led, chapter)["states"]
    (state / "status.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def _fingerprint(led: Ledger, p: Project, chapter: int, kind: str) -> str:
    keys = []
    for s in p.chapter(chapter):
        for j in led.jobs(s.id):
            if j["status"] == "done" and (j["verdict"] == "pass" or j["selected"] or j["kind"] in ("variant", kind)):
                keys.append(j["key"])
    return hashlib.sha1("|".join(sorted(keys)).encode()).hexdigest()[:12]


def _cut(p: Project, led: Ledger, chapter: int, state: Path, four_k: bool) -> str:
    """Assemble a chapter cut once per set of approved shots, upload it, return its link."""
    tag = "4k" if four_k else "animatic"
    marker = state / f"{tag}_ch{chapter}.txt"
    fp = _fingerprint(led, p, chapter, "upscale" if four_k else "take")
    made = asm.assemble(p, led, chapter, four_k=four_k) if not (marker.exists() and marker.read_text() == fp) else None
    f = p.media / "animatic" / f"chapter_{chapter}{'_4k' if four_k else ''}.mp4"
    if made and p.store is not None:
        p.store.upload(p, f, clobber=True)
    marker.write_text(fp)
    return (p.store.url(p, f) if p.store is not None else str(f)) or str(f)


def notice(p: Project, led: Ledger, chapter: int, state: Path, out: dict) -> dict:
    owner = os.environ.get("GITHUB_REPOSITORY_OWNER", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    at = f"@{owner} " if owner else ""
    shots = p.chapter(chapter)
    st = {s.id: steps.state(p, led, s) for s in shots}
    acts = out["actions"]
    if acts and acts[-1].get("action") == Action.SKIPPED:
        return {"kind": "setup", "text": f"{at}Add the KAGGLE_API_TOKEN and KAGGLE_USERNAME repository secrets "
                                         "(Settings → Secrets and variables → Actions) so the producer can run."}
    errors = [a for a in acts if a.get("action") == Action.ERROR]
    running = list(led.active_batches())
    if errors and not running:
        e = errors[-1]
        return {"kind": "error", "text": f"{at}Chapter {chapter}: the {e.get('backend', 'Kaggle').title()} step "
                                         f"stopped: {e['error']}"}
    if running:
        parts = [f"{b['backend'].title()} batch {b['id']} ({b['stage']}) is {b['status']}" for b in running]
        tail = "".join(f" {e.get('backend', '').title()} hit a problem: {e['error'][:200]}" for e in errors)
        tail += "".join(f" {a.get('backend', '').title()} batch {a['batch']}: {a['error'][:200]}"
                        for a in acts if a.get("status") == Status.COLLECTING)
        tail += "".join(f" {a.get('backend', '').title()} batch {a['batch']} was handed over, but its queued run "
                        "could not be cancelled there and may still start."
                        for a in acts if a.get("action") == Action.HANDED_OVER and not a.get("cancelled"))
        return {"kind": "running", "text": f"Chapter {chapter}: " + "; ".join(parts) + "." + tail}
    rs = {r["name"]: steps.ref_state(p, led, r) for r in p.refs_missing(chapter)
          if steps.ref_state(p, led, r) != "waiting_source"}
    human = [f"ref {n}" for n, x in rs.items() if x == "needs_human"] + [i for i, x in st.items() if x == "needs_human"]
    human_text = (f"{at}These shots need you: {', '.join(human)}. They used their take or failure cap. Change the "
                  "prompt in prompts/<shot>.txt, simplify the shot, or route it to a paid model in shots.csv.")
    if out.get("review", {}).get("items"):
        text = f"Chapter {chapter}: {out['review']['items']} items wait for Claude's review."
        return {"kind": "review", "text": text + (" " + human_text if human else "")}
    if human:
        return {"kind": "human", "text": human_text}
    blob = f"https://github.com/{repo}/blob/state/review/ch{chapter}"
    if rs and all(x == "selected" for x in rs.values()):
        names = review.refs_sheet(p, led, state / "review" / f"ch{chapter}" / "refs.jpg", chapter)
        return {"kind": "refs", "text": f"{at}The reference stills are ready: {blob}/refs.jpg (in order: "
                                        f"{', '.join(names)}). Comment `/approve refs` to make them the film's look, or "
                                        "`/reject <name>` for any you don't want, and new candidates will be painted."}
    if rs:
        return {"kind": "idle", "text": f"Reference stills: {rs}"}
    if all(led.selected(s.id) for s in shots) and not led.approved(chapter, "board"):
        ids = review.board_sheet(p, led, chapter, state / "review" / f"ch{chapter}" / "board.jpg")
        return {"kind": "board", "text": f"{at}The chapter {chapter} board is ready: {blob}/board.jpg "
                                         f"(shots in order: {', '.join(ids)}). Comment `/approve board {chapter}` "
                                         "to start the takes."}
    if all(x in DONE for x in st.values()) and not led.approved(chapter, "animatic"):
        link = _cut(p, led, chapter, state, four_k=False)
        return {"kind": "animatic", "text": f"{at}The chapter {chapter} animatic is ready: {link} . "
                                            f"Comment `/approve animatic {chapter}` to start the 4K pass."}
    if led.approved(chapter, "animatic") and all(steps.upscale_state(p, led, s) == "done" for s in shots):
        link = _cut(p, led, chapter, state, four_k=True)
        nxt = f" Comment `/chapter {chapter + 1}` to start the next chapter." if chapter < 5 else ""
        return {"kind": "done", "text": f"{at}Chapter {chapter} is finished in 4K: {link} .{nxt}"}
    counts: dict[str, int] = {}
    for x in st.values():
        counts[x] = counts.get(x, 0) + 1
    return {"kind": "idle", "text": f"Chapter {chapter}: {counts}"}


COMMAND = re.compile(r"^/(approve)\s+(board|animatic)\s+([1-5])\b|^/(chapter)\s+([1-5])\b|^/(status)\b"
                     r"|^/approve\s+(refs)\b|^/reject\s+([a-z0-9_]+)", re.I | re.M)


def command(p: Project, led: Ledger, state: Path, body: str, author: str, owner: str) -> str:
    """Run the owner's command from an issue comment. Anyone else is refused."""
    if not owner or author != owner:
        return f"Only @{owner} can give production commands."
    m = COMMAND.search(body or "")
    if not m:
        return ("Commands: `/approve refs`, `/reject <ref name>`, `/approve board N`, `/approve animatic N`, "
                "`/chapter N`, `/status`.")
    if m.group(7):
        done, waiting = [], []
        for r in p.refs_missing():
            sel = led.selected(ref_shot(r["name"]).id)
            if not sel:
                waiting.append(r["name"])
                continue
            dest = p.ref_path(r["name"])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p.fetch(sel["path"]), dest)
            done.append(r["name"])
        if not done:
            return "Nothing to approve: no reference still has been selected yet."
        tail = f" Still waiting: {', '.join(waiting)}." if waiting else " Keyframes start on the next producer run."
        return f"Approved and added to style_bible/refs: {', '.join(done)}.{tail}"
    if m.group(8):
        name = m.group(8).lower()
        sel = led.selected(ref_shot(name).id)
        if not sel:
            return f"`{name}` has no selected reference still to reject."
        led.unselect(sel["key"], "rejected by the owner")
        return f"Rejected the chosen `{name}`. Claude will pick another candidate, or new ones will be painted."
    if m.group(1):
        gate, chapter = m.group(2).lower(), int(m.group(3))
        shots = p.chapter(chapter)
        if gate == "board":
            missing = [s.id for s in shots if not led.selected(s.id)]
            if missing:
                return f"Not approved: no keyframe is selected yet for {', '.join(missing)}."
        else:
            if not led.approved(chapter, "board"):
                return f"Not approved: the chapter {chapter} board is not approved yet."
            open_ = [s.id for s in shots if steps.state(p, led, s) not in DONE]
            if open_:
                return f"Not approved: these shots have no passing take yet: {', '.join(open_)}."
        led.approve(chapter, gate, "human")
        return f"Approved: chapter {chapter} {gate}. The producer will continue on its next run."
    if m.group(4):
        (state / "chapter.txt").write_text(m.group(5))
        return f"The producer now works on chapter {m.group(5)}."
    s = steps.status(p, led, chapter_of(state))
    return f"Chapter {chapter_of(state)}: {s['states']}, 4K: {s['4k']}"
