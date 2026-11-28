"""The chapter, unattended: voices, renders, the board and the cut with dialogue and subtitles.

    run(p, chapter, stage="all", shard=(i, n))

Stages, each safe to repeat (only what changed is redone):
    voices   every line of the chapter voiced in every language, placed, lip-synced, mixed per shot
    render   this shard's shots: built and rendered (previews, or --final 4K), and each speaking shot
             again per other language (<shot>.<language>.mp4, config cgi.dubs); a shot whose plan
             has not changed since its last render is skipped
    finish   the storyboard, the chapter cut with its dialogue tracks and .srt files, a dubbed cut per
             other language (chapter_N.<language>.mp4: its own lips and voice), and a review sheet
             (review.json, with the continuity findings at every cut) for Claude's reviewer run

Shards let many machines share a chapter (GitHub's runners: .github/workflows/cgi.yml). Nothing here
waits for a person: acting sheets in "draft" render like approved ones; a sheet set to "hold" is
skipped (the owner's veto).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..audio import dialogue
from ..log import log as _log
from . import continuity
from . import plan as planner
from .acting import arc
from .acting import direction as dirn
from .config import settings, work_dir
from .render import runner as rnd


def log(event: str, msg: str = "", level: str = "info", **kw) -> None:
    _log("auto", event, msg, level, **kw)


def plan_hash(plan: dict) -> str:
    """What the picture depends on: the plan without where its files go."""
    keep = {k: v for k, v in plan.items() if k not in ("output",)}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def mine(shots: list[str], shard: tuple[int, int]) -> list[str]:
    i, n = shard
    return [s for k, s in enumerate(shots) if k % max(1, n) == i]


def held(p, shot_id: str) -> bool:
    return dirn.direction_for(p, shot_id).status == "hold"


def stage_voices(p, chapter: int, engine: str | None, shots: list[str]) -> dict:
    out: dict[str, list] = {"voiced": [], "overruns": [], "failed": []}
    langs = dialogue.load(p).get("languages", ["en"])
    for sid in shots:
        if not dialogue.lines_for(p, sid):
            continue
        try:
            r = dialogue.voice_shot(p, sid, langs, engine)
        except Exception as e:  # noqa: BLE001 - a language that fails must not silence the others
            log("voices.retry", f"{sid}: {e}; trying the primary language alone", level="warn", shot=sid)
            try:
                r = dialogue.voice_shot(p, sid, [dialogue.load(p).get("primary", "en")], engine)
            except Exception as e2:  # noqa: BLE001
                log("voices.failed", f"{sid}: {e2}", level="error", shot=sid)
                out["failed"].append(sid)
                continue
        out["voiced"].append(sid)
        out["overruns"] += [{"shot": sid, **o} for o in r.get("overruns", [])]
        log(
            "voices.done", f"{sid}: {r['lines']} lines", shot=sid, lines=r["lines"], overruns=len(r.get("overruns", []))
        )
    return out


def prune_frames(folder: Path, keep: int = 3) -> int:
    """Keep the first, middle and last frame of each frames folder (enough for the board and the
    review); the clip holds the rest. Returns how many frames were removed."""
    gone = 0
    for d in folder.glob("frames-*"):
        frames = sorted(d.glob("*.png"))
        if len(frames) <= keep:
            continue
        want = {0, len(frames) // 2, len(frames) - 1}
        for i, f in enumerate(frames):
            if i not in want:
                f.unlink()
                gone += 1
    return gone


def dub_languages(p, shot_id: str) -> list[str]:
    """The languages a shot is rendered in besides the primary one (config cgi.dubs)."""
    if settings(p).get("dubs", "all") != "all":
        return []
    return dialogue.dubbed_languages(dialogue.shot_manifest(p, shot_id))


def _render_one(p, sid: str, final: bool, size, force: bool, language: str | None) -> dict | None:
    """Render one shot in one language, unless its plan is unchanged since its last render (None)."""
    cfg = settings(p)
    tag = "final" if final else "preview"
    pl = planner.build(p, sid, preview=not final, size=size, language=language)
    h = plan_hash(pl)
    stamp = work_dir(p, cfg, sid) / f"render-{tag}{pl['output'].get('clip_suffix', '')}.json"
    prev = json.loads(stamp.read_text()) if stamp.exists() else {}
    if not force and prev.get("hash") == h and Path(prev.get("clip", "")).exists():
        return None
    t0 = time.time()
    r = rnd.render_shot(p, sid, preview=not final, size=size, language=language)
    stamp.write_text(json.dumps({"hash": h, "clip": r["clip"], "seconds": round(time.time() - t0, 1),
                                 "seconds_per_frame": r.get("seconds_per_frame"), "language": r.get("language"),
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return {"shot": sid, "clip": r["clip"], "seconds_per_frame": r.get("seconds_per_frame"),
            "language": r.get("language")}


def stage_render(p, chapter: int, shots: list[str], final: bool, size, force: bool, prune: bool = False) -> dict:
    """Render the shots (and each speaking shot's dubs); a shot whose plan is unchanged is skipped."""
    cfg = settings(p)
    out: dict[str, list] = {"rendered": [], "skipped": [], "held": [], "failed": []}
    for sid in shots:
        if held(p, sid):
            out["held"].append(sid)
            continue
        for language in [None, *dub_languages(p, sid)]:
            name = sid if language is None else f"{sid}.{language}"
            try:
                done = _render_one(p, sid, final, size, force, language)
                if done is None:
                    out["skipped"].append(name)
                else:
                    out["rendered"].append(done)
            except Exception as e:  # noqa: BLE001 - one bad shot must not stop the chapter
                log("render.failed", f"{name}: {e}", level="error", shot=sid, language=language)
                out["failed"].append({"shot": sid, "language": language, "error": str(e)[:400]})
        if prune:
            prune_frames(work_dir(p, cfg, sid))
    return out


def review_sheet(p, chapter: int, final: bool) -> Path:
    """What Claude's reviewer run looks at: per shot, the acting sheet's intent, the frames to judge
    (start, middle, end), the dialogue placed in it, any line that overruns the shot, each spoken
    line's lip sync in every language (lipsync.sync_report: ok means within broadcast tolerance), and
    the continuity findings at the cut into it (continuity.check), its feeling against the intended
    emotional arc (arc.report), and the culture pack's rules that shaped its beats."""
    cfg = settings(p)
    items = []
    cuts: dict[str, list] = {}
    feel: dict[str, dict] = {}
    try:
        for f in continuity.check(p, chapter):
            cuts.setdefault(f.shot, []).append({k: v for k, v in f.to_dict().items() if k != "shot"})
        feel = {x["shot"]: {k: v for k, v in x.items() if k != "shot"} for x in arc.report(p, chapter)["shots"]}
    except Exception as e:  # noqa: BLE001 - the review sheet is still worth having without them
        log("review.checks_failed", str(e)[:300], level="error")
    for s in p.chapter(chapter):
        d, played = dirn.directed(p, s.id)
        folder = work_dir(p, cfg, s.id)
        frames = sorted((f for fd in rnd.frame_folders(folder) for f in fd.glob("*.png")),
                        key=lambda f: f.stat().st_mtime)
        man = dialogue.shot_manifest(p, s.id) or {}
        items.append({
            "shot": s.id, "type": s.type, "description": s.shot, "status": d.status,
            "camera": f"{d.camera.framing} {d.camera.angle} {int(d.camera.lens)}mm {d.camera.move}",
            "beats": [f"{b.t}s {b.who or 'all'}: {b.emotion} {b.intensity}"
                      + (f" held {b.restraint}" if b.restraint else "") + (f" ({b.action})" if b.action else "")
                      for b in d.beats],
            "lines": [f"{ln['who']}: {ln['lines'].get(man.get('primary', 'en'), {}).get('text', '')}"
                      for ln in man.get("lines", [])],
            "overruns": man.get("overruns", []),
            "continuity": cuts.get(s.id, []),
            "arc": feel.get(s.id, {}),
            "culture": sorted({c.rule for c in played}),
            "lipsync": [{"who": ln["who"], "language": lang, **rep}
                        for ln in man.get("lines", []) if ln.get("on_screen", True) and not ln.get("nonverbal")
                        for lang, rep in ln.get("sync", {}).items()],
            "frames": [str(f.relative_to(p.root)) for f in (frames[:1] + frames[len(frames) // 2:len(frames) // 2 + 1]
                                                             + frames[-1:])] if frames else [],
        })
    out = p.root / cfg["work"] / f"review-ch{chapter}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"chapter": chapter, "quality": "final" if final else "preview",
                               "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shots": items},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def stage_finish(p, chapter: int, final: bool) -> dict:
    from .. import assemble as asm  # noqa: PLC0415
    from .. import steps  # noqa: PLC0415
    out: dict = {}
    shots = [s.id for s in p.chapter(chapter)]
    try:
        out["board"] = str(rnd.board(p, shots, chapter))
    except Exception as e:  # noqa: BLE001
        out["board_error"] = str(e)[:300]
    cfg = dict(p.config.get("cgi", {}))
    p.config.setdefault("cgi", {})["use_in_cut"] = True         # this cut is the CG one
    try:
        led = steps.open_ledger(p)
        out["cut"] = asm.assemble(p, led, chapter, four_k=final)
        dubs = sorted({lang for sid in shots for lang in dub_languages(p, sid)})
        out["dubs"] = {lang: asm.assemble(p, led, chapter, four_k=final, language=lang) for lang in dubs}
        for lang, cut in out["dubs"].items():
            if cut.get("undubbed"):
                log("finish.undubbed", f"{lang} cut: {', '.join(cut['undubbed'])} keep the primary lips (no {lang} "
                    "render)", level="warn", language=lang, shots=cut["undubbed"])
    finally:
        p.config["cgi"] = cfg
    out["review"] = str(review_sheet(p, chapter, final))
    return out


def run(p, chapter: int, stage: str = "all", final: bool = False, shard: tuple[int, int] = (0, 1),
        engine: str | None = None, size=None, force: bool = False, only: list[str] | None = None,
        prune: bool = False) -> dict:
    shots = [s.id for s in p.chapter(chapter) if not only or s.id in only]
    todo = mine(shots, shard)
    result: dict = {"chapter": chapter, "shard": f"{shard[0]}/{shard[1]}", "shots": todo}
    if stage in ("voices", "all"):
        result["voices"] = stage_voices(p, chapter, engine, shots if stage == "voices" else todo)
    if stage in ("render", "all"):
        result["render"] = stage_render(p, chapter, todo, final, size, force, prune)
    if stage in ("finish", "all"):
        result["finish"] = stage_finish(p, chapter, final)
    failed = (result.get("voices", {}).get("failed", []) + result.get("render", {}).get("failed", []))
    result["failed"] = failed
    log("auto.done", f"chapter {chapter} shard {shard[0]}/{shard[1]}: "
        f"{len(result.get('render', {}).get('rendered', []))} rendered, {len(failed)} failed",
        chapter=chapter, failed=len(failed))
    return result
