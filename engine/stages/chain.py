"""The chain: every stage of making a film, ordered by what each consumes and produces.

    python -m engine chain               where every chapter of the film stands, stage by stage
    python -m engine chain --chapter 3   one chapter

Two ways make a shot's picture, side by side: the 3D path (voices -> render), the main one, and the
generative path (refs -> keyframes -> video). Both end in the same cut. The owner's approvals
(gates) stop the chain until they are given. Planned stages are listed, marked not built.

Each stage says how it skips work that has not changed, so a change upstream redoes only what
depends on it: a job's key holds a hash of its prompt, a render's stamp the hash of its plan, a
voiced line is cached by its text and voice.
"""
from __future__ import annotations

from graphlib import TopologicalSorter

from .. import assemble, steps
from ..audio import dialogue
from ..cgi import config as cgi_config
from .stage import ANIMATIC_GATE, ASSEMBLE, BOARD_GATE, CGI, REFS_GATE, SCHEDULER, Stage

# What the film's own folder provides (films/<name>/): the chain's inputs
FILM_INPUTS = frozenset({
    "shot sheet", "style", "reference prompts", "scenes", "acting sheets", "sets", "lights", "cast",
    "dialogue script",
    "music cues",            # planned: style_bible/sound.json, when the music stage is built
})


# ---------------------------------------------------------------- where a chapter stands
def _refs_wanted(p, led, chapter) -> bool:
    return any(steps.ref_state(p, led, r) in ("todo", "needs_candidates") for r in p.refs_missing(chapter))


def _refs_unapproved(p, led, chapter) -> bool:
    return bool(p.refs_missing(chapter))      # gate 1: the reference stills are not approved yet


def _states(p, led, chapter) -> list[str]:
    return [steps.state(p, led, s) for s in p.chapter(chapter)]


def _keyframes_wanted(p, led, chapter) -> bool:
    return any(s in ("todo", "needs_keyframes") for s in _states(p, led, chapter))


def _video_wanted(p, led, chapter) -> bool:
    return led.approved(chapter, BOARD_GATE) and any(s in ("needs_take", "needs_variants") for s in _states(p, led, chapter))


def _upscale_wanted(p, led, chapter) -> bool:
    return led.approved(chapter, ANIMATIC_GATE) and any(
        steps.upscale_state(p, led, s) == "needed" for s in p.chapter(chapter))


def _refs_progress(p, led, chapter) -> str:
    missing = p.refs_missing(chapter)
    if not missing:
        return "approved"
    states = [steps.ref_state(p, led, r) for r in missing]
    picked = sum(s == "selected" for s in states)
    return f"{picked}/{len(missing)} picked, waiting for /approve refs" if picked == len(missing) else \
        f"{picked}/{len(missing)} picked"


def _keyframes_progress(p, led, chapter) -> str:
    shots = p.chapter(chapter)
    chosen = sum(1 for s in shots if led.selected(s.id))
    gate = "board approved" if led.approved(chapter, BOARD_GATE) else "board not approved"
    return f"{chosen}/{len(shots)} chosen; {gate}"


def _video_progress(p, led, chapter) -> str:
    moving = [s for s in p.chapter(chapter) if p.model(s.model).get("generates_video")]
    passed = sum(1 for s in moving if steps.state(p, led, s) == "take_passed")
    return f"{passed}/{len(moving)} takes passed"


def _upscale_progress(p, led, chapter) -> str:
    states = [steps.upscale_state(p, led, s) for s in p.chapter(chapter)]
    return f"{states.count('done')}/{len(states)} in 4K"


def _voices_progress(p, led, chapter) -> str:
    speaking = [s.id for s in p.chapter(chapter) if dialogue.lines_for(p, s.id)]
    voiced = sum(1 for sid in speaking if dialogue.shot_manifest(p, sid))
    return f"{voiced}/{len(speaking)} speaking shots voiced" if speaking else "no lines"


def _render_progress(p, led, chapter) -> str:
    cfg = cgi_config.settings(p)
    shots = p.chapter(chapter)
    done = sum(1 for s in shots if any((cgi_config.work_dir(p, cfg, s.id) / f"{s.id}{x}.mp4").exists()
                                       for x in ("", "-preview")))
    return f"{done}/{len(shots)} rendered"


def _cut_progress(p, led, chapter) -> str:
    made = assemble.cut_path(p, chapter).exists()
    gate = "approved" if led.approved(chapter, ANIMATIC_GATE) else "not approved"
    return f"made; {gate}" if made else "not made"


def _final_progress(p, led, chapter) -> str:
    return "made" if assemble.cut_path(p, chapter, four_k=True).exists() else "not made"


# ---------------------------------------------------------------- the stages
CHAIN: tuple[Stage, ...] = (
    Stage("refs", "paints candidate reference stills; the reviewer picks one per subject",
          consumes=("shot sheet", "style", "reference prompts"), produces=("reference stills",),
          runner=SCHEDULER, generators=("refs",),
          skips_unchanged="a job's key holds its prompt's hash; approved stills are never painted again",
          wants=_refs_wanted, halts=_refs_unapproved, plan=steps.refs, progress=_refs_progress),
    Stage("voices", "voices every line, places it in its shot, makes the mouths",
          consumes=("dialogue script", "shot sheet"), produces=("voiced lines", "mouths", "shot dialogue tracks"),
          runner=CGI, generators=("voice", "lipsync"),
          skips_unchanged="each line is cached by its text and voice", progress=_voices_progress),
    Stage("render", "builds every shot in Blender from its acting sheet, set, lights and cast",
          consumes=("shot sheet", "acting sheets", "sets", "lights", "cast", "mouths"), produces=("3D renders",),
          runner=CGI, generators=("render",),
          skips_unchanged="a render's stamp holds its plan's hash", progress=_render_progress),
    Stage("keyframes", "draws keyframe candidates from the reference stills; the reviewer picks one per shot",
          consumes=("reference stills", "shot sheet", "scenes"), produces=("keyframes",),
          runner=SCHEDULER, generators=("keyframe",), gate=REFS_GATE,
          skips_unchanged="a job's key holds its prompt's hash", wants=_keyframes_wanted, plan=steps.keyframes,
          progress=_keyframes_progress),
    Stage("video", "animates each chosen keyframe; only a failed take gets another",
          consumes=("keyframes",), produces=("takes",),
          runner=SCHEDULER, generators=("video",), gate=BOARD_GATE,
          skips_unchanged="a take's key holds its prompt's hash", wants=_video_wanted, plan=steps.video,
          progress=_video_progress),
    Stage("ai_finish", "repaints a 3D render in the reference stills' look, guided by its depth, lines and masks",
          consumes=("3D renders", "reference stills"), produces=("finished shots",),
          runner=SCHEDULER, generators=("ai_finish",), built=False),
    Stage("music", "scores the chapter from its music cues",
          consumes=("music cues", "shot sheet"), produces=("music track",),
          runner=SCHEDULER, generators=("music",), built=False),
    Stage("cut", "cuts the chapter: the best picture per shot, its dialogue tracks and subtitles",
          consumes=("3D renders", "takes", "keyframes", "shot dialogue tracks"), produces=("chapter cut", "subtitles"),
          runner=ASSEMBLE, generators=("subtitles",),
          skips_unchanged="cheap: a cut is remade from what exists", progress=_cut_progress),
    Stage("upscale", "upscales the approved cut's takes and keyframes to 4K",
          consumes=("chapter cut", "takes", "keyframes"), produces=("4K shots",),
          runner=SCHEDULER, generators=("upscale",), gate=ANIMATIC_GATE,
          skips_unchanged="a 4K job's key holds its source's key", wants=_upscale_wanted, plan=steps.upscale,
          progress=_upscale_progress),
    Stage("final", "cuts the chapter in 4K",
          consumes=("4K shots", "shot dialogue tracks"), produces=("4K chapter",),
          runner=ASSEMBLE, gate=ANIMATIC_GATE, progress=_final_progress),
)
BY_NAME = {s.name: s for s in CHAIN}


def dependencies() -> dict[str, set[str]]:
    """For each stage, the stages that make what it consumes."""
    makers = {a: s.name for s in CHAIN for a in s.produces}
    return {s.name: {makers[a] for a in s.consumes if a in makers} for s in CHAIN}


def order() -> list[Stage]:
    """The stages in an order their inputs allow: every stage after the stages it consumes from, and
    otherwise as CHAIN lists them. Raises graphlib.CycleError if the chain ever loops."""
    deps = dependencies()
    tuple(TopologicalSorter(deps).static_order())          # proves there is no loop
    rank = {s.name: i for i, s in enumerate(CHAIN)}
    out: list[Stage] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        for d in sorted(deps[name], key=rank.get):
            visit(d)
        seen.add(name)
        out.append(BY_NAME[name])

    for s in CHAIN:
        visit(s.name)
    return out


def gpu_stages() -> list[Stage]:
    """The built stages the scheduler runs on GPU providers, in chain order."""
    return [s for s in order() if s.runner == SCHEDULER and s.built]


def next_gpu_stage(p, led, chapter: int) -> str | None:
    """What the chapter needs next from the GPU, or None if it is waiting on a person."""
    for s in gpu_stages():
        if s.wants(p, led, chapter):
            return s.name
        if s.halts and s.halts(p, led, chapter):
            return None
    return None


def report(p, led, chapters: list[int]) -> str:
    """Every stage against every chapter asked for, as a text table."""
    rows = [("stage", "runs on", "gate", *[f"chapter {c}" for c in chapters])]
    for s in order():
        cells = ["not built"] * len(chapters) if not s.built else \
            [s.progress(p, led, c) if s.progress else "" for c in chapters]
        rows.append((s.name, s.runner, s.gate or "", *cells))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    nxt = {c: next_gpu_stage(p, led, c) for c in chapters}
    lines += ["", "next GPU stage: " + "; ".join(f"chapter {c}: {n or 'waiting for a person'}" for c, n in nxt.items())]
    return "\n".join(lines)
