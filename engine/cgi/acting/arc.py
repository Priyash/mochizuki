"""The emotional arc: how a chapter feels from shot to shot, set against how it is meant to feel.

Every emotion of the acting sheets has a place on Russell's circumplex of affect (J. A. Russell, "A
circumplex model of affect", Journal of Personality and Social Psychology 39, 1980; placed after Posner,
Russell and Peterson, Development and Psychopathology 17, 2005): valence, from unpleasant (-1) to pleasant
(+1), and arousal, from calm (-1) to excited (+1). A beat at intensity k sits at k times its emotion's
place. A shot feels like its beats, each for as long as it lasts, averaged over its characters. Strung
together shot by shot, that is the chapter's realized arc.

The intended arc is the film's own (style_bible/arc.json). For each chapter it is either points in the
chapter's time (0 at its start, 1 at its end) with a valence and, optionally, an arousal, or one of the
six shapes most stories follow (A. J. Reagan et al., "The emotional arcs of stories are dominated by six
basic shapes", EPJ Data Science 5, 2016): rags_to_riches, tragedy, man_in_a_hole, icarus, cinderella,
oedipus. A shape gives valence only.

    {"chapters": {"1": {"points": [[0.0, 0.6, 0.5], [0.5, -0.4, -0.3], [1.0, 0.4, -0.2]]},
                  "2": {"shape": "man_in_a_hole", "amplitude": 0.6}},
     "fill_drafts": true}

`report` sets the two side by side: each shot's distance from the intended arc, the correlation of the
chapter's valence with it, and the shape the realized arc is closest to. With "fill_drafts", a drafted
shot whose description gave no feeling (its one neutral beat) takes the arc's feeling at that moment
instead (`suggest`). Hand-written beats are never changed.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ...film_check import read_checked
from ...project import ARC_FILE
from . import direction as dirn

CIRCUMPLEX = {                      # emotion -> (valence, arousal)
    "neutral": (0.0, 0.0),
    "joy": (0.8, 0.5),
    "tenderness": (0.6, -0.35),
    "wonder": (0.55, 0.45),
    "curiosity": (0.35, 0.35),
    "resolve": (0.3, 0.45),
    "surprise": (0.15, 0.85),
    "bittersweet": (0.0, -0.4),
    "sorrow": (-0.7, -0.45),
    "fear": (-0.65, 0.7),
    "anger": (-0.7, 0.75),
}
SHAPES = {                          # Reagan et al.'s six, as valence over the story's time
    "rags_to_riches": [(0.0, -1.0), (1.0, 1.0)],
    "tragedy": [(0.0, 1.0), (1.0, -1.0)],
    "man_in_a_hole": [(0.0, 0.5), (0.5, -1.0), (1.0, 0.8)],
    "icarus": [(0.0, -0.5), (0.5, 1.0), (1.0, -0.8)],
    "cinderella": [(0.0, -0.5), (0.3, 0.8), (0.65, -0.6), (1.0, 1.0)],
    "oedipus": [(0.0, 0.5), (0.3, -0.8), (0.65, 0.6), (1.0, -1.0)],
}
AMPLITUDE = 0.6                     # a shape's valence reaches this, unless the chapter says otherwise
VALENCE_TOLERANCE = 0.4             # a shot this far from the intended valence is off the arc ...
AROUSAL_TOLERANCE = 0.4             # ... or this far from its arousal, where one is intended
MIN_INTENSITY = 0.2                 # the weakest feeling worth playing
NEUTRAL_RADIUS = 0.15               # an intended point this near the centre is played neutral
MIN_SHOTS = 3                       # a correlation needs at least this many shots that feel something


@dataclass
class ShotArc:
    shot: str
    at: float                                    # the shot's middle, 0..1 through the chapter
    realized: tuple[float, float] | None         # (valence, arousal); None for a shot of things
    intended: tuple[float, float | None] | None  # None when the chapter has no intended arc
    off: str = ""                                # how it strays from the intended arc

    def to_dict(self) -> dict:
        return asdict(self)


def place(emotion: str, intensity: float) -> tuple[float, float]:
    v, a = CIRCUMPLEX.get(emotion, (0.0, 0.0))
    return v * intensity, a * intensity


def shot_feeling(d: dirn.Direction) -> tuple[float, float] | None:
    """How a shot feels: each character's beats, each held until their next one, averaged over time;
    then the characters averaged. None when nobody in it feels anything."""
    felt = []
    for who in d.characters:
        beats = sorted((b for b in d.beats if b.who in ("", who)), key=lambda b: b.t)
        if not beats:
            continue
        v = a = total = 0.0
        for k, b in enumerate(beats):
            span = (beats[k + 1].t if k + 1 < len(beats) else d.seconds) - b.t
            span = max(span, 1e-3)
            pv, pa = place(b.emotion, b.intensity)
            v, a, total = v + pv * span, a + pa * span, total + span
        felt.append((v / total, a / total))
    if not felt:
        return None
    return sum(x for x, _ in felt) / len(felt), sum(y for _, y in felt) / len(felt)


def _interp(points: list[tuple[float, float]], x: float) -> float:
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / max(1e-9, x1 - x0)
    return pts[-1][1]


def load(p, check: bool = True) -> dict:
    """The film's file ({} when it has none). With `check`, a file with problems is refused, all of them
    named at once (FilmError)."""
    return read_checked(p.style_bible / ARC_FILE, problems if check else None)


def problems(spec: dict) -> list[str]:
    """What is wrong with arc.json, in words."""
    out = []
    for ch, c in spec.get("chapters", {}).items():
        if not isinstance(c, dict):
            out.append(f"chapters.{ch}: an object with points or a shape")
        elif "points" in c:
            for pt in c["points"]:
                if not (isinstance(pt, list) and len(pt) in (2, 3) and all(isinstance(x, (int, float)) for x in pt)
                        and 0 <= pt[0] <= 1 and all(-1 <= x <= 1 for x in pt[1:])):
                    out.append(f"chapters.{ch}.points: {pt} is not [time 0..1, valence -1..1, arousal -1..1]")
        elif c.get("shape") not in SHAPES:
            out.append(f"chapters.{ch}.shape {c.get('shape')!r} is not one of {', '.join(SHAPES)}")
    return out


def intended(spec: dict | None, x: float) -> tuple[float, float | None] | None:
    """The intended (valence, arousal) at `x` through the chapter; arousal None when not intended."""
    if not spec:
        return None
    if "points" in spec:
        pts = spec["points"]
        v = _interp([(pt[0], pt[1]) for pt in pts], x)
        with_a = [(pt[0], pt[2]) for pt in pts if len(pt) > 2]
        return v, (_interp(with_a, x) if with_a else None)
    shape = SHAPES.get(spec.get("shape", ""))
    if shape is None:
        return None
    return _interp(shape, x) * float(spec.get("amplitude", AMPLITUDE)), None


def chapter_spec(p, chapter: int) -> dict | None:
    return load(p).get("chapters", {}).get(str(chapter))


def _times(p, chapter: int) -> dict[str, float]:
    """Each shot's middle, 0..1 through its chapter."""
    shots = p.chapter(chapter)
    total = sum(s.seconds for s in shots) or 1.0
    out, t = {}, 0.0
    for s in shots:
        out[s.id] = (t + s.seconds / 2) / total
        t += s.seconds
    return out


def _corr(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < MIN_SHOTS:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-9 or sy < 1e-9:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def reads_as(at: list[float], valence: list[float]) -> tuple[str, float] | None:
    """Which of the six shapes a realized arc is closest to, and how closely (correlation)."""
    scored: list[tuple[str, float]] = []
    for name, pts in SHAPES.items():
        r = _corr(valence, [_interp(pts, x) for x in at])
        if r is not None:
            scored.append((name, r))
    return max(scored, key=lambda x: x[1]) if scored else None


def report(p, chapter: int) -> dict:
    """The chapter's realized arc beside its intended one."""
    spec = chapter_spec(p, chapter)
    times = _times(p, chapter)
    shots = []
    for sid, x in times.items():
        realized = shot_feeling(dirn.direction_for(p, sid))
        want = intended(spec, x)
        off = []
        if realized and want:
            if abs(realized[0] - want[0]) > VALENCE_TOLERANCE:
                off.append(f"valence {realized[0]:+.2f}, the arc wants {want[0]:+.2f}")
            if want[1] is not None and abs(realized[1] - want[1]) > AROUSAL_TOLERANCE:
                off.append(f"arousal {realized[1]:+.2f}, the arc wants {want[1]:+.2f}")
        shots.append(ShotArc(sid, round(x, 3), realized and (round(realized[0], 3), round(realized[1], 3)),
                             want and (round(want[0], 3), None if want[1] is None else round(want[1], 3)),
                             "; ".join(off)))
    felt = [s for s in shots if s.realized]
    with_intent = [s for s in felt if s.intended]
    r = _corr([s.realized[0] for s in with_intent if s.realized], [s.intended[0] for s in with_intent if s.intended])
    shape = reads_as([s.at for s in felt], [s.realized[0] for s in felt if s.realized])
    return {"chapter": chapter, "intended": spec, "shots": [s.to_dict() for s in shots],
            "valence_r": None if r is None else round(r, 3),
            "reads_as": None if shape is None else {"shape": shape[0], "r": round(shape[1], 3)},
            "off": [s.shot for s in shots if s.off]}


def suggest(valence: float, arousal: float | None) -> tuple[str, float]:
    """The emotion, and how strong, that best plays an intended point of the arc."""
    target = (valence, 0.0 if arousal is None else arousal)
    if math.hypot(*target) < NEUTRAL_RADIUS:
        return "neutral", 0.3
    best = ("neutral", 0.3, math.inf)
    for emo, (cv, ca) in CIRCUMPLEX.items():
        norm = cv * cv + (ca * ca if arousal is not None else 0.0)
        if emo == "neutral" or norm < 1e-9:
            continue
        k = (target[0] * cv + (target[1] * ca if arousal is not None else 0.0)) / norm
        k = max(MIN_INTENSITY, min(1.0, k))
        dist = math.hypot(target[0] - k * cv, (target[1] - k * ca) if arousal is not None else 0.0)
        if dist < best[2]:
            best = (emo, round(k, 2), dist)
    return best[0], best[1]


def fill_draft(p, chapter: int, d: dirn.Direction) -> dirn.Direction:
    """A drafted shot that found no feeling in its description (one neutral beat) plays the arc's
    feeling at its moment, when arc.json says "fill_drafts"."""
    spec = load(p)
    if not spec.get("fill_drafts") or len(d.beats) != 1 or d.beats[0].emotion != "neutral":
        return d
    want = intended(spec.get("chapters", {}).get(str(chapter)), _times(p, chapter).get(d.shot, 0.5))
    if want is None:
        return d
    emotion, intensity = suggest(*want)
    d.beats[0].emotion, d.beats[0].intensity = emotion, intensity
    d.notes = (d.notes + " " if d.notes else "") + f"[arc: {emotion} {intensity} from arc.json]"
    return d
