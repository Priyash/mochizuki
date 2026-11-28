"""The continuity supervisor: what a script supervisor and an editor check at every cut, measured on the
shots' plans (the cameras, marks, eyelines, light and props Blender will render), before anything is
rendered.

At each cut between two shots of one scene:

    axis      The 180-degree rule. The camera stays on one side of the line between two characters who
              share the scene, so each keeps their side of the screen. A shot on the line (a neutral
              shot) or a camera move across it sets the line again; a cutaway does not.
    jump      The 30-degree rule. A new shot of the same subject moves the camera at least JUMP_DEG, or
              changes the shot size at least JUMP_SIZE times; otherwise the cut reads as a jump.
    eyeline   Two characters who look at each other across a cut look in opposite screen directions
              (the same direction, if the same person looks again). Someone looking at a person off
              screen does not look into the lens.
    light     The key light (its direction, colour and strength), the ambient light and the practical
              lights hold through a scene.
    props     A prop seen on both sides of a cut has not moved or changed, unless the first shot acts
              on it.
    position  A character seen on both sides of a cut has not jumped or turned, unless walking.

Sources: D. Arijon, "Grammar of the Film Language" (1976), on the line of interest, neutral shots and
eyelines; D. Bordwell and K. Thompson, "Film Art", ch. 6, on continuity editing and the 30-degree rule.

A finding is accepted, not fixed, when the shot after the cut gives the reason in its acting sheet:
    "continuity": {"axis": "she turns her back on the room: the line turns with her"}
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from . import plan as planner
from .acting import direction as dirn
from .config import settings

RULES = dirn.CONTINUITY_RULES
NEUTRAL_DEG = 15.0        # a camera this close to the line between two characters is on it: a neutral shot
JUMP_DEG = 30.0           # the 30-degree rule
JUMP_SIZE = 1.5           # ... or the frame this many times bigger or smaller (one shot size is about 1.7x)
LENS_DEG = 10.0           # a look this close to the lens reads as a look at the audience
LIGHT_DEG = 15.0          # the key light may move this much within a scene ...
LIGHT_COLOUR = 0.08       # ... change colour this much (per channel, 0..1) ...
LIGHT_ENERGY = 1.25       # ... and strength this many times
PROP_MOVE_M = 0.05        # a prop may sit this far from where it was
PROP_TURN_DEG = 10.0      # ... and turn this much
POSITION_M = 0.3          # a character may stand this far from where they were
TURN_DEG = 45.0           # ... and turn this much
WALKING = "walking"       # a posture that moves the character between shots
STILL_STATE = ("kind", "at", "rot", "motion")   # a prop's keys that are not its state


@dataclass(frozen=True)
class Finding:
    shot: str              # the shot after the cut
    after: str             # the shot before it ("" for the first shot of a scene)
    rule: str              # one of RULES
    message: str
    accepted: str = ""     # why the cut breaks the rule on purpose, from the shot's acting sheet

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Shot:
    """One shot as continuity sees it: from its plan and its acting sheet."""
    id: str
    scene: str
    cams: tuple[list[float], list[float]]          # the camera at the shot's first and last frame
    targets: tuple[list[float], list[float]]       # ... and where it looks
    frame_h: float
    subject: str                                  # who or what the camera frames (a part's owner)
    on_screen: dict                               # name -> the plan's character entry
    gaze: tuple[dict, dict]                       # name -> whom they look at first, and last
    marks: dict                                   # name -> a mark this shot moves someone to
    rig: dict
    props: dict                                   # the props the sheet names -> their state
    acts_on: set = field(default_factory=set)     # props this shot's actions move
    walking: set = field(default_factory=set)
    accept: dict = field(default_factory=dict)    # rule -> why, from the acting sheet


def shot_view(p, shot_id: str) -> Shot | None:
    """The shot as continuity sees it; None for a black frame (nothing to keep continuous)."""
    pl = planner.build(p, shot_id, preview=True)
    if pl["blackout"]:
        return None
    d = dirn.direction_for(p, shot_id)
    keys = pl["camera"]["keys"]
    on_screen = {c["name"]: c for c in pl["characters"]}
    names = set(settings(p)["characters"]) | set(on_screen)

    def looks(first: bool) -> dict:
        out: dict = {}
        for b in sorted(d.beats, key=lambda b: b.t, reverse=not first):
            for who in ([b.who] if b.who else list(on_screen)):
                if who in on_screen and who not in out and b.gaze in names and b.gaze != who:
                    out[who] = b.gaze
        return out

    props = {x["kind"]: {k: v for k, v in x.items() if k != "motion"} for x in pl["props"] if x["kind"] in d.props}
    return Shot(id=shot_id, scene=pl["scene"], cams=(keys[0]["loc"], keys[-1]["loc"]),
                targets=(keys[0]["target"], keys[-1]["target"]), frame_h=float(pl["camera"]["frame_h"]),
                subject=pl["camera"]["subject"].split(":")[0], on_screen=on_screen, gaze=(looks(True), looks(False)),
                marks=dict(d.marks), rig=pl["rig"], props=props,
                acts_on={a.get("prop", "") for a in d.actions if a.get("do") != "sway"},
                walking={n for n, c in on_screen.items() if c["posture"] == WALKING},
                accept=dict(d.continuity))


# ---------------------------------------------------------------- geometry
def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _angle(u: list[float], v: list[float]) -> float:
    """Degrees between two vectors."""
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(u[i] * v[i] for i in range(3)) / (nu * nv)))))


def side(cam: list[float], a: list[float], b: list[float]) -> int:
    """Which side of the line from a to b (seen from above) the camera is on: +1 or -1, or 0 when it
    stands on the line, within NEUTRAL_DEG of it, seen from between them (a neutral shot)."""
    ab = (b[0] - a[0], b[1] - a[1])
    v = (cam[0] - (a[0] + b[0]) / 2, cam[1] - (a[1] + b[1]) / 2)
    cross = ab[0] * v[1] - ab[1] * v[0]
    dot = ab[0] * v[0] + ab[1] * v[1]
    if math.degrees(math.atan2(abs(cross), abs(dot))) < NEUTRAL_DEG:
        return 0
    return 1 if cross > 0 else -1


def screen_x(cam: list[float], target: list[float], v: list[float]) -> float:
    """How far a direction `v` points to the right of the frame (+) or left (-), seen through a camera
    at `cam` looking at `target`."""
    f = _sub(target, cam)
    return v[0] * f[1] - v[1] * f[0]                # v . (f x up), up = +z


def _look(sh: Shot, k: int, who: str, at: dict) -> tuple[float, float]:
    """Someone's look at `at` in frame k (0 first, 1 last): how far it points across the frame (-1 left
    .. +1 right), and its angle from the lens."""
    h, t = planner.head_point(sh.on_screen[who]), planner.head_point(at)
    v = _sub(t, h)
    f = _sub(sh.targets[k], sh.cams[k])
    norm = math.sqrt(sum(x * x for x in v)) * math.hypot(f[0], f[1])
    return (screen_x(sh.cams[k], sh.targets[k], v) / norm if norm > 1e-9 else 0.0), _angle(v, _sub(sh.cams[k], h))


def _deg(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


# ---------------------------------------------------------------- the rules
def axis(sh: Shot, where: dict, lines: dict) -> list[tuple[str, str]]:
    """The 180-degree rule, between everyone on screen and everyone else in the scene. `lines` holds,
    through the scene, each pair's side of the line, the shot that set it, and which way round they were."""
    out = []
    for a, b in sorted({tuple(sorted((x, y))) for x in sh.on_screen for y in where if y != x}):
        pa, pb = where[a]["at"], where[b]["at"]
        start, end = side(sh.cams[0], pa, pb), side(sh.cams[1], pa, pb)
        before, set_by, was_right = lines.get((a, b), (0, "", True))
        if before and start and start != before:
            now = "right" if not was_right else "left"
            out.append(("axis", f"the camera crosses the line between {a} and {b}: {b} was to the "
                                f"{'right' if was_right else 'left'} of {a} in {set_by}, here to the {now}. Keep "
                                f"the camera on {set_by}'s side, or go through a shot on the line"))
        lines[(a, b)] = (end, sh.id, screen_x(sh.cams[1], sh.targets[1], _sub(pb, pa)) > 0)
    return out


def jump(prev: Shot, sh: Shot) -> list[tuple[str, str]]:
    """The 30-degree rule."""
    if sh.subject != prev.subject or sh.subject == "vista":
        return []
    turn = _angle(_sub(prev.cams[1], prev.targets[1]), _sub(sh.cams[0], sh.targets[0]))
    size = max(prev.frame_h, sh.frame_h) / max(1e-6, min(prev.frame_h, sh.frame_h))
    if turn >= JUMP_DEG or size >= JUMP_SIZE:
        return []
    return [("jump", f"the same subject ({sh.subject}) only {turn:.0f} degrees round and {size:.2f}x the size: "
                     f"the cut jumps. Move the camera at least {JUMP_DEG:.0f} degrees, or change the size "
                     f"at least {JUMP_SIZE}x")]


def eyeline(prev: Shot | None, sh: Shot, where: dict) -> list[tuple[str, str]]:
    """Looks that meet across the cut, and looks into the lens."""
    out = []
    for k in (0, 1):                                         # at either end of the shot
        for who, at in sh.gaze[k].items():
            if at in sh.on_screen or at not in where:
                continue
            _, lens = _look(sh, k, who, where[at])
            if lens < LENS_DEG:
                out.append(("eyeline", f"{who} looks at {at} (off screen) {lens:.0f} degrees from the lens: it "
                                       f"reads as a look at the audience. Move the camera off {who}'s eyeline"))
    if prev is None:
        return list(dict.fromkeys(out))
    sideways = math.sin(math.radians(LENS_DEG))             # a look that points across the frame at all
    for who, at in prev.gaze[1].items():                     # the last look before the cut ...
        if at not in where:
            continue
        before, _ = _look(prev, 1, who, where[at])
        for looker, other, same in ((at, who, False), (who, at, True)):   # ... and the first after it
            if sh.gaze[0].get(looker) != other or looker not in sh.on_screen or other not in where:
                continue
            after, _ = _look(sh, 0, looker, where[other])
            if abs(before) < sideways or abs(after) < sideways:
                continue                                     # into or out of the depth: no screen direction
            if ((before > 0) != (after > 0)) == same:        # the same look flips, or a look back does not
                out.append(("eyeline", f"{who} looks {'right' if before > 0 else 'left'} at {at} in {prev.id}, "
                                       f"and {looker} looks {'right' if after > 0 else 'left'} at {other} here: "
                                       + ("the same look should keep its direction" if same
                                          else "looks that meet should point opposite ways")))
    return list(dict.fromkeys(out))


def light(prev: Shot, sh: Shot) -> list[tuple[str, str]]:
    a, b = prev.rig, sh.rig
    ka, kb = a.get("key", {}), b.get("key", {})
    changes = []
    if (_deg(ka.get("azimuth", 0), kb.get("azimuth", 0)) > LIGHT_DEG
            or abs(ka.get("elevation", 0) - kb.get("elevation", 0)) > LIGHT_DEG):
        changes.append(f"the key light moves (azimuth {ka.get('azimuth')} -> {kb.get('azimuth')}, "
                       f"elevation {ka.get('elevation')} -> {kb.get('elevation')})")
    if max((abs(x - y) for x, y in zip(ka.get("color", []), kb.get("color", []))), default=0.0) > LIGHT_COLOUR:
        changes.append(f"the key light's colour changes ({ka.get('color')} -> {kb.get('color')})")
    ea, eb = float(ka.get("energy", 1.0)), float(kb.get("energy", 1.0))
    if max(ea, eb) > LIGHT_ENERGY * max(1e-6, min(ea, eb)):
        changes.append(f"the key light's strength changes ({ea} -> {eb})")
    if max((abs(x - y) for x, y in zip(a.get("world", []), b.get("world", []))), default=0.0) > LIGHT_COLOUR:
        changes.append(f"the ambient light changes ({a.get('world')} -> {b.get('world')})")
    if bool(a.get("practical")) != bool(b.get("practical")):
        changes.append("a practical light " + ("goes out" if a.get("practical") else "comes on"))
    return [("light", f"{c} within the scene") for c in changes]


def props(prev: Shot, sh: Shot) -> list[tuple[str, str]]:
    out = []
    for name in sorted(set(prev.props) & set(sh.props) - prev.acts_on):
        a, b = prev.props[name], sh.props[name]
        moved = math.dist(a.get("at", [0, 0, 0]), b.get("at", [0, 0, 0]))
        turned = max((_deg(x, y) for x, y in zip(a.get("rot", []), b.get("rot", []))), default=0.0)
        if moved > PROP_MOVE_M or turned > PROP_TURN_DEG:
            out.append(("props", f"the {name} moves {moved:.2f} m and turns {turned:.0f} degrees across the cut"))
        for k in sorted((set(a) | set(b)) - set(STILL_STATE)):
            if a.get(k) != b.get(k):
                out.append(("props", f"the {name}'s {k} changes across the cut ({a.get(k)} -> {b.get(k)})"))
    return out


def position(prev: Shot, sh: Shot) -> list[tuple[str, str]]:
    out = []
    for name in sorted(set(prev.on_screen) & set(sh.on_screen) - prev.walking - sh.walking):
        a, b = prev.on_screen[name], sh.on_screen[name]
        moved, turned = math.dist(a["at"], b["at"]), _deg(a["yaw"], b["yaw"])
        if moved > POSITION_M or turned > TURN_DEG:
            out.append(("position", f"{name} jumps {moved:.2f} m and turns {turned:.0f} degrees across the cut"))
    return out


# ---------------------------------------------------------------- a chapter
def check(p, chapter: int) -> list[Finding]:
    """Every continuity finding at the cuts of a chapter, in order; accepted ones carry their reason."""
    cfg = settings(p)
    out: list[Finding] = []
    prev: Shot | None = None
    where: dict = {}
    lines: dict = {}
    for s in p.chapter(chapter):
        sh = shot_view(p, s.id)
        if sh is None:
            continue
        if prev is None or sh.scene != prev.scene:           # a new scene: everyone back on the set's marks
            prev, lines = None, {}
            where = {n: planner.mark_entry(cfg, n, m, s.chapter)
                     for n, m in planner.set_for(p, sh.scene).get("marks", {}).items()}
        where.update({n: planner.mark_entry(cfg, n, m, s.chapter) for n, m in sh.marks.items()})
        where.update(sh.on_screen)
        found = axis(sh, where, lines) + eyeline(prev, sh, where)
        if prev is not None:
            found += jump(prev, sh) + light(prev, sh) + props(prev, sh) + position(prev, sh)
        out += [Finding(sh.id, prev.id if prev else "", rule, msg, sh.accept.get(rule, "")) for rule, msg in found]
        prev = sh
    return out
