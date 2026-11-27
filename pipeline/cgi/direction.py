"""The acting sheet of a shot: what the camera does, who feels what and when, and how it is timed.

A text prompt cannot direct a performance; an acting sheet can. Each shot gets one:

    camera     framing (ECU CU MCU MS MLS WIDE INSERT), angle, lens, move, subject
    beats      moments of feeling: when (seconds), who, which emotion, how strong, how much
               it is held in (restraint), where they look, what they do, how long they hold it
    timing     ones / twos / threes (characters on twos by default), impact holds
    performance  procedural (from the beats), capture (a recorded human performance) or reference

The categories are fixed words rather than free text, so they drive the rig directly; the
free text (action, notes) is for the person acting the shot. This borrows the idea of
AniMatrix (arXiv 2605.03652): anime has conventions, not physics, so make them explicit
directives instead of hoping a model infers them.

`draft(p, shot)` writes a first sheet from the shot description with plain rules; the
hand-written sheets in style_bible/acting.json win over the draft, field by field. Every
sheet is a proposal until the owner sets "status": "approved".
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

EMOTIONS = ("neutral", "joy", "sorrow", "tenderness", "wonder", "fear", "resolve", "curiosity",
            "surprise", "anger", "bittersweet")
FRAMINGS = ("ECU", "CU", "MCU", "MS", "MLS", "WIDE", "INSERT")
ANGLES = ("eye", "three-quarter", "profile", "behind", "low", "high", "overhead")
MOVES = ("static", "push", "pull", "pan", "tilt", "drift", "track")
TIMINGS = ("ones", "twos", "threes", "mixed")
PERFORMANCES = ("procedural", "capture", "reference")
LENS = {"ECU": 100, "CU": 85, "MCU": 65, "MS": 50, "MLS": 35, "WIDE": 24, "INSERT": 100}

CHARACTER_NAMES = {"tsumugi": "tsumugi", "sayo": "sayo", "akari": "akari", "child": "child", "children": "child"}
PROP_WORDS = {"mallet": "mallet", "mortar": "mortar", "bowl": "bowl", "lantern": "lantern", "lanterns": "lantern",
              "rice": "rice", "mochi": "rice", "dumplings": "dumplings", "tray": "tray", "cushion": "cushion",
              "pampas": "pampas", "silver grass": "pampas", "radio": "radio", "hearth": "hearth",
              "jacket": "jacket", "ink line": "ink_line", "screen": "screen", "seed cloth": "seed_cloth"}
LANDSCAPE = re.compile(r"\b(paddies|paddy|sea|moon|sky|rocket|horizon|village|colony|crater|milky way|terraces|"
                       r"launch tower|riverbank|coast|earth|stars|cave)\b", re.I)

# words in a shot description -> beats. (emotion, intensity, restraint, gaze, action)
LEXICON = [
    (r"\blaugh(s|ing|ter)?\b", ("joy", 0.85, 0.0, "", "laughs")),
    (r"\bsmil(e|es|ing)\b", ("joy", 0.55, 0.0, "", "smiles")),
    (r"holding herself together", ("sorrow", 0.75, 0.85, "", "holds herself together")),
    (r"not turning", ("sorrow", 0.45, 0.9, "away", "does not turn")),
    (r"\btraces?\b", ("tenderness", 0.6, 0.2, "bowl", "traces it slowly")),
    (r"looks? up", ("wonder", 0.65, 0.0, "up", "looks up")),
    (r"\bpeeks?\b", ("curiosity", 0.6, 0.0, "", "peeks")),
    (r"\bcopies\b", ("tenderness", 0.45, 0.2, "", "copies the gesture")),
    (r"presses .* into her hands", ("resolve", 0.7, 0.6, "", "presses it into her hands")),
    (r"(cannot|can't) run", ("fear", 0.75, 0.0, "", "tries and cannot")),
    (r"\b(moonquake|shudders|falls)\b", ("fear", 0.6, 0.0, "", "")),
    (r"\bmends?\b", ("resolve", 0.5, 0.4, "", "mends it")),
    (r"\bher face\b", ("bittersweet", 0.6, 0.5, "", "")),
    (r"\bwatching\b", ("bittersweet", 0.5, 0.4, "", "watches")),
    (r"\bnames\b", ("tenderness", 0.6, 0.1, "", "names it")),
    (r"\bcatches\b", ("surprise", 0.6, 0.0, "", "catches it")),
]


@dataclass
class Beat:
    t: float                     # seconds from the start of the shot
    emotion: str = "neutral"
    intensity: float = 0.5       # 0..1
    who: str = ""                # character, "" = everyone in the shot
    restraint: float = 0.0       # 0..1: feeling held in. The face moves less; the eyes do more
    gaze: str = ""               # camera, up, down, away, a character or a prop
    action: str = ""             # for the person acting the shot
    hold: float = 0.0            # seconds the pose is held once reached


@dataclass
class Camera:
    framing: str = "MS"
    angle: str = "eye"
    lens: float = 50.0
    move: str = "static"
    amount: float = 0.0          # push/pull: fraction of the distance; pan/tilt: degrees; drift: strength
    subject: str = ""            # a character, "a+b" for a two-shot, a prop, or "name:hands"
    focus: str = ""              # what is sharp (default: the subject)
    fstop: float = 2.8


@dataclass
class Direction:
    shot: str
    seconds: float
    type: str
    scene: str = ""
    characters: list = field(default_factory=list)
    props: list = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)
    beats: list = field(default_factory=list)
    timing: str = "twos"
    impacts: list = field(default_factory=list)     # seconds where a hit lands: the pose is held there
    performance: str = "procedural"
    posture: dict = field(default_factory=dict)     # character -> standing, kneeling, seiza, walking
    props_state: dict = field(default_factory=dict)  # e.g. {"bowl": {"seams": 2}, "mallet": {"at": [...]}}
    actions: list = field(default_factory=list)     # what props do: {"prop": "mallet", "do": "strike", "at": 1.9}
    marks: dict = field(default_factory=dict)       # this shot only: {"sayo": {"at": [x, y, 0], "yaw": 145}}
    blackout: bool = False                          # a black frame (the film opens on one)
    dialogue: str = ""
    status: str = "draft"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Direction:
        d = dict(d)
        cam = Camera(**{k: v for k, v in (d.pop("camera", None) or {}).items() if k in Camera.__dataclass_fields__})
        beats = [Beat(**{k: v for k, v in b.items() if k in Beat.__dataclass_fields__}) for b in d.pop("beats", [])]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(camera=cam, beats=beats, **known)


# ---------------------------------------------------------------- drafting from the shot sheet
def _characters(text: str, chapter: int) -> list[str]:
    found = []
    for word, name in CHARACTER_NAMES.items():
        if re.search(rf"\b{word}\b", text, re.I) and name not in found:
            found.append(name)
    if not found and re.search(r"\b(she|her)\b", text, re.I) and chapter >= 3:
        found.append("tsumugi")              # from chapter 3 on, an unnamed "she" is Tsumugi
    if not found and re.search(r"\b(both|grandmother)\b", text, re.I):
        found = ["sayo", "tsumugi"]
    if re.search(r"\bboth\b", text, re.I) and chapter <= 2:
        found = list(dict.fromkeys(found + ["sayo", "tsumugi"]))
    return found


def _props(text: str) -> list[str]:
    out = []
    for word, name in PROP_WORDS.items():
        if re.search(rf"\b{word}\b", text, re.I) and name not in out:
            out.append(name)
    return out


def _camera(text: str, s_type: str, chars: list[str], props: list[str]) -> Camera:
    t = text
    framing = None
    if re.search(r"\bCU\b", t) and re.search(r"\b(eyes|hands|finger|feet)\b", t, re.I):
        framing = "INSERT" if re.search(r"\b(hands|finger|feet)\b", t, re.I) else "ECU"
    for f in ("ECU", "MCU", "CU", "MS", "MLS"):
        if framing is None and re.search(rf"\b{f}\b", t):
            framing = f
    if framing is None and re.search(r"two-shot", t, re.I):
        framing = "MS"
    if framing is None and re.search(r"\binsert\b", t, re.I):
        framing = "INSERT"
    if framing is None and re.search(r"\bwide\b|^interior\b", t, re.I):
        framing = "WIDE" if re.search(r"\bwide\b", t, re.I) else "MLS"
    if framing is None:
        framing = "MS" if chars else ("WIDE" if LANDSCAPE.search(t) or not props else "INSERT")
    angle = "eye"
    if re.search(r"from behind|\bbehind\b|not turning", t, re.I):
        angle = "behind"
    elif re.search(r"^low\b|\blow:", t, re.I):
        angle = "low"
    elif re.search(r"two-shot", t, re.I):
        angle = "three-quarter"
    move, amount = "static", 0.0
    if s_type == "Still":
        move, amount = "push", 0.05          # a held frame still breathes: a slow push with real parallax
    elif re.search(r"^ACTION\b", t):
        move, amount = "drift", 0.35
    elif framing in ("CU", "ECU") and s_type == "Hero":
        move, amount = "push", 0.06
    elif re.search(r"\bwalks?\b|\bchases?\b|\bcarry|\bcarries\b", t, re.I):
        move, amount = "pan", 25.0
    elif re.search(r"\b(climbs|rises|clears)\b", t, re.I):
        move, amount = "tilt", 8.0
    subject = ""
    if framing == "INSERT" and re.search(r"\b(hands|finger)\b", t, re.I) and chars:
        subject = f"{chars[0]}:hands"
    elif re.search(r"two-shot|\bboth\b", t, re.I) and len(chars) >= 2:
        subject = f"{chars[0]}+{chars[1]}"
    elif chars and framing != "INSERT":
        subject = chars[-1] if re.search(r"\bturns\b", t) and len(chars) > 1 else chars[0]
    elif re.search(r"\bmoon\b", t, re.I) and not props:
        subject = "moon"
    elif framing in ("WIDE", "MLS"):
        subject = "vista"                             # a wide of a place frames the place, not one thing in it
    elif props:
        subject = props[0]
    return Camera(framing=framing, angle=angle, lens=float(LENS[framing]), move=move, amount=amount,
                  subject=subject, fstop=2.0 if framing in ("ECU", "INSERT", "CU") else 2.8)


def _beats(text: str, seconds: float, chars: list[str]) -> list[Beat]:
    beats: list[Beat] = []
    if not chars:
        return beats                                  # a shot of things: they have no feelings to time
    who = chars[0]
    for pattern, (emo, inten, restr, gaze, action) in LEXICON:
        if re.search(pattern, text, re.I):
            t = round(min(seconds * 0.25, 1.0), 2) if not beats else round(min(seconds - 0.5, beats[-1].t + 1.2), 2)
            beats.append(Beat(t=t, emotion=emo, intensity=inten, who=who, restraint=restr, gaze=gaze,
                              action=action, hold=round(min(1.0, seconds * 0.2), 2)))
    if chars and not beats:
        beats.append(Beat(t=0.0, emotion="neutral", intensity=0.3, who=who, hold=round(seconds * 0.5, 2)))
    return beats


def _posture(text: str, chars: list[str], chapter: int) -> dict:
    out = {}
    for c in chars:
        if re.search(r"\bkneel", text, re.I) or (chapter == 1 and c == "tsumugi" and re.search(r"rice|mortar|hands", text, re.I)):
            out[c] = "kneeling"
        elif re.search(r"\bwalks?\b|\bchases?\b|\bcarr", text, re.I):
            out[c] = "walking"
        else:
            out[c] = "standing"
    return out


def draft(p, s, scene: str = "") -> Direction:
    """A first acting sheet for shot `s`, from its description alone."""
    text = s.shot
    chars = _characters(text, s.chapter)
    props = _props(text)
    timing = "mixed" if re.match(r"ACTION\b", text) else ("twos" if chars else "ones")
    impacts, actions = [], []
    if "mallet" in props and re.search(r"\b(falls|swings|thud|striking|rises|lifts)\b", text, re.I):
        impacts = [round(s.seconds * 0.45, 2)]
        actions.append({"prop": "mallet", "do": "strike", "at": impacts[0]})
    if "lantern" in props:
        actions.append({"prop": "lantern", "do": "sway"})
    if props and re.search(r"\bbreeze\b|\bmoving in\b", text, re.I):
        actions.append({"prop": props[0], "do": "sway"})
    dialogue = s.sound if ":" in s.sound and not s.sound.upper().startswith(("SFX", "AMBIENCE", "SCORE")) else ""
    return Direction(shot=s.id, seconds=float(s.seconds), type=s.type, scene=scene, characters=chars, props=props,
                     camera=_camera(text, s.type, chars, props), beats=_beats(text, s.seconds, chars),
                     timing=timing, impacts=impacts, posture=_posture(text, chars, s.chapter), actions=actions,
                     blackout=bool(re.match(r"black screen", text, re.I)), dialogue=dialogue)


# ---------------------------------------------------------------- the hand-written sheets
def acting_file(p) -> Path:
    return p.root / "style_bible" / "acting.json"


def load_acting(p) -> dict:
    f = acting_file(p)
    if not f.exists():
        return {}
    data = json.loads(f.read_text(encoding="utf-8"))
    return {k: v for k, v in data.get("shots", {}).items()}


def merge(base: Direction, override: dict) -> Direction:
    """Fields in the hand-written sheet replace the draft's; camera is merged key by key."""
    d = base.to_dict()
    for k, v in override.items():
        if k.startswith("_"):
            continue
        if k == "camera" and isinstance(v, dict):
            d["camera"] = {**d["camera"], **v}
            if "framing" in v and "lens" not in v:
                d["camera"]["lens"] = float(LENS.get(v["framing"], d["camera"]["lens"]))
        else:
            d[k] = v
    return Direction.from_dict(d)


def scene_of(p, shot_id: str) -> str:
    f = p.root / "style_bible" / "scenes.json"
    if not f.exists():
        return ""
    for name, sc in json.loads(f.read_text(encoding="utf-8")).get("scenes", {}).items():
        if shot_id in sc.get("shots", []):
            return name
    return ""


def direction_for(p, shot_id: str) -> Direction:
    """The acting sheet the shot renders with: the draft, with style_bible/acting.json on top."""
    s = next((x for x in p.shots if x.id == shot_id), None)
    if s is None:
        raise KeyError(f"no shot {shot_id} in shots.csv")
    d = draft(p, s, scene_of(p, shot_id))
    over = load_acting(p).get(shot_id)
    return merge(d, over) if over else d


def problems(d: Direction) -> list[str]:
    """What is wrong with a sheet, in words. Empty when it can be rendered."""
    out = []
    c = d.camera
    if c.framing not in FRAMINGS:
        out.append(f"camera.framing {c.framing!r} is not one of {FRAMINGS}")
    if c.angle not in ANGLES:
        out.append(f"camera.angle {c.angle!r} is not one of {ANGLES}")
    if c.move not in MOVES:
        out.append(f"camera.move {c.move!r} is not one of {MOVES}")
    if d.timing not in TIMINGS:
        out.append(f"timing {d.timing!r} is not one of {TIMINGS}")
    if d.performance not in PERFORMANCES:
        out.append(f"performance {d.performance!r} is not one of {PERFORMANCES}")
    for b in d.beats:
        if b.emotion not in EMOTIONS:
            out.append(f"beat at {b.t}s: emotion {b.emotion!r} is not one of {EMOTIONS}")
        if not 0 <= b.t <= d.seconds:
            out.append(f"beat at {b.t}s is outside the shot (0-{d.seconds}s)")
        if not 0 <= b.intensity <= 1 or not 0 <= b.restraint <= 1:
            out.append(f"beat at {b.t}s: intensity and restraint go from 0 to 1")
        if b.who and b.who not in d.characters:
            out.append(f"beat at {b.t}s is for {b.who!r}, who is not in the shot {d.characters}")
    for t in d.impacts:
        if not 0 <= t <= d.seconds:
            out.append(f"impact at {t}s is outside the shot")
    return out
