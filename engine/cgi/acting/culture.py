"""The culture pack: how a feeling is shown, in the film's culture, by whom and in front of whom.

People feel alike; cultures differ in what they let show. Ekman and Friesen called these display rules
("The repertoire of nonverbal behavior", Semiotica 1, 1969): to show a feeling as it is, to play it down,
to mask it. Matsumoto measured them across cultures ("Cultural similarities and differences in display
rules", Motivation and Emotion 14, 1990; "Mapping expressive differences around the world", Journal of
Cross-Cultural Psychology 39, 2008): in Japan, sadness and anger are played down more than almost
anywhere, the more so before others; a child's open feeling is allowed. The pack writes such rules as
data (style_bible/culture.json), and the acting sheets follow them.

    {"culture": "ja",
     "rules": [{"name": "elders hold grief in",
                "when": {"emotion": ["sorrow", "bittersweet"], "age": [60, 120]},
                "then": {"restraint_min": 0.8, "gaze": "down"},
                "why": "..."}]}

when (every condition must hold; a beat for "" is everyone's, so who and age need one character):
    emotion      the beat's emotion is one of these
    who          the beat is one of these characters'
    age          [min, max]: the character's age in this chapter (config cgi.characters.<name>.look)
    with_age     [min, max]: someone else in the shot is this old
    intensity    [min, max]: the beat's intensity
then (in order, rule after rule):
    restraint_min, restraint_max   hold the feeling in at least, or at most, this much
    intensity_scale                show it this many times as strongly
    gaze                           look here, when the beat looks nowhere in particular
    hold_min                       hold the pose at least this long (ma: the pause that lets it land)

A hand-written beat keeps every field it gives: a rule fills or changes only what the sheet left to the
draft. Restraint is played the anime way (acting/retarget.py): the mouth and brows move less, the eyes
do more.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...film_check import read_checked
from ...project import CULTURE_FILE
from ..config import by_chapter, settings
from . import direction as dirn

WHEN = ("emotion", "who", "age", "with_age", "intensity")
THEN = ("restraint_min", "restraint_max", "intensity_scale", "gaze", "hold_min")
OWNS = {"restraint_min": "restraint", "restraint_max": "restraint", "intensity_scale": "intensity",
        "gaze": "gaze", "hold_min": "hold"}          # the beat field each effect changes
UNKNOWN_AGE = None


@dataclass
class Change:
    shot: str
    beat: float                  # the beat's time
    who: str
    rule: str
    fields: dict = field(default_factory=dict)   # field -> [before, after]


def load(p, check: bool = True) -> dict:
    """The film's file ({} when it has none). With `check`, a file with problems is refused, all of them
    named at once (FilmError)."""
    return read_checked(p.style_bible / CULTURE_FILE, problems if check else None)


def _range(x) -> bool:
    return isinstance(x, list) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x) and x[0] <= x[1]


def problems(spec: dict) -> list[str]:
    """What is wrong with culture.json, in words."""
    out = []
    for k, r in enumerate(spec.get("rules", [])):
        where = f"rules[{k}] ({r.get('name', 'no name')})" if isinstance(r, dict) else f"rules[{k}]"
        if not isinstance(r, dict) or not isinstance(r.get("when", {}), dict) or not isinstance(r.get("then"), dict):
            out.append(f"{where}: a rule is {{\"name\", \"when\": {{...}}, \"then\": {{...}}, \"why\"}}")
            continue
        for key, val in r.get("when", {}).items():
            if key not in WHEN:
                out.append(f"{where}: when.{key} is not one of {WHEN}")
            elif key == "emotion" and not set(val) <= set(dirn.EMOTIONS):
                out.append(f"{where}: when.emotion {sorted(set(val) - set(dirn.EMOTIONS))} are not emotions")
            elif key in ("age", "with_age", "intensity") and not _range(val):
                out.append(f"{where}: when.{key} is [min, max]")
        for key, val in r["then"].items():
            if key not in THEN:
                out.append(f"{where}: then.{key} is not one of {THEN}")
            elif key in ("restraint_min", "restraint_max") and not (isinstance(val, (int, float)) and 0 <= val <= 1):
                out.append(f"{where}: then.{key} goes from 0 to 1")
            elif key in ("intensity_scale", "hold_min") and not (isinstance(val, (int, float)) and val >= 0):
                out.append(f"{where}: then.{key} is a number, 0 or more")
        if not r.get("why"):
            out.append(f"{where}: say why (the convention it follows)")
    return out


def ages(p, chapter: int, names: list[str]) -> dict[str, float | None]:
    """Each character's age in this chapter, from their look; None when it is not given."""
    chars = settings(p).get("characters", {})
    out: dict[str, float | None] = {}
    for n in names:
        look = by_chapter(chars.get(n, {}).get("look", {}), chapter, {}) or {}
        out[n] = float(look["age"]) if "age" in look else UNKNOWN_AGE
    return out


def _within(x: float | None, rng: list) -> bool:
    return x is not None and rng[0] <= x <= rng[1]


def _matches(when: dict, b: dirn.Beat, who: str, age: dict, others: list[str]) -> bool:
    if "emotion" in when and b.emotion not in when["emotion"]:
        return False
    if "who" in when and who not in when["who"]:
        return False
    if "age" in when and not _within(age.get(who), when["age"]):
        return False
    if "with_age" in when and not any(_within(age.get(o), when["with_age"]) for o in others):
        return False
    return not ("intensity" in when and not _within(b.intensity, when["intensity"]))


def _effect(effect: str, before, val):
    """A beat field after one effect."""
    if effect == "restraint_min" or effect == "hold_min":
        return max(before, val)
    if effect == "restraint_max":
        return min(before, val)
    if effect == "intensity_scale":
        return round(min(1.0, before * val), 3)
    return before or val                                # gaze: only where the beat looks nowhere in particular


def apply(p, chapter: int, d: dirn.Direction, explicit: list[set] | None = None) -> tuple[dirn.Direction, list[Change]]:
    """The sheet with the culture's display rules played into its beats, and what they changed.
    `explicit`: for each beat, the fields its hand-written sheet gives (those are never changed)."""
    rules = load(p).get("rules", [])
    changes: list[Change] = []
    if not rules or not d.beats:
        return d, changes
    age = ages(p, chapter, d.characters)
    for k, b in enumerate(d.beats):
        given = explicit[k] if explicit is not None and k < len(explicit) else set()
        who = b.who or (d.characters[0] if len(d.characters) == 1 else "")
        others = [c for c in d.characters if c != who]
        for r in rules:
            if not _matches(r.get("when", {}), b, who, age, others):
                continue
            done: dict = {}
            for effect, val in r["then"].items():
                owned = OWNS.get(effect, "")
                if not owned or owned in given:
                    continue
                before = getattr(b, owned)
                after = _effect(effect, before, val)
                if after != before:
                    setattr(b, owned, after)
                    done[owned] = [before, after]
            if done:
                changes.append(Change(d.shot, b.t, who, r.get("name", ""), done))
    if changes:
        names = sorted({c.rule for c in changes})
        d.notes = (d.notes + " " if d.notes else "") + f"[culture: {'; '.join(names)}]"
    return d, changes


def explicit_fields(override: dict | None) -> list[set] | None:
    """The fields each hand-written beat gives; None when the sheet leaves the beats to the draft."""
    if not override or "beats" not in override:
        return None
    return [set(b) for b in override["beats"]]
