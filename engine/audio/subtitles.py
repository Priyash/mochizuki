"""Subtitles: an .srt per language for a chapter, timed on the cut, from the lines dialogue.py placed."""
from __future__ import annotations

from pathlib import Path

from ..cgi.config import settings
from .dialogue import offsets, shot_manifest

TAIL_S = 0.4          # a cue stays this long after its line ends ...
GAP_S = 0.05          # ... but always ends this long before the next cue starts
SUBS_DIR = "subs"


def _ts(t: float, sep: str = ",") -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    sec, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d}{sep}{ms:03d}"


def write_srt(p, chapter: int, language: str = "en") -> Path:
    """An .srt for the chapter in `language`, from the placed lines (laughs are not subtitled)."""
    cfg = settings(p)
    off = offsets(p, chapter)
    cues = []
    for sid, t0 in off.items():
        man = shot_manifest(p, sid)
        if not man:
            continue
        for ln in man["lines"]:
            if ln["nonverbal"] or language not in ln["lines"]:
                continue
            x = ln["lines"][language]
            cues.append((t0 + x["start"], t0 + x["end"] + TAIL_S, x["text"]))
    cues.sort()
    out = (p.root / cfg["work"] / SUBS_DIR / f"chapter_{chapter}.{language}.srt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{i}\n{_ts(a)} --> {_ts(min(b, cues[i][0] - GAP_S) if i < len(cues) else b)}\n{text}\n\n"
                           for i, (a, b, text) in enumerate(cues, start=1)), encoding="utf-8")
    return out
