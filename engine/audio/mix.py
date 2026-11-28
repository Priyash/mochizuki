"""Mixing: the chapter's audio tracks, one per language. Today the dialogue, shot after shot; music and
sound effects join here when their generators exist."""
from __future__ import annotations

from pathlib import Path

from ..cgi.config import settings, work_dir
from . import lipsync, voice

AUDIO_DIR = "audio"


def chapter_dialogue(p, chapter: int, language: str = "en") -> Path | None:
    """The chapter's dialogue track: each shot's track, or its silence, end to end. None if no shot speaks."""
    cfg = settings(p)
    parts, any_voice = [], False
    for s in p.chapter(chapter):
        f = work_dir(p, cfg, s.id) / f"audio-{language}.wav"
        if f.exists():
            any_voice = True
            samples, rate = lipsync.read_wav(f)
            need = int(round(s.seconds * rate))
            parts.extend(samples[:need] + [0.0] * max(0, need - len(samples)))
        else:
            parts.extend([0.0] * int(round(s.seconds * voice.RATE)))
    if not any_voice:
        return None
    return lipsync.write_wav(p.root / cfg["work"] / AUDIO_DIR / f"chapter_{chapter}.{language}.wav", parts, voice.RATE)
