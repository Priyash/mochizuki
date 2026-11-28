"""The film's dialogue, voiced, placed in its shots, lip-synced and subtitled. No actors needed.

    voice_shot(p, "1-11")        voices every line of the shot in every language, places them in time
                                 (never overlapping, after their `at`), makes the mouth curves for the
                                 primary language, and mixes one shot-length track per language
    offsets(p, chapter)          where each shot starts in the chapter's cut
The chapter's .srt files are subtitles.py; its dialogue track is mix.py.

What it writes per shot, in media/cgi/<shot>/:
    dialogue.json        every line: who, start and end per language, on screen or not, mouth curves
    audio-<lang>.wav     the shot's dialogue track (silence where nobody speaks)
Voiced lines are cached in media/cgi/voices/, so only changed lines are voiced again.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..project import DIALOGUE_FILE

from ..cgi.config import by_chapter, settings, work_dir
from . import lipsync, voice

GAP = 0.25            # seconds between two lines in the same shot
TAIL = 0.05           # a line should end this long before the cut


@dataclass
class Line:
    shot: str
    who: str
    at: float
    en: str
    ja: str = ""
    ja_kana: str = ""
    emotion: str = "neutral"
    intensity: float = 0.5
    restraint: float = 0.0
    delivery: str = ""
    on_screen: bool = True
    nonverbal: bool = False

    def text(self, language: str) -> str:
        return self.ja if language == "ja" and self.ja else self.en


def script_file(p) -> Path:
    return p.style_bible / DIALOGUE_FILE


def load(p) -> dict:
    f = script_file(p)
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"lines": [], "voices": {}, "cast": {}}


def lines_for(p, shot_id: str, script: dict | None = None) -> list[Line]:
    script = script or load(p)
    keep = Line.__dataclass_fields__
    found = [Line(**{k: v for k, v in ln.items() if k in keep}) for ln in script.get("lines", [])
             if ln.get("shot") == shot_id]
    return sorted(found, key=lambda ln: ln.at)


def voice_name(script: dict, who: str, chapter: int) -> str:
    cast = script.get("cast", {}).get(who, who)
    return by_chapter(cast, chapter, who) if isinstance(cast, dict) else cast


def voice_spec(script: dict, name: str, language: str, engine: str | None = None) -> dict:
    v = script.get("voices", {}).get(name)
    if v is None:
        raise KeyError(f"no voice {name!r} in style_bible/dialogue.json")
    spec = dict(v.get(language) or v.get("en") or {})
    if engine:
        spec["engine"] = engine
    spec.setdefault("describe", v.get("describe", ""))
    return spec


def instruct_for(line: Line) -> str:
    """How to say it, for engines that take directions (Qwen3-TTS and the like)."""
    held = " Hold the feeling back." if line.restraint >= 0.6 else ""
    return f"{line.emotion}, intensity {line.intensity:.1f}: {line.delivery}.{held}".strip()


@dataclass
class Placed:
    who: str
    on_screen: bool
    nonverbal: bool
    voice: str
    lines: dict = field(default_factory=dict)          # language -> {"wav", "start", "end", "text"}
    mouth: dict = field(default_factory=dict)          # channel -> per-frame values for the whole shot


def _duration(wav: Path) -> float:
    samples, rate = lipsync.read_wav(wav)
    return len(samples) / rate


def voice_shot(p, shot_id: str, languages: list[str] | None = None, engine: str | None = None,
               python: str | None = None) -> dict:
    cfg = settings(p)
    script = load(p)
    s = next((x for x in p.shots if x.id == shot_id), None)
    if s is None:
        raise KeyError(f"no shot {shot_id} in shots.csv")
    lines = lines_for(p, shot_id, script)
    out_dir = work_dir(p, cfg, shot_id)
    if not lines:
        return {"shot": shot_id, "lines": 0}
    languages = languages or script.get("languages", ["en"])
    primary = script.get("primary", languages[0])
    fps = int(cfg["fps"])
    frames = int(round(s.seconds * fps))
    cache = p.root / cfg["work"] / "voices" if not Path(cfg["work"]).is_absolute() else Path(cfg["work"]) / "voices"
    placed: list[Placed] = []
    overruns = []
    wavs: dict = {}
    for ln in lines:
        name = voice_name(script, ln.who, s.chapter)
        for lang in languages:
            spec = voice_spec(script, name, lang, engine)
            wavs[(id(ln), lang)] = voice.speak(spec, ln.text(lang), lang, cache / name / lang, kana=ln.ja_kana,
                                               instruct=instruct_for(ln), python=python)
    timing = {lang: fit(lines, [wavs[(id(ln), lang)] for ln in lines], s.seconds, cache / "_fitted" / lang)
              for lang in languages}
    for lang, t in timing.items():
        overruns += [{"who": lines[i].who, "language": lang, "text": lines[i].text(lang)[:40], "by_seconds": over}
                     for i, over in t["overruns"]]
    for k, ln in enumerate(lines):
        pl = Placed(ln.who, ln.on_screen, ln.nonverbal, voice_name(script, ln.who, s.chapter))
        for lang in languages:
            wav, start, end = timing[lang]["placed"][k]
            pl.lines[lang] = {"wav": str(wav), "start": round(start, 3), "end": round(end, 3), "text": ln.text(lang)}
        lp = pl.lines[primary]
        m = lipsync.for_line(Path(lp["wav"]), ln.text(primary), primary, fps, ln.ja_kana)
        first = int(round(lp["start"] * fps))
        for ch, vals in m.items():
            row = [0.0] * frames
            for i, v in enumerate(vals):
                if 0 <= first + i < frames:
                    row[first + i] = v
            pl.mouth[ch] = row
        placed.append(pl)
    tracks = {}
    for lang in languages:
        tracks[lang] = str(mix(out_dir / f"audio-{lang}.wav", s.seconds,
                               [(Path(pl.lines[lang]["wav"]), pl.lines[lang]["start"]) for pl in placed]))
    manifest = {"shot": shot_id, "primary": primary, "languages": languages, "fps": fps, "frames": frames,
                "lines": [asdict(pl) for pl in placed], "audio": tracks, "overruns": overruns}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dialogue.json").write_text(json.dumps(manifest), encoding="utf-8")
    return {"shot": shot_id, "lines": len(placed), "overruns": overruns, "audio": tracks}


MAX_SPEEDUP = 1.15   # beyond this, a line sounds rushed: the reviewer rewrites it instead


def fit(lines: list[Line], wavs: list[Path], seconds: float, scratch: Path) -> dict:
    """Place lines in a shot: each at its `at` or after the previous line plus a breath. If the last
    one would run past the cut, first pull the lines earlier (into the silence before the first one),
    then tighten the breaths, then speed the speech up a little (at most MAX_SPEEDUP). Whatever is
    still too long is reported, for the reviewer to rewrite."""
    durs = [_duration(w) for w in wavs]
    limit = seconds - TAIL

    def place(gap: float, speed: float = 1.0) -> list[tuple[float, float]]:
        t, out = 0.0, []
        for ln, d in zip(lines, durs):
            start = max(ln.at, t + (gap if out else 0.0))
            out.append((start, start + d / speed))
            t = start + d / speed
        return out

    spans = place(GAP)
    over = spans[-1][1] - limit if spans else 0.0
    if over > 0 and spans:
        pull = min(over, max(0.0, spans[0][0] - 0.2))
        spans = [(a - pull, b - pull) for a, b in spans]
        over -= pull
    if over > 0:
        spans = place(0.12)
        pull = min(max(0.0, spans[-1][1] - limit), max(0.0, spans[0][0] - 0.2))
        spans = [(a - pull, b - pull) for a, b in spans]
        over = spans[-1][1] - limit
    speed = 1.0
    if over > 0:
        talk = sum(durs)
        speed = min(MAX_SPEEDUP, talk / max(0.1, talk - over))
        spans = place(0.12, speed)
        pull = min(max(0.0, spans[-1][1] - limit), max(0.0, spans[0][0] - 0.2))
        spans = [(a - pull, b - pull) for a, b in spans]
        over = spans[-1][1] - limit
    out_wavs = list(wavs)
    if speed > 1.0:
        out_wavs = [voice.tempo(w, speed, scratch) for w in wavs]
    overruns = [(len(lines) - 1, round(over, 2))] if over > 0.01 else []
    return {"placed": [(w, a, b) for w, (a, b) in zip(out_wavs, spans)], "speed": round(speed, 3),
            "overruns": overruns}


def mix(out: Path, seconds: float, parts: list[tuple[Path, float]], rate: int = voice.RATE) -> Path:
    """One shot-length mono track with each line placed at its start (pure Python)."""
    track = [0.0] * int(round(seconds * rate))
    for wav, start in parts:
        samples, r = lipsync.read_wav(wav)
        if r != rate:
            raise ValueError(f"{wav}: {r} Hz, expected {rate} Hz")
        a = int(round(start * rate))
        for i, x in enumerate(samples):
            if a + i >= len(track):
                break
            track[a + i] += x
    return lipsync.write_wav(out, [max(-1.0, min(1.0, x)) for x in track], rate)


def shot_manifest(p, shot_id: str) -> dict | None:
    f = work_dir(p, settings(p), shot_id) / "dialogue.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


# ---------------------------------------------------------------- the chapter
def offsets(p, chapter: int) -> dict:
    t, out = 0.0, {}
    for s in p.chapter(chapter):
        out[s.id] = t
        t += s.seconds
    return out
