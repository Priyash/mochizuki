"""Voices: text to speech, then the voice's treatment (pitch, speed, the lantern's timbre).

Engines (pick per voice and language in style_bible/dialogue.json):

    kokoro    Kokoro-82M (Apache 2.0): English and Japanese voices, fast on a CPU, so it runs on
              GitHub's free runners as well as the Mac. pip install kokoro "misaki[ja]" soundfile
    command   any other engine as a command template (Qwen3-TTS through mlx-audio on the Mac,
              Chatterbox, CosyVoice...): placeholders {text} {voice} {lang} {speed} {out} {instruct}
    stub      a speech-like test tone, no model: tests and dry runs

Every line is cached by a hash of everything that shapes it, so a re-run only voices what changed.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path

from .lipsync import read_wav, vowels_en, vowels_ja, write_wav

RATE = 48000
VERSION = 2          # bump when the treatment changes, so cached lines are voiced again
KOKORO_SCRIPT = Path(__file__).parent / "tts" / "kokoro_say.py"


def line_key(spec: dict, text: str, language: str, instruct: str = "") -> str:
    blob = json.dumps({"spec": spec, "text": text, "lang": language, "instruct": instruct, "v": VERSION},
                      sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def kokoro_lang(voice: str) -> str:
    """Kokoro's language code from the voice name: af_/am_ American, bf_/bm_ British, jf_/jm_ Japanese."""
    return {"a": "a", "b": "b", "j": "j", "z": "z", "e": "e", "f": "f", "h": "h", "i": "i", "p": "p"}.get(voice[:1], "a")


def stub_speech(text: str, language: str, kana: str = "", rate: int = RATE, seed: int = 0) -> list[float]:
    """A test voice: one voiced burst per vowel, a pause at each comma and full stop."""
    rng = random.Random(seed)
    out: list[float] = [0.0] * int(0.08 * rate)
    formant = {"A": (730, 1090), "I": (270, 2290), "U": (300, 870), "E": (530, 1840), "O": (570, 840)}
    tokens = list(kana) if language == "ja" and kana else text.split()
    vowels = vowels_ja(kana) if language == "ja" and kana else vowels_en(text)
    per = max(1, len(vowels) // max(1, len(tokens)))
    vi = 0
    for t in tokens:
        for _ in range(per if language != "ja" else 1):
            if vi >= len(vowels):
                break
            f1, f2 = formant[vowels[vi]]
            vi += 1
            n = int(rate * rng.uniform(0.11, 0.16))
            for i in range(n):
                env = math.sin(math.pi * i / n) ** 0.7
                x = i / rate
                out.append(env * (0.45 * math.sin(2 * math.pi * 140 * x) + 0.25 * math.sin(2 * math.pi * f1 * x)
                                  + 0.12 * math.sin(2 * math.pi * f2 * x)))
            out.extend([0.0] * int(rate * 0.03))
        if any(p in t for p in ",.?!、。？！"):
            out.extend([0.0] * int(rate * 0.22))
    out.extend([0.0] * int(0.12 * rate))
    return out


def _ffmpeg_treatment(src: Path, dst: Path, spec: dict) -> Path:
    """Pitch in semitones (duration kept), speed, the lantern's timbre, loudness to -18 LUFS, 48 kHz mono."""
    if not shutil.which("ffmpeg"):
        shutil.copy(src, dst)
        return dst
    filters = []
    semis = float(spec.get("pitch", 0.0))
    if abs(semis) > 0.01:
        k = 2 ** (semis / 12)
        filters += [f"asetrate={int(RATE * k)}", f"aresample={RATE}", f"atempo={1 / k:.5f}"]
    if spec.get("fx") == "lantern":                       # a little hollow and shimmering: a voice from inside paper
        filters += ["chorus=0.6:0.85:45|58:0.35|0.28:0.30|0.42:1.8|1.1", "aecho=0.8:0.6:9:0.22",
                    "highpass=f=140"]
    filters += ["loudnorm=I=-18:TP=-2:LRA=7", f"aresample={RATE}"]
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-af", ",".join(filters),
                    "-ac", "1", "-ar", str(RATE), "-sample_fmt", "s16", str(dst)], check=True)
    return dst


def trim_silence(wav: Path, threshold: float = 0.012, pad: float = 0.04) -> Path:
    """Cut the silence engines leave before and after a line (a breath of `pad` is kept), so a line's
    length is its speech and it lands where it is placed."""
    samples, rate = read_wav(wav)
    loud = [i for i, x in enumerate(samples) if abs(x) > threshold]
    if not loud:
        return wav
    a = max(0, loud[0] - int(pad * rate))
    b = min(len(samples), loud[-1] + int(pad * rate) + 1)
    if a > 0 or b < len(samples):
        write_wav(wav, samples[a:b], rate)
    return wav


def speak(spec: dict, text: str, language: str, out_dir: Path, kana: str = "", instruct: str = "",
          python: str | None = None) -> Path:
    """Voice one line with `spec` ({"engine", "voice", "speed", "pitch", "fx", ...}); returns the WAV path.
    Already voiced lines (same text and settings) are reused."""
    key = line_key(spec, text, language, instruct)
    final = out_dir / f"{key}.wav"
    if final.exists():
        return final
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{key}.raw.wav"
    engine = spec.get("engine", "kokoro")
    if engine == "stub":
        write_wav(raw, stub_speech(text, language, kana, seed=int(key[:6], 16)), RATE)
    elif engine == "kokoro":
        subprocess.run([python or sys.executable, str(KOKORO_SCRIPT), "--text", text, "--voice", spec["voice"],
                        "--lang", kokoro_lang(spec["voice"]), "--speed", str(spec.get("speed", 1.0)),
                        "--out", str(raw)], check=True)
    elif engine == "command":
        fill = {"text": text, "voice": spec.get("voice", ""), "lang": language, "speed": str(spec.get("speed", 1.0)),
                "out": str(raw), "instruct": instruct}
        subprocess.run([part.format(**fill) for part in spec["cmd"]], check=True)
    else:
        raise ValueError(f"unknown voice engine {engine!r} (kokoro, command or stub)")
    if not raw.exists() or raw.stat().st_size < 1000:
        raise RuntimeError(f"the {engine} engine wrote no audio for: {text[:60]}")
    treated = dict(spec)
    if engine == "stub":
        treated.pop("speed", None)
    _ffmpeg_treatment(raw, final, treated)
    raw.unlink(missing_ok=True)
    return trim_silence(final)


def tempo(wav: Path, factor: float, out_dir: Path) -> Path:
    """The same line, spoken `factor` times faster (pitch kept)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{wav.stem}.x{factor:.3f}.wav"
    if not out.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav), "-af", f"atempo={factor:.4f}",
                        "-ac", "1", "-ar", str(RATE), "-sample_fmt", "s16", str(out)], check=True)
    return out
