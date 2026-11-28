"""Mouths that move with the voice, the way anime does it (kuchipaku).

From a line's audio and its text: the loudness per frame says when the mouth is open and how
far; the vowels of the text say which shape (A I U E O), spread over the voiced stretches in
order. Openness is cut to anime's few steps (closed, half, open) and held on twos, so it reads
as drawn lip flaps rather than a puppet. Japanese vowels come straight from the kana; English
vowels from spelling, which is close enough for anime flaps.

Pure Python: the WAV is read with the standard library.
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SHAPES = ("A", "I", "U", "E", "O")
CHANNEL = {v: f"Fcl_MTH_{v}" for v in SHAPES}

# hiragana and katakana -> vowel (the small tsu, the long-vowel mark and ん close the mouth)
_KANA_ROWS = {
    "A": "あかさたなはまやらわがざだばぱぁゃアカサタナハマヤラワガザダバパァャ",
    "I": "いきしちにひみりぎじぢびぴぃイキシチニヒミリギジヂビピィ",
    "U": "うくすつぬふむゆるぐずづぶぷぅゅゥウクスツヌフムユルグズヅブプュ",
    "E": "えけせてねへめれげぜでべぺぇエケセテネヘメレゲゼデベペェ",
    "O": "おこそとのほもよろをごぞどぼぽぉょオコソトノホモヨロヲゴゾドボポォョ",
}
KANA = {ch: v for v, row in _KANA_ROWS.items() for ch in row}
SMALL_Y = set("ゃゅょャュョ")       # きゃ is one syllable: the small ya replaces the i


def vowels_ja(kana: str) -> list[str]:
    out: list[str] = []
    for ch in kana:
        v = KANA.get(ch)
        if v is None:
            if ch in "ーっッ" and out:
                continue                                   # long vowel / pause: no new shape
            continue
        if ch in SMALL_Y and out:
            out[-1] = v
            continue
        out.append(v)
    return out


_EN_DIGRAPHS = [("ee", "I"), ("ea", "I"), ("ie", "I"), ("oo", "U"), ("ou", "A"), ("ow", "O"), ("oa", "O"),
                ("ai", "E"), ("ay", "E"), ("ei", "E"), ("au", "O"), ("aw", "O"), ("oi", "O"), ("oy", "O")]
_EN_SINGLE = {"a": "A", "e": "E", "i": "I", "o": "O", "u": "U", "y": "I"}


def vowels_en(text: str) -> list[str]:
    """Rough vowel shapes from English spelling: one per vowel group, digraphs first."""
    out: list[str] = []
    for word in "".join(c.lower() if c.isalpha() else " " for c in text).split():
        if len(word) > 2 and word.endswith("e") and word[-2] not in "aeiouy":
            word = word[:-1]                               # silent e
        i = 0
        while i < len(word):
            pair = word[i:i + 2]
            hit = next((v for d, v in _EN_DIGRAPHS if pair == d), None)
            if hit:
                out.append(hit)
                i += 2
                continue
            v = _EN_SINGLE.get(word[i])
            if v and (word[i] != "y" or i > 0):
                out.append(v)
                i += 1
                while i < len(word) and word[i] in "aeiouy":
                    i += 1                                 # a vowel run is one syllable
                continue
            i += 1
    return out or ["A"]


def read_wav(path: Path) -> tuple[list[float], int]:
    """Mono samples in -1..1 and the sample rate (16-bit PCM; stereo is averaged)."""
    with wave.open(str(path), "rb") as w:
        n, ch, width, rate = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"{path}: 16-bit PCM WAV expected, got {width * 8}-bit")
    ints = struct.unpack(f"<{n * ch}h", raw)
    vals = [sum(ints[i:i + ch]) / ch for i in range(0, len(ints), ch)] if ch > 1 else list(ints)
    return [v / 32768.0 for v in vals], rate


def write_wav(path: Path, samples: list[float], rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack(f"<{len(samples)}h", *(max(-32767, min(32767, int(s * 32767))) for s in samples)))
    return path


def energy(samples: list[float], rate: int, fps: int = 24) -> list[float]:
    """Loudness per video frame (RMS), scaled so the line's loud parts reach about 1."""
    hop = rate / fps
    n = int(math.ceil(len(samples) / hop))
    out = []
    for f in range(n):
        a, b = int(f * hop), int(min(len(samples), (f + 1) * hop))
        seg = samples[a:b] or [0.0]
        out.append(math.sqrt(sum(s * s for s in seg) / len(seg)))
    ref = sorted(out)[int(len(out) * 0.9)] if out else 1.0
    return [min(1.0, x / ref) if ref > 1e-6 else 0.0 for x in out]


def mouth(samples: list[float], rate: int, vowels: list[str], fps: int = 24, on: int = 2,
          threshold: float = 0.18) -> dict:
    """Per-frame mouth channels (Fcl_MTH_A..O, 0..1) and 'voice' (loudness) for one line."""
    e = energy(samples, rate, fps)
    voiced = [x > threshold for x in e]
    n_voiced = sum(voiced) or 1
    out = {CHANNEL[v]: [0.0] * len(e) for v in SHAPES}
    k = 0
    for f, x in enumerate(e):
        if not voiced[f]:
            continue
        v = vowels[min(len(vowels) - 1, int(k * len(vowels) / n_voiced))] if vowels else "A"
        k += 1
        level = 1.0 if x > 0.6 else 0.55                  # anime's steps: half open, open
        out[CHANNEL[v]][f] = level * (1.0 if v in ("A", "O") else 0.8)
    for name, vals in out.items():                        # held on twos, like drawn flaps
        out[name] = [vals[(i // on) * on] for i in range(len(vals))]
    out["voice"] = e
    return out


def for_line(wav: Path, text: str, language: str, fps: int = 24, kana: str = "") -> dict:
    samples, rate = read_wav(wav)
    v = vowels_ja(kana) if language == "ja" and kana else vowels_en(text)
    return mouth(samples, rate, v, fps)
