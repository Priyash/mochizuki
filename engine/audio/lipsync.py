"""Mouths that move with the voice, the way anime does it (kuchipaku), timed for dubbing.

For one voiced line, from its audio and its text:

1. Syllables from the text: one vowel shape (A I U E O) per syllable, and whether a bilabial (m, b, p,
   and Japanese ん) shuts the lips before or after it. Japanese comes from the kana; other languages
   from spelling, which is close enough for anime's five shapes. Bilabial closures matter most: they
   are the mouth shapes viewers notice when they are missing.
2. The vowel band (VOWEL_BAND_HZ): everything below is measured on the sound between 500 Hz and
   3 kHz, where the vowels' formants are. The murmur of a shut-lipped m b p lies below it and the hiss
   of s sh f above, so neither opens the mouth, however loud.
3. Syllable nuclei: the peaks of its loudness contour, each kept only if a dip of at least MIN_DIP_DB
   separates it from the last (de Jong & Wempe, "Praat script to detect syllable nuclei and measure
   speech rate automatically", Behavior Research Methods 41, 2009). The text's syllables are laid on
   the peaks in order; where there are fewer peaks than syllables, the missing ones share the voiced
   time between their neighbours.
4. Closures: the lips are shut from where the vowel before stops to where the vowel after is
   released (where the band's loudness falls or rises most over EDGE_S), so a closure lasts as long
   as its murmur or silence.
5. Frames: the shape of the syllable sounding then, at anime's steps (closed, half, open), held on
   twos like drawn flaps. Each frame shows the sound LEAD_FRAMES later, and a line's mouth starts on
   the shot's frame before its sound, timed to the sample: viewers notice sound before picture at
   about 45 ms, picture before sound only at about 125 ms (ITU-R BT.1359). A closure shuts its whole
   held pair, unless that would hide a vowel (fast speech keeps its vowels).

`sync_report` measures the result against those limits where a viewer can time it: at every phrase's
start and end, at 10 ms; whether every closure is on screen; whether the mouth is ever open in silence.

Pure Python: the WAV is read with the standard library.
"""
from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

SHAPES = ("A", "I", "U", "E", "O")
CHANNEL = {v: f"Fcl_MTH_{v}" for v in SHAPES}

HOP_S = 0.010                  # the loudness contour: 10 ms steps ...
WINDOW_S = 0.025               # ... of 25 ms windows
MIN_DIP_DB = 2.0               # de Jong & Wempe: a nucleus needs this dip before it
VOICED = 0.18                  # loudness (of the line's loud parts) above which the mouth is open
OPEN = 0.6                     # ... and above which it is fully open
CLOSURE_S = 0.08               # a bilabial closure lasts about this long
LEAD_FRAMES = 1                # each frame shows the sound this many frames ahead (see above)
SOUND_FIRST_MS = 45.0          # ITU-R BT.1359: sound ahead of picture is noticed beyond this ...
PICTURE_FIRST_MS = 125.0       # ... picture ahead of sound beyond this
HALF_OPEN = 0.55               # anime's steps: closed, half open (quieter sound), open
NARROW = 0.8                   # I U E open less than A O
LOUD_PERCENTILE = 0.9          # loudness is relative to the line's loud parts: this percentile is 1
PAUSE_S = 0.4                  # syllable nuclei further apart than this have a pause between them
MAX_LAG_FRAMES = 6             # sync_report searches this far either way for the best match
MIN_PAUSE_S = 0.15             # a dip in the sound this long is a pause between phrases, not a consonant
EDGE_S = 0.03                  # a vowel's release (or stop) is where its loudness rises (falls) most over this
VOWEL_BAND_HZ = (500.0, 3000.0)   # the vowels' formants; m b p murmur below, s sh f hiss above
SILENCE = 0.05                 # loudness below this is silence ...
SPEECH_MARGIN_FRAMES = 2       # ... but the mouth may lead or close this near speech
JA = "ja"                      # read from its kana, not its spelling
KANA_BLOCK = ("\u3040", "\u30ff")   # hiragana and katakana: text with anything else needs a kana reading

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
BILABIAL_KANA = set("まみむめもばびぶべぼぱぴぷぺぽマミムメモバビブベボパピプペポ")
NASAL_N = set("んン")


@dataclass(frozen=True)
class Syllable:
    vowel: str                     # A I U E O
    closes_before: bool = False    # a bilabial closes the lips before the vowel (ma, ba, pa)
    closes_after: bool = False     # ... or after it (him, rub; ん)


def syllables_ja(kana: str) -> list[Syllable]:
    out: list[Syllable] = []
    for ch in kana:
        if ch in NASAL_N and out:                          # ん is drawn closed in anime, and is [m] before m b p
            out[-1] = Syllable(out[-1].vowel, out[-1].closes_before, True)
            continue
        v = KANA.get(ch)
        if v is None:
            continue                                       # ー っ and punctuation: no new shape
        if ch in SMALL_Y and out:
            out[-1] = Syllable(v, out[-1].closes_before, out[-1].closes_after)
            continue
        out.append(Syllable(v, ch in BILABIAL_KANA))
    return out


def vowels_ja(kana: str) -> list[str]:
    return [s.vowel for s in syllables_ja(kana)]


_EN_DIGRAPHS = [("ee", "I"), ("ea", "I"), ("ie", "I"), ("oo", "U"), ("ou", "A"), ("ow", "O"), ("oa", "O"),
                ("ai", "E"), ("ay", "E"), ("ei", "E"), ("au", "O"), ("aw", "O"), ("oi", "O"), ("oy", "O")]
_EN_SINGLE = {"a": "A", "e": "E", "i": "I", "o": "O", "u": "U", "y": "I"}
_VOWEL_LETTERS = "aeiouy"


def _bilabial(cluster: str) -> bool:
    """Does a consonant cluster hold an m, b or p that is said (ph is f)?"""
    c = cluster.replace("ph", "")
    return any(ch in c for ch in "mbp")


def _silent_e(word: str) -> bool:
    """make, home: a final e after a consonant is silent, unless it is the word's only vowel (the, she)
    or ends a syllabic -le (table, simple)."""
    if len(word) < 3 or not word.endswith("e") or word[-2] in _VOWEL_LETTERS:
        return False
    if not any(c in _VOWEL_LETTERS for c in word[:-1]):
        return False
    return not (word[-2] == "l" and word[-3] not in _VOWEL_LETTERS)


def syllables_en(text: str) -> list[Syllable]:
    """Rough syllables from English spelling: one per vowel group (digraphs first, a silent e dropped),
    with the bilabials in the consonants before and after it."""
    out: list[Syllable] = []
    for word in "".join(c.lower() if c.isalpha() else " " for c in text).split():
        if _silent_e(word):
            word = word[:-1]
        groups: list[tuple[int, int, str]] = []            # (start, end, shape) of each vowel group
        i = 0
        while i < len(word):
            pair = word[i:i + 2]
            hit = next((v for d, v in _EN_DIGRAPHS if pair == d), None)
            if hit:
                groups.append((i, i + 2, hit))
                i += 2
                continue
            v = _EN_SINGLE.get(word[i])
            if v and (word[i] != "y" or i > 0):
                j = i + 1
                while j < len(word) and word[j] in _VOWEL_LETTERS:
                    j += 1                                 # a vowel run is one syllable
                groups.append((i, j, v))
                i = j
                continue
            i += 1
        for k, (a, b, v) in enumerate(groups):
            before = word[groups[k - 1][1]:a] if k else word[:a]
            after = word[b:] if k == len(groups) - 1 else ""   # a word's last consonants close its last syllable
            out.append(Syllable(v, _bilabial(before), _bilabial(after)))
    return out or [Syllable("A")]


def vowels_en(text: str) -> list[str]:
    """Rough vowel shapes from English spelling: one per vowel group, digraphs first."""
    return [s.vowel for s in syllables_en(text)]


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


def _rms(samples: list[float], a: int, b: int) -> float:
    seg = samples[max(0, a):max(0, b)]
    return math.sqrt(sum(s * s for s in seg) / len(seg)) if seg else 0.0


def _scale(values: list[float]) -> list[float]:
    """Loudness relative to the line's loud parts (its 90th percentile): those reach about 1."""
    ref = sorted(values)[int(len(values) * LOUD_PERCENTILE)] if values else 1.0
    return [min(1.0, x / ref) if ref > 1e-6 else 0.0 for x in values]


def energy(samples: list[float], rate: int, fps: int = 24, shift_frames: float = 0.0) -> list[float]:
    """Loudness per video frame (RMS of the frame's own stretch of sound, `shift_frames` later),
    scaled so the line's loud parts reach about 1."""
    hop = rate / fps
    n = int(math.ceil(len(samples) / hop))
    raw = [_rms(samples, int((f + shift_frames) * hop), int((f + shift_frames + 1) * hop)) for f in range(n)]
    return _scale(raw)


def _biquad(samples: list[float], rate: int, hz: float, high: bool) -> list[float]:
    """A second-order Butterworth high- or low-pass (the RBJ audio-EQ cookbook biquad)."""
    w0 = 2 * math.pi * hz / rate
    alpha = math.sin(w0) / math.sqrt(2)                     # Q = 1/sqrt(2): Butterworth
    cw = math.cos(w0)
    a0 = 1 + alpha
    edge = (1 + cw) if high else (1 - cw)
    b0, b1, b2 = edge / 2 / a0, (-edge if high else edge) / a0, edge / 2 / a0
    a1, a2 = -2 * cw / a0, (1 - alpha) / a0
    out = [0.0] * len(samples)
    x1 = x2 = y1 = y2 = 0.0
    for i, x in enumerate(samples):
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i], x2, x1, y2, y1 = y, x1, x, y1, y
    return out


def vowel_band(samples: list[float], rate: int) -> list[float]:
    """The sound between VOWEL_BAND_HZ: the vowels' formants, which the open mouth shapes. The murmur
    of a shut-lipped m b p lies below it and the hiss of s sh f above, so neither opens the mouth."""
    lo, hi = VOWEL_BAND_HZ
    return _biquad(_biquad(samples, rate, lo, high=True), rate, hi, high=False)


def contour(samples: list[float], rate: int) -> list[float]:
    """The loudness contour in dB, every HOP_S, lightly smoothed (three steps)."""
    hop, win = int(rate * HOP_S), int(rate * WINDOW_S)
    n = max(1, int(math.ceil(len(samples) / hop)))
    db = [20 * math.log10(_rms(samples, i * hop - win // 2, i * hop + win // 2) + 1e-9) for i in range(n)]
    return [sum(db[max(0, i - 1):i + 2]) / len(db[max(0, i - 1):i + 2]) for i in range(n)]


def nuclei(db: list[float], voiced: list[bool]) -> list[int]:
    """Syllable peaks in the contour (indices): local maxima in voiced stretches, each kept only if a
    dip of MIN_DIP_DB separates it from the last kept one; when two are not separated, the louder stays."""
    peaks: list[int] = []
    for i in range(1, len(db) - 1):
        if not voiced[i] or not (db[i] >= db[i - 1] and db[i] > db[i + 1]):
            continue
        if peaks:
            dip = min(db[peaks[-1]:i + 1])
            if db[i] - dip < MIN_DIP_DB or db[peaks[-1]] - dip < MIN_DIP_DB:
                if db[i] > db[peaks[-1]]:
                    peaks[-1] = i                          # one syllable: keep its louder peak
                continue
        peaks.append(i)
    return peaks


def _voiced_steps(db_voiced: list[bool]) -> list[int]:
    return [i for i, v in enumerate(db_voiced) if v]


def align(n_syllables: int, peaks: list[int], voiced_steps: list[int]) -> list[float]:
    """Where each syllable's nucleus is (in contour steps). Peaks are taken in order; if there are more
    than syllables, the extra are the least prominent ones (handled by `nuclei` merging) and the
    surplus is spread; if fewer, the syllables between two found peaks share the voiced time between
    them evenly, on a clock that runs only while the line is voiced."""
    if n_syllables <= 0:
        return []
    if not voiced_steps:
        return [0.0] * n_syllables
    clock = {s: k for k, s in enumerate(voiced_steps)}     # contour step -> position on the voiced clock
    total = len(voiced_steps)

    def at_clock(x: float) -> float:
        return float(voiced_steps[min(total - 1, max(0, int(round(x))))])

    anchors = [clock[p] for p in peaks if p in clock]
    if len(anchors) > n_syllables:                          # keep n, spread evenly through the found peaks
        anchors = [anchors[round(k * (len(anchors) - 1) / max(1, n_syllables - 1))] for k in range(n_syllables)]
    if len(anchors) == n_syllables:
        return [at_clock(a) for a in anchors]
    if not anchors:                                         # no evidence of syllables: even on the voiced clock
        return [at_clock((k + 0.5) * total / n_syllables) for k in range(n_syllables)]
    # some peaks: give each to the syllable at its share of the voiced clock, in order
    owner: list[int] = []
    for a in anchors:
        k = min(n_syllables - 1, max(0, int(a * n_syllables / total)))
        k = max(k, owner[-1] + 1 if owner else 0)
        owner.append(min(k, n_syllables - (len(anchors) - len(owner))))
    pos = {k: float(a) for k, a in zip(owner, anchors)}
    known = sorted(pos)
    out: list[float] = []
    for k in range(n_syllables):                            # the rest, evenly between their known neighbours
        if k in pos:
            out.append(pos[k])
            continue
        left = max((j for j in known if j < k), default=None)
        right = min((j for j in known if j > k), default=None)
        lo = pos[left] if left is not None else 0.0
        hi = pos[right] if right is not None else float(total)
        span_l = left if left is not None else -1
        span_r = right if right is not None else n_syllables
        out.append(lo + (hi - lo) * (k - span_l) / (span_r - span_l))
    return [at_clock(x) for x in out]


def _steepest(loud: list[float], a: int, b: int, rise: bool) -> int:
    """The step in [a, b] where the loudness rises (or falls) most over EDGE_S around it: where a
    vowel is released from a closure (or stopped by one), on the vowel band's loudness, where a
    murmured [m] is quiet. Over EDGE_S, so a vowel that swells in still outweighs a murmur's sudden
    start; ties go to the sharpest single step."""
    n = len(loud)
    a, b = max(1, a), min(n - 1, b)
    if a > b:
        return min(max(0, b), n - 1)
    h = int(EDGE_S / HOP_S) // 2
    sign = 1.0 if rise else -1.0
    return max(range(a, b + 1), key=lambda i: (round(sign * (loud[min(n - 1, i + h)] - loud[max(0, i - h - 1)]), 6),
                                               sign * (loud[i] - loud[i - 1])))


def _closures(syls: list[Syllable], centres: list[float], loud: list[float]) -> list[tuple[float, float]]:
    """When the lips are shut for each bilabial, (start, end) in seconds: from where the vowel before
    stops to where the vowel after is released, both where the vowel band's loudness (`loud`) falls
    or rises fastest, so a closure lasts as long as its murmur or silence. Next to a pause or the
    line's edge, it lasts CLOSURE_S. Neighbours further than PAUSE_S are across a pause."""
    out = []
    reach = int(PAUSE_S / HOP_S)
    cs = [int(c) for c in centres]

    def at(i: int) -> float:
        return (i - 0.5) * HOP_S                            # between contour steps i - 1 and i

    for k, (s, ci) in enumerate(zip(syls, cs)):
        prev = cs[k - 1] if k and ci - reach <= cs[k - 1] < ci else None
        nxt = cs[k + 1] if k + 1 < len(cs) and ci < cs[k + 1] <= ci + reach else None
        if s.closes_before:
            release = _steepest(loud, ci - reach if prev is None else prev + 1, ci, rise=True)
            start = at(release) - CLOSURE_S if prev is None else at(_steepest(loud, prev + 1, release - 1, rise=False))
            out.append((start, at(release)))
        if s.closes_after:
            stop = _steepest(loud, ci + 1, ci + reach if nxt is None else nxt - 1, rise=False)
            end = at(stop) + CLOSURE_S if nxt is None else at(_steepest(loud, stop + 1, nxt, rise=True))
            out.append((at(stop), end))
    return sorted((max(0.0, a), max(a, b, 0.0) + (HOP_S if b <= a else 0.0)) for a, b in out)


def mouth(samples: list[float], rate: int, syllables: list, fps: int = 24, on: int = 2,
          lead_frames: int = LEAD_FRAMES, closures: list[float] | None = None, display0: float = 0.0,
          band: list[float] | None = None) -> dict:
    """Per-frame mouth channels (Fcl_MTH_A..O, 0..1) and 'voice' (loudness) for one line, all from
    the vowel band (see vowel_band; pass `band` if it is already made).

    `syllables` are Syllable or plain vowel letters. Pass `closures` (a list) to be given the seconds
    where the lips close. `display0`: when frame 0 is on screen, in frames after the sound starts
    (line_mouth starts a line's mouth a frame early, on the shot's frame grid)."""
    syls = [s if isinstance(s, Syllable) else Syllable(str(s)) for s in syllables] or [Syllable("A")]
    band = vowel_band(samples, rate) if band is None else band
    db = contour(band, rate)
    loud = _scale([10 ** (x / 20) for x in db])             # the contour's loudness, 0..1
    voiced = [x > VOICED for x in loud]
    centres = align(len(syls), nuclei(db, voiced), _voiced_steps(voiced))
    shut = _closures(syls, centres, loud)
    if closures is not None:
        closures.extend((a + b) / 2 for a, b in shut)
    shift = display0 + lead_frames                          # each frame shows the sound this much later

    def heard_at(i: int) -> float:
        return (i + shift + 0.5) / fps                      # the moment frame i shows

    e = energy(band, rate, fps, shift_frames=shift)
    out = {CHANNEL[v]: [0.0] * len(e) for v in SHAPES}
    mids = [(centres[k] + centres[k + 1]) / 2 for k in range(len(centres) - 1)]
    for f, x in enumerate(e):
        if x <= VOICED:
            continue
        t_step = heard_at(f) / HOP_S                        # this frame's sound, on the contour
        k = sum(1 for m in mids if t_step >= m)              # the syllable sounding then
        v = syls[min(k, len(syls) - 1)].vowel
        level = 1.0 if x > OPEN else HALF_OPEN
        out[CHANNEL[v]][f] = level * (1.0 if v in ("A", "O") else NARROW)
    nuclei_s = [c * HOP_S for c in centres]
    shut_frames: set[int] = set()                           # a closure shuts the frames showing it (at least
    for a, b in shut:                                       # the one at its middle) and the rest of their
        mid = int(round((a + b) / 2 * fps - shift - 0.5))   # held groups, but never a frame whose sound is
        span = range(int(math.ceil(a * fps - shift - 0.5)), int(math.floor(b * fps - shift - 0.5)) + 1)
        for g in {mid, *span}:                              # nearer a vowel's nucleus: fast speech keeps
            if not 0 <= g < len(e):                         # its vowels (before frame 0 the mouth is shut)
                continue
            shut_frames.add(g)
            for i in range((g // on) * on, min((g // on + 1) * on, len(e))):
                gap = max(a - heard_at(i), heard_at(i) - b, 0.0)
                if gap <= min((abs(heard_at(i) - c) for c in nuclei_s), default=math.inf):
                    shut_frames.add(i)
    broken = {i // on for i in shut_frames}                 # a group a closure splits is drawn on ones
    for name, vals in out.items():                          # held on `on`s, like drawn flaps
        out[name] = [0.0 if i in shut_frames else vals[i] if i // on in broken else vals[(i // on) * on]
                     for i in range(len(vals))]
    out["voice"] = [e[(i // on) * on] for i in range(len(e))]
    return out


def openness(m: dict) -> list[float]:
    n = len(m.get("voice", []))
    return [max(m[CHANNEL[v]][f] for v in SHAPES) for f in range(n)]


def _phrases(loud: list[float]) -> list[tuple[int, int]]:
    """The line's phrases on the contour (first, last step): voiced stretches, joined across the
    shorter dips of consonants (under MIN_PAUSE_S)."""
    gap = int(MIN_PAUSE_S / HOP_S)
    runs: list[tuple[int, int]] = []
    for i, x in enumerate(loud):
        if x <= VOICED:
            continue
        if runs and i - runs[-1][1] <= gap:
            runs[-1] = (runs[-1][0], i)
        else:
            runs.append((i, i))
    return runs


def sync_report(samples: list[float], rate: int, m: dict, fps: int = 24, closures: list[float] | None = None,
                display0: float = 0.0, band: list[float] | None = None) -> dict:
    """How well a line's mouth matches its sound, by the limits viewers notice (ITU-R BT.1359: the
    sound may come at most SOUND_FIRST_MS before the picture, the picture PICTURE_FIRST_MS before it).

    timing_ms: at every phrase's start and end (after or before a pause, where the mouth must open or
    shut and a viewer can time it), how far the picture's change comes before the sound's: positive is
    picture first, as it should be. [earliest-sound, latest-sound] of all of them; picture_first_ms is
    their median. An edge at a bilabial is judged by its closure instead. closures: [shown, total],
    a shut mouth within the limits of each. open_in_silence: frames open away from any sound.
    unmoved: phrases the mouth never opens for. Measured at 10 ms, not by the frame: ITU's limits
    are about a frame wide."""
    opened = openness(m)
    n = len(opened)
    t0 = display0 / fps                                     # frame f is on screen from t0 + f / fps

    def shown(f: int) -> float:
        return t0 + f / fps

    band = vowel_band(samples, rate) if band is None else band   # phrases: where vowels sound
    phrases = [((a - 0.5) * HOP_S, (b + 0.5) * HOP_S)
               for a, b in _phrases(_scale([10 ** (x / 20) for x in contour(band, rate)]))]
    shut = closures or []
    at_bilabial = lambda t: any(abs(c - t) <= CLOSURE_S for c in shut)   # noqa: E731
    timing: list[float] = []
    unmoved = 0
    for k, (on_s, off_s) in enumerate(phrases):
        lo = (phrases[k - 1][1] + on_s) / 2 if k else -math.inf
        hi = (off_s + phrases[k + 1][0]) / 2 if k + 1 < len(phrases) else math.inf
        mine = [f for f in range(n) if opened[f] > 0 and lo <= shown(f) + 0.5 / fps < hi]
        if not mine:
            unmoved += 1
            continue
        if not at_bilabial(on_s):
            timing.append((on_s - shown(mine[0])) * 1000)
        if not at_bilabial(off_s):
            timing.append((off_s - shown(mine[-1] + 1)) * 1000)
    seen = 0
    for t in shut:                                          # a shut frame on screen within the limits
        a, b = t - PICTURE_FIRST_MS / 1000, t + SOUND_FIRST_MS / 1000
        outside = a < shown(0) or b > shown(n)              # before or after the line: shut anyway
        seen += outside or any(opened[f] == 0.0 and shown(f) < b and shown(f + 1) > a for f in range(n))
    heard = energy(samples, rate, fps, shift_frames=display0)
    near: set[int] = set()
    for f in (f for f in range(n) if heard[f] > VOICED):
        near.update(range(f - SPEECH_MARGIN_FRAMES, f + SPEECH_MARGIN_FRAMES + 1))
    silent_open = sum(1 for f in range(n) if opened[f] > 0 and heard[f] < SILENCE and f not in near)
    in_limits = all(-SOUND_FIRST_MS <= x <= PICTURE_FIRST_MS for x in timing)
    ok = in_limits and seen == len(shut) and silent_open == 0 and unmoved == 0
    timing.sort()
    return {"picture_first_ms": round(timing[len(timing) // 2], 1) if timing else None,
            "timing_ms": [round(timing[0], 1), round(timing[-1], 1)] if timing else [],
            "closures": [seen, len(shut)], "open_in_silence": silent_open, "unmoved": unmoved, "ok": ok}


def syllables_for(text: str, language: str, kana: str = "") -> list[Syllable]:
    """The syllables a line's mouth is drawn with. Japanese from its kana reading (kanji give no
    reading of their own, so a line with any needs `kana`); every other language from its spelling."""
    if language != JA:
        return syllables_en(text)
    if not kana and any(ch.isalpha() and not KANA_BLOCK[0] <= ch <= KANA_BLOCK[1] for ch in text):
        raise ValueError(f"a Japanese line needs its kana reading (ja_kana) for lip sync: {text!r}")
    return syllables_ja(kana or text)


def first_frame(start_s: float, fps: int, lead_frames: int = LEAD_FRAMES) -> int:
    """The shot frame a line's mouth begins on: the frame its sound starts in, less the lead."""
    return int(round(start_s * fps)) - lead_frames


def line_mouth(wav: Path, text: str, language: str, fps: int = 24, kana: str = "", on: int = 2,
               lead_frames: int = LEAD_FRAMES, start_s: float = 0.0) -> tuple[dict, dict, int]:
    """One line's mouth channels, its sync report, and the shot frame the channels begin on. The
    line's sound starts `start_s` into the shot; its mouth `lead_frames` before, on the shot's
    frames, so it is timed to the sound to the sample, not to the nearest frame."""
    samples, rate = read_wav(wav)
    first = first_frame(start_s, fps, lead_frames)
    display0 = first - start_s * fps
    shut: list[float] = []
    band = vowel_band(samples, rate)
    m = mouth(samples, rate, syllables_for(text, language, kana), fps, on, lead_frames, closures=shut,
              display0=display0, band=band)
    return m, sync_report(samples, rate, m, fps, shut, display0, band), first
