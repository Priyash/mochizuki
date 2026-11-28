"""Lip sync accuracy, measured on synthetic speech whose timing is known to the sample: where the lips
close, which shape each syllable gets, and whether the result is in sync by the broadcast limits
(ITU-R BT.1359). The check itself is tested too: a mouth put out of sync must fail it."""
from __future__ import annotations

import math

import pytest

from engine.audio import lipsync as L

RATE = 48000
FPS = 24
TOLERANCE_S = 0.03            # a closure within this of its murmur's middle (ITU: sound first is seen at 45 ms)
FORMANTS = {"A": (730, 1090), "I": (270, 2290), "U": (300, 870), "E": (530, 1840), "O": (570, 840)}
PITCH_HZ = 140
QUIET, LOUD = 0.04, 0.15      # an [m] murmur well under the vowel, and one only ~7 dB under it
SUB_FRAME = (-1.5, -1.0, -0.5)   # where a line's sound can start between two frames (see line_mouth)


def vowel(sec: float, v: str, amp: float = 0.5) -> list[float]:
    """A vowel: its pitch and two formants, swelling in and out."""
    f1, f2 = FORMANTS[v]
    n = int(sec * RATE)
    return [amp * math.sin(math.pi * i / n) ** 0.5 * (0.6 * math.sin(2 * math.pi * PITCH_HZ * i / RATE)
                                                      + 0.3 * math.sin(2 * math.pi * f1 * i / RATE)
                                                      + 0.1 * math.sin(2 * math.pi * f2 * i / RATE)) for i in range(n)]


def murmur(sec: float, amp: float) -> list[float]:
    """The hum of a shut-lipped m: the pitch alone."""
    return [amp * math.sin(2 * math.pi * PITCH_HZ * i / RATE) for i in range(int(sec * RATE))]


def silence(sec: float) -> list[float]:
    return [0.0] * int(sec * RATE)


# name: (sound, syllables, the middle of each lips-shut stretch, vowels as (shape, start, end))
def lines(m_amp: float) -> dict:
    return {
        "mama": (silence(0.20) + murmur(0.08, m_amp) + vowel(0.18, "A") + murmur(0.08, m_amp) + vowel(0.18, "A")
                 + silence(0.25), L.syllables_ja("まま"), [0.24, 0.50], [("A", 0.28, 0.46), ("A", 0.54, 0.72)]),
        "come home": (silence(0.20) + vowel(0.24, "O") + murmur(0.08, m_amp) + silence(0.18) + vowel(0.24, "O")
                      + murmur(0.08, m_amp) + silence(0.25), L.syllables_en("come home"), [0.48, 0.98],
                      [("O", 0.20, 0.44), ("O", 0.70, 0.94)]),
        "hello, moon": (silence(0.20) + vowel(0.16, "E") + murmur(0.04, 0.1) + vowel(0.22, "O") + silence(0.18)
                        + murmur(0.08, m_amp) + vowel(0.28, "U") + silence(0.25), L.syllables_en("hello, moon"),
                        [0.84], [("E", 0.20, 0.36), ("O", 0.40, 0.62), ("U", 0.88, 1.16)]),
    }


CASES = [(name, amp) for amp in (QUIET, LOUD) for name in lines(amp)]


def run(sound, syllables, display0=-1.0):
    shut: list[float] = []
    m = L.mouth(sound, RATE, syllables, FPS, closures=shut, display0=display0)
    return m, shut, L.sync_report(sound, RATE, m, FPS, shut, display0)


def shown(f: int, display0: float) -> float:
    """The moment of sound frame f shows (it is on screen LEAD_FRAMES earlier)."""
    return (f + display0 + L.LEAD_FRAMES + 0.5) / FPS


@pytest.mark.parametrize(("name", "amp"), CASES)
def test_each_closure_lands_on_its_murmur(name, amp):
    sound, syllables, expected, _ = lines(amp)[name]
    _, shut, _ = run(sound, syllables)
    assert len(shut) == len(expected)
    for got, want in zip(sorted(shut), expected):
        assert abs(got - want) <= TOLERANCE_S, (name, amp, shut)


@pytest.mark.parametrize(("name", "amp"), CASES)
def test_shut_through_each_closure_and_the_right_shape_on_each_vowel(name, amp):
    sound, syllables, expected, vowels = lines(amp)[name]
    m, _, _ = run(sound, syllables)
    opened = L.openness(m)
    for t in expected:                                       # the frame showing a murmur's middle is shut
        f = min(range(len(opened)), key=lambda i: abs(shown(i, -1.0) - t))
        assert opened[f] == 0.0, (name, t)
    for shape, a, b in vowels:                               # a vowel's middle shows its own shape
        f = min(range(len(opened)), key=lambda i: abs(shown(i, -1.0) - (a + b) / 2))
        assert m[L.CHANNEL[shape]][f] > 0, (name, shape)


@pytest.mark.parametrize(("name", "amp"), CASES)
@pytest.mark.parametrize("display0", SUB_FRAME)
def test_in_sync_by_the_broadcast_limits(name, amp, display0):
    sound, syllables, *_ = lines(amp)[name]
    _, shut, rep = run(sound, syllables, display0)
    assert rep["ok"], rep
    assert rep["timing_ms"] and -L.SOUND_FIRST_MS <= rep["timing_ms"][0] <= rep["timing_ms"][1] <= L.PICTURE_FIRST_MS
    assert rep["closures"] == [len(shut), len(shut)] and rep["open_in_silence"] == 0 and rep["unmoved"] == 0


def test_fast_speech_keeps_its_vowel_between_two_closures():
    sound = murmur(0.03, QUIET) + vowel(0.07, "A") + murmur(0.05, QUIET) + vowel(0.20, "A") + silence(0.1)
    m, shut, rep = run(sound, L.syllables_ja("まま"))
    opened = L.openness(m)
    shut_frames = [i for i, x in enumerate(opened) if x == 0.0]
    between = [i for i in range(len(opened)) if shown(i, -1.0) < shut[1] and opened[i] > 0]
    assert len(shut) == 2 and between and shut_frames                # closed, open, closed, open: at 9 syllables/s
    assert rep["ok"], rep


def test_a_vowel_change_without_a_dip_changes_the_shape():
    m, _, rep = run(silence(0.20) + vowel(0.20, "A") + vowel(0.20, "I") + silence(0.25), L.syllables_ja("あい"))
    a = next(f for f in range(len(m["voice"])) if m[L.CHANNEL["A"]][f] > 0)
    i = next(f for f in range(len(m["voice"])) if m[L.CHANNEL["I"]][f] > 0)
    assert a < i and abs(shown(i, -1.0) - 0.40) <= 1.5 / FPS and rep["ok"]


def test_a_line_starts_a_frame_early_on_the_shots_frames(tmp_path):
    sound, *_ = lines(QUIET)["mama"]
    wav = L.write_wav(tmp_path / "mama.wav", sound, RATE)
    start = 1.013                                            # between frames 24 and 25
    m, rep, first = L.line_mouth(wav, "mama", "ja", FPS, kana="まま", start_s=start)
    assert first == L.first_frame(start, FPS) == round(start * FPS) - L.LEAD_FRAMES
    assert len(m["voice"]) == math.ceil(len(sound) / (RATE / FPS)) and rep["ok"], rep


# ---------------------------------------------------------------- the check catches what is wrong
def test_the_check_fails_a_mouth_out_of_sync():
    sound, syllables, *_ = lines(QUIET)["hello, moon"]
    m, shut, _ = run(sound, syllables)
    late = {ch: [0.0] * 4 + v[:-4] for ch, v in m.items()}     # four frames late: the sound comes 167 ms first
    rep = L.sync_report(sound, RATE, late, FPS, shut, -1.0)
    assert not rep["ok"] and rep["timing_ms"][0] < -L.SOUND_FIRST_MS
    early = {ch: v[5:] + [0.0] * 5 for ch, v in m.items()}     # five early: the picture 208 ms first
    rep = L.sync_report(sound, RATE, early, FPS, shut, -1.0)
    assert not rep["ok"] and rep["timing_ms"][1] > L.PICTURE_FIRST_MS


def test_the_check_fails_a_missing_closure_and_a_mouth_open_in_silence():
    sound, syllables, *_ = lines(QUIET)["come home"]
    m, shut, _ = run(sound, syllables)
    flat = {ch: [max(v)] * len(v) if ch != "voice" else v for ch, v in m.items()}   # open all the time
    rep = L.sync_report(sound, RATE, flat, FPS, shut, -1.0)
    assert not rep["ok"] and rep["closures"][0] < len(shut) and rep["open_in_silence"] > 0
    shut_none = {ch: [0.0] * len(v) for ch, v in m.items()}    # never opens
    rep = L.sync_report(sound, RATE, shut_none, FPS, shut, -1.0)
    assert not rep["ok"] and rep["unmoved"] > 0


# ---------------------------------------------------------------- reading the text
def test_english_syllables_and_their_closures():
    def read(w):
        return [(s.vowel, s.closes_before, s.closes_after) for s in L.syllables_en(w)]
    assert read("the") == [("E", False, False)] and read("she") == [("E", False, False)]   # the e is the vowel
    assert read("make") == [("A", True, False)]                    # silent e; m closes before
    assert read("home") == [("O", False, True)]                    # m closes after
    assert read("table") == [("A", False, False), ("E", True, False)]   # syllabic -le, and its b
    assert read("phone") == [("O", False, False)]                  # ph is f: no closure


def test_japanese_reads_its_kana():
    assert [(s.vowel, s.closes_before, s.closes_after) for s in L.syllables_ja("さんま")] == [
        ("A", False, True), ("A", True, False)]                    # ん shuts the lips, and so does ま
    assert L.syllables_for("ありがとう", "ja") == L.syllables_ja("ありがとう")   # kana only: read as written
    with pytest.raises(ValueError, match="kana"):
        L.syllables_for("前より", "ja")                            # kanji: needs its reading
    assert L.syllables_for("前より", "ja", "まえより")[0].closes_before
