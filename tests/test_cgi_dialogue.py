"""The automated dialogue: the script, voices (the stub engine: no model), placing lines in their
shot, lip sync, subtitles, the chapter's tracks, the voice driving the face in the plan, and the
cut carrying one audio track per language."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path

import pytest

from engine.cgi import auto, dialogue, lipsync, voice
from engine.cgi import plan as planner
from engine.project import Project

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def tone(path: Path, seconds: float, rate: int = voice.RATE) -> Path:
    return lipsync.write_wav(path, [0.3 * math.sin(2 * math.pi * 180 * i / rate)
                                    for i in range(int(seconds * rate))], rate)


def line(at: float, who: str = "sayo", en: str = "line") -> dialogue.Line:
    return dialogue.Line("9-99", who, at, en)


# ---------------------------------------------------------------- the script
def test_the_script_is_castable(proj):
    script = dialogue.load(proj)
    shots = {s.id: s for s in proj.shots}
    assert script["primary"] in script["languages"]
    for ln in script["lines"]:
        s = shots[ln["shot"]]                                         # a shot that exists
        name = dialogue.voice_name(script, ln["who"], s.chapter)
        for lang in script["languages"]:
            assert dialogue.voice_spec(script, name, lang)            # a voice for every language
        assert 0 <= ln["at"] < s.seconds, ln
        if ln.get("ja") and not ln.get("nonverbal"):
            assert lipsync.vowels_ja(ln["ja_kana"]), ln               # Japanese lip sync needs the kana
    assert dialogue.voice_name(script, "tsumugi", 1) != dialogue.voice_name(script, "tsumugi", 5)   # she ages


# ---------------------------------------------------------------- lip sync
def test_kana_gives_the_mouth_shapes():
    assert lipsync.vowels_ja("ありがとう") == ["A", "I", "A", "O", "U"]
    assert lipsync.vowels_ja("きょう") == ["O", "U"]                  # kyo is one syllable
    assert lipsync.vowels_ja("ちょっと") == ["O", "O"]                # the small tsu is a pause
    assert lipsync.vowels_ja("ラーメン") == ["A", "E"]                # long mark and n: no new shape


def test_english_shapes_from_spelling():
    assert lipsync.vowels_en("moon") == ["U"]
    assert lipsync.vowels_en("Hello") == ["E", "O"]
    assert lipsync.vowels_en("make") == ["A"]                         # silent e
    assert lipsync.vowels_en("...") == ["A"]


def test_the_mouth_flaps_with_the_voice_on_twos():
    rate = 48000
    speech = [0.5 * math.sin(2 * math.pi * 200 * i / rate) for i in range(rate // 2)]
    samples = [0.0] * (rate // 4) + speech + [0.0] * (rate // 4)     # quiet, half a second of voice, quiet
    m = lipsync.mouth(samples, rate, ["A", "O"], fps=24)
    opened = [max(m[c][f] for c in lipsync.CHANNEL.values()) for f in range(24)]
    assert not any(opened[:6]) and not any(opened[-5:])
    assert all(v > 0 for v in opened[6:18])
    for ch in lipsync.CHANNEL.values():                               # drawn flaps: held on twos
        assert all(m[ch][i] == m[ch][i - 1] for i in range(1, 24, 2))
    first = {v: next(f for f in range(24) if m[f"Fcl_MTH_{v}"][f] > 0) for v in "AO"}
    assert first["A"] < first["O"]                                    # the shapes come in the text's order


# ---------------------------------------------------------------- voices
@needs_ffmpeg
def test_a_voiced_line_is_cached_trimmed_and_48k(tmp_path):
    spec = {"engine": "stub", "voice": "x", "pitch": 2}
    a = voice.speak(spec, "The soup is still warm.", "en", tmp_path)
    assert voice.speak(spec, "The soup is still warm.", "en", tmp_path) == a
    assert voice.speak(spec, "Pickled plum.", "en", tmp_path) != a
    samples, rate = lipsync.read_wav(a)
    assert rate == voice.RATE and 0.5 < len(samples) / rate < 5
    lead = next(i for i, x in enumerate(samples) if abs(x) > 0.012)
    assert lead / rate < 0.06                                         # the engine's leading silence is gone


@needs_ffmpeg
def test_the_lantern_has_its_own_timbre(tmp_path):
    plain = voice.speak({"engine": "stub", "voice": "x"}, "Tsumugi.", "en", tmp_path)
    paper = voice.speak({"engine": "stub", "voice": "x", "fx": "lantern"}, "Tsumugi.", "en", tmp_path)
    assert lipsync.read_wav(plain)[0] != lipsync.read_wav(paper)[0]


def test_an_unknown_engine_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown voice engine"):
        voice.speak({"engine": "nope"}, "hi", "en", tmp_path)


# ---------------------------------------------------------------- placing lines
def test_lines_wait_for_their_cue_and_never_overlap(tmp_path):
    r = dialogue.fit([line(0.5), line(1.0, "tsumugi")], [tone(tmp_path / "a.wav", 1.2), tone(tmp_path / "b.wav", 0.8)],
                     6.0, tmp_path / "fit")
    (_, a1, b1), (_, a2, b2) = r["placed"]
    assert a1 == 0.5 and b1 == pytest.approx(1.7)
    assert a2 == pytest.approx(b1 + dialogue.GAP)                    # after the first line, and a breath
    assert r["speed"] == 1.0 and not r["overruns"]


@needs_ffmpeg
def test_a_long_exchange_is_pulled_in_then_quickened_then_reported(tmp_path):
    lines = [line(1.0), line(2.5, "tsumugi")]
    wavs = [tone(tmp_path / "a.wav", 1.5), tone(tmp_path / "b.wav", 1.5)]
    r = dialogue.fit(lines, wavs, 4.0, tmp_path / "fit")              # fits by starting earlier
    assert r["speed"] == 1.0 and not r["overruns"]
    assert r["placed"][0][1] >= 0.2 and r["placed"][-1][2] <= 4.0 - dialogue.TAIL + 1e-6
    r = dialogue.fit(lines, wavs, 3.2, tmp_path / "fit")              # fits a little quicker
    assert 1.0 < r["speed"] <= dialogue.MAX_SPEEDUP and not r["overruns"]
    quick, _ = lipsync.read_wav(r["placed"][0][0])
    assert len(quick) / voice.RATE == pytest.approx(1.5 / r["speed"], abs=0.03)
    r = dialogue.fit(lines, wavs, 2.0, tmp_path / "fit")              # cannot fit: the reviewer rewrites it
    assert r["speed"] == dialogue.MAX_SPEEDUP and r["overruns"] and r["overruns"][0][1] > 0.5


def test_the_mix_is_as_long_as_the_shot(tmp_path):
    out = dialogue.mix(tmp_path / "m.wav", 2.0, [(tone(tmp_path / "a.wav", 0.5), 1.0)])
    samples, rate = lipsync.read_wav(out)
    assert len(samples) == 2 * rate
    assert max(map(abs, samples[: rate - 1])) == 0 and max(map(abs, samples[rate: rate + rate // 2])) > 0.2


# ---------------------------------------------------------------- a shot, a chapter, the cut
@needs_ffmpeg
def test_a_shot_is_voiced_in_both_languages(proj):
    r = dialogue.voice_shot(proj, "1-11", engine="stub")
    man = dialogue.shot_manifest(proj, "1-11")
    s = next(x for x in proj.shots if x.id == "1-11")
    assert r["lines"] == len(man["lines"]) == 2 and man["languages"] == ["en", "ja"]
    for lang in ("en", "ja"):
        spans = [(ln["lines"][lang]["start"], ln["lines"][lang]["end"]) for ln in man["lines"]]
        assert all(b1 <= a2 for (_, b1), (a2, _) in zip(spans, spans[1:]))       # never two at once
        assert spans[-1][1] <= s.seconds
        samples, rate = lipsync.read_wav(Path(man["audio"][lang]))
        assert len(samples) == round(s.seconds * rate)
    assert all(len(v) == man["frames"] for ln in man["lines"] for v in ln["mouth"].values())


@needs_ffmpeg
def test_subtitles_and_the_chapter_track_follow_the_cut(proj):
    for sid in ("1-04", "1-11"):
        dialogue.voice_shot(proj, sid, engine="stub")
    off = dialogue.offsets(proj, 1)
    srt = dialogue.subtitles(proj, 1, "en").read_text(encoding="utf-8")
    cues = [c.split("\n") for c in srt.strip().split("\n\n")]
    assert len(cues) == 2                                             # the two lines; the laugh is not subtitled
    first = dialogue.shot_manifest(proj, "1-11")["lines"][0]["lines"]["en"]
    h, m, rest = cues[0][1].split(" --> ")[0].split(":")
    assert int(h) * 3600 + int(m) * 60 + float(rest.replace(",", ".")) == pytest.approx(off["1-11"] + first["start"], abs=0.002)
    ends = [c[1].split(" --> ") for c in cues]
    assert ends[0][1] <= ends[1][0]                                   # one subtitle at a time
    ja = dialogue.subtitles(proj, 1, "ja").read_text(encoding="utf-8")
    assert "前より" in ja and ja.count(" --> ") == 2
    track = dialogue.chapter_audio(proj, 1, "en")
    samples, rate = lipsync.read_wav(track)
    assert len(samples) == round(sum(s.seconds for s in proj.chapter(1)) * rate)
    t = off["1-11"] + first["start"] + 0.2
    assert max(map(abs, samples[int(t * rate): int(t * rate) + 2400])) > 0.05      # her voice is where the cut says


@needs_ffmpeg
def test_the_voice_moves_her_mouth_in_the_plan(proj):
    silent = planner.build(proj, "1-11", preview=True)
    dialogue.voice_shot(proj, "1-11", engine="stub")
    pl = planner.build(proj, "1-11", preview=True)
    sayo = next(c for c in pl["characters"] if c["name"] == "sayo")
    before = next(c for c in silent["characters"] if c["name"] == "sayo")
    assert sayo["face"]["_source"].endswith("+voice") and not before["face"]["_source"].endswith("+voice")
    talk = dialogue.shot_manifest(proj, "1-11")["lines"][0]
    f = int(talk["lines"]["en"]["start"] * 24) + 4
    flaps = [sum(sayo["face"][ch][g] for ch in planner.MOUTHS) for g in range(f, f + 12)]
    assert max(flaps) > 0.4 and min(flaps) < max(flaps)              # open and close as she speaks
    assert set(pl["audio"]) == {"en", "ja"}
    assert auto.plan_hash(pl) != auto.plan_hash(silent)               # a new line means a new render


def test_akari_speaks_in_light():
    ch = planner.overlay_talk({"akari.glow": [0.5] * 4}, {"voice": [0.0, 1.0, 1.0, 0.0]})
    assert ch["akari.glow"] == [0.5, 1.1, 1.1, 0.5]


def test_feeling_shows_around_the_words():
    face = {"Fcl_MTH_A": [0.0] * 4, "Fcl_MTH_Joy": [1.0] * 4}
    talk = {"voice": [0.0, 0.9, 0.9, 0.0], "Fcl_MTH_A": [0.0, 1.0, 1.0, 0.0]}
    out = planner.overlay_talk(face, talk)
    assert out["Fcl_MTH_A"] == [0.0, 1.0, 1.0, 0.0]
    assert out["Fcl_MTH_Joy"] == [1.0, 0.6, 0.6, 1.0]                # the smile softens while she talks


@needs_ffmpeg
def test_the_cut_carries_a_track_and_subtitles_per_language(proj):
    from engine import assemble as asm
    from engine import steps
    dialogue.voice_shot(proj, "1-11", engine="stub")
    clip_dir = proj.root / "media" / "cgi" / "1-02"
    clip_dir.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=24",
                    "-t", "4", "-pix_fmt", "yuv420p", str(clip_dir / "1-02-preview.mp4")], check=True)
    proj.config.setdefault("cgi", {})["use_in_cut"] = True
    r = asm.assemble(proj, steps.open_ledger(proj), 1)
    assert r["audio"] == ["en", "ja"]
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", r["file"]],
                                      capture_output=True, text=True, check=True).stdout)
    langs = [s.get("tags", {}).get("language") for s in probe["streams"] if s["codec_type"] == "audio"]
    assert langs == ["eng", "jpn"]
    assert all(Path(f).exists() for f in r["subtitles"].values())
