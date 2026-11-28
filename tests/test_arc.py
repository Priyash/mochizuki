"""The emotional arc: feelings placed on the circumplex, a shot's feeling over time, the intended arc,
the comparison, and drafted shots that take their feeling from the arc."""
from __future__ import annotations

import json

import pytest

from engine.cgi import auto
from engine.cgi.acting import arc
from engine.cgi.acting import direction as dirn
from engine.project import ARC_FILE


def test_every_emotion_has_its_place_on_the_circumplex():
    assert set(arc.CIRCUMPLEX) == set(dirn.EMOTIONS)
    quadrant = {e: (v > 0, a > 0) for e, (v, a) in arc.CIRCUMPLEX.items() if e != "neutral"}
    assert quadrant["joy"] == (True, True) and quadrant["tenderness"] == (True, False)       # Russell 1980
    assert quadrant["sorrow"] == (False, False) and quadrant["fear"] == (False, True)
    assert all(-1 <= x <= 1 for va in arc.CIRCUMPLEX.values() for x in va)


def test_a_shot_feels_like_its_beats_for_as_long_as_each_lasts():
    d = dirn.Direction(shot="9-99", seconds=4.0, type="Hero", characters=["a"],
                       beats=[dirn.Beat(t=0.0, emotion="joy", intensity=1.0, who="a"),
                              dirn.Beat(t=1.0, emotion="sorrow", intensity=1.0, who="a")])
    v, a = arc.shot_feeling(d)
    assert v == pytest.approx((0.8 * 1 + -0.7 * 3) / 4) and a == pytest.approx((0.5 * 1 + -0.45 * 3) / 4)
    assert arc.shot_feeling(dirn.Direction(shot="9-98", seconds=3.0, type="Still")) is None   # a shot of things


def test_the_intended_arc_from_points_or_a_shape():
    spec = {"points": [[0.0, 0.6, 0.4], [1.0, -0.6]]}
    assert arc.intended(spec, 0.5) == (pytest.approx(0.0), 0.4)
    hole = {"shape": "man_in_a_hole", "amplitude": 0.5}
    assert arc.intended(hole, 0.5) == (pytest.approx(-0.5), None) and arc.intended(hole, 0.0)[0] > 0
    assert arc.intended(None, 0.5) is None
    assert arc.problems({"chapters": {"1": {"shape": "hero"}, "2": {"points": [[2, 0]]}}}) == [
        f"chapters.1.shape 'hero' is not one of {', '.join(arc.SHAPES)}",
        "chapters.2.points: [2, 0] is not [time 0..1, valence -1..1, arousal -1..1]"]


def test_a_realized_arc_reads_as_its_shape():
    at = [i / 10 for i in range(11)]
    fall_rise = [arc._interp(arc.SHAPES["man_in_a_hole"], x) * 0.4 + 0.05 * (-1) ** i for i, x in enumerate(at)]
    assert arc.reads_as(at, fall_rise)[0] == "man_in_a_hole"
    assert arc.reads_as(at, [0.8 - 0.1 * i for i in range(11)])[0] == "tragedy"


def test_the_arc_suggests_a_feeling_for_a_point():
    assert arc.suggest(0.75, 0.45)[0] == "joy"
    assert arc.suggest(0.5, -0.3)[0] == "tenderness"
    assert arc.suggest(-0.6, -0.4) == ("sorrow", pytest.approx(0.86, abs=0.02))
    assert arc.suggest(0.05, 0.05) == ("neutral", 0.3)
    assert arc.suggest(-0.5, None)[0] in ("sorrow", "fear", "anger")       # a shape gives valence only


def _write_arc(proj, spec: dict) -> None:
    (proj.style_bible / ARC_FILE).write_text(json.dumps(spec), encoding="utf-8")


def test_the_films_arc_and_the_chapter_report(proj):
    assert arc.problems(arc.load(proj)) == []
    r = arc.report(proj, 1)
    assert len(r["shots"]) == len(proj.chapter(1)) and r["valence_r"] is not None
    assert all(s["intended"] for s in r["shots"]) and r["reads_as"]["shape"] in arc.SHAPES
    _write_arc(proj, {"chapters": {"1": {"shape": "tragedy", "amplitude": 1.0}}})   # a joyful start, a sad end
    off = arc.report(proj, 1)
    assert off["off"] and all("valence" in s["off"] for s in off["shots"] if s["shot"] in off["off"])
    sheet = json.loads(auto.review_sheet(proj, 1, False).read_text(encoding="utf-8"))
    assert [s["shot"] for s in sheet["shots"] if s["arc"].get("off")] == off["off"]


def test_drafted_shots_take_their_feeling_from_the_arc(proj):
    _write_arc(proj, {"fill_drafts": True, "chapters": {"1": {"points": [[0.0, -0.6, -0.4], [1.0, -0.6, -0.4]]}}})
    filled = dirn.direction_for(proj, "1-09")                                      # a draft: "Sayo shapes the mochi"
    assert [(b.emotion, b.intensity) for b in filled.beats] == [arc.suggest(-0.6, -0.4)]
    sheet = dirn.direction_for(proj, "1-11")                                       # hand-written: never changed
    assert [b.emotion for b in sheet.beats] == [b["emotion"] for b in dirn.load_acting(proj)["1-11"]["beats"]]
    assert "[arc:" not in sheet.notes and "[arc:" not in dirn.direction_for(proj, "1-16").notes
    _write_arc(proj, {"fill_drafts": False, "chapters": {"1": {"points": [[0.0, -0.6, -0.4], [1.0, -0.6]]}}})
    assert [b.emotion for b in dirn.direction_for(proj, "1-09").beats] == ["neutral"]
