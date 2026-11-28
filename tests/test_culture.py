"""The culture pack: display rules played into the beats, by who feels and who is there, never over what a
hand-written sheet gives."""
from __future__ import annotations

import json

import pytest

from engine.cgi import cli
from engine.cgi.acting import culture
from engine.cgi.acting import direction as dirn
from engine.film_check import FilmError
from engine.project import CULTURE_FILE

RULES = {"rules": [
    {"name": "child open", "when": {"emotion": ["joy"], "age": [0, 12]}, "then": {"restraint_max": 0.1}, "why": "x"},
    {"name": "elder grief", "when": {"emotion": ["sorrow"], "age": [60, 120]},
     "then": {"restraint_min": 0.8, "gaze": "down"}, "why": "x"},
    {"name": "before a child", "when": {"emotion": ["sorrow"], "age": [18, 120], "with_age": [0, 12]},
     "then": {"intensity_scale": 0.5}, "why": "x"},
    {"name": "ma", "when": {"intensity": [0.7, 1.0]}, "then": {"hold_min": 0.5}, "why": "x"},
]}


def _sheet(*beats: dirn.Beat, who: tuple = ("sayo", "tsumugi")) -> dirn.Direction:
    return dirn.Direction(shot="1-11", seconds=6.0, type="Hero", characters=list(who), beats=list(beats))


def _use(proj, spec: dict) -> None:
    (proj.style_bible / CULTURE_FILE).write_text(json.dumps(spec), encoding="utf-8")


def test_the_films_pack_is_valid_and_mistakes_are_named(proj):
    assert culture.problems(culture.load(proj)) == []
    bad = {"rules": [{"name": "r", "when": {"mood": ["x"], "emotion": ["glee"], "age": [9]},
                      "then": {"restraint_min": 2, "shout": 1}}]}
    assert culture.problems(bad) == [
        "rules[0] (r): when.mood is not one of " + str(culture.WHEN),
        "rules[0] (r): when.emotion ['glee'] are not emotions",
        "rules[0] (r): when.age is [min, max]",
        "rules[0] (r): then.restraint_min goes from 0 to 1",
        "rules[0] (r): then.shout is not one of " + str(culture.THEN),
        "rules[0] (r): say why (the convention it follows)"]


def test_rules_follow_who_feels_and_who_is_there(proj):
    _use(proj, RULES)                                                   # chapter 1: Sayo is 78, Tsumugi 7
    d, changes = culture.apply(proj, 1, _sheet(dirn.Beat(t=0.0, emotion="sorrow", intensity=0.6, who="sayo"),
                                               dirn.Beat(t=2.0, emotion="joy", intensity=0.9, who="tsumugi",
                                                         restraint=0.5)))
    grief, joy = d.beats
    assert (grief.restraint, grief.gaze, grief.intensity) == (0.8, "down", 0.3)   # held in, eyes down, played down
    assert (joy.restraint, joy.hold) == (0.1, 0.5)                                # a child's joy, open; ma
    assert [c.rule for c in changes] == ["elder grief", "before a child", "child open", "ma"]
    alone, _ = culture.apply(proj, 1, _sheet(dirn.Beat(t=0.0, emotion="sorrow", intensity=0.6, who="sayo"),
                                             who=("sayo",)))
    assert alone.beats[0].intensity == 0.6                                        # no child there: not played down


def test_a_hand_written_field_is_never_changed(proj):
    _use(proj, RULES)
    beat = dirn.Beat(t=0.0, emotion="sorrow", intensity=0.6, who="sayo", restraint=0.3, gaze="tsumugi")
    d, changes = culture.apply(proj, 1, _sheet(beat), explicit=[{"t", "emotion", "intensity", "restraint", "gaze"}])
    assert (d.beats[0].restraint, d.beats[0].gaze, d.beats[0].intensity) == (0.3, "tsumugi", 0.6) and not changes


def test_the_sheets_play_the_pack(proj):
    drafted = dirn.direction_for(proj, "1-22")                          # Sayo alone, presenting the dishes
    assert drafted.beats[0].restraint >= 0.8 and drafted.beats[0].gaze == "down"
    written = dirn.load_acting(proj)["1-11"]["beats"]
    played = dirn.direction_for(proj, "1-11").beats
    for given, b in zip(written, played):                              # every field the sheet gives, as given
        assert all(getattr(b, k) == v for k, v in given.items() if k in dirn.Beat.__dataclass_fields__)
    (proj.style_bible / CULTURE_FILE).unlink()
    assert dirn.direction_for(proj, "1-22").beats[0].restraint == 0.0   # no pack, no rules


def test_a_broken_file_is_refused_with_every_problem_named(proj):
    _use(proj, {"rules": [{"name": "r", "when": {"mood": ["x"]}, "then": {"shout": 1}}]})
    with pytest.raises(FilmError, match="3 problem"):
        dirn.direction_for(proj, "1-22")                                # not a KeyError inside a render
    assert cli.culture_cmd(proj, 1) == 3
    (proj.style_bible / CULTURE_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(FilmError, match="not valid JSON"):
        culture.load(proj)
