"""Procedural anime characters: the figure's proportions and postures (pure Python), and the shot plan
framing the face the builder will make. The Blender side runs in test_cgi.py's real-Blender test."""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from pipeline.cgi import figure
from pipeline.cgi import plan as planner
from pipeline.project import Project

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def proj(tmp_path) -> Project:
    shutil.copy(REPO / "config.json", tmp_path / "config.json")
    shutil.copy(REPO / "shots.csv", tmp_path / "shots.csv")
    shutil.copytree(REPO / "style_bible", tmp_path / "style_bible")
    return Project(tmp_path)


def test_anime_proportions_by_age():
    kid, adult, old = (figure.skeleton(h, a) for h, a in ((1.2, 7), (1.58, 27), (1.45, 80)))
    assert kid["heads"] < old["heads"] < adult["heads"]          # a child is mostly head; age shortens the body
    for s, h in ((kid, 1.2), (adult, 1.58), (old, 1.45)):
        top = s["head"]["center"][2] + s["H"] / 2
        assert top < h and h - top < 0.15 * s["H"]                 # hair (and a stoop) make up the rest
        assert 0.8 * h < s["eyes"][2] < 0.95 * h
    assert old["stoop"] > 0 and adult["stoop"] == 0 and old["head"]["center"][1] < 0   # the head bows forward


def test_arms_keep_their_length_and_bend_at_the_elbow():
    for posture in ("standing", "seiza"):
        s = figure.skeleton(1.58, 27, posture)
        j, pr = s["joints"], figure.proportions(27)
        for side in "LR":
            upper = math.dist(j[f"shoulder_{side}"], j[f"elbow_{side}"])
            fore = math.dist(j[f"elbow_{side}"], j[f"wrist_{side}"])
            assert upper == pytest.approx(pr["upper_arm"] * s["H"], rel=1e-3)
            assert fore == pytest.approx(pr["forearm"] * s["H"], rel=0.08)


def test_seiza_sits_on_the_heels_with_the_hands_on_the_lap():
    s = figure.skeleton(1.2, 7, "seiza")
    j = s["joints"]
    assert s["eyes"][2] < 0.6 * 1.2
    assert j["knee_L"][1] < -0.15 and j["knee_L"][2] < 0.1         # knees forward, on the floor
    assert j["ankle_L"][1] > j["knee_L"][1] + 0.1                    # shins folded back under her
    assert abs(s["hands"][2] - j["pelvis"][2]) < 0.15                # hands rest on the lap


def test_figures_turn_with_their_mark():
    p = figure.to_world([0.0, -1.0, 0.5], [2.0, 3.0, 0.0], 90.0)   # facing -Y, turned 90: now facing +X
    assert p == pytest.approx([3.0, 3.0, 0.5])


def test_the_camera_frames_the_face_the_builder_makes(proj):
    pl = planner.build(proj, "1-05", preview=True)
    c = pl["characters"][0]
    assert c["figure"] and c["look"]["age"] == 7 and c["look"]["hair_style"] == "bob"
    eyes = figure.to_world(c["figure"]["eyes"], c["at"], c["yaw"])
    assert planner.head_point(c) == eyes
    target = pl["camera"]["keys"][0]["target"]
    assert math.dist(target, eyes) < 0.15


def test_she_ages_between_chapters(proj):
    young = next(c for c in planner.build(proj, "1-05", preview=True)["characters"] if c["name"] == "tsumugi")
    old = next(c for c in planner.build(proj, "4-16", preview=True)["characters"] if c["name"] == "tsumugi")
    assert young["look"]["age"] < 10 < 80 < old["look"]["age"]
    assert old["look"]["hair_style"] == "bun" and old["figure"]["stoop"] > 0


def test_a_model_file_replaces_the_figure(proj):
    model = proj.root / "assets" / "characters" / "tsumugi-07.vrm"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"glTF")
    c = planner.build(proj, "1-05", preview=True)["characters"][0]
    assert c["model"] and c["figure"] is None


def test_the_maquette_is_still_there_if_wanted(proj):
    proj.config.setdefault("cgi", {})["stand_in"] = "maquette"
    c = planner.build(proj, "1-05", preview=True)["characters"][0]
    assert c["figure"] is None and c["stand_in"] == "maquette"
