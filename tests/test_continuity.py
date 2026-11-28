"""The continuity supervisor: each rule on staged geometry where the answer is known, then the real film."""
from __future__ import annotations

import json

from engine.cgi import auto, cli
from engine.cgi import continuity as C
from engine.cgi.acting import direction as dirn
from engine.project import ACTING_FILE

RIG = {"key": {"color": [1.0, 0.96, 0.9], "energy": 3.0, "azimuth": -35, "elevation": 40}, "world": [0.2, 0.2, 0.25]}
A_AT, B_AT = [-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]     # two people face to face along x
FRONT, BACK = [0.0, -3.0, 1.2], [0.0, 3.0, 1.2]    # the two sides of the line between them
ON_LINE = [2.5, 0.15, 1.2]                         # behind b, on the line: a neutral shot
MID = [0.0, 0.0, 1.2]


def person(name: str, at: list[float], yaw: float, posture: str = "standing") -> dict:
    return {"name": name, "at": list(at), "yaw": yaw, "posture": posture, "height": 1.5, "procedural": ""}


A, B = person("a", A_AT, 90), person("b", B_AT, -90)
WHERE = {"a": A, "b": B}


def shot(sid: str, cam: list[float], target: list[float] = MID, *, on: tuple = ("a", "b"), first: dict | None = None,
         last: dict | None = None, cam_end: list[float] | None = None, frame_h: float = 0.95, subject: str = "a",
         rig: dict | None = None, props: dict | None = None, acts_on: tuple = (), people: dict | None = None) -> C.Shot:
    who = people or WHERE
    return C.Shot(id=sid, scene="S", cams=(cam, cam_end or cam), targets=(target, target), frame_h=frame_h,
                  subject=subject, on_screen={n: who[n] for n in on}, gaze=(first or {}, last or first or {}),
                  marks={}, rig=rig or RIG, props=props or {}, acts_on=set(acts_on),
                  walking={n for n in on if who[n]["posture"] == C.WALKING})


def axis_run(*shots: C.Shot) -> list[str]:
    lines: dict = {}
    return [sh.id for sh in shots for _ in C.axis(sh, WHERE, lines)]


# ---------------------------------------------------------------- the 180-degree rule
def test_which_side_of_the_line():
    assert C.side(FRONT, A_AT, B_AT) == -C.side(BACK, A_AT, B_AT) != 0
    assert C.side(ON_LINE, A_AT, B_AT) == 0


def test_crossing_the_line_is_found():
    assert axis_run(shot("1", FRONT), shot("2", FRONT), shot("3", BACK)) == ["3"]


def test_a_neutral_shot_or_a_move_across_sets_the_line_again():
    assert axis_run(shot("1", FRONT), shot("2", ON_LINE), shot("3", BACK)) == []
    assert axis_run(shot("1", FRONT), shot("2", FRONT, cam_end=BACK), shot("3", BACK)) == []


def test_a_cutaway_does_not():
    insert = shot("2", [0.0, -1.0, 0.5], [0.0, 0.0, 0.0], on=(), subject="bowl")
    assert axis_run(shot("1", FRONT), insert, shot("3", BACK)) == ["3"]
    assert axis_run(shot("1", FRONT, on=("a",)), shot("2", BACK, on=("b",))) == ["2"]   # singles hold the line too


# ---------------------------------------------------------------- the 30-degree rule
def test_a_cut_to_the_same_subject_needs_a_new_angle_or_size():
    near = [0.5, -3.0, 1.2]                                 # about 10 degrees round from FRONT
    assert [r for r, _ in C.jump(shot("1", FRONT), shot("2", near))] == ["jump"]
    assert C.jump(shot("1", FRONT), shot("2", [-2.5, -2.0, 1.2])) == []             # 50 degrees round
    assert C.jump(shot("1", FRONT), shot("2", near, frame_h=0.32)) == []             # MS to CU
    assert C.jump(shot("1", FRONT), shot("2", near, subject="b")) == []              # someone else


# ---------------------------------------------------------------- eyelines
def test_looks_that_meet_point_opposite_ways():
    a_looks = shot("1", FRONT, on=("a",), first={"a": "b"})
    b_back = shot("2", FRONT, on=("b",), first={"b": "a"})
    assert C.eyeline(a_looks, b_back, WHERE) == []
    b_wrong = shot("2", BACK, on=("b",), first={"b": "a"})                           # the reverse from across the line
    assert [r for r, _ in C.eyeline(a_looks, b_wrong, WHERE)] == ["eyeline"]


def test_the_same_look_keeps_its_direction():
    a_looks = shot("1", FRONT, on=("a",), first={"a": "b"})
    assert C.eyeline(a_looks, shot("2", [-1.0, -2.5, 1.2], on=("a",), first={"a": "b"}), WHERE) == []
    assert [r for r, _ in C.eyeline(a_looks, shot("2", BACK, on=("a",), first={"a": "b"}), WHERE)] == ["eyeline"]


def test_a_look_off_screen_does_not_go_down_the_lens():
    b_eyes = [0.45, 0.0, 1.35]                              # the camera where b's eyes are
    found = C.eyeline(None, shot("1", b_eyes, [-0.5, 0.0, 1.35], on=("a",), first={"a": "b"}), WHERE)
    assert [r for r, _ in found] == ["eyeline"] and "lens" in found[0][1]
    assert C.eyeline(None, shot("1", FRONT, on=("a",), first={"a": "b"}), WHERE) == []


# ---------------------------------------------------------------- light, props, positions
def test_the_light_holds_through_a_scene():
    def lit(**key) -> dict:
        return {**RIG, "key": {**RIG["key"], **key}}
    assert C.light(shot("1", FRONT), shot("2", FRONT, rig=lit(azimuth=-30))) == []   # 5 degrees: the same light
    assert len(C.light(shot("1", FRONT), shot("2", FRONT, rig=lit(azimuth=10)))) == 1
    assert len(C.light(shot("1", FRONT), shot("2", FRONT, rig=lit(color=[0.6, 0.7, 1.0], energy=6.0)))) == 2
    practical = {**RIG, "practical": {"energy": 45}}
    assert "comes on" in C.light(shot("1", FRONT), shot("2", FRONT, rig=practical))[0][1]


def test_a_prop_stays_put_unless_the_shot_moves_it():
    bowl = {"kind": "bowl", "at": [0.0, 0.0, 0.4], "rot": [0, 0, 0], "seams": 1}
    moved = {**bowl, "at": [0.2, 0.0, 0.4]}
    assert len(C.props(shot("1", FRONT, props={"bowl": bowl}), shot("2", FRONT, props={"bowl": moved}))) == 1
    assert C.props(shot("1", FRONT, props={"bowl": bowl}, acts_on=("bowl",)),
                   shot("2", FRONT, props={"bowl": moved})) == []
    mended = {**bowl, "seams": 2}
    assert "seams" in C.props(shot("1", FRONT, props={"bowl": bowl}), shot("2", FRONT, props={"bowl": mended}))[0][1]


def test_nobody_jumps_across_a_cut_unless_walking():
    stepped = {"a": person("a", [0.1, 0.0, 0.0], 90), "b": B}
    assert len(C.position(shot("1", FRONT), shot("2", FRONT, people=stepped))) == 1
    walked = {"a": person("a", [0.1, 0.0, 0.0], 90, "walking"), "b": B}
    assert C.position(shot("1", FRONT), shot("2", FRONT, people=walked)) == []


# ---------------------------------------------------------------- the real film
def _accept_all(proj, found) -> None:
    f = proj.style_bible / ACTING_FILE
    data = json.loads(f.read_text(encoding="utf-8"))
    for x in found:
        data["shots"].setdefault(x.shot, {}).setdefault("continuity", {})[x.rule] = f"meant: test {x.rule}"
    f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_every_chapter_is_checked_and_a_finding_can_be_accepted(proj):
    shots = {s.id for s in proj.shots}
    for chapter in sorted({s.chapter for s in proj.shots}):
        found = C.check(proj, chapter)
        assert found == C.check(proj, chapter)                                      # the same every time
        assert all(x.shot in shots and x.rule in C.RULES and x.message for x in found)
    found = C.check(proj, 1)
    assert found and cli.continuity_cmd(proj, 1) == 3                               # the 180-degree crossings
    _accept_all(proj, found)
    again = C.check(proj, 1)
    assert [(x.shot, x.rule) for x in again] == [(x.shot, x.rule) for x in found]
    assert all(x.accepted.startswith("meant") for x in again) and cli.continuity_cmd(proj, 1) == 0
    sheet = json.loads(auto.review_sheet(proj, 1, False).read_text(encoding="utf-8"))
    at_cuts = [(s["shot"], c["rule"]) for s in sheet["shots"] for c in s["continuity"]]
    assert at_cuts == [(x.shot, x.rule) for x in again]


def test_an_acceptance_must_name_a_rule_and_say_why():
    d = dirn.Direction(shot="9-99", seconds=4, type="Simple", continuity={"axes": "x", "jump": " "})
    probs = dirn.problems(d)
    assert any("continuity.axes: not a rule" in x for x in probs)
    assert any("continuity.jump: say why" in x for x in probs)
