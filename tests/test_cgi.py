"""The 3D pipeline's decisions, without Blender: acting sheets, camera layout, anime timing, face
capture and retargeting, the shot plan, and the render runner (with a stand-in Blender).
One test runs real Blender when MOCHI_BPY_PYTHON names a Python with the bpy module."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from pipeline.cgi import capture as cap
from pipeline.cgi import direction as dirn
from pipeline.cgi import layout, retarget, timing
from pipeline.cgi import plan as planner
from pipeline.cgi import render as rnd
from pipeline.project import Project, Shot

REPO = Path(__file__).resolve().parents[1]
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@pytest.fixture()
def proj(tmp_path) -> Project:
    shutil.copy(REPO / "config.json", tmp_path / "config.json")
    shutil.copy(REPO / "shots.csv", tmp_path / "shots.csv")
    shutil.copytree(REPO / "style_bible", tmp_path / "style_bible")
    return Project(tmp_path)


def shot(text: str, chapter: int = 1, seconds: int = 4, kind: str = "Hero", sound: str = "") -> Shot:
    return Shot("9-99", chapter, text, seconds, kind, "Editor", 0, sound)


# ---------------------------------------------------------------- acting sheets
def test_draft_reads_framing_angle_and_feeling(proj):
    d = dirn.draft(proj, shot("CU Tsumugi laughing, flour on her nose", seconds=3))
    assert (d.camera.framing, d.camera.lens, d.camera.subject) == ("CU", 85.0, "tsumugi")
    assert d.camera.move == "push"                       # an emotional close-up breathes in
    assert d.beats and d.beats[0].emotion == "joy" and d.beats[0].intensity > 0.8

    behind = dirn.draft(proj, shot("From behind: both look up at the moon", seconds=7))
    assert behind.camera.angle == "behind"
    assert behind.characters == ["sayo", "tsumugi"] and behind.camera.subject == "sayo+tsumugi"
    assert any(b.emotion == "wonder" and b.gaze == "up" for b in behind.beats)

    hands = dirn.draft(proj, shot("CU Sayo's hands present each dish", seconds=6))
    assert hands.camera.framing == "INSERT" and hands.camera.subject == "sayo:hands"

    action = dirn.draft(proj, shot("ACTION. Moonquake: the cave shudders, lamps swing", chapter=4))
    assert action.timing == "mixed" and action.camera.move == "drift"

    still = dirn.draft(proj, shot("Wide: the full moon clears the sea", kind="Still"))
    assert still.camera.framing == "WIDE" and still.camera.move == "push" and still.beats == []


def test_draft_dialogue_mallet_and_blackout(proj):
    d = dirn.draft(proj, shot("Two-shot: Sayo hands her the bowl", sound="Sayo: It broke the year your mother was born."))
    assert d.dialogue.startswith("Sayo:") and d.camera.angle == "three-quarter"
    m = dirn.draft(proj, shot("CU mallet falls into the stone mortar, lamplight", kind="Simple"))
    assert m.impacts and any(a["do"] == "strike" for a in m.actions) and m.characters == [] and m.beats == []
    assert dirn.draft(proj, shot("Black screen. Mallet thuds begin", kind="Still")).blackout


def test_hand_written_sheet_wins_field_by_field(proj):
    d = dirn.direction_for(proj, "1-11")
    assert d.performance == "capture" and len(d.beats) >= 4
    assert any(b.restraint >= 0.8 for b in d.beats)       # Sayo holds the memory in
    assert d.scene == "1A"
    over = dirn.merge(d, {"camera": {"framing": "CU"}})
    assert over.camera.lens == 85.0 and over.camera.angle == d.camera.angle


def test_every_shot_has_a_renderable_sheet(proj):
    bad = {s.id: dirn.problems(dirn.direction_for(proj, s.id)) for s in proj.shots}
    assert {k: v for k, v in bad.items() if v} == {}


def test_problems_are_reported_in_words(proj):
    d = dirn.direction_for(proj, "1-05")
    d.beats.append(dirn.Beat(t=99, emotion="smug", who="nobody"))
    msgs = " ".join(dirn.problems(d))
    assert "smug" in msgs and "outside the shot" in msgs and "nobody" in msgs


# ---------------------------------------------------------------- camera
def test_framing_fills_the_frame_whatever_the_lens():
    for lens in (24, 50, 85, 100):
        d = layout.distance_for(0.32, lens)
        assert math.isclose(2 * d * math.tan(layout.vfov(lens) / 2), 0.32, rel_tol=1e-9)
    assert layout.distance_for(0.32, 100) > layout.distance_for(0.32, 50)


def test_angles_place_the_camera_where_they_say():
    face = [0, 0, 1.3]
    front, _, _ = layout.place(face, 0, "CU", "eye", 85)
    assert front[1] < -0.5                                   # yaw 0 faces -Y: the camera is there
    back, _, _ = layout.place(face, 0, "CU", "behind", 85)
    assert back[1] > 0.5
    low, _, _ = layout.place([0, 0, 0.3], 0, "CU", "low", 85)
    assert layout.FLOOR_CLEARANCE <= low[2] < 0.3            # below the face, never under the floor
    turned, _, _ = layout.place(face, 90, "CU", "eye", 85)
    assert turned[0] > 0.5                                   # facing +X: the camera moves with her


def test_moves():
    loc, tgt = (0, -2, 1), (0, 0, 1)
    push = layout.keys(loc, tgt, "push", 0.1, 48, 0.32)
    assert math.dist(push[-1]["loc"], tgt) == pytest.approx(1.8, abs=1e-4)
    assert math.dist(push[0]["loc"], tgt) == pytest.approx(2.0, abs=1e-4)
    static = layout.keys(loc, tgt, "static", 0, 10, 0.32)
    assert len({tuple(k["loc"]) for k in static}) == 1
    drift = layout.keys(loc, tgt, "drift", 0.35, 96, 0.95, seed=3)
    spread = max(math.dist(k["loc"], loc) for k in drift)
    assert 0 < spread < 0.01                                 # a held camera, not a shake
    assert layout.keys(loc, tgt, "drift", 0.35, 96, 0.95, seed=3) == drift


# ---------------------------------------------------------------- anime timing
def test_key_poses_keep_the_extremes():
    tri = [abs(10 - i) for i in range(21)]
    keys = timing.key_poses(tri, 0.02)
    assert {0, 10, 20} <= set(keys) and len(keys) <= 5
    assert timing.key_poses([i * 0.5 for i in range(30)], 0.01) == [0, 29]


def test_twos_threes_and_hit_stop():
    vals = [[float(i)] for i in range(10)]
    twos = timing.step(vals, 2)
    assert [v[0] for v in twos] == [0, 0, 2, 2, 4, 4, 6, 6, 8, 9]
    stopped = timing.hit_stop(vals, 4, 3)
    assert [v[0] for v in stopped] == [0, 1, 2, 3, 4, 4, 4, 4, 5, 6] and len(stopped) == len(vals)
    slow_then_fast = [[i * 0.01] for i in range(10)] + [[0.1 + i * 0.3] for i in range(1, 6)]
    m = timing.mixed(slow_then_fast)
    assert m[2] == m[3] and m[-2] != m[-3]                   # slow part on twos, the swing on ones


def test_emotion_curves_follow_the_beats():
    beats = [dirn.Beat(t=0.0, emotion="tenderness", intensity=0.5, who="sayo"),
             dirn.Beat(t=1.0, emotion="sorrow", intensity=0.6, who="sayo", restraint=0.9),
             dirn.Beat(t=2.0, emotion="bittersweet", intensity=1.0, who="sayo"),
             dirn.Beat(t=0.5, emotion="joy", intensity=1.0, who="tsumugi")]
    c = timing.emotion_curves(beats, 72, 24, dirn.EMOTIONS, "sayo")
    assert c["tenderness"][20] == pytest.approx(0.5, abs=0.02)
    assert c["sorrow"][46] == pytest.approx(0.6, abs=0.02) and c["restraint"][46] == pytest.approx(0.9, abs=0.02)
    assert c["joy"][71] > 0.5 and c["sorrow"][71] > 0.7      # bittersweet is both at once
    assert max(c["joy"][:47]) == 0.0                          # Tsumugi's joy is not Sayo's


def test_blinks_land_on_thoughts_and_are_repeatable():
    b = timing.blinks(240, 24, seed=5, per_minute=10, thoughts=[3.0, 6.0])
    assert b == timing.blinks(240, 24, seed=5, per_minute=10, thoughts=[3.0, 6.0])
    assert all(0 <= x <= 1 for x in b)
    assert max(b[74:80]) == 1.0 and max(b[146:152]) == 1.0   # a blink just after each beat
    awed = {"wonder": [1.0] * 240, "tenderness": [0.0] * 240}
    slow = timing.blinks(240, 24, seed=5, thoughts=[3.0], curves=awed)
    assert sum(1 for x in slow[74:90] if x > 0.5) > sum(1 for x in b[74:90] if x > 0.5)   # a slow, awed blink


# ---------------------------------------------------------------- capture and faces
LLF_HEADER = ("Timecode,BlendShapeCount," + ",".join(n[0].upper() + n[1:] for n in cap.ARKIT)
              + ",HeadYaw,HeadPitch,HeadRoll\n")


def write_llf(path: Path, rows: int = 120, fps: int = 60) -> Path:
    lines = [LLF_HEADER]
    for i in range(rows):
        vals = [0.0] * len(cap.ARKIT)
        vals[cap.ARKIT.index("jawOpen")] = 0.5 + 0.4 * math.sin(i / 6)
        vals[cap.ARKIT.index("eyeBlinkLeft")] = 1.0 if 30 <= i < 36 else 0.0
        tc = f"10:00:{i // fps:02d}:{i % fps:02d}.000"
        lines.append(f"{tc},52," + ",".join(f"{v:.4f}" for v in vals) + ",0.1,-0.05,0.0\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_live_link_face_csv(tmp_path):
    tr = cap.read_live_link_csv(write_llf(tmp_path / "take.csv"))
    assert tr.fps == pytest.approx(60, rel=0.01) and len(tr) == 120
    assert "jawOpen" in tr.names and "headYaw" in tr.names
    r = cap.resample(tr, 24)
    assert len(r) == 48 and r.fps == 24
    assert max(r.channel("eyeBlinkLeft")) > 0.5


def test_one_euro_steadies_a_still_face():
    rnd_ = random.Random(1)
    noisy = cap.Track(60, ["browInnerUp"], [[0.4 + rnd_.uniform(-0.05, 0.05)] for _ in range(300)])
    sm = cap.smooth(noisy)
    var = lambda xs: sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs)  # noqa: E731
    assert var(sm.channel("browInnerUp")[50:]) < var(noisy.channel("browInnerUp")[50:]) / 4


def test_capture_maps_onto_the_characters_face():
    tr = cap.Track(24, ["jawOpen", "eyeBlinkLeft", "headYaw"], [[0.5, 1.0, -0.3]])
    m = retarget.map_capture(tr)
    assert m.channel("Fcl_MTH_A")[0] == pytest.approx(0.575)
    assert m.channel("Fcl_EYE_Close_L")[0] == 1.0 and m.channel("head.yaw")[0] == pytest.approx(-0.3)


def test_restraint_moves_feeling_from_the_mouth_to_the_eyes():
    open_ = {"sorrow": [0.8], "restraint": [0.0]}
    held = {"sorrow": [0.8], "restraint": [0.9]}
    a, b = retarget.emotion_face(open_), retarget.emotion_face(held)
    assert b.channel("Fcl_MTH_Sorrow")[0] < a.channel("Fcl_MTH_Sorrow")[0] * 0.5
    assert b.channel("anime.tear")[0] > a.channel("anime.tear")[0]
    assert b.channel("Fcl_EYE_Sorrow")[0] > a.channel("Fcl_EYE_Sorrow")[0]
    laugh = retarget.emotion_face({"joy": [1.0]})
    assert laugh.channel("Fcl_MTH_A")[0] > 0.4               # joy past 0.75 opens into a laugh


def test_learned_mapping_recovers_a_face():
    src = ["jawOpen", "mouthSmileLeft", "browInnerUp"]
    rng = random.Random(2)
    ex = []
    for _ in range(12):
        x = {n: rng.random() for n in src}
        ex.append((x, {"MouthOpen": 0.9 * x["jawOpen"] + 0.05, "Sad": 0.7 * x["browInnerUp"]}))
    table = retarget.fit_mapping(src, ["MouthOpen", "Sad"], ex, ridge=1e-6)
    w = dict(table["MouthOpen"])
    assert w["jawOpen"] == pytest.approx(0.9, abs=0.01) and w["_bias"] == pytest.approx(0.05, abs=0.01)
    assert dict(table["Sad"])["browInnerUp"] == pytest.approx(0.7, abs=0.01)


def test_akari_acts_with_light():
    frames = 10
    happy = retarget.lantern_channels({"joy": [1.0] * frames}, frames, [0.0] * frames)
    sad = retarget.lantern_channels({"sorrow": [1.0] * frames}, frames, [0.0] * frames)
    assert happy.channel("akari.glow")[0] > sad.channel("akari.glow")[0] * 2


# ---------------------------------------------------------------- the plan
def _project_point(cam: dict, pt, aspect=16 / 9):
    k = cam["keys"][0]
    loc, tgt = k["loc"], k["target"]
    f = [tgt[i] - loc[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in f))
    f = [x / n for x in f]
    r = [f[1], -f[0], 0.0]
    rn = math.hypot(r[0], r[1])
    r = [x / rn for x in r]
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    v = [pt[i] - loc[i] for i in range(3)]
    z = sum(a * b for a, b in zip(v, f))
    th = math.tan(cam["vfov"] / 2)
    return (sum(a * b for a, b in zip(v, r)) / z / (th * aspect), sum(a * b for a, b in zip(v, u)) / z / th, z)


def test_mallet_falls_on_ones_and_holds_the_hit(proj):
    pl = planner.build(proj, "1-02", preview=True)
    mallet = next(x for x in pl["props"] if x["kind"] == "mallet")
    z = [m["at"][2] for m in mallet["motion"]]
    hit = timing.seconds_to_frame(1.9, 24)
    assert min(z) == pytest.approx(0.5, abs=0.01)
    lowest = z.index(min(z))
    assert abs(lowest - hit) <= 1 and z[lowest:lowest + 4] == [z[lowest]] * 4     # the hit is held
    assert z[lowest - 3] > z[lowest - 2] > z[lowest - 1] > z[lowest]              # the fall is on ones
    assert pl["characters"] == [] and pl["cheated_out"] == []


def test_two_shot_frames_both_faces(proj):
    pl = planner.build(proj, "1-11", preview=True)
    cam = pl["camera"]
    for c in pl["characters"]:
        x, y, z = _project_point(cam, planner.head_point(c))
        assert abs(x) < 0.95 and abs(y) < 0.95 and z > 0, c["name"]
        assert c["face"]["_source"] == "beats" and len(c["face"]["Fcl_EYE_Close"]) == pl["frames"]


def test_from_behind_means_behind(proj):
    pl = planner.build(proj, "1-14", preview=True)
    cam = pl["camera"]["keys"][0]["loc"]
    for c in pl["characters"]:
        f = layout.facing_vec(c["yaw"])
        to_cam = [cam[i] - c["at"][i] for i in range(2)]
        assert f[0] * to_cam[0] + f[1] * to_cam[1] < 0, c["name"]       # the camera sees their backs


def test_props_that_hide_a_face_are_cheated_out(proj):
    pl = planner.build(proj, "1-05", preview=True)
    assert "mortar" in pl["cheated_out"]
    kept = planner.build(proj, "1-03", preview=True)
    assert "mortar" not in kept["cheated_out"]                           # the hands are in it


def test_continuity_is_data(proj):
    ch1 = planner.build(proj, "1-10", preview=True)
    assert next(x for x in ch1["props"] if x["kind"] == "bowl")["seams"] == 1
    d = dirn.direction_for(proj, "3-15")
    assert "bowl" in d.props
    cfg = planner.settings(proj)
    assert planner.by_chapter(cfg["props_state"]["bowl"], 3)["seams"] == 2


def test_eyeline_to_someone_off_screen(proj):
    pl = planner.build(proj, "1-05", preview=True)
    yaw = pl["characters"][0]["face"]["head.yaw"]
    assert abs(yaw[-1]) > 0.02 or abs(pl["characters"][0]["face"]["head.pitch"][-1]) > 0.02  # she looks to Sayo


def test_a_captured_take_replaces_the_procedural_face(proj, tmp_path):
    work = Path(planner.work_dir(proj, planner.settings(proj), "1-05"))
    tr = cap.read_live_link_csv(write_llf(tmp_path / "t.csv", rows=200))
    cap.Track.save(tr, work / "face-tsumugi.json")
    pl = planner.build(proj, "1-05", preview=True)
    face = pl["characters"][0]["face"]
    assert face["_source"] == "capture"
    assert max(face["Fcl_MTH_A"]) > 0.5 and "anime.tear" in face
    assert len(face["Fcl_MTH_A"]) == pl["frames"]


def test_plans_for_the_whole_film_build(proj):
    for s in proj.shots:
        pl = planner.build(proj, s.id, preview=True)
        assert pl["frames"] == round(s.seconds * 24) and len(pl["camera"]["keys"]) == pl["frames"]


# ---------------------------------------------------------------- the runner
FAKE_BLENDER = textwrap.dedent('''
    import json, sys, struct, zlib, os
    a = sys.argv
    plan = json.load(open(a[a.index("--plan") + 1]))
    def png(path, w, h, flat):
        raw = b"".join(b"\\x00" + b"".join(bytes([(x * 7 + y * 3) % 256 if not flat else 40, 90, 160])
                                          for x in range(w)) for y in range(h))
        ck = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
        open(path, "wb").write(b"\\x89PNG\\r\\n\\x1a\\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                               + ck(b"IDAT", zlib.compress(raw)) + ck(b"IEND", b""))
    print("CGI " + json.dumps({"event": "build.done", "msg": "built", "props": len(plan["props"])}))
    os.makedirs(os.path.dirname(plan["output"]["blend"]), exist_ok=True)
    open(plan["output"]["blend"], "w").write("blend")
    if "--no-render" not in a:
        out = plan["output"]["frames"]; os.makedirs(out, exist_ok=True)
        first, last = plan["frame_range"]
        for f in range(first, last + 1):
            png(os.path.join(out, f"{f:04d}.png"), 64, 36, os.environ.get("FAKE_FLAT") == "1")
            print("CGI " + json.dumps({"event": "render.frame", "frame": f, "of": last, "seconds": 0.01}))
    print("CGI " + json.dumps({"event": "done", "msg": plan["shot"]}))
''')


@pytest.fixture()
def fake_blender(tmp_path, monkeypatch):
    exe = tmp_path / "fakepython"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {tmp_path / 'fake_blender.py'} \"$@\"\n")
    exe.chmod(0o755)
    (tmp_path / "fake_blender.py").write_text(FAKE_BLENDER)
    monkeypatch.setenv("BLENDER", str(exe))
    monkeypatch.setattr(rnd, "BUILD", tmp_path / "unused-build.py")
    monkeypatch.setattr(rnd, "MAC_BLENDER", str(tmp_path / "no-such-blender"))
    return exe


def test_command_line_for_blender_and_for_bpy(tmp_path):
    plan = tmp_path / "plan.json"
    b = rnd.command("/Applications/Blender.app/Contents/MacOS/Blender", plan)
    assert b[1:5] == ["-b", "--python-exit-code", "1", "--python"] and b[-2:] == ["--plan", str(plan)]
    py = rnd.command("/opt/bpy/bin/python3", plan, no_render=True)
    assert py[0] == "/opt/bpy/bin/python3" and py[-1] == "--no-render"


@needs_ffmpeg
def test_render_checks_and_encodes(proj, fake_blender, monkeypatch):
    monkeypatch.setattr(rnd, "command", lambda exe, plan_file, no_render=False:
                        [str(exe), "--plan", str(plan_file)] + (["--no-render"] if no_render else []))
    r = rnd.render_shot(proj, "1-02", preview=True, frames=(1, 12))
    assert Path(r["clip"]).exists() and r["frames"] == 12 and Path(r["blend"]).exists()
    monkeypatch.setenv("FAKE_FLAT", "1")
    with pytest.raises(RuntimeError, match="blank"):
        rnd.render_shot(proj, "1-05", preview=True, frames=(1, 4))
    built = rnd.render_shot(proj, "1-06", preview=True, no_render=True)
    assert "clip" not in built and Path(built["plan"]).exists()


@needs_ffmpeg
def test_board_tiles_the_shots(proj, fake_blender, monkeypatch):
    monkeypatch.setattr(rnd, "command", lambda exe, plan_file, no_render=False: [str(exe), "--plan", str(plan_file)])
    for s in ("1-02", "1-03", "1-04"):
        rnd.render_shot(proj, s, preview=True, frames=(1, 2))
    out = rnd.board(proj, ["1-02", "1-03", "1-04"], 1, cols=2, width=64)
    assert out.exists() and out.stat().st_size > 100


BPY = os.environ.get("MOCHI_BPY_PYTHON")


@pytest.mark.skipif(not BPY, reason="set MOCHI_BPY_PYTHON to a Python with the bpy module to run real Blender")
@needs_ffmpeg
def test_real_blender_builds_and_renders_a_frame(proj, monkeypatch):
    cfg = json.loads((proj.root / "config.json").read_text())
    cfg["cgi"] = {"blender": BPY, "engine": "BLENDER_WORKBENCH", "preview_resolution": [96, 54]}
    (proj.root / "config.json").write_text(json.dumps(cfg))
    p = Project(proj.root)
    r = rnd.render_shot(p, "1-11", preview=True, frames=(70, 70))
    assert Path(r["blend"]).exists() and Path(r["clip"]).exists()
    subprocess.run([BPY, "-c", "import bpy"], check=True)


@needs_ffmpeg
def test_the_cut_uses_cg_renders_when_switched_on(proj):
    from pipeline import assemble as asm
    from pipeline import steps
    clip_dir = proj.root / "media" / "cgi" / "1-02"
    clip_dir.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=24",
                    "-t", "4", "-pix_fmt", "yuv420p", str(clip_dir / "1-02-preview.mp4")], check=True)
    led = steps.open_ledger(proj)
    assert asm.assemble(proj, led, 1)["from"]["cgi"] == 0          # off by default
    cfg = json.loads((proj.root / "config.json").read_text())
    cfg["cgi"] = {"use_in_cut": True}
    (proj.root / "config.json").write_text(json.dumps(cfg))
    p = Project(proj.root)
    r = asm.assemble(p, steps.open_ledger(p), 1)
    assert r["from"]["cgi"] == 1 and r["shots"] == len(p.chapter(1))
