"""The AI finish: a shot's guided 3D render, repainted on a GPU provider (stand-in engines here)."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess

import pytest
from test_pipeline import copy_project

from engine import assemble as asm
from engine import scheduler as sched
from engine import stages, steps
from engine.cgi import config as cgi_config
from engine.providers.base import WORKER
from engine.providers.kaggle import LocalKaggle

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
SHOT = "1-05"          # the laugh: a Hero close-up of 3 s


def _guide(proj, shot: str = SHOT, source: str = "testsrc=size=160x90:rate=24", control: str = "depth"):
    """A stand-in for `cgi render --guides`: the shot's guide clip where the render puts it."""
    f = cgi_config.work_dir(proj, cgi_config.settings(proj), shot) / f"{shot}.{control}.mp4"
    f.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", source, "-t", "3",
                    "-pix_fmt", "yuv420p", str(f)], check=True)
    return f


def _finished(proj, led):
    local = LocalKaggle(proj.media)
    built = sched.build(proj, led, "ai_finish", 1, SHOT, provider=local, engine_set="stub")
    sched.submit(proj, led, local, built["batch"])
    return built, sched.collect(proj, led, local, built["batch"])


@needs_ffmpeg
def test_only_shots_with_a_guided_render_get_a_finish(tmp_path):
    proj = copy_project(tmp_path)
    led = steps.open_ledger(proj)
    assert steps.finish_state(proj, led, proj.shot(SHOT)) == "no_render"
    _guide(proj)
    assert steps.finish_state(proj, led, proj.shot(SHOT)) == "needed"
    others = [s for s in steps.finish_shots(proj, 1) if s.id != SHOT]
    assert others and all(s.type == "Hero" for s in others)                    # engines.ai_finish.shots: "Hero"
    assert all(steps.finish_state(proj, led, s) == "no_render" for s in others)
    assert [s.id for s in steps.finish_shots(proj, 1, only=SHOT)] == [SHOT]


@needs_ffmpeg
def test_a_finish_goes_to_a_provider_and_comes_back(tmp_path):
    proj = copy_project(tmp_path)
    led = steps.open_ledger(proj)
    _guide(proj)
    built, res = _finished(proj, led)
    job = json.loads((sched.batch_dir(proj, built["batch"]) / "dataset" / "manifest.json").read_text())["jobs"][0]
    assert job["engine"] == "ai_finish" and job["kind"] == "finish" and job["media"] == "video"
    assert job["control"].endswith(f"{SHOT}.depth.mp4") and job["refs"]       # shipped beside the inputs
    assert res["made"] == 1 and res["failed"] == 0, res
    out = proj.root / led.jobs(SHOT, "finish")[0]["path"]
    assert out.exists() and out.parent.parent.name == "finished"
    assert steps.finish_state(proj, led, proj.shot(SHOT)) == "done"


@needs_ffmpeg
def test_a_new_render_is_finished_again_and_the_same_render_never_is(tmp_path):
    proj = copy_project(tmp_path)
    led = steps.open_ledger(proj)
    _guide(proj)
    _finished(proj, led)
    again = sched.build(proj, led, "ai_finish", 1, SHOT, provider=LocalKaggle(proj.media), engine_set="stub")
    assert again["batch"] is None                                              # nothing changed: no new job
    _guide(proj, source="color=c=red:size=160x90:rate=24")                     # the shot was rendered again
    assert steps.finish_state(proj, led, proj.shot(SHOT)) == "needed"
    _finished(proj, led)
    keys = [j["key"] for j in led.jobs(SHOT, "finish")]
    assert len(keys) == 2 and keys[0] != keys[1]


@needs_ffmpeg
def test_the_finish_stays_out_of_the_producers_loop_until_switched_on(tmp_path):
    proj = copy_project(tmp_path)
    led = steps.open_ledger(proj)
    _guide(proj)
    stage = stages.BY_NAME["ai_finish"]
    assert not stage.wants(proj, led, 1) and "auto off" in stage.progress(proj, led, 1)
    proj.config["engines"]["ai_finish"]["auto"] = True
    assert stage.wants(proj, led, 1)



@needs_ffmpeg
def test_no_finish_without_approved_reference_stills(tmp_path):
    proj = copy_project(tmp_path)
    for f in proj.refs_dir.glob("*.png"):
        f.unlink()                                                             # gate 1 not passed yet
    led = steps.open_ledger(proj)
    _guide(proj)
    out = steps.finish(proj, led, steps.BatchExecutor("b", 3600), 1, SHOT)
    assert out["made"] == 0 and out["no_reference_stills"] == [SHOT]

def test_a_film_without_the_finish_is_told_so(tmp_path):
    proj = copy_project(tmp_path, cfg_edit=lambda c: c["engines"].pop("ai_finish"))
    led = steps.open_ledger(proj)
    with pytest.raises(ValueError, match="not set up"):
        steps.finish(proj, led, steps.BatchExecutor("b", 3600), 1)
    assert stages.BY_NAME["ai_finish"].progress(proj, led, 1) == "not set up"


@needs_ffmpeg
def test_the_cut_takes_a_finish_only_when_switched_on_and_never_a_failed_one(tmp_path):
    proj = copy_project(tmp_path)
    led = steps.open_ledger(proj)
    _guide(proj)
    _finished(proj, led)
    assert asm.finished_clip(proj, led, SHOT) is None                           # use_in_cut is off
    proj.config["engines"]["ai_finish"]["use_in_cut"] = True
    assert asm.finished_clip(proj, led, SHOT).endswith(".mp4")
    led.verdict(led.jobs(SHOT, "finish")[0]["key"], "fail", "human", "off model")
    assert asm.finished_clip(proj, led, SHOT) is None


def _worker():
    spec = importlib.util.spec_from_file_location("worker_for_finish", WORKER)
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    return w


@needs_ffmpeg
def test_the_worker_reads_a_guide_clip_at_the_finishs_size_and_frame_rate(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    w = _worker()
    clip = tmp_path / "g.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=24",
                    "-t", "1", "-pix_fmt", "yuv420p", str(clip)], check=True)
    frames = w.guide_frames(clip, 25, 64, 48, 16)                               # 1 s at 16 fps, asked for 25
    assert len(frames) == 25 and all(f.size == (64, 48) for f in frames)
    assert frames[-1] is frames[-2]                                             # a short clip holds its last frame
    tall = tmp_path / "tall.png"
    Image.new("RGB", (20, 80), (10, 20, 30)).save(tall)
    ref = w.letterbox(tall, 64, 48)
    assert ref.size == (64, 48) and ref.getpixel((0, 0)) == (255, 255, 255)    # whole, on white, never cropped
    assert w.generator_for("ai_finish", [{"kind": "finish"}]) == "ai_finish"
