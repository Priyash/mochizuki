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

def _no_finish(cfg):
    for name in [n for n in cfg["engines"] if n.startswith("ai_finish")]:
        cfg["engines"].pop(name)


def test_a_film_without_the_finish_is_told_so(tmp_path):
    proj = copy_project(tmp_path, cfg_edit=_no_finish)
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


@needs_ffmpeg
def test_the_finish_paints_the_stills_look_not_the_toon_render(tmp_path):
    proj = copy_project(tmp_path)
    led = steps.open_ledger(proj)
    _guide(proj)
    ex = steps.BatchExecutor("b", 3600)
    assert steps.finish(proj, led, ex, 1, SHOT)["made"] == 1
    prompt = ex.specs[0]["prompt"]
    style_txt = proj.style_file.read_text(encoding="utf-8").strip()
    assert "cel-shaded" in style_txt and "cel-shaded" not in prompt            # style.txt is the 3D render's look
    assert prompt.startswith(proj.engine("ai_finish")["style"])
    assert proj.engine("ai_finish")["who"]["tsumugi_ch1"] in prompt            # who the still shows
    assert proj.shot(SHOT).shot in prompt
    assert proj.prompt(proj.shot(SHOT)).startswith(style_txt)                  # keyframes keep style.txt


@needs_ffmpeg
def test_the_finish_never_takes_the_style_scene_and_cuts_a_sheet_to_one_view(tmp_path):
    image = pytest.importorskip("PIL.Image")
    proj = copy_project(tmp_path)
    s = proj.shot(SHOT)
    assert [r.stem for r in proj.refs_for(s)] == ["style", "tsumugi_ch1"]            # keyframes: style first
    cfg = steps.finish_config(proj)
    refs = steps.finish_refs(proj, s, cfg)
    assert len(refs) == 1 and refs[0].name.startswith("tsumugi_ch1.")                # no style scene
    assert refs[0].parent == proj.media / "finished" / "refs"                         # never in the refs release
    sheet = image.open(proj.ref_path("tsumugi_ch1")).size
    assert image.open(refs[0]).size == (round(sheet[0] * 0.32), sheet[1])            # the front view only
    clip = _guide(proj)
    key = steps.finish_spec(proj, s, 0, clip)["key"]
    proj.config["engines"]["ai_finish"]["crops"]["tsumugi_ch1"] = [0.0, 0.0, 0.3, 1.0]
    moved = steps.finish_refs(proj, s, steps.finish_config(proj))
    assert moved[0] != refs[0] and steps.finish_spec(proj, s, 0, clip)["key"] != key  # a new box is a new job
    del proj.config["engines"]["ai_finish"]["crops"]["tsumugi_ch1"]
    assert steps.finish_refs(proj, s, steps.finish_config(proj)) == [proj.ref_path("tsumugi_ch1")]   # no box: whole


@needs_ffmpeg
def test_a_variant_finish_is_its_own_job_at_its_own_size(tmp_path):
    proj = copy_project(tmp_path)
    s = proj.shot(SHOT)
    clip = _guide(proj)
    base = steps.finish_spec(proj, s, 0, clip)
    big = steps.finish_spec(proj, s, 0, clip, engine="ai_finish_720")
    assert big["engine"] == "ai_finish_720" and big["key"] != base["key"] and big["refs"] == base["refs"]
    cfg = steps.finish_config(proj, "ai_finish_720")
    assert (cfg["width"], cfg["height"], cfg["fps"]) == (1280, 720, proj.engine("ai_finish")["fps"])
    lines = _guide(proj, control="lines")
    assert steps.finish_spec(proj, s, 0, lines, control="lines")["key"] != base["key"]   # each guide its own job


def _fake_gpu_stack(calls: dict, gpu_gb: float = 15.0, capability=(7, 5)):
    """Stand-ins for torch and diffusers that record what the finish engine asks of them (a T4 by default)."""
    import types
    torch = types.ModuleType("torch")
    torch.float16, torch.float32, torch.bfloat16 = "fp16", "fp32", "bf16"
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: True, get_device_capability=lambda *a: capability,
        is_bf16_supported=lambda: capability[0] >= 8, memory_allocated=lambda *a: 0,
        get_device_properties=lambda i: types.SimpleNamespace(total_memory=gpu_gb * 2**30))
    torch.device = lambda name: f"device:{name}"
    torch.Generator = lambda dev: types.SimpleNamespace(manual_seed=lambda seed: ("generator", dev, seed))
    import contextlib
    torch.inference_mode = contextlib.nullcontext
    d = types.ModuleType("diffusers")

    class Module:
        def __init__(self, name):
            self.name = name

        def to(self, dev):
            calls.setdefault("moved", []).append((self.name, dev))
            return self

        def enable_tiling(self):
            pass

        def enable_group_offload(self, **kw):
            calls["group_offload"] = kw

    class Pipe:
        _execution_device = "cuda:0"

        def __init__(self, local, **kw):
            calls["pipeline"] = (local, kw)
            self.transformer = kw.get("transformer") or Module("transformer")
            self.vae, self.scheduler = kw["vae"], types.SimpleNamespace(config={"use_flow_sigmas": True})

        def __call__(self, **kw):
            import numpy as np
            calls.setdefault("calls", []).append(kw)
            n, h, w = kw["num_frames"], kw["height"], kw["width"]
            frames = np.linspace(0, 1, n * h * w * 3, dtype="float32").reshape(n, h, w, 3)
            return types.SimpleNamespace(frames=[frames])

        def to(self, dev):
            calls["pipeline_to"] = dev
            return self

        def enable_model_cpu_offload(self):
            calls["model_offload"] = True

        def set_progress_bar_config(self, **kw):
            pass

    def scheduler(kind):
        return types.SimpleNamespace(from_config=lambda config, **kw: calls.__setitem__("scheduler", (kind, kw))
                                     or (kind, kw))
    d.AutoencoderKLWan = types.SimpleNamespace(from_pretrained=lambda *a, **k: Module("vae"))
    d.WanVACEPipeline = types.SimpleNamespace(from_pretrained=lambda local, **kw: Pipe(local, **kw))
    d.GGUFQuantizationConfig = lambda compute_dtype: ("gguf", compute_dtype)
    d.WanVACETransformer3DModel = types.SimpleNamespace(
        from_single_file=lambda path, **kw: calls.__setitem__("single_file", (path, kw)) or Module("gguf"))
    d.UniPCMultistepScheduler, d.FlowMatchEulerDiscreteScheduler = scheduler("unipc"), scheduler("euler")
    return torch, d


def _finish_engine(monkeypatch, cfg: dict, gguf_in: str = "QuantStack", gpu_gb: float = 15.0, capability=(7, 5)):
    import sys
    w, calls = _worker(), {}
    torch, diffusers = _fake_gpu_stack(calls, gpu_gb, capability)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setattr(w, "snapshot", lambda model, skip: calls.__setitem__("snapshot", (model, skip)) or "/hf")
    class Embeds:
        def __init__(self, text):
            self.text = text

        def to(self, dev):
            return (self.text, dev)
    monkeypatch.setattr(w, "encode_prompts", lambda local, texts, device="cpu": calls.__setitem__(
        "encoded_on", device) or {t: Embeds(t) for t in texts})
    monkeypatch.setattr(w, "gpu_room_gb", lambda headroom: gpu_gb - headroom)
    monkeypatch.setattr(w, "find_gguf", lambda repo, sub, prefer, room: calls.setdefault("gguf", []).append(
        (repo, room)) or (f"/{repo}.gguf" if gguf_in in repo else None))
    calls["engine"] = w.WanFinish(cfg, "auto", ["a prompt"], "a negative")
    return calls


def test_the_14b_finish_comes_as_a_gguf_and_goes_to_a_t4_a_block_at_a_time(tmp_path, monkeypatch):
    proj = copy_project(tmp_path)
    calls = _finish_engine(monkeypatch, steps.finish_config(proj, "ai_finish_720"))
    assert calls["snapshot"] == ("Wan-AI/Wan2.1-VACE-14B-diffusers", ["transformer"])   # the GGUF replaces it
    assert calls["gguf"] == [("QuantStack/Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF", 15)]  # sized by RAM
    path, kw = calls["single_file"]
    assert path.startswith("/QuantStack") and kw["config"] == "/hf" and kw["subfolder"] == "transformer"
    assert kw["quantization_config"] == ("gguf", "fp16")                    # a T4 computes in fp16
    assert calls["pipeline"][1]["transformer"].name == "gguf" and calls["pipeline"][1]["text_encoder"] is None
    assert calls["group_offload"]["offload_type"] == "block_level" and "model_offload" not in calls
    assert ("vae", "device:cuda") in calls["moved"] and "pipeline_to" not in calls
    assert calls["encoded_on"] == "cpu"                                     # UMT5 overflows in fp16
    assert calls["scheduler"] == ("unipc", {"solver_order": 1, "disable_corrector": [0, 1, 2, 3], "flow_shift": 5.0})


def test_on_an_a100_the_whole_finish_sits_on_the_gpu_as_the_stills_engine_does(tmp_path, monkeypatch):
    proj = copy_project(tmp_path)
    calls = _finish_engine(monkeypatch, steps.finish_config(proj, "ai_finish_720"), gpu_gb=40.0, capability=(8, 0))
    assert calls["pipeline_to"] == "cuda" and "group_offload" not in calls and "model_offload" not in calls
    assert calls["encoded_on"] == "cuda"                                    # seconds instead of minutes
    assert calls["single_file"][1]["quantization_config"] == ("gguf", "bf16")
    small = _finish_engine(monkeypatch, steps.finish_config(proj), gpu_gb=40.0, capability=(8, 0))
    assert small["pipeline_to"] == "cuda" and "gguf" not in small


def test_a_missing_gguf_repo_falls_back_to_the_next(tmp_path, monkeypatch):
    proj = copy_project(tmp_path)
    calls = _finish_engine(monkeypatch, steps.finish_config(proj, "ai_finish_720"), gguf_in="lym00")
    assert [r for r, _ in calls["gguf"]] == ["QuantStack/Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF",
                                            "lym00/Wan2.1_T2V_14B_LightX2V_StepCfgDistill_VACE-GGUF"]
    assert calls["single_file"][0].startswith("/lym00")


def test_the_small_finish_loads_whole_and_offloads_by_model_on_a_t4(tmp_path, monkeypatch):
    proj = copy_project(tmp_path)
    calls = _finish_engine(monkeypatch, steps.finish_config(proj))
    assert calls["snapshot"] == ("Wan-AI/Wan2.1-VACE-1.3B-diffusers", [])
    assert "gguf" not in calls and "transformer" not in calls["pipeline"][1]
    assert calls["model_offload"] and "group_offload" not in calls and calls["encoded_on"] == "cpu"
    assert calls["scheduler"] == ("unipc", {"flow_shift": 3.0})


def test_no_gguf_anywhere_is_a_clear_error(tmp_path, monkeypatch):
    proj = copy_project(tmp_path)
    with pytest.raises(RuntimeError, match="no GGUF in QuantStack/.*, lym00/"):
        _finish_engine(monkeypatch, steps.finish_config(proj, "ai_finish_720"), gguf_in="nowhere")


class _KaggleHere:
    """Kaggle for the finish test, run here: the notebook's archive comes from the fake release and the
    worker runs with its stand-in engines."""

    def __init__(self, release_dir, tmp):
        self.release_dir, self.tmp, self.log, self.pushed = release_dir, tmp, [], []

    def push(self, folder, accelerator, timeout_s):
        import re
        import sys
        import tarfile
        script = (folder / "finish_test.py").read_text()
        name = re.search(r"urlretrieve\('([^']+)'", script).group(1).rsplit("/", 1)[1]
        test_id = re.search(r'"--expect-batch", \'([^\']+)\'', script).group(1)
        self.pushed.append((json.loads((folder / "kernel-metadata.json").read_text()), accelerator, timeout_s))
        inputs = self.tmp / "in"
        with tarfile.open(self.release_dir / name) as t:
            t.extractall(inputs, filter="data")
        subprocess.run([sys.executable, str(inputs / "worker.py"), "--manifest", str(inputs / "manifest.json"),
                        "--inputs", str(inputs), "--out", str(self.tmp / "working" / "out"), "--engine-set", "stub",
                        "--expect-batch", test_id], check=True)

    def status(self, ref):
        return "complete"

    def output(self, ref, dest):
        shutil.copytree(self.tmp / "working", dest, dirs_exist_ok=True)


def _rendered(tmp_path):
    """A film with shot 1-05 rendered: its 3D render and both guides."""
    (tmp_path / "film").mkdir()
    proj = copy_project(tmp_path / "film")
    for control in ("depth", "lines"):
        _guide(proj, control=control)
    render = cgi_config.work_dir(proj, cgi_config.settings(proj), SHOT) / f"{SHOT}.mp4"   # the 3D render itself
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "smptebars=size=320x180:rate=24",
                    "-t", "3", "-pix_fmt", "yuv420p", str(render)], check=True)
    return proj


@needs_ffmpeg
def test_a_finish_test_runs_one_shot_outside_the_ledger_and_lays_it_beside_the_render(tmp_path):
    pytest.importorskip("PIL")
    from test_mac import FakeRelease

    from engine.providers.kaggle import finish_test as ft
    from engine.providers.release import Release
    proj = _rendered(tmp_path)
    fr = FakeRelease(tmp_path / "release")
    kaggle = _KaggleHere(fr.dir, tmp_path / "kaggle")
    r = ft.run(proj, kaggle, Release("o/r", gh=fr.gh, tag=ft.RELEASE), SHOT, ["ai_finish_720"], ["depth", "lines"],
               tmp_path / "work", minutes=120, poll=0, sleep=lambda s: None)
    assert set(r["jobs"]) == {"ai_finish_720/depth", "ai_finish_720/lines"}, r
    assert all(j["ok"] for j in r["jobs"].values()), r
    assert r["jobs"]["ai_finish_720/depth"]["size"] == "1280x720" and r["backend"] == "kaggle"
    assert r["engines"] == {"ai_finish_720": "Wan-AI/Wan2.1-VACE-14B-diffusers"}
    meta, accelerator, timeout = kaggle.pushed[0]
    assert meta["id"] == "tester/film-finish-test" and meta["enable_internet"] == "true"   # its own notebook
    assert accelerator == "NvidiaTeslaT4" and timeout == 120 * 60
    test = r["test"]
    names = set(fr.names())
    assert {f"{test}-inputs.tar.gz", f"{test}-ai_finish_720-depth.mp4", f"{test}-ai_finish_720-depth-side.mp4",
            f"{test}-ai_finish_720-depth-middle.png", f"{test}-ai_finish_720-lines-middle.png",
            f"{test}-summary.json", f"{test}-worker.jsonl"} <= names
    assert all(u.startswith(f"https://github.com/o/r/releases/download/{ft.RELEASE}/") for u in r["files"])
    from PIL import Image
    middle = Image.open(fr.dir / f"{test}-ai_finish_720-depth-middle.png")
    assert middle.height == 720 and middle.width > 2 * 1280                         # render, finish, the still
    assert steps.open_ledger(proj).jobs(SHOT, "finish") == []                         # the ledger never knew


class _ColabRunnerHere:
    """The scheduled Colab runner, run here when the test waits: it takes the waiting batch from the
    batches release, runs the worker with its stand-in engines and uploads the results as the runner does."""

    def __init__(self, rel, tmp, take=True):
        self.rel, self.tmp, self.take, self.taken = rel, tmp, take, []

    def __call__(self, seconds):
        import sys
        import tarfile
        names = self.rel.names()
        for name in names:
            if not (name.startswith("colab-") and name.endswith(".tar.gz")) or name.endswith(".results.tar.gz"):
                continue
            bid = name[len("colab-"):-len(".tar.gz")]
            if not self.take or f"colab-{bid}.results.tar.gz" in names:
                continue
            d = self.tmp / bid
            with tarfile.open(self.rel.dir / name) as t:
                t.extractall(d / "in", filter="data")
            assert not (d / "in" / "worker.py").exists()                         # the runner brings its own
            shutil.copyfile(WORKER, d / "in" / "worker.py")
            claim = d / f"colab-{bid}.claim.json"
            claim.write_text("{}")
            self.rel.gh(["release", "upload", "batches", str(claim)])
            subprocess.run([sys.executable, str(d / "in" / "worker.py"), "--manifest", str(d / "in" / "manifest.json"),
                            "--inputs", str(d / "in"), "--out", str(d / "out"), "--engine-set", "stub",
                            "--expect-batch", bid], check=True)
            res = json.loads((d / "out" / "results.json").read_text())
            res.update(vm_seconds=900, gpu="L4", units_used=0.4)
            (d / "out" / "results.json").write_text(json.dumps(res))
            results = d / f"colab-{bid}.results.tar.gz"
            with tarfile.open(results, "w:gz") as t:
                t.add(d / "out", arcname="out")
            self.rel.gh(["release", "upload", "batches", str(results)])
            self.taken.append(bid)


@needs_ffmpeg
def test_a_finish_test_on_colab_waits_for_the_runner_and_tidies_up(tmp_path):
    pytest.importorskip("PIL")
    from test_mac import FakeRelease

    from engine.providers.kaggle import finish_test as ft
    from engine.providers.release import Release
    proj = _rendered(tmp_path)
    fr, fb = FakeRelease(tmp_path / "finish-tests"), FakeRelease(tmp_path / "batches")
    runner = _ColabRunnerHere(fb, tmp_path / "vm")
    r = ft.run(proj, None, Release("o/r", gh=fr.gh, tag=ft.RELEASE), SHOT, ["ai_finish", "ai_finish_720"], ["depth"],
               tmp_path / "work", poll=0, sleep=runner, backend="colab", batches=Release("o/r", gh=fb.gh))
    assert runner.taken == [r["test"]] and r["backend"] == "colab"             # both engines in one VM session
    assert set(r["jobs"]) == {"ai_finish/depth", "ai_finish_720/depth"} and all(j["ok"] for j in r["jobs"].values())
    assert r["jobs"]["ai_finish/depth"]["size"] == "832x480" and r["jobs"]["ai_finish_720/depth"]["size"] == "1280x720"
    assert (r["gpu"], r["units_used"]) == ("L4", 0.4)                           # what the owner's units paid for
    assert fb.names() == []                                                     # nothing left for a later run
    assert {f"{r['test']}-ai_finish-depth-middle.png", f"{r['test']}-ai_finish_720-depth-middle.png"} <= set(fr.names())


@needs_ffmpeg
def test_a_colab_test_no_runner_takes_is_taken_down(tmp_path, monkeypatch):
    from test_mac import FakeRelease

    from engine.providers.kaggle import finish_test as ft
    from engine.providers.release import Release
    proj = _rendered(tmp_path)
    fr, fb = FakeRelease(tmp_path / "finish-tests"), FakeRelease(tmp_path / "batches")
    monkeypatch.setattr(ft, "COLAB_PICKUP_MINUTES", -1)                         # the wait is over at once
    r = ft.run(proj, None, Release("o/r", gh=fr.gh, tag=ft.RELEASE), SHOT, ["ai_finish"], ["depth"],
               tmp_path / "work", poll=0, sleep=_ColabRunnerHere(fb, tmp_path / "vm", take=False),
               backend="colab", batches=Release("o/r", gh=fb.gh))
    assert "no Colab runner took the batch" in r["error"] and not r["jobs"]["ai_finish/depth"]["ok"]
    assert fb.names() == []                                                     # no unit is spent on it later


@needs_ffmpeg
def test_a_finish_test_kaggle_will_not_start_says_why(tmp_path):
    from test_mac import FakeRelease

    from engine.providers.kaggle import KaggleError
    from engine.providers.kaggle import finish_test as ft
    from engine.providers.release import Release

    class Busy(_KaggleHere):
        def push(self, folder, accelerator, timeout_s):
            raise KaggleError("kaggle kernels push failed: Maximum batch GPU session count of 2 reached.")
    proj = copy_project(tmp_path)
    _guide(proj)
    fr = FakeRelease(tmp_path / "release")
    r = ft.run(proj, Busy(fr.dir, tmp_path / "k"), Release("o/r", gh=fr.gh, tag=ft.RELEASE), SHOT, ["ai_finish"],
               ["depth"], tmp_path / "work", poll=0, sleep=lambda s: None)
    assert "GPU session count" in r["error"] and r["statuses"] == ["not started"]
    assert not r["jobs"]["ai_finish/depth"]["ok"] and f"{r['test']}-summary.json" in fr.names()


@needs_ffmpeg
def test_the_finish_hands_the_pipeline_its_prompt_on_the_pipelines_device(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    proj = copy_project(tmp_path)
    clip = _guide(proj)
    refs = steps.finish_refs(proj, proj.shot(SHOT), steps.finish_config(proj))
    shutil.copyfile(clip, tmp_path / "guide.mp4")
    shutil.copyfile(refs[0], tmp_path / "still.png")
    calls = _finish_engine(monkeypatch, steps.finish_config(proj), gpu_gb=40.0, capability=(8, 0))
    job = {"key": "k", "seconds": 3, "control": "guide.mp4", "refs": ["still.png"], "prompt": "a prompt"}
    info = calls["engine"].run(job, tmp_path, tmp_path / "out.mp4")
    kw = calls["calls"][0]
    assert kw["prompt_embeds"] == ("a prompt", "cuda:0")                 # given embeddings: moved here, not there
    assert kw["negative_prompt_embeds"] == ("a negative", "cuda:0")
    assert len(kw["video"]) == kw["num_frames"] == 49 and len(kw["reference_images"]) == 1
    assert (kw["width"], kw["height"], kw["num_inference_steps"]) == (832, 480, 30)
    assert info["frames"] == 49 and (tmp_path / "out.mp4").exists()


@needs_ffmpeg
def test_a_kept_colab_test_waits_for_a_gpu_and_is_collected_when_it_comes(tmp_path):
    pytest.importorskip("PIL")
    from test_mac import FakeRelease

    from engine.providers.kaggle import finish_test as ft
    from engine.providers.release import Release
    proj = _rendered(tmp_path)
    fr, fb = FakeRelease(tmp_path / "finish-tests"), FakeRelease(tmp_path / "batches")
    rel, batches = Release("o/r", gh=fr.gh, tag=ft.RELEASE), Release("o/r", gh=fb.gh)
    r = ft.run(proj, None, rel, SHOT, ["ai_finish"], ["depth"], tmp_path / "work", poll=0,
               sleep=_ColabRunnerHere(fb, tmp_path / "vm", take=False), backend="colab", batches=batches,
               wait=-1, keep=True)                                          # Colab has no GPU tonight
    test = r["test"]
    assert f"--collect {test}" in r["error"] and fb.names() == [f"colab-{test}.tar.gz"]   # still posted
    assert ft.collect(proj, rel, batches, test, tmp_path / "later")["error"] == "no results yet (waiting)"
    _ColabRunnerHere(fb, tmp_path / "vm")(0)                                # a runner gets a GPU at last
    done = ft.collect(proj, rel, batches, test, tmp_path / "later")
    assert done["jobs"]["ai_finish/depth"]["ok"] and done["statuses"] == ["collected"]
    assert f"{test}-ai_finish-depth-middle.png" in fr.names() and fb.names() == []
