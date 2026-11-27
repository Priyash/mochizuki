"""The Mac package's parts on their own: engines router, video engine, LoRA training, settings."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

from pipeline.mac import config as mc
from pipeline.mac import train as mt
from pipeline.mac.engines import Engines, StubImageEngine, StubVideoEngine, build, lora_args, segments
from pipeline.mac.engines import video as mv
from pipeline.mac.engines.image import lora_triggers
from pipeline.mac.process import run_streaming

FAKE_WAN = textwrap.dedent('''
    import sys, time, subprocess
    a = sys.argv
    out = a[a.index("--output-path") + 1]; n = int(a[a.index("--num-frames") + 1])
    for i in range(1, 4):
        sys.stderr.write(f"\\rDiffusion: {i*33}%|##| {i}/3 [00:0{i}<00:00, 0.1s/it]"); sys.stderr.flush(); time.sleep(0.05)
    print()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size=128x64:rate=24",
                    "-frames:v", str(n), "-pix_fmt", "yuv420p", out], check=True)
    print("Video saved to", out)
''')

FAKE_TRAIN = textwrap.dedent('''
    import sys, json, zipfile, time
    from pathlib import Path
    a = sys.argv
    if "--dry-run" in a:
        print("Training config validated."); sys.exit(0)
    conf = json.loads(Path(a[a.index("--config") + 1]).read_text())
    for i in range(1, 4):
        sys.stderr.write(f"\\r{i}/3 [00:0{i}<00:00, 0.1s/it]"); sys.stderr.flush(); time.sleep(0.05)
    out = Path(conf["checkpoint"]["output_path"]) / "checkpoints"; out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out / "0000003_checkpoint.zip", "w") as z:
        z.writestr("0000003_adapter.safetensors", b"lora-weights")
    print("done")
''')


def _still(path: Path) -> Path:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180", "-frames:v",
                    "1", str(path)], check=True)
    return path


def test_router_sends_takes_to_video_and_the_rest_to_images(tmp_path):
    seen = []

    class Img:
        def run(self, job, inputs, out):
            seen.append(("image", job["key"]))
            return {}

    class Vid:
        def run(self, job, inputs, out):
            seen.append(("video", job["key"]))
            return {}
    made = []
    e = Engines(lambda: made.append("image") or Img(), lambda: made.append("video") or Vid())
    e.run({"key": "a", "kind": "ref"}, tmp_path, tmp_path / "a.png")
    e.run({"key": "b", "kind": "take"}, tmp_path, tmp_path / "b.mp4")
    e.run({"key": "c", "kind": "keyframe"}, tmp_path, tmp_path / "c.png")
    assert seen == [("image", "a"), ("video", "b"), ("image", "c")]
    assert made == ["image", "video"]                     # each engine made once, when first needed
    assert e.suffix({"kind": "take"}) == ".mp4" and e.suffix({"kind": "ref"}) == ".png"


def test_stub_engines_make_real_files(tmp_path):
    StubImageEngine().run({"key": "s"}, tmp_path, tmp_path / "s.png")
    StubVideoEngine().run({"key": "v"}, tmp_path, tmp_path / "v.mp4")
    assert (tmp_path / "s.png").stat().st_size > 0 and (tmp_path / "v.mp4").stat().st_size > 1000
    assert isinstance(build({}, "stub"), Engines)


def test_wan_segments_chain_long_shots():
    assert segments(2.0, 24, 81) == [49]
    s = segments(6.0, 24, 81)                             # 144 frames: 81 + a chained clip
    assert s[0] == 81 and sum(s) - (len(s) - 1) >= 144 and all((f - 1) % 4 == 0 for f in s)


def test_video_engine_makes_and_chains_clips(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_video", type(sys)("mlx_video"))
    fake = tmp_path / "fake_wan.py"
    fake.write_text(FAKE_WAN)
    cfg = {"engines": {"video": {**mc.DEFAULTS["engines"]["video"], "cmd": [sys.executable, str(fake)],
                                 "width": 128, "height": 64, "fps": 24, "max_frames": 25}}}
    cfg["engines"]["video"].pop("download")
    eng = mv.MlxVideoEngine(cfg)
    _still(tmp_path / "start.png")
    info = eng.run({"key": "take-1", "kind": "take", "prompt": "p", "image": "start.png", "seconds": 1.5},
                   tmp_path, tmp_path / "take.mp4")
    assert len(info["segments"]) == 2                     # 36 frames at max 25 a clip: two chained clips
    frames = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                             "stream=nb_read_frames", "-of", "csv=p=0", str(tmp_path / "take.mp4")],
                            capture_output=True, text=True).stdout.strip()
    assert int(frames) == sum(info["segments"]) - 1       # the shared frame is not repeated


def test_video_engine_says_how_to_install_mlx_video(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_video", None)
    with pytest.raises(RuntimeError, match="mlx-video is not installed"):
        mv.MlxVideoEngine({"engines": mc.DEFAULTS["engines"]})


def test_training_makes_a_lora_and_registers_it(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "WORK", tmp_path / "work")
    monkeypatch.setattr(mt, "LORAS", tmp_path / "work" / "loras")
    monkeypatch.setattr(mt, "LOCAL_LORAS", tmp_path / "work" / "loras.json")
    tool = tmp_path / "mflux-train"
    tool.write_text(f"#!{sys.executable}\n" + FAKE_TRAIN)
    tool.chmod(0o755)
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(16):
        _still(imgs / f"look{i}.png")
    (imgs / "look0.txt").write_text("the grandmother at the mortar")
    cfg = {"train": mc.DEFAULTS["train"]}
    r = mt.train_lora(cfg, "style3d", imgs, trigger="mchz3d", caption="3D cel-shaded anime still",
                      preview="key art", tool=str(tool))
    lora = tmp_path / "work" / "loras" / "style3d.safetensors"
    assert r["ok"] and lora.read_bytes() == b"lora-weights"
    data = tmp_path / "work" / "train" / "style3d" / "data"
    assert (data / "01.txt").read_text() == "mchz3d, the grandmother at the mortar"   # its own caption
    assert (data / "02.txt").read_text() == "mchz3d, 3D cel-shaded anime still"
    assert (data / "preview_1.txt").read_text() == "mchz3d, key art"
    reg = json.loads((tmp_path / "work" / "loras.json").read_text())
    assert reg[0]["name"] == "style3d" and reg[0]["trigger"] == "mchz3d"


def test_training_refuses_too_few_images(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(3):
        _still(imgs / f"x{i}.png")
    with pytest.raises(ValueError, match="at least 8"):
        mt.train_lora({"train": mc.DEFAULTS["train"]}, "x", imgs, dry_run=True)


def test_train_config_fits_a_32_gb_mac():
    c = mt.train_config(mc.DEFAULTS["train"], 20)
    assert c["model"] == "z-image" and c["quantize"] == 8 and c["low_ram"] and c["gradient_checkpointing"]
    assert c["training_loop"]["num_epochs"] * 20 >= mc.DEFAULTS["train"]["steps_total"]


def test_loras_trained_on_this_mac_reach_the_engines(tmp_path):
    lora = tmp_path / "style3d.safetensors"
    lora.write_bytes(b"x")
    reg = tmp_path / "loras.json"
    reg.write_text(json.dumps([{"name": "style3d", "path": str(lora), "trigger": "mchz3d", "scale": 0.8,
                                "roles": ["refs"]}]))
    cfg = mc.settings(None, local_loras=reg)
    assert lora_args(cfg, "refs") == ["--lora-paths", str(lora), "--lora-scales", "0.8"]
    assert lora_args(cfg, "keyframe") == []
    assert lora_triggers(cfg, "refs") == "mchz3d"
    lora.unlink()                                          # moved away: render without it, no crash
    assert lora_args(cfg, "refs") == [] and lora_triggers(cfg, "refs") == ""


def test_engine_sets_swap_only_what_they_name():
    cfg = {**mc.DEFAULTS, "engines_zimage_base": {"refs": {"cmd": ["mflux-generate-z-image"]}}}
    alt = mc.with_engine_set(cfg, "zimage_base")
    assert alt["engines"]["refs"]["cmd"] == ["mflux-generate-z-image"]
    assert alt["engines"]["video"] == mc.DEFAULTS["engines"]["video"]
    with pytest.raises(KeyError):
        mc.with_engine_set(cfg, "nope")


def test_streaming_logs_steps_and_other_output(tmp_path, capsys):
    script = tmp_path / "tool.py"
    script.write_text(textwrap.dedent('''
        import sys, time
        print("WARNING: something odd", flush=True)
        for i in range(1, 4):
            sys.stderr.write(f"\\r {i*33}%|##| {i}/3 [00:0{i}<00:00]"); sys.stderr.flush(); time.sleep(0.05)
        print(); print("saved")
    '''))
    code, tail = run_streaming([sys.executable, str(script)], 30, label="t:", kind="train", tool="mflux-train")
    out = capsys.readouterr().out
    assert code == 0 and "saved" in tail
    assert "train.loaded" in out and out.count("train.step") == 3
    assert "WARN  mac    mflux-train.out" in out
    assert "%|" not in out                                # the raw bar is not echoed
