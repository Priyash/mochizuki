"""The generators: every AI model and tool the engine uses to make a film, in one catalog.

Each generator makes one kind of thing (images, video, voices ...) for one stage of the chain
(engine/stages). Its model is named and configured in the film's config.json, never in code, so the
best model for a job can change with a config edit; `config` says where. `setup` is what it installs
or downloads before it runs, `needs` what a GPU must have, `runs_on` where it can run.

Where the code is: GPU generators live in the worker (worker/worker.py, the one file every cloud
provider runs; its GENERATORS use these names) and, for the Mac, in engine/mac/generators. The others
run on any CPU: the 3D render (engine/cgi) and sound (engine/audio).

Also here: small rules the generators share, so they are written once (the worker keeps its own
copies, since it ships alone; a test keeps them equal).
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

# What a generator makes
IMAGE, VIDEO, UPSCALE, RENDER, VOICE, LIPSYNC, SUBTITLES, MUSIC = (
    "image", "video", "upscale", "3d render", "voice", "lip sync", "subtitles", "music")
# Where it runs
GPU = "gpu"              # on a provider: Kaggle, Colab or the Mac (engine/providers)
CPU = "cpu"              # anywhere: a GitHub runner, the Mac, this machine
# What a GPU must have
BF16 = "bf16"            # bfloat16: Ampere or newer (A100, L4), or Apple Silicon. Not Kaggle's T4s.
CAPABILITIES = frozenset({BF16})
# The 3D render's guide passes for the AI finish (engine/cgi/render/blender/guides.py renders them)
GUIDES = ("depth", "lines")
FINISH_KEYS = ("model", "width", "height", "fps", "max_frames")    # what engines.ai_finish must say


@dataclass(frozen=True)
class Generator:
    name: str
    makes: str
    stage: str
    config: str                     # where its model and settings live in the film's config.json
    runs_on: tuple[str, ...]
    setup: str                      # what it installs or downloads before it runs
    code: str                       # where it is implemented
    needs: frozenset[str] = field(default_factory=frozenset)
    built: bool = True


CATALOG: dict[str, Generator] = {g.name: g for g in (
    Generator("refs", IMAGE, "refs", "engines.refs", (GPU,), needs=frozenset({BF16}),
              setup="the worker's GPU packages; Qwen-Image (GGUF) from Hugging Face",
              code="worker/worker.py QwenStills; Mac: engine/mac/generators/image.py"),
    Generator("keyframe", IMAGE, "keyframes", "engines.keyframe", (GPU,), needs=frozenset({BF16}),
              setup="the worker's GPU packages; Qwen-Image-Edit (GGUF) from Hugging Face",
              code="worker/worker.py QwenKeyframes; Mac: engine/mac/generators/image.py"),
    Generator("video", VIDEO, "video", "engines.<the shot's model in shots.csv>", (GPU,),
              setup="the worker's GPU packages; Wan 2.2 (5B, or 14B as GGUF) from Hugging Face",
              code="worker/worker.py WanVideo; Mac: engine/mac/generators/video.py"),
    Generator("upscale", UPSCALE, "upscale", "engines.upscale", (GPU,),
              setup="the worker's GPU packages; Real-ESRGAN and Real-CUGAN weights",
              code="worker/worker.py Upscale4K"),
    Generator("render", RENDER, "render", "cgi", (CPU,),
              setup="Blender 5, or a Python with the bpy module", code="engine/cgi/render/runner.py"),
    Generator("voice", VOICE, "voices", "style_bible/dialogue.json -> voices", (CPU,),
              setup="Kokoro-82M (engine/audio/tts/requirements.txt), or any command", code="engine/audio/voice.py"),
    Generator("lipsync", LIPSYNC, "voices", "(none)", (CPU,), setup="nothing", code="engine/audio/lipsync.py"),
    Generator("subtitles", SUBTITLES, "cut", "style_bible/dialogue.json -> languages", (CPU,),
              setup="nothing", code="engine/audio/subtitles.py"),
    Generator("ai_finish", VIDEO, "ai_finish", "engines.ai_finish", (GPU,),
              setup="the worker's GPU packages; a Wan VACE model (video-to-video, guided) from Hugging Face",
              code="worker/worker.py WanFinish; guides: engine/cgi/render/blender/guides.py"),
    Generator("music", MUSIC, "music", "engines.music", (GPU, CPU), built=False,
              setup="a music model whose licence allows a published film", code="(not built)"),
)}


def for_stage(stage: str) -> list[Generator]:
    return [g for g in CATALOG.values() if g.stage == stage]


def needs_for(stage: str) -> frozenset[str]:
    """What a GPU must have to run a stage's built generators."""
    out: frozenset[str] = frozenset()
    for g in for_stage(stage):
        if g.built:
            out |= g.needs
    return out


# ---------------------------------------------------------------- shared rules
def wan_frames(seconds: float, fps: int, max_frames: int) -> int:
    """Wan needs 4k+1 frames. Smallest such count covering the shot, capped by the model."""
    need = max(1, math.ceil(seconds * fps))
    frames = ((need - 1 + 3) // 4) * 4 + 1
    cap = ((max_frames - 1) // 4) * 4 + 1
    return max(5, min(frames, cap))


def wan_segments(seconds: float, fps: int, max_frames: int) -> list[int]:
    """Frame counts of the clips that cover a shot. Each is 4k+1 frames and at most max_frames.

    Long shots are chained: each next clip starts from the previous clip's last frame,
    so it shares one frame with it. Total new frames >= the shot's length.
    """
    need = max(1, math.ceil(seconds * fps))
    first = wan_frames(seconds, fps, max_frames)
    segs, have = [first], first
    while have < need:
        nxt = wan_frames((need - have + 1) / fps, fps, max_frames)
        segs.append(nxt)
        have += nxt - 1
    return segs


def seed_for(key: str) -> int:
    """A stable seed per key, so a rerun of the same job (or shot) draws the same picture."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)
