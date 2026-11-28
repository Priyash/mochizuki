"""The Mac's engines, and the router that sends each job to the right one.

  images  MfluxImageEngine   reference stills (text to image) and keyframes (image edit)
  video   MlxVideoEngine     takes (image to video, Wan 2.2)
Engines load lazily: a batch of stills never needs mlx-video installed, and the other way round.
"""
from __future__ import annotations

from pathlib import Path

from .base import Engine, StubImageEngine, StubVideoEngine, check_image, check_video, is_video_job
from .image import MfluxImageEngine, lora_args
from .video import MlxVideoEngine, frames_for, segments

__all__ = ["Engine", "Engines", "MfluxImageEngine", "MlxVideoEngine", "StubImageEngine", "StubVideoEngine",
           "build", "check_image", "check_video", "frames_for", "is_video_job", "lora_args", "segments"]


class Engines:
    """One `run(job, inputs, out)` for every kind of job; each engine is made on first use."""

    def __init__(self, make_image, make_video):
        self._make = {"image": make_image, "video": make_video}
        self._made: dict[str, Engine] = {}

    def engine(self, kind: str) -> Engine:
        if kind not in self._made:
            self._made[kind] = self._make[kind]()
        return self._made[kind]

    def suffix(self, job: dict) -> str:
        return ".mp4" if is_video_job(job) else ".png"

    def run(self, job: dict, inputs: Path, out: Path) -> dict:
        return self.engine("video" if is_video_job(job) else "image").run(job, inputs, out)


def build(cfg: dict, engine_set: str = "gpu") -> Engines:
    """The real engines ("gpu"), or stand-ins with no model ("stub")."""
    if engine_set == "stub":
        return Engines(StubImageEngine, StubVideoEngine)
    return Engines(lambda: MfluxImageEngine(cfg), lambda: MlxVideoEngine(cfg))
