"""Your own Apple Silicon Mac as a free, offline renderer: stills, keyframes, video, LoRA training.

Modules, each with one job:

  config     settings (config.json → mac) and engine sets
  common     structured log, seeds, paths under ~/mochizuki-mac, ffmpeg
  sysinfo    what the Mac has, and how it is coping (memory, swap, heat)
  models     model downloads with progress, and checks before a model is used
  process    running a model tool and turning its output into structured log lines
  engines/   image (mflux: Z-Image, FLUX.2 klein), video (mlx-video: Wan 2.2), stand-ins, the router
  worker     claim a batch from the `batches` release, render it, upload results
  selftest   the checks you run before real work (still, keyframe, video)
  train      LoRA training with mflux-train (the film's look, the characters)
  backend    producer side: the Mac as a backend the pipeline sends batches to
  cli        the `mac-worker` command

The producer and the Mac talk only through the `batches` release (engine/release.py).
"""
from .backend import MacBackend, from_env
from .config import DEFAULTS, NAME, settings, with_engine_set

__all__ = ["DEFAULTS", "NAME", "MacBackend", "from_env", "settings", "with_engine_set"]
