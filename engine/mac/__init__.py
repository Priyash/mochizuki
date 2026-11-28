"""Your own Apple Silicon Mac as a free, offline renderer: stills, keyframes, video, LoRA training.

Sub-components:
  config        settings (the film's config.json -> mac) and engine sets
  generators/   image (mflux: Z-Image, FLUX.2 klein), video (mlx-video: Wan 2.2), stand-ins, the router
  runtime/      shared (workspace, log, ffmpeg), downloads (models), tools (running them), health
  worker/       loop (claim a batch, render it, upload results), selftest (checks before real work)
  training/     lora (LoRA training with mflux-train: the film's look, the characters)
  cli           the `mac-worker` command
The producer and the Mac talk only through the `batches` release (engine/providers/release.py); the
producer side, the Mac as a provider, is engine/providers/mac.py.
"""
from .config import DEFAULTS, NAME, settings, with_engine_set

__all__ = ["DEFAULTS", "NAME", "settings", "with_engine_set"]
