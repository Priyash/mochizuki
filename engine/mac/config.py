"""The Mac's settings: config.json → `mac`, over these defaults.

Engines are commands, so a model can be swapped in config.json without touching code:
  engines.refs      text to image (reference stills)
  engines.keyframe  image edit from 1-3 reference images (derived stills, keyframes)
  engines.video     image to video (takes)
Other engine sets live beside them as engines_<name> (engines_zimage_base, engines_qwen4bit) and
are picked with `mac-worker --engines <name>`. `loras` lists trained LoRAs the engines apply.
"""
from __future__ import annotations

import json
from pathlib import Path

NAME = "mac"

DEFAULTS = {
    "enabled": False,
    "stages": ["refs", "keyframes"],
    "batch_hours": 3,
    "setup_minutes": 5,
    "max_wait_hours": 72,
    "stale_minutes": 45,
    "heartbeat_minutes": 5,
    "poll_seconds": 60,
    "seconds_per_job": 300,
    "engines": {
        "refs": {"cmd": ["mflux-generate-z-image-turbo", "--model", "z-image-turbo", "--steps", "9"],
                 "hf_repo": "Tongyi-MAI/Z-Image-Turbo"},
        "keyframe": {"cmd": ["mflux-generate-flux2-edit", "--model", "flux2-klein-4b", "--steps", "4"],
                     "hf_repo": "black-forest-labs/FLUX.2-klein-4B", "image_flag": "--image-paths", "max_refs": 3},
        "video": {"cmd": ["{python}", "-m", "mlx_video.models.wan_2.generate", "--model-dir", "{model_dir}",
                          "--steps", "20", "--guide-scale", "5.0"],
                  "download": {"repo": "Anes1032/Wan2.2-TI2V-5B-mlx-q8",
                               "require": ["model.safetensors", "t5_encoder.safetensors", "vae.safetensors",
                                           "config.json"]},
                  "width": 1280, "height": 704, "fps": 24, "max_frames": 81, "seconds_per_clip": 1800},
    },
    "loras": [],
    "train": {"model": "z-image", "steps_total": 1500, "rank": 16, "learning_rate": 1e-4, "quantize": 8,
              "max_resolution": 1024, "low_ram": True, "gradient_checkpointing": True},
}


def settings(p, local_loras: Path | None = None) -> dict:
    """config.json's `mac` over the defaults; each engine and the train block merged one level deep.
    LoRAs trained on this Mac (`local_loras`, ~/mochizuki-mac/loras.json) are added to `loras`."""
    cfg = {**DEFAULTS, **(p.config.get("mac", {}) if p is not None else {})}
    cfg["engines"] = {**DEFAULTS["engines"], **cfg.get("engines", {})}
    cfg["train"] = {**DEFAULTS["train"], **cfg.get("train", {})}
    if local_loras is not None and local_loras.exists():
        try:
            mine = json.loads(local_loras.read_text())
        except ValueError:
            mine = []
        names = {lo.get("name") for lo in mine}
        cfg["loras"] = [lo for lo in cfg.get("loras", []) if lo.get("name") not in names] + mine
    return cfg


def with_engine_set(cfg: dict, name: str) -> dict:
    """cfg with engines_<name> in place of the default engines (missing roles keep the default)."""
    alt = cfg.get(f"engines_{name}")
    if not alt:
        raise KeyError(f"no engine set mac.engines_{name} in config.json")
    return {**cfg, "engines": {**cfg["engines"], **alt}}
