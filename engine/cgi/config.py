"""Settings of the 3D pipeline: the engine's defaults below, overridden by the film's config.json -> "cgi"."""
from __future__ import annotations

import copy
from pathlib import Path

from ..project import MEDIA_DIR

DEFAULTS: dict = {
    # Blender 5.x: its binary, or a Python that has the bpy module. Empty: look for it
    # (/Applications/Blender.app, then `blender` on PATH, then the BLENDER environment variable).
    "blender": "",
    "fps": 24,
    "resolution": [3840, 2160],          # final frames are rendered natively in 4K
    "preview_resolution": [960, 540],    # the animatic and quick looks
    "engine": "BLENDER_EEVEE",           # real-time renderer: seconds a frame, no noise, no flicker
    "samples": 16,
    "preview_samples": 4,
    "character_step": 2,                 # characters animate on twos, like hand-drawn anime
    "camera_step": 1,                    # the camera always moves on ones
    "outline_px": 2.2,                   # ink line width, in pixels of the frame height
    "sensor_mm": 36.0,
    "blinks_per_minute": 10,             # anime blinks less than people do, and on purpose
    "encode": {"crf": 12, "preview_crf": 20, "codec": "libx264"},
    "work": f"{MEDIA_DIR}/cgi",          # plans, .blend files, frames and clips per shot
    "use_in_cut": False,                 # assemble: prefer a shot's CG render over takes and keyframes
    "stand_in": "anime",                 # without a model file: "anime" (made from code) or "maquette"
    "dubs": "all",                       # "all": each script language its own lip-synced render of every shot
                                         # where someone speaks on screen (1-11.ja.mp4); "none": one render
    # Who is who, and continuity that is data (a prop's state by chapter): each film's own, in its
    # config.json -> cgi.characters / cgi.props_state. A character entry looks like
    #   "name": {"model": {"<chapter>": "assets/characters/name.vrm"}, "height_m": {"<chapter>": 1.5},
    #            "colour": [r, g, b], "hair": [r, g, b], "look": {"<chapter>": {"age": 30, "hair_style": "bun"}}}
    # or {"procedural": "lantern", ...} for a character built from code. Tables keyed by chapter
    # fall back to the nearest earlier chapter (by_chapter).
    "characters": {},
    "props_state": {},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k.startswith("_"):
            continue
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def settings(p) -> dict:
    """DEFAULTS with config.json's "cgi" section on top. Notes (keys starting with _) are dropped."""
    return _merge(DEFAULTS, p.config.get("cgi", {}))


def work_dir(p, cfg: dict, shot_id: str) -> Path:
    d = Path(cfg["work"])
    d = d if d.is_absolute() else p.root / d
    return d / shot_id


def by_chapter(table: dict, chapter: int, default=None):
    """table is keyed by chapter number as a string; the nearest earlier chapter wins."""
    for c in range(chapter, 0, -1):
        if str(c) in table:
            return table[str(c)]
    return default
