"""The light of a shot: style_bible/lights.json, by scene, else by chapter.

A rig is a few lights with colours and directions, a world colour, and the toon settings
that make the light read as cel shading: how dark the shadow band is and what colour it
leans to, and the colour of the ink line. Keeping this as data is what keeps a scene's
light the same from its first shot to its last.
"""
from __future__ import annotations

import copy
import json

from ..project import LIGHTS_FILE

DEFAULT_RIG = {
    "key": {"type": "SUN", "color": [1.0, 0.96, 0.9], "energy": 3.0, "azimuth": -35, "elevation": 40},
    "fill": {"type": "SUN", "color": [0.6, 0.7, 1.0], "energy": 0.6, "azimuth": 140, "elevation": 20},
    "world": [0.18, 0.2, 0.26],
    "toon": {"bands": [0.42, 0.78], "shadow_tint": [0.55, 0.55, 0.75], "mid": 0.86, "rim": [1.0, 1.0, 1.0],
             "rim_strength": 0.25, "outline": [0.1, 0.08, 0.12]},
}


def load(p) -> dict:
    f = p.style_bible / LIGHTS_FILE
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k.startswith("_"):
            continue
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def rig_for(p, chapter: int, scene: str = "", shot_id: str = "") -> dict:
    """Default rig <- chapter <- scene <- shot, each layer changing only what it names."""
    data = load(p)
    rig = copy.deepcopy(DEFAULT_RIG)
    for key in (str(chapter), scene, shot_id):
        if key and key in data.get("rigs", {}):
            rig = _merge(rig, data["rigs"][key])
    return rig
