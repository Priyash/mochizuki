"""Where the camera goes. Plain geometry in metres, Blender's axes (Z up).

A character's facing is a yaw angle: 0 faces -Y, 90 faces +X. The camera is placed around
its subject by framing (how much of the subject fills the frame height), angle (azimuth and
elevation around the subject's facing) and lens; the distance follows from the lens's field
of view, so a CU is always a CU whatever the lens. Moves are baked per frame, on ones.
"""
from __future__ import annotations

import math
import random

# how many metres of the subject the frame height holds
FRAME_HEIGHT = {"ECU": 0.14, "CU": 0.32, "MCU": 0.55, "MS": 0.95, "MLS": 1.7, "WIDE": 7.0, "INSERT": 0.24}
# (azimuth from the subject's facing, elevation), degrees
ANGLE = {"eye": (0.0, 2.0), "three-quarter": (35.0, 4.0), "profile": (90.0, 2.0), "behind": (180.0, 8.0),
         "low": (15.0, -14.0), "high": (10.0, 28.0), "overhead": (0.0, 75.0)}
FLOOR_CLEARANCE = 0.06


def vfov(lens_mm: float, sensor_mm: float = 36.0, aspect: float = 16 / 9) -> float:
    """Vertical field of view in radians; Blender fits the sensor width to a landscape frame."""
    return 2 * math.atan((sensor_mm / aspect) / 2 / lens_mm)


def distance_for(frame_h: float, lens_mm: float, sensor_mm: float = 36.0, aspect: float = 16 / 9) -> float:
    """Camera distance at which `frame_h` metres fill the frame height."""
    return (frame_h / 2) / math.tan(vfov(lens_mm, sensor_mm, aspect) / 2)


def facing_vec(yaw_deg: float) -> tuple[float, float, float]:
    y = math.radians(yaw_deg)
    return (math.sin(y), -math.cos(y), 0.0)


def orbit_vec(yaw_deg: float, azimuth_deg: float, elevation_deg: float) -> tuple[float, float, float]:
    """Unit vector from the subject towards the camera."""
    y = math.radians(yaw_deg + azimuth_deg)
    e = math.radians(elevation_deg)
    return (math.sin(y) * math.cos(e), -math.cos(y) * math.cos(e), math.sin(e))


def _add(a, b, k=1.0):
    return tuple(a[i] + k * b[i] for i in range(3))


def _sub(a, b):
    return tuple(a[i] - b[i] for i in range(3))


def _len(a):
    return math.sqrt(sum(x * x for x in a))


def _lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def smooth(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


def place(target, yaw_deg: float, framing: str, angle: str, lens_mm: float, sensor_mm: float = 36.0,
          aspect: float = 16 / 9, frame_h: float | None = None):
    """Camera location for a subject at `target` that faces `yaw_deg`. Returns (location, target, frame_h)."""
    fh = frame_h or FRAME_HEIGHT.get(framing, 0.95)
    d = distance_for(fh, lens_mm, sensor_mm, aspect)
    az, el = ANGLE.get(angle, ANGLE["eye"])
    loc = _add(target, orbit_vec(yaw_deg, az, el), d)
    if loc[2] < FLOOR_CLEARANCE:                      # a low angle never goes through the floor
        loc = (loc[0], loc[1], FLOOR_CLEARANCE)
    return loc, tuple(target), fh


def _rotate_about(point, centre, deg_yaw=0.0, deg_pitch=0.0):
    """Turn `point` around `centre`: yaw about Z, then pitch about the horizontal axis."""
    v = _sub(point, centre)
    y = math.radians(deg_yaw)
    v = (v[0] * math.cos(y) - v[1] * math.sin(y), v[0] * math.sin(y) + v[1] * math.cos(y), v[2])
    if deg_pitch:
        h = math.hypot(v[0], v[1]) or 1e-9
        p0 = math.atan2(v[2], h)
        r = _len(v)
        p = p0 + math.radians(deg_pitch)
        k = math.cos(p) * r / h
        v = (v[0] * k, v[1] * k, math.sin(p) * r)
    return _add(centre, v)


def keys(loc, target, move: str, amount: float, frames: int, frame_h: float, seed: int = 0) -> list[dict]:
    """The camera on every frame (1-based): {"frame", "loc", "target"}."""
    rng = random.Random(seed)
    phases = [rng.uniform(0, math.tau) for _ in range(12)]
    out = []
    for f in range(1, frames + 1):
        t = 0.0 if frames <= 1 else (f - 1) / (frames - 1)
        e = smooth(t)
        cl, ct = tuple(loc), tuple(target)
        if move == "push":
            cl = _lerp(loc, target, amount * e)
        elif move == "pull":
            cl = _add(target, _sub(loc, target), 1 + amount * e)
        elif move == "pan":
            ct = _rotate_about(target, loc, deg_yaw=amount * (e - 0.5))
        elif move == "tilt":
            ct = _rotate_about(target, loc, deg_pitch=amount * e)
        elif move == "track":
            side = _sub(target, loc)
            side = (-side[1], side[0], 0.0)
            n = _len(side) or 1.0
            cl = _add(loc, side, amount * frame_h * (e - 0.5) / n)
            ct = _add(target, side, amount * frame_h * (e - 0.5) / n)
        elif move == "drift":                         # a held camera: slow, small, never a shake
            s = f / 24.0
            a = amount * frame_h * 0.012
            n1 = sum(math.sin(s * w + phases[i]) for i, w in enumerate((0.7, 1.3, 2.1))) / 3
            n2 = sum(math.sin(s * w + phases[3 + i]) for i, w in enumerate((0.5, 1.1, 1.9))) / 3
            n3 = sum(math.sin(s * w + phases[6 + i]) for i, w in enumerate((0.6, 1.7, 2.3))) / 3
            cl = _add(loc, (n1 * a, n2 * a, n3 * a * 0.6))
        out.append({"frame": f, "loc": [round(x, 5) for x in cl], "target": [round(x, 5) for x in ct]})
    return out


def outline_m(frame_h: float, res_y: int, px: float) -> float:
    """Ink line width in metres at the subject, so lines are the same thickness on screen in every shot."""
    return frame_h * px / max(1, res_y)
