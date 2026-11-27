"""Anime timing: what turns smooth motion into motion that reads like anime.

Captured or simulated motion is smooth on every frame; anime is not. It is pose to pose:
clear key poses, eased in and out, animated on twos (or threes), with holds on the poses
that matter and a stop on every hit. These functions do that to any curves (a face
channel, a prop's transform), as plain lists of frames, so they run anywhere.

    key_poses(values, tol)        the frames that carry the motion (Ramer-Douglas-Peucker)
    pose_to_pose(values, keys)    the motion rebuilt from those poses, eased in and out
    step(values, on)              on twos / threes: each pose held for `on` frames
    mixed(values)                 fast moves on ones, the rest on twos
    hit_stop(values, frame, n)    the pose held for n frames where a hit lands
    emotion_curves(...)           the acting sheet's beats as 0..1 curves per emotion
    blinks(...)                   blinks that land on changes of thought, not at random only
"""
from __future__ import annotations

import math
import random

Frames = list  # list of frames; each frame is a list of floats


def _as_frames(values) -> Frames:
    return [list(v) if isinstance(v, (list, tuple)) else [float(v)] for v in values]


def _ranges(values: Frames) -> list[float]:
    dims = len(values[0]) if values else 0
    out = []
    for d in range(dims):
        col = [v[d] for v in values]
        out.append(max(col) - min(col) or 1.0)
    return out


def key_poses(values, tol: float = 0.03) -> list[int]:
    """Indices of the frames needed to redraw `values` within `tol` (a fraction of each channel's range).
    The first and last frames are always kept."""
    vals = _as_frames(values)
    n = len(vals)
    if n <= 2:
        return list(range(n))
    rng = _ranges(vals)
    norm = [[x / rng[d] for d, x in enumerate(v)] for v in vals]
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        worst, at = -1.0, -1
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            err = max(abs(norm[i][d] - (norm[a][d] + (norm[b][d] - norm[a][d]) * t)) for d in range(len(norm[i])))
            if err > worst:
                worst, at = err, i
        if worst > tol:
            keep.add(at)
            stack += [(a, at), (at, b)]
    return sorted(keep)


def ease(t: float, favour: float = 0.6) -> float:
    """Slow in, slow out. favour=0 is linear; 1 is a full smoothstep that lingers on the poses."""
    t = min(1.0, max(0.0, t))
    return (1 - favour) * t + favour * t * t * (3 - 2 * t)


def pose_to_pose(values, keys: list[int], favour: float = 0.6) -> Frames:
    vals = _as_frames(values)
    out = [list(v) for v in vals]
    for a, b in zip(keys, keys[1:]):
        for i in range(a, b + 1):
            t = ease((i - a) / (b - a) if b > a else 0.0, favour)
            out[i] = [vals[a][d] + (vals[b][d] - vals[a][d]) * t for d in range(len(vals[a]))]
    return out


def step(values, on: int = 2) -> Frames:
    """Hold each sampled pose for `on` frames (on twos: frames 0,0,2,2,4,4...). The last frame is kept."""
    vals = _as_frames(values)
    if on <= 1:
        return [list(v) for v in vals]
    out = [list(vals[(i // on) * on]) for i in range(len(vals))]
    if vals:
        out[-1] = list(vals[-1])
    return out


def speeds(values) -> list[float]:
    vals = _as_frames(values)
    rng = _ranges(vals) if vals else []
    sp = [0.0]
    for i in range(1, len(vals)):
        sp.append(max(abs(vals[i][d] - vals[i - 1][d]) / rng[d] for d in range(len(vals[i]))))
    return sp


def mixed(values, fast: float = 0.08) -> Frames:
    """Fast moves stay on ones (a swing, a fall); everything else goes on twos."""
    vals = _as_frames(values)
    sp = speeds(vals)
    twos = step(vals, 2)
    return [list(vals[i]) if sp[i] > fast or (i + 1 < len(sp) and sp[i + 1] > fast) else twos[i]
            for i in range(len(vals))]


def hit_stop(values, frame: int, n: int = 3) -> Frames:
    """Hold the pose at `frame` for n extra frames; the rest of the motion comes n frames later.
    The length does not change: the tail is trimmed."""
    vals = _as_frames(values)
    if not 0 <= frame < len(vals) or n <= 0:
        return [list(v) for v in vals]
    out = vals[:frame + 1] + [list(vals[frame]) for _ in range(n)] + vals[frame + 1:]
    return [list(v) for v in out[:len(vals)]]


def timed(values, mode: str = "twos", tol: float = 0.03, impacts: list[int] | None = None) -> Frames:
    """The whole treatment: key poses, eased, then stepped by `mode`, then hit stops on impacts."""
    vals = _as_frames(values)
    if not vals:
        return []
    out = pose_to_pose(vals, key_poses(vals, tol))
    out = {"ones": lambda v: v, "twos": lambda v: step(v, 2), "threes": lambda v: step(v, 3),
           "mixed": mixed}.get(mode, lambda v: step(v, 2))(out)
    for f in sorted(impacts or [], reverse=True):
        out = hit_stop(out, f, 3)
    return out


# ---------------------------------------------------------------- acting sheet -> curves
def attack_seconds(intensity: float) -> float:
    """Strong feelings arrive faster than gentle ones."""
    return max(0.12, 0.55 - 0.35 * intensity)


def emotion_curves(beats: list, frames: int, fps: int, emotions: tuple, who: str = "") -> dict:
    """Per-frame 0..1 curves for each emotion, plus 'restraint', for one character.
    Each beat moves the face from where it is to its emotion over its attack time and keeps it
    there until the next beat; a bittersweet beat is joy and sorrow at once."""
    mine = sorted((b for b in beats if not b.who or not who or b.who == who), key=lambda b: b.t)
    curves = {e: [0.0] * frames for e in emotions}
    curves["restraint"] = [0.0] * frames
    state = {e: 0.0 for e in curves}
    if not mine:
        return curves
    for i, b in enumerate(mine):
        target = {e: 0.0 for e in curves}
        if b.emotion == "bittersweet":
            target["joy"], target["sorrow"] = b.intensity * 0.6, b.intensity * 0.8
        elif b.emotion in target:
            target[b.emotion] = b.intensity
        target["restraint"] = b.restraint
        start = int(round(b.t * fps))
        end = frames if i + 1 == len(mine) else int(round(mine[i + 1].t * fps))
        att = max(1, int(round(attack_seconds(b.intensity) * fps)))
        before = dict(state)
        for f in range(max(0, start), min(frames, end)):
            k = ease(min(1.0, (f - start) / att), 0.8)
            for e in curves:
                curves[e][f] = before[e] + (target[e] - before[e]) * k
        state = {e: curves[e][min(frames, end) - 1] if min(frames, end) > max(0, start) else before[e]
                 for e in curves}
    return curves


BLINK = [0.55, 1.0, 1.0, 0.7, 0.35, 0.1]                     # fast close, hold, open: 6 frames at 24 fps
SLOW_BLINK = [0.25, 0.55, 0.85, 1.0, 1.0, 1.0, 0.85, 0.6, 0.35, 0.15]   # tender or awed


def blinks(frames: int, fps: int, seed: int, per_minute: float = 10.0, thoughts: list[float] | None = None,
           curves: dict | None = None) -> list[float]:
    """Eyelid closure per frame (0 open, 1 shut). A blink lands on each change of thought (a beat)
    unless one has just happened; more blinks when afraid or holding feelings in, fewer when awed,
    and slow ones when tender or in wonder."""
    rng = random.Random(seed)
    out = [0.0] * frames
    curves = curves or {}

    def at(name, f):
        c = curves.get(name)
        return c[min(f, len(c) - 1)] if c else 0.0

    starts: list[int] = []
    for t in thoughts or []:
        f = int(round(t * fps)) + 2                      # a beat's blink comes just after the thought
        if 0 <= f < frames and all(abs(f - s) > fps * 0.6 for s in starts):
            starts.append(f)
    f = int(rng.uniform(0.8, 2.5) * fps)
    while f < frames:
        if all(abs(f - s) > fps * 0.8 for s in starts):
            starts.append(f)
        rate = per_minute * (1 + 0.8 * at("restraint", f) + 0.6 * at("fear", f) - 0.5 * at("wonder", f))
        gap = 60.0 / max(2.0, rate)
        f += int(max(1.4, rng.expovariate(1 / gap)) * fps)
    for s in sorted(starts):
        slow = at("tenderness", s) + at("wonder", s) > 0.5
        shape = SLOW_BLINK if slow else BLINK
        scale = fps / 24.0
        for k in range(int(len(shape) * scale)):
            if s + k < frames:
                out[s + k] = max(out[s + k], shape[min(len(shape) - 1, int(k / scale))])
    return out


def seconds_to_frame(t: float, fps: int) -> int:
    return int(round(t * fps))


def ease_curve(frames: int, keys: list[tuple[int, float]], favour: float = 0.7) -> list[float]:
    """A single channel through (frame, value) keys, eased between them, held after the last."""
    if not keys:
        return [0.0] * frames
    keys = sorted(keys)
    out = []
    for f in range(frames):
        if f <= keys[0][0]:
            out.append(keys[0][1])
            continue
        for (fa, va), (fb, vb) in zip(keys, keys[1:]):
            if fa <= f <= fb:
                out.append(va + (vb - va) * ease((f - fa) / max(1, fb - fa), favour))
                break
        else:
            out.append(keys[-1][1])
    return out


def wobble(frames: int, fps: int, seed: int, amount: float, rate: float = 0.35) -> list[float]:
    """A slow, smooth wander (breathing, a lantern on a string): a sum of sines, never noise."""
    rng = random.Random(seed)
    ph = [rng.uniform(0, math.tau) for _ in range(3)]
    return [amount * (math.sin(math.tau * rate * f / fps + ph[0]) * 0.6
                      + math.sin(math.tau * rate * 1.9 * f / fps + ph[1]) * 0.3
                      + math.sin(math.tau * rate * 3.3 * f / fps + ph[2]) * 0.1) for f in range(frames)]
