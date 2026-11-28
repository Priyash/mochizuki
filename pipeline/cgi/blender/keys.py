"""Keyframing helpers shared by the character builders. Runs inside Blender."""
from __future__ import annotations


def key_curve(datablock, path: str, values: list, index: int = -1, frame0: int = 1, step_tol: float = 1e-4):
    """Keyframe `values` on a property, one key per change (constant interpolation is set by the caller)."""
    last = None
    for f, v in enumerate(values):
        if last is not None and abs(v - last) < step_tol and f + 1 < len(values):
            continue
        if index >= 0:
            getattr(datablock, path)[index] = v
            datablock.keyframe_insert(path, index=index, frame=frame0 + f)
        else:
            if path.startswith('["'):
                datablock[path[2:-2]] = v
            else:
                setattr(datablock, path, v)
            datablock.keyframe_insert(path, frame=frame0 + f)
        last = v


def ch(face: dict, name: str, frames: int) -> list:
    """A face channel as a list of `frames` values (zeros if the plan has none)."""
    v = face.get(name)
    return list(v) if isinstance(v, list) else [0.0] * frames


def mix(*chans: list) -> list:
    """The per-frame maximum of several channels."""
    return [max(vals) for vals in zip(*chans)]
