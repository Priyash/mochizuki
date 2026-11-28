"""The body of a procedural anime character, as numbers: proportions by age, joints by posture.

    skeleton(height=1.20, age=7, posture="seiza")

Pure Python, so the shot plan (pipeline/cgi/plan.py) knows exactly where a character's eyes and
hands are before Blender draws anything: the camera frames the face the builder will make, and
Blender (blender/anime.py) only puts meshes on these joints. Units are metres; the character
stands at the origin facing -Y, its left side at +X.

Proportions are anime ones, in head heights (H): a child of 7 is about 4.7 heads tall with eyes
low in a round face; an adult woman about 6.6 heads; age shortens the spine and legs a little,
rounds the back and bows the head forward.
"""
from __future__ import annotations

import math

# depth below the top of the head, in head heights, for a standing figure
CHILD = {"heads": 4.7, "eye": 0.60, "chin": 1.0, "shoulder": 1.13, "chest": 1.55, "waist": 2.05, "hip": 2.45,
         "knee": 3.5, "ankle": 4.5, "shoulder_w": 0.55, "hip_w": 0.30, "upper_arm": 0.85, "forearm": 0.75,
         "hand": 0.34, "head_w": 0.92, "head_d": 0.98}
ADULT = {"heads": 6.6, "eye": 0.56, "chin": 1.0, "shoulder": 1.32, "chest": 1.85, "waist": 2.6, "hip": 3.25,
         "knee": 4.75, "ankle": 6.3, "shoulder_w": 0.74, "hip_w": 0.40, "upper_arm": 1.3, "forearm": 1.15,
         "hand": 0.45, "head_w": 0.84, "head_d": 0.95}
OLD = {"heads": 5.7, "eye": 0.58, "chin": 1.0, "shoulder": 1.25, "chest": 1.7, "waist": 2.35, "hip": 2.85,
       "knee": 4.1, "ankle": 5.4, "shoulder_w": 0.84, "hip_w": 0.5, "upper_arm": 1.12, "forearm": 0.98,
       "hand": 0.42, "head_w": 0.9, "head_d": 0.96}
HAIR_TOP = 0.06          # hair above the skull, in head heights


def _lerp(a: dict, b: dict, t: float) -> dict:
    t = max(0.0, min(1.0, t))
    return {k: a[k] + (b[k] - a[k]) * t for k in a}


def proportions(age: float) -> dict:
    if age <= 8:
        return dict(CHILD)
    if age <= 17:
        return _lerp(CHILD, ADULT, (age - 8) / 9)
    if age <= 60:
        return dict(ADULT)
    return _lerp(ADULT, OLD, (age - 60) / 25)


def stoop_for(age: float) -> float:
    """How far the back rounds: none until 65, a clear bow at 90."""
    return max(0.0, min(1.0, (age - 65) / 25))


def _ik(shoulder, hand, upper: float, fore: float, pole) -> list[float]:
    """The elbow of a two-bone arm from shoulder to hand, bent towards `pole`."""
    d = [hand[i] - shoulder[i] for i in range(3)]
    dist = math.sqrt(sum(x * x for x in d)) or 1e-6
    reach = max(abs(upper - fore) + 1e-4, min(upper + fore - 1e-4, dist))
    u = [x / dist for x in d]
    p = [pole[i] - sum(pole[j] * u[j] for j in range(3)) * u[i] for i in range(3)]       # pole without u
    pn = math.sqrt(sum(x * x for x in p)) or 1e-6
    v = [x / pn for x in p]
    cos_a = (upper * upper + reach * reach - fore * fore) / (2 * upper * reach)
    a = math.acos(max(-1.0, min(1.0, cos_a)))
    return [shoulder[i] + upper * (math.cos(a) * u[i] + math.sin(a) * v[i]) for i in range(3)]


def skeleton(height: float, age: float, posture: str = "standing", arms: str = "", stoop: float | None = None) -> dict:
    """Joints, head size and placement for a character `height` metres tall (to the top of the hair)."""
    pr = proportions(age)
    H = height / (pr["heads"] + HAIR_TOP)
    bow = stoop_for(age) if stoop is None else stoop

    def z(depth: float) -> float:                         # standing height of a landmark
        return (pr["heads"] - depth) * H

    seated = posture in ("kneeling", "seiza")
    hip_z = z(pr["hip"])
    drop = 0.0
    if seated:                                            # sitting on the heels
        seat = (0.78 if age <= 8 else 0.86) * H
        drop = hip_z - seat
        hip_z = seat
    lift = -drop

    def up(depth: float) -> float:                        # the upper body moves down with the hips
        return z(depth) + lift

    # the spine bows forward with age: the chest a little, the neck and head more
    chest_fwd, neck_fwd, head_fwd = -0.10 * bow * H, -0.22 * bow * H, -0.30 * bow * H
    head_drop = 0.10 * bow * H
    head_c = [0.0, head_fwd, up(0.5) - head_drop]
    eye_z = up(pr["eye"]) - head_drop
    j = {
        "pelvis": [0.0, 0.0, hip_z + 0.12 * H],
        "waist": [0.0, chest_fwd * 0.3, up(pr["waist"])],
        "chest": [0.0, chest_fwd, up(pr["chest"])],
        "neck": [0.0, neck_fwd, up(pr["shoulder"]) + 0.05 * H],
        "head_base": [0.0, head_fwd + 0.06 * H, up(pr["chin"]) - head_drop + 0.12 * H],
    }
    sw = pr["shoulder_w"] * H / 2
    for side, sx in (("L", 1), ("R", -1)):
        j[f"shoulder_{side}"] = [sx * sw, chest_fwd * 0.8, up(pr["shoulder"]) - 0.08 * H]
        j[f"hip_{side}"] = [sx * pr["hip_w"] * H / 2, 0.0, hip_z]
        thigh = (pr["knee"] - pr["hip"]) * H
        shin = (pr["ankle"] - pr["knee"]) * H
        if seated:
            knee_z = 0.24 * H
            fwd = math.sqrt(max(0.0, thigh * thigh - (hip_z - knee_z) ** 2))
            j[f"knee_{side}"] = [sx * pr["hip_w"] * H * 0.6, -fwd, knee_z]
            j[f"ankle_{side}"] = [sx * pr["hip_w"] * H * 0.45, -fwd + shin * 0.97, 0.20 * H]
            j[f"toe_{side}"] = [sx * pr["hip_w"] * H * 0.3, -fwd + shin * 0.97 + 0.3 * H, 0.12 * H]
        else:
            j[f"knee_{side}"] = [sx * pr["hip_w"] * H * 0.52, -0.02 * H, z(pr["knee"])]
            j[f"ankle_{side}"] = [sx * pr["hip_w"] * H * 0.5, 0.0, z(pr["ankle"])]
            j[f"toe_{side}"] = [sx * pr["hip_w"] * H * 0.55, -0.36 * H, 0.06 * H]
    pose = arms or ("lap" if seated else "down")
    upper, fore = pr["upper_arm"] * H, pr["forearm"] * H
    for side, sx in (("L", 1), ("R", -1)):
        s = j[f"shoulder_{side}"]
        if pose == "lap":                                 # hands resting on the thighs
            k = j[f"knee_{side}"]
            hand = [k[0] * 0.75, k[1] * 0.55, hip_z + 0.12 * H]
            pole = [sx * 0.6, 0.3, -1.0]
        elif pose == "front":                             # hands together in front, as if holding something
            hand = [sx * 0.10 * H, -0.62 * H + chest_fwd, j["pelvis"][2] + 0.45 * H]
            pole = [sx * 0.8, 0.4, -1.0]
        else:                                             # arms down, a little forward
            hand = [sx * (sw + 0.06 * H), -0.12 * H, s[2] - (upper + fore) * 0.94]
            pole = [sx * 0.3, 0.6, -0.2]
        j[f"elbow_{side}"] = _ik(s, hand, upper, fore, pole)
        j[f"wrist_{side}"] = hand
    w = (pr["head_w"] * H, pr["head_d"] * H, H)
    return {"H": H, "age": age, "posture": posture, "arms": pose, "stoop": bow, "heads": pr["heads"],
            "head": {"center": head_c, "size": list(w)}, "eyes": [0.0, head_c[1] - w[1] * 0.46, eye_z],
            "hands": [(j["wrist_L"][i] + j["wrist_R"][i]) / 2 for i in range(3)], "joints": j}


def to_world(point: list[float], at: list[float], yaw_deg: float) -> list[float]:
    """A point of the figure (character space) in the set, for a character on `at` turned by `yaw_deg`."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    x, y, zz = point
    return [at[0] + c * x - s * y, at[1] + s * x + c * y, at[2] + zz]
