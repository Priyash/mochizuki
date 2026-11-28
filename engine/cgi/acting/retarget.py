"""From feelings and performances to the character's face shapes.

Two sources meet here, frame by frame:

  * the acting sheet's emotion curves (timing.emotion_curves), turned into the face's shape
    keys by a table per emotion, with restraint applied the way anime plays held feelings:
    the mouth and brows move less, the eyes do more (wet, bigger highlights, more blinks);
  * a captured performance (capture.Track of ARKit shapes), mapped onto the same shape keys
    and exaggerated a little, because a real face under-reads on an anime face.

Shape key names default to the ones VRoid Studio exports (Fcl_EYE_Close, Fcl_MTH_A, ...).
A character with other names maps them in config.json -> cgi.characters.<name>.face_names.
`fit_mapping` learns a capture-to-character table from a few example poses (ridge regression),
for faces whose shapes do not line up one to one with ARKit's.
"""
from __future__ import annotations

from .capture import Track

# ARKit shape -> VRoid shape key, as weighted sums
CAPTURE_MAP: dict[str, list[tuple[str, float]]] = {
    "Fcl_EYE_Close_L": [("eyeBlinkLeft", 1.0)],
    "Fcl_EYE_Close_R": [("eyeBlinkRight", 1.0)],
    "Fcl_EYE_Joy_L": [("cheekSquintLeft", 0.8), ("eyeSquintLeft", 0.3)],
    "Fcl_EYE_Joy_R": [("cheekSquintRight", 0.8), ("eyeSquintRight", 0.3)],
    "Fcl_EYE_Surprised": [("eyeWideLeft", 0.5), ("eyeWideRight", 0.5)],
    "Fcl_BRW_Angry": [("browDownLeft", 0.5), ("browDownRight", 0.5)],
    "Fcl_BRW_Sorrow": [("browInnerUp", 1.0)],
    "Fcl_BRW_Surprised": [("browOuterUpLeft", 0.5), ("browOuterUpRight", 0.5)],
    "Fcl_MTH_A": [("jawOpen", 1.0)],
    "Fcl_MTH_O": [("mouthFunnel", 1.0)],
    "Fcl_MTH_U": [("mouthPucker", 1.0)],
    "Fcl_MTH_E": [("mouthStretchLeft", 0.4), ("mouthStretchRight", 0.4)],
    "Fcl_MTH_I": [("mouthSmileLeft", 0.15), ("mouthSmileRight", 0.15)],
    "Fcl_MTH_Joy": [("mouthSmileLeft", 0.5), ("mouthSmileRight", 0.5)],
    "Fcl_MTH_Sorrow": [("mouthFrownLeft", 0.5), ("mouthFrownRight", 0.5)],
    "Fcl_MTH_Angry": [("mouthPressLeft", 0.3), ("mouthPressRight", 0.3), ("noseSneerLeft", 0.2),
                      ("noseSneerRight", 0.2)],
    "head.yaw": [("headYaw", 1.0)],
    "head.pitch": [("headPitch", 1.0)],
    "head.roll": [("headRoll", 1.0)],
    "eyes.yaw": [("eyeLookOutLeft", 0.25), ("eyeLookInLeft", -0.25), ("eyeLookInRight", 0.25),
                 ("eyeLookOutRight", -0.25)],
    "eyes.pitch": [("eyeLookUpLeft", 0.2), ("eyeLookUpRight", 0.2), ("eyeLookDownLeft", -0.2),
                   ("eyeLookDownRight", -0.2)],
}
SIGNED = {"head.yaw", "head.pitch", "head.roll", "eyes.yaw", "eyes.pitch"}

# emotion -> shape keys at full intensity
EMOTION_FACE: dict[str, dict[str, float]] = {
    "joy": {"Fcl_BRW_Joy": 0.6, "Fcl_EYE_Joy": 0.5, "Fcl_MTH_Joy": 1.0},
    "sorrow": {"Fcl_BRW_Sorrow": 1.0, "Fcl_EYE_Sorrow": 0.7, "Fcl_MTH_Sorrow": 0.8},
    "tenderness": {"Fcl_BRW_Sorrow": 0.25, "Fcl_EYE_Joy": 0.3, "Fcl_MTH_Joy": 0.4},
    "wonder": {"Fcl_EYE_Surprised": 0.5, "Fcl_BRW_Surprised": 0.5, "Fcl_MTH_O": 0.15},
    "fear": {"Fcl_BRW_Sorrow": 0.6, "Fcl_BRW_Surprised": 0.4, "Fcl_EYE_Surprised": 0.8, "Fcl_MTH_Sorrow": 0.4,
             "Fcl_MTH_E": 0.3},
    "resolve": {"Fcl_BRW_Angry": 0.3, "Fcl_EYE_Angry": 0.15},
    "curiosity": {"Fcl_BRW_Surprised": 0.4, "Fcl_EYE_Spread": 0.3, "Fcl_MTH_U": 0.1},
    "surprise": {"Fcl_BRW_Surprised": 1.0, "Fcl_EYE_Surprised": 1.0, "Fcl_MTH_Surprised": 0.8},
    "anger": {"Fcl_BRW_Angry": 1.0, "Fcl_EYE_Angry": 0.8, "Fcl_MTH_Angry": 0.8},
}
LAUGH = {"Fcl_EYE_Joy": 0.9, "Fcl_MTH_A": 0.45}             # joy past 0.75 opens into a laugh
OUTWARD = ("Fcl_MTH_", "Fcl_BRW_")                          # what restraint holds back
ANIME = ("anime.tear", "anime.highlight", "anime.blush")


def _clamp(x: float, signed: bool = False) -> float:
    return max(-1.0, min(1.0, x)) if signed else max(0.0, min(1.0, x))


def map_capture(tr: Track, table: dict | None = None, amplify: float = 1.15) -> Track:
    """ARKit curves -> shape-key curves. Rotations pass through (radians); shapes are exaggerated
    by `amplify` and clamped to 0..1."""
    table = table or CAPTURE_MAP
    idx = {n: i for i, n in enumerate(tr.names)}
    names = list(table)
    vals = []
    for row in tr.values:
        out = []
        for target, parts in table.items():
            v = sum(row[idx[s]] * w for s, w in parts if s in idx)
            out.append(_clamp(v, True) if target in SIGNED else _clamp(v * amplify))
        vals.append(out)
    return Track(tr.fps, names, vals, tr.source, {**tr.meta, "mapped": "arkit->vroid"})


def emotion_face(curves: dict, blink: list[float] | None = None) -> Track:
    """The acting sheet's curves (emotion -> per-frame 0..1, plus 'restraint') as shape-key curves,
    with restraint and the anime layer. `blink` adds eyelid closure."""
    frames = len(next(iter(curves.values()))) if curves else 0
    names: list[str] = []
    for table in list(EMOTION_FACE.values()) + [LAUGH]:
        names += [k for k in table if k not in names]
    names += [n for n in ("Fcl_EYE_Close",) + ANIME if n not in names]
    vals = []
    for f in range(frames):
        r = curves.get("restraint", [0.0] * frames)[f]
        v = dict.fromkeys(names, 0.0)
        for emo, table in EMOTION_FACE.items():
            k = curves.get(emo, [0.0] * frames)[f]
            for shape, w in table.items():
                v[shape] += w * k
        joy = curves.get("joy", [0.0] * frames)[f]
        if joy > 0.75:
            for shape, w in LAUGH.items():
                v[shape] = max(v[shape], w * (joy - 0.75) / 0.25)
        for shape in v:
            if shape.startswith(OUTWARD):
                v[shape] *= 1 - 0.65 * r
        sorrow = curves.get("sorrow", [0.0] * frames)[f]
        wonder = curves.get("wonder", [0.0] * frames)[f]
        tender = curves.get("tenderness", [0.0] * frames)[f]
        v["Fcl_EYE_Sorrow"] += 0.3 * r * sorrow                # the eyes carry what the face holds back
        v["anime.tear"] = 0.9 * sorrow * max(r, 0.3)            # tears welling, not falling
        v["anime.highlight"] = 0.5 * wonder + 0.5 * r * sorrow + 0.2 * tender
        v["anime.blush"] = 0.3 * joy + 0.4 * tender
        v["Fcl_EYE_Close"] = blink[f] if blink and f < len(blink) else 0.0
        vals.append([_clamp(v[n]) for n in names])
    return Track(24.0, names, vals, "acting-sheet", {"kind": "face", "from": "beats"})


def combine(base: Track, extra: Track, weight: float = 1.0) -> Track:
    """Channel-wise sum of two face tracks of the same length (extra x weight), clamped.
    Used to lay the anime layer (and a lighter emotion layer) over a captured performance."""
    names = list(base.names) + [n for n in extra.names if n not in base.names]
    bi = {n: i for i, n in enumerate(base.names)}
    ei = {n: i for i, n in enumerate(extra.names)}
    vals = []
    for f in range(max(len(base), len(extra))):
        b = base.values[min(f, len(base) - 1)] if len(base) else []
        e = extra.values[min(f, len(extra) - 1)] if len(extra) else []
        row = []
        for n in names:
            x = (b[bi[n]] if n in bi and b else 0.0) + (e[ei[n]] * weight if n in ei and e else 0.0)
            row.append(_clamp(x, n in SIGNED))
        vals.append(row)
    return Track(base.fps or extra.fps, names, vals, base.source, {**base.meta, "layered": extra.source})


def rename(tr: Track, names: dict[str, str]) -> Track:
    """Use a character's own shape key names (config cgi.characters.<name>.face_names)."""
    return Track(tr.fps, [names.get(n, n) for n in tr.names], [list(r) for r in tr.values], tr.source, dict(tr.meta))


def lantern_channels(curves: dict, frames: int, wobble: list[float]) -> Track:
    """Akari, a paper lantern, acts with light and lean: brighter with joy, dimmer with sorrow,
    a flicker of fear, a tilt of curiosity, a bob that quickens when happy."""
    names = ["akari.glow", "akari.tilt", "akari.bob"]
    vals = []
    for f in range(frames):
        g = lambda e, f=f: curves.get(e, [0.0] * frames)[f]  # noqa: E731 - this frame's value of an emotion
        glow = 0.75 + 0.5 * g("joy") + 0.3 * g("wonder") + 0.2 * g("tenderness") - 0.4 * g("sorrow")
        glow += 0.12 * g("fear") * (1 if (f // 2) % 2 else -1)
        tilt = 14.0 * g("curiosity") + 6.0 * g("tenderness") - 8.0 * g("sorrow")
        bob = wobble[f] * (1 + g("joy"))
        vals.append([max(0.05, glow), tilt, bob])
    return Track(24.0, names, vals, "acting-sheet", {"kind": "lantern"})


# ---------------------------------------------------------------- learning a mapping
def _solve(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Solve a x = b for x (Gauss-Jordan with partial pivoting); a is n x n, b is n x k."""
    n, k = len(a), len(b[0])
    m = [list(a[i]) + list(b[i]) for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        if abs(m[p][c]) < 1e-12:
            raise ValueError("the examples do not pin the mapping down: add more varied poses")
        m[c], m[p] = m[p], m[c]
        piv = m[c][c]
        m[c] = [x / piv for x in m[c]]
        for r in range(n):
            if r != c and m[r][c]:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [row[n:n + k] for row in m]


def fit_mapping(src_names: list[str], dst_names: list[str], examples: list[tuple[dict, dict]],
                ridge: float = 0.05) -> dict[str, list[tuple[str, float]]]:
    """Learn target = sum(w * source) + bias from example poses: [(arkit values, character values)].
    Six to ten poses (neutral, smile, laugh, sad, surprised, angry, each viseme) are enough; ridge keeps
    the weights small where the examples say little. Returns a table for map_capture (bias as '_bias')."""
    d = len(src_names)
    xs = [[ex[0].get(n, 0.0) for n in src_names] + [1.0] for ex in examples]
    ys = [[ex[1].get(n, 0.0) for n in dst_names] for ex in examples]
    xtx = [[sum(x[i] * x[j] for x in xs) + (ridge if i == j and i < d else 0.0) for j in range(d + 1)]
           for i in range(d + 1)]
    xty = [[sum(x[i] * y[k] for x, y in zip(xs, ys)) for k in range(len(dst_names))] for i in range(d + 1)]
    w = _solve(xtx, xty)
    table = {}
    for k, dst in enumerate(dst_names):
        parts = [(src_names[i], round(w[i][k], 5)) for i in range(d) if abs(w[i][k]) > 1e-4]
        if abs(w[d][k]) > 1e-4:
            parts.append(("_bias", round(w[d][k], 5)))
        table[dst] = parts
    return table


def with_bias(tr: Track) -> Track:
    """A copy of tr with a constant '_bias' channel of 1, so learned tables can carry their bias."""
    return Track(tr.fps, list(tr.names) + ["_bias"], [list(r) + [1.0] for r in tr.values], tr.source, dict(tr.meta))
