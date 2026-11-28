"""One shot, fully decided: the JSON file Blender builds and renders, and nothing else.

    build(p, shot_id, preview=False) -> dict

folds together the acting sheet (direction), the scene's set (style_bible/sets.json), the
light (palette), the camera (layout), the props' motion and the faces (timing, capture,
retarget). Every number Blender needs is in the plan, so a shot renders the same every time
and the whole decision can be read, diffed and tested without Blender.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..audio import dialogue
from ..generators import GUIDES, seed_for
from ..project import SETS_FILE
from .acting import direction as dirn
from .acting import retarget, timing
from .acting.capture import Track, resample, smooth
from .config import by_chapter, settings, work_dir
from .staging import figure, layout, palette

POSTURE_HEAD = {"standing": 0.9, "walking": 0.9, "kneeling": 0.54, "seiza": 0.52}   # eye height / body height
POSTURE_HANDS = {"standing": 0.55, "walking": 0.55, "kneeling": 0.36, "seiza": 0.34}

PROP_CENTRE = {"mortar": 0.4, "mallet": 0.0, "bowl": 0.03, "lantern": 0.0, "rice": 0.04, "cushion": 0.04,
               "tray": 0.03, "dumplings": 0.12, "pampas": 0.5}
PROP_FRAME = {"mortar": 1.1, "bowl": 0.2, "lantern": 0.5, "rice": 0.45, "cushion": 0.7, "tray": 0.45,
              "dumplings": 0.38, "pampas": 1.2, "mallet": 0.9, "hearth": 1.2}   # an insert shows the whole thing
CAMERA_HOME = (0.0, -3.2, 1.0)          # where a camera stands to look at something far away (the moon)
CROWD_SPACING = 0.9


def load_sets(p) -> dict:
    f = p.style_bible / SETS_FILE
    return json.loads(f.read_text(encoding="utf-8")).get("sets", {}) if f.exists() else {}


def set_for(p, scene: str) -> dict:
    sets = load_sets(p)
    return sets.get(scene) or sets.get("_default") or {"front_yaw": 0, "pieces": [], "props": {}, "marks": {},
                                                       "vista": [0, 20, 3]}


def _add(a, b, k=1.0):
    return [a[i] + k * b[i] for i in range(3)]


# ---------------------------------------------------------------- characters
def character_entries(p, cfg: dict, d, chapter: int, the_set: dict) -> list[dict]:
    """Who is in the shot, where, in what posture, and as what: a model file if there is one, else a
    procedural anime figure (engine/cgi/staging/figure.py) whose joints the camera and eyelines use."""
    out = []
    for i, name in enumerate(d.characters):
        spec = cfg["characters"].get(name, {})
        mark = d.marks.get(name) or the_set.get("marks", {}).get(name)
        if mark is None:                                   # no mark yet: stand side by side, facing the front
            mark = {"at": [(i - (len(d.characters) - 1) / 2) * CROWD_SPACING, 0, 0], "yaw": 0}
        posture = d.posture.get(name) or mark.get("posture") or "standing"
        height = by_chapter(spec.get("height_m", {}), chapter, 1.5)
        model = by_chapter(spec.get("model", {}), chapter, "") or ""
        model_path = (p.root / model) if model else None
        has_model = bool(model_path and model_path.exists())
        look = dict(by_chapter(spec.get("look", {}), chapter, {}) or {})
        fig = None
        stand_in = cfg.get("stand_in", "anime")
        if not spec.get("procedural") and not has_model and stand_in == "anime":
            arms = "front" if d.camera.subject == f"{name}:hands" and posture in ("standing", "walking") else ""
            fig = figure.skeleton(height, float(look.get("age", 30)), posture, arms=arms)
        out.append({"name": name, "at": list(mark["at"]), "yaw": float(mark.get("yaw", 0)), "posture": posture,
                    "height": height, "colour": spec.get("colour", [0.5, 0.5, 0.5]),
                    "hair": spec.get("hair", [0.1, 0.1, 0.1]),
                    "procedural": spec.get("procedural", ""),
                    "model": str(model_path) if has_model else "",
                    "model_wanted": model, "face_names": spec.get("face_names", {}),
                    "eye_height_m": spec.get("eye_height_m", {}), "look": look, "figure": fig,
                    "stand_in": stand_in})
    return out


def head_point(c: dict) -> list[float]:
    """Where the eyes are. A character's measured eye heights (config cgi.characters.<name>.eye_height_m,
    by posture) win over the anime-proportion defaults."""
    if c.get("procedural") == "lantern":
        return [c["at"][0], c["at"][1], c["at"][2] + 1.1]
    if c.get("figure"):
        return figure.to_world(c["figure"]["eyes"], c["at"], c["yaw"])
    eye = (c.get("eye_height_m") or {}).get(c["posture"])
    z = eye if eye else c["height"] * POSTURE_HEAD.get(c["posture"], 0.9)
    return [c["at"][0], c["at"][1], c["at"][2] + z]


def hands_point(c: dict) -> list[float]:
    if c.get("figure"):
        return figure.to_world(c["figure"]["hands"], c["at"], c["yaw"])
    f = layout.facing_vec(c["yaw"])
    reach = 0.28 * c["height"] / 1.45
    return [c["at"][0] + f[0] * reach, c["at"][1] + f[1] * reach,
            c["at"][2] + c["height"] * POSTURE_HANDS.get(c["posture"], 0.55)]


def gaze_point(gaze: str, me: dict, chars: dict, props: dict, cam: list[float]) -> list[float] | None:
    h = head_point(me)
    f = layout.facing_vec(me["yaw"])
    if not gaze:
        return None
    if gaze == "camera":
        return list(cam)
    if gaze == "up":
        return [h[0] + f[0] * 2, h[1] + f[1] * 2, h[2] + 2.2]
    if gaze == "down":
        return [h[0] + f[0] * 0.6, h[1] + f[1] * 0.6, h[2] - 0.8]
    if gaze == "away":
        return [h[0] - f[1] * 2, h[1] + f[0] * 2, h[2]]
    if gaze in chars:
        return head_point(chars[gaze])
    if gaze in props:
        at = props[gaze]["at"]
        return [at[0], at[1], at[2] + PROP_CENTRE.get(gaze, 0.05)]
    return None


def head_angles_to(me: dict, target: list[float]) -> tuple[float, float]:
    """Yaw and pitch (radians, relative to the body's facing) that point the face at `target`."""
    h = head_point(me)
    v = [target[i] - h[i] for i in range(3)]
    face = math.radians(me["yaw"])
    world_yaw = math.atan2(v[0], -v[1])
    rel = (world_yaw - face + math.pi) % (2 * math.pi) - math.pi
    pitch = math.atan2(v[2], math.hypot(v[0], v[1]) or 1e-6)
    return rel, pitch


def gaze_curves(d, me: dict, chars: dict, props: dict, cam: list[float], frames: int, fps: int) -> dict:
    """head.yaw / head.pitch / eyes.yaw / eyes.pitch per frame: the head turns most of the way and the
    eyes lead it, as people (and good animation) do."""
    keys_h, keys_p = [(0, 0.0)], [(0, 0.0)]
    for b in sorted((b for b in d.beats if b.who in ("", me["name"])), key=lambda b: b.t):
        tgt = gaze_point(b.gaze, me, chars, props, cam)
        if tgt is None:
            continue
        yaw, pitch = head_angles_to(me, tgt)
        f0 = timing.seconds_to_frame(b.t, fps)
        keys_h += [(f0, keys_h[-1][1]), (f0 + int(0.3 * fps), max(-1.05, min(1.05, yaw)))]
        keys_p += [(f0, keys_p[-1][1]), (f0 + int(0.3 * fps), max(-0.7, min(0.7, pitch)))]
    yaw_c = timing.ease_curve(frames, keys_h)
    pitch_c = timing.ease_curve(frames, keys_p)
    lead = int(0.12 * fps)                                 # the eyes arrive a few frames before the head
    eyes_y = [yaw_c[min(frames - 1, f + lead)] - yaw_c[f] * 0.65 for f in range(frames)]
    eyes_p = [pitch_c[min(frames - 1, f + lead)] - pitch_c[f] * 0.65 for f in range(frames)]
    return {"head.yaw": [y * 0.65 for y in yaw_c], "head.pitch": [x * 0.65 for x in pitch_c],
            "eyes.yaw": eyes_y, "eyes.pitch": eyes_p}


MOUTHS = ("Fcl_MTH_A", "Fcl_MTH_I", "Fcl_MTH_U", "Fcl_MTH_E", "Fcl_MTH_O")


def talking(manifest: dict | None, who: str, frames: int, language: str | None = None) -> dict:
    """This character's lip-sync curves in the shot (from cgi voices), summed over their on-screen lines,
    for `language` (default: the primary one, whose curves every manifest has)."""
    out: dict = {}
    for ln in (manifest or {}).get("lines", []):
        if ln["who"] != who or not ln.get("on_screen", True):
            continue
        mouths = (ln.get("mouths") or {}).get(language) if language else None
        for ch, vals in (mouths or ln["mouth"]).items():
            row = out.setdefault(ch, [0.0] * frames)
            for i in range(min(frames, len(vals))):
                row[i] = max(row[i], vals[i])
    return out


def overlay_talk(channels: dict, talk: dict) -> dict:
    """The voice drives the mouth while someone speaks; the feeling still shows around it. For Akari
    the voice is light: its glow pulses as it speaks."""
    if not talk:
        return channels
    voice = talk.get("voice", [])
    if "akari.glow" in channels:
        channels["akari.glow"] = [g + 0.6 * (voice[i] if i < len(voice) else 0.0)
                                  for i, g in enumerate(channels["akari.glow"])]
        return channels
    n = len(next(iter(channels.values()))) if channels else len(voice)
    speaking = [i < len(voice) and voice[i] > 0.18 for i in range(n)]
    for ch in MOUTHS:
        base = channels.get(ch, [0.0] * n)
        flap = talk.get(ch, [0.0] * n)
        channels[ch] = [flap[i] if speaking[i] else base[i] for i in range(n)]
    for ch in ("Fcl_MTH_Joy", "Fcl_MTH_Sorrow", "Fcl_MTH_Angry", "Fcl_MTH_Surprised"):
        if ch in channels:
            channels[ch] = [v * (0.6 if speaking[i] else 1.0) for i, v in enumerate(channels[ch])]
    return channels


def face_track(p, cfg: dict, d, me: dict, frames: int, fps: int, gaze: dict, shot_dir: Path,
               talk: dict | None = None) -> dict:
    """Per-frame face channels for one character: a captured performance if there is one
    (media/cgi/<shot>/face-<name>.json, from `cgi capture`), with the anime layer from the
    beats on top; else the beats alone. Returns {channel: [values]}."""
    curves = timing.emotion_curves(d.beats, frames, fps, dirn.EMOTIONS, me["name"])
    thoughts = [b.t for b in d.beats if b.who in ("", me["name"])]
    blink = timing.blinks(frames, fps, seed_for(d.shot + me["name"]), cfg["blinks_per_minute"], thoughts, curves)
    if me.get("procedural") == "lantern":
        wob = timing.wobble(frames, fps, seed_for(d.shot + "bob"), 0.03)
        tr = retarget.lantern_channels(curves, frames, wob)
        return {**overlay_talk({n: tr.channel(n) for n in tr.names}, talk or {}), "_source": "beats"}
    emo = retarget.emotion_face(curves, blink)
    captured = shot_dir / f"face-{me['name']}.json"
    source = "beats"
    if captured.exists():
        cap = Track.load(captured)
        cap = resample(smooth(cap), fps, frames)
        mapped = retarget.map_capture(cap)
        anime = Track(emo.fps, [n for n in emo.names if n.startswith("anime.")],
                      [[row[emo.names.index(n)] for n in emo.names if n.startswith("anime.")] for row in emo.values])
        tr = retarget.combine(mapped, anime)
        source = "capture"
    else:
        tr = emo
    channels: dict[str, Any] = {n: tr.channel(n) for n in tr.names}
    if source == "beats":
        channels.update(gaze)
    on = cfg["character_step"]
    for n, vals in channels.items():
        if n.startswith("Fcl_EYE_Close"):
            continue                                       # blinks stay on ones: snappy
        channels[n] = [v[0] for v in timing.timed(vals, {1: "ones", 2: "twos", 3: "threes"}.get(on, "twos"))]
    channels = overlay_talk(channels, talk or {})
    if me.get("face_names"):
        channels = {me["face_names"].get(n, n): v for n, v in channels.items()}
    channels["_source"] = source + ("+voice" if talk else "")
    return channels


# ---------------------------------------------------------------- props and their motion
def mallet_motion(frames: int, fps: int, strikes: list[float]) -> list[dict]:
    """The mallet's head (origin) and tilt on every frame: lifted, a little higher (anticipation),
    a fall on ones that accelerates, the hit held, a small rebound, then a slow lift."""
    up = {"at": [0.26, 0.0, 1.06], "rot": [0, 38, 0]}         # the handle leans towards whoever swings (+X)
    high = {"at": [0.31, 0.0, 1.14], "rot": [0, 30, 0]}
    hit = {"at": [0.02, 0.0, 0.5], "rot": [0, 72, 0]}
    rebound = {"at": [0.05, 0.0, 0.57], "rot": [0, 67, 0]}
    keys: list[tuple[int, dict, str]] = [(0, up, "ease")]
    for t in sorted(strikes):
        fi = timing.seconds_to_frame(t, fps)
        keys += [(max(1, fi - 9), high, "ease"), (fi, hit, "fall"), (fi + 3, hit, "hold"),
                 (fi + 9, rebound, "ease"), (fi + 30, up, "ease")]
    keys = sorted(((f, k, how) for f, k, how in keys), key=lambda x: x[0])
    out: list[dict[str, Any]] = []
    for f in range(frames):
        prev = max((k for k in keys if k[0] <= f), key=lambda k: k[0])
        nxt = min((k for k in keys if k[0] > f), key=lambda k: k[0], default=None)
        if nxt is None:
            pose = prev[1]
        else:
            t = (f - prev[0]) / (nxt[0] - prev[0])
            k = t * t if nxt[2] == "fall" else (0.0 if nxt[2] == "hold" else timing.ease(t, 0.8))
            pose = {key: [prev[1][key][i] + (nxt[1][key][i] - prev[1][key][i]) * k for i in range(3)]
                    for key in ("at", "rot")}
        out.append({"frame": f + 1, "at": [round(x, 4) for x in pose["at"]], "rot": [round(x, 3) for x in pose["rot"]]})
    vals = [o["at"] + o["rot"] for o in out]
    stepped = timing.mixed(vals, fast=0.06)
    return [{"frame": i + 1, "at": v[:3], "rot": v[3:]} for i, v in enumerate(stepped)]


def sway(frames: int, fps: int, seed: int, base: dict, degrees: float = 3.0) -> list[dict]:
    a = timing.wobble(frames, fps, seed, degrees, 0.3)
    b = timing.wobble(frames, fps, seed + 1, degrees * 0.7, 0.23)
    vals = [[base["at"][0], base["at"][1], base["at"][2], a[f], b[f], 0.0] for f in range(frames)]
    vals = timing.step(vals, 2)
    return [{"frame": f + 1, "at": v[:3], "rot": v[3:]} for f, v in enumerate(vals)]


def rise(frames: int, base: dict, by: float = 9.0) -> list[dict]:
    out = []
    for f in range(frames):
        k = 1 - (1 - f / max(1, frames - 1)) ** 2
        out.append({"frame": f + 1, "at": [base["at"][0], base["at"][1], base["at"][2] - by * (1 - k)],
                    "rot": base.get("rot", [0, 0, 0])})
    return out


def prop_entries(p, cfg: dict, d, chapter: int, the_set: dict, frames: int, fps: int) -> dict:
    props = {}
    for name, spec in the_set.get("props", {}).items():
        props[name] = {"kind": name, "at": list(spec.get("at", [0, 0, 0])), "rot": list(spec.get("rot", [0, 0, 0]))}
    for name in d.props:
        props.setdefault(name, {"kind": name, "at": [0, 0, 0], "rot": [0, 0, 0]})
    for name, state in cfg.get("props_state", {}).items():
        if name in props:
            props[name].update(by_chapter(state, chapter, {}) or {})
    for name, state in d.props_state.items():
        props.setdefault(name, {"kind": name, "at": [0, 0, 0], "rot": [0, 0, 0]}).update(state)
    for name, pr in props.items():
        strikes = [a["at"] for a in d.actions if a.get("prop") == name and a.get("do") == "strike"]
        if name == "mallet" and strikes:
            pr["motion"] = mallet_motion(frames, fps, strikes)
        if any(a.get("prop") == name and a.get("do") == "sway" for a in d.actions):
            pr["motion"] = sway(frames, fps, seed_for(d.shot + name), pr)
    return props


def piece_entries(the_set: dict, d, frames: int) -> list[dict]:
    out = []
    for piece in the_set.get("pieces", []):
        pc = dict(piece)
        pc.setdefault("rot", [0, 0, 0])
        if any(a.get("prop") == pc["kind"] and a.get("do") == "rise" for a in d.actions):
            pc["motion"] = rise(frames, pc)
        out.append(pc)
    return out


# ---------------------------------------------------------------- camera
def camera_entry(cfg: dict, d, the_set: dict, chars: dict, props: dict, pieces: list, frames: int, res: list) -> dict:
    c = d.camera
    aspect = res[0] / res[1]
    subject = c.subject or (d.characters[0] if d.characters else (d.props[0] if d.props else "vista"))
    yaw = float(the_set.get("front_yaw", 0))
    fh = None
    far = None
    if subject == "vista":
        target = list(the_set.get("vista", [0, 20, 3]))
        loc, target, fh = list(CAMERA_HOME), target, layout.FRAME_HEIGHT.get(c.framing, 7.0)
        far = True
    elif "+" in subject:
        a, b = (chars.get(x) for x in subject.split("+", 1))
        if not a or not b:
            raise ValueError(f"camera.subject {subject!r}: both must be in the shot {list(chars)}")
        ha, hb = head_point(a), head_point(b)
        v = [hb[0] - ha[0], hb[1] - ha[1]]
        n = [-v[1], v[0]]
        front = layout.facing_vec(yaw)
        if n[0] * front[0] + n[1] * front[1] < 0:
            n = [-n[0], -n[1]]
        yaw = math.degrees(math.atan2(n[0], -n[1]))
        dy = (a["yaw"] - b["yaw"] + 180) % 360 - 180
        if abs(dy) < 60:                                   # side by side, looking the same way: frame by that
            yaw = a["yaw"] - dy / 2
        span = math.hypot(*v)
        fh = max(layout.FRAME_HEIGHT.get(c.framing, 0.95), (span + 0.7) / aspect, abs(ha[2] - hb[2]) + 0.6)
        target = [(ha[i] + hb[i]) / 2 for i in range(3)]
        target[2] = max(ha[2], hb[2]) + 0.2 - fh / 2       # head room above the taller one
        az, _el = layout.ANGLE.get(c.angle, (0.0, 0.0))
        if az and c.angle in ("three-quarter", "profile"):  # favour the first named: look past the second at them
            side = layout.orbit_vec(yaw, az, 0)
            if side[0] * (hb[0] - ha[0]) + side[1] * (hb[1] - ha[1]) < 0:
                yaw = yaw - 2 * az
    elif ":" in subject:
        who, part = subject.split(":", 1)
        me = chars.get(who)
        if not me:
            raise ValueError(f"camera.subject {subject!r}: {who} is not in the shot")
        target = hands_point(me) if part == "hands" else head_point(me)
        yaw = me["yaw"]
    elif subject in chars:
        me = chars[subject]
        target = head_point(me)
        if c.framing in ("MS", "MLS", "WIDE"):
            target[2] -= 0.25 * layout.FRAME_HEIGHT.get(c.framing, 0.95)
        yaw = me["yaw"]
    elif subject in props:
        at = props[subject]["at"]
        target = [at[0], at[1], at[2] + PROP_CENTRE.get(subject, 0.05)]
        if c.framing in ("INSERT", "CU", "ECU"):
            fh = PROP_FRAME.get(subject, layout.FRAME_HEIGHT[c.framing]) * {"ECU": 0.45, "CU": 0.7}.get(c.framing, 1.0)
    else:
        piece = next((x for x in pieces if x["kind"] == subject), None)
        if piece is None:                                  # not built in this set yet: frame the place instead
            target, loc, far = list(the_set.get("vista", [0, 20, 3])), list(CAMERA_HOME), True
            subject = "vista"
        else:
            target, loc, far = list(piece["at"]), list(CAMERA_HOME), True
    if far:
        frame_h = fh or layout.FRAME_HEIGHT.get(c.framing, 7.0)
        if c.angle == "low":
            loc = [loc[0], loc[1], 0.45]
    else:
        loc, target, frame_h = layout.place(target, yaw, c.framing, c.angle, c.lens, cfg["sensor_mm"], aspect, fh)
        loc = list(loc)
    keys = layout.keys(loc, target, c.move, c.amount, frames, frame_h, seed_for(d.shot + "cam"))
    focus = c.focus or subject
    focus_pt = target
    if focus in chars:
        focus_pt = head_point(chars[focus])
    return {"lens": c.lens, "sensor_mm": cfg["sensor_mm"], "fstop": c.fstop, "keys": keys,
            "focus_point": [round(x, 4) for x in focus_pt], "frame_h": round(frame_h, 4), "subject": subject,
            "vfov": layout.vfov(c.lens, cfg["sensor_mm"], aspect)}


def follow_head(cam: dict, c: dict, eye_radius: float = 0.1) -> None:
    """In a close-up the eyes stay where the camera aimed: the camera moves with the head's average
    turn over the shot (a face looking up at the moon lifts its eyes out of a fixed frame)."""
    face = c["face"]
    n = len(face.get("head.pitch", [])) or 1
    pitch = sum(face.get("head.pitch", [0.0])) / n
    yaw = sum(face.get("head.yaw", [0.0])) / n
    if abs(pitch) < 0.02 and abs(yaw) < 0.02:
        return
    f = layout.facing_vec(c["yaw"])
    side = (-f[1], f[0])
    dz = eye_radius * math.sin(pitch)
    dside = eye_radius * math.sin(yaw)
    back = eye_radius * (1 - math.cos(pitch) * math.cos(yaw))
    delta = [side[0] * dside - f[0] * back, side[1] * dside - f[1] * back, dz]
    for k in cam["keys"]:
        k["loc"] = [round(k["loc"][i] + delta[i], 5) for i in range(3)]
        k["target"] = [round(k["target"][i] + delta[i], 5) for i in range(3)]
    cam["focus_point"] = [round(cam["focus_point"][i] + delta[i], 4) for i in range(3)]


# ---------------------------------------------------------------- keeping the view clear
PROP_SIZE = {"mortar": (0.21, 0.37), "rice": (0.0, 0.16), "bowl": (0.03, 0.08), "lantern": (0.0, 0.17),
             "cushion": (0.04, 0.3), "tray": (0.02, 0.2), "dumplings": (0.1, 0.13), "pampas": (0.5, 0.25)}


def _seg_dist(p, a, b) -> float:
    ab = [b[i] - a[i] for i in range(3)]
    ap = [p[i] - a[i] for i in range(3)]
    t = max(0.0, min(1.0, sum(x * y for x, y in zip(ap, ab)) / (sum(x * x for x in ab) or 1e-9)))
    return math.dist(p, [a[i] + ab[i] * t for i in range(3)])


def prop_points(pr: dict) -> list[tuple[list, float]]:
    """Points along a prop with a radius each: what can block the camera."""
    at = pr["at"]
    if pr["kind"] == "mallet":
        rot = pr.get("rot", [0, 0, 0])
        rx, ry, rz = (math.radians(x) for x in rot)
        a = [math.cos(rx) * math.sin(ry), -math.sin(rx), math.cos(rx) * math.cos(ry)]   # the handle: XYZ euler of +Z
        axis = [a[0] * math.cos(rz) - a[1] * math.sin(rz), a[0] * math.sin(rz) + a[1] * math.cos(rz), a[2]]
        pts = [(list(at), 0.1)]
        pts += [([at[i] + axis[i] * (0.1 + 0.05 * k) for i in range(3)], 0.03) for k in range(19)]
        return pts
    up, r = PROP_SIZE.get(pr["kind"], (0.05, 0.1))
    return [([at[0], at[1], at[2] + up], r)]


def occludes(pt, r: float, loc, subj, subj_r: float) -> bool:
    """True when the sphere (pt, r) sits between the camera and the sphere (subj, subj_r) and covers
    part of it on screen: their angular discs overlap and it is nearer."""
    vp = [pt[i] - loc[i] for i in range(3)]
    vs = [subj[i] - loc[i] for i in range(3)]
    dp, ds = math.sqrt(sum(x * x for x in vp)), math.sqrt(sum(x * x for x in vs))
    if dp < 0.02 or dp >= ds:
        return False
    cosang = max(-1.0, min(1.0, sum(a * b for a, b in zip(vp, vs)) / (dp * ds)))
    return math.acos(cosang) < math.atan(r / dp) + math.atan(subj_r / ds)


def subject_points(cam: dict, chars: dict) -> list[tuple[list, float]]:
    """What the shot is about, as spheres: each framed character's head and chest, or the target."""
    pts = []
    if ":" in cam["subject"]:                              # a part (hands): just that point
        who, part = cam["subject"].split(":", 1)
        c = chars.get(who)
        if c:
            return [(hands_point(c) if part == "hands" else head_point(c), 0.1)]
    names = cam["subject"].split("+")
    for n in names:
        c = chars.get(n)
        if c:
            h = head_point(c)
            pts += [(h, 0.13), ([h[0], h[1], h[2] - 0.3 * c["height"] / 1.45], 0.2)]
    return pts or [(cam["keys"][0]["target"], 0.15)]


def cheat_props(props: dict, cam: dict, d, chars: dict | None = None) -> list[str]:
    """A prop that would hide what the shot is about (a face, a body, the prop it frames) is taken out
    of the shot (hidden), the way a camera crew cheats a prop out of frame. Props the shot names,
    its subject, moving props, and anything in a wide never are."""
    hidden: list[str] = []
    if d.camera.framing == "WIDE" or cam["subject"] == "vista":
        return hidden                                      # a wide shows the place as it is
    loc = cam["keys"][0]["loc"]
    targets = subject_points(cam, chars or {})
    for name, pr in props.items():
        if name in d.props or name == cam["subject"] or pr.get("motion"):
            continue
        if any(occludes(pt, r, loc, sp, sr) for pt, r in prop_points(pr) for sp, sr in targets):
            pr["hidden"] = True
            hidden.append(name)
    return hidden


# ---------------------------------------------------------------- the plan
def build(p, shot_id: str, preview: bool = False, frames_only: tuple[int, int] | None = None,
          size: tuple[int, int] | None = None, guides: tuple[str, ...] = (), language: str | None = None) -> dict:
    """One shot's plan. `language`: whose lips and voice it renders (default: the dialogue's primary
    language); another language is a dub, whose frames and clip carry its code (1-11.ja.mp4)."""
    unknown = set(guides) - set(GUIDES)
    if unknown:
        raise ValueError(f"no guide pass called {', '.join(sorted(unknown))}; there are {', '.join(GUIDES)}")
    cfg = settings(p)
    s = next((x for x in p.shots if x.id == shot_id), None)
    if s is None:
        raise KeyError(f"no shot {shot_id} in shots.csv")
    d = dirn.direction_for(p, shot_id)
    bad = dirn.problems(d)
    if bad:
        raise ValueError(f"{shot_id}: the acting sheet has problems:\n  " + "\n  ".join(bad))
    fps = int(cfg["fps"])
    frames = max(1, int(round(d.seconds * fps)))
    res = list(size) if size else list(cfg["preview_resolution"] if preview else cfg["resolution"])
    the_set = set_for(p, d.scene)
    rig = palette.rig_for(p, s.chapter, d.scene, shot_id)
    chars = {c["name"]: c for c in character_entries(p, cfg, d, s.chapter, the_set)}
    props = prop_entries(p, cfg, d, s.chapter, the_set, frames, fps)
    pieces = piece_entries(the_set, d, frames)
    cam = camera_entry(cfg, d, the_set, chars, props, pieces, frames, res)
    shot_dir = work_dir(p, cfg, shot_id)
    hidden = cheat_props(props, cam, d, chars)
    # eyelines match across cuts: someone on their mark off screen can still be looked at
    seen_by = dict(chars)
    for name, mark in {**the_set.get("marks", {}), **d.marks}.items():
        if name not in seen_by:
            spec = cfg["characters"].get(name, {})
            seen_by[name] = {"name": name, "at": list(mark["at"]), "yaw": float(mark.get("yaw", 0)),
                             "posture": mark.get("posture", "standing"),
                             "height": by_chapter(spec.get("height_m", {}), s.chapter, 1.5),
                             "procedural": spec.get("procedural", "")}
    spoken = dialogue.shot_manifest(p, shot_id)
    primary = (spoken or {}).get("primary")
    if language and spoken and language not in spoken.get("languages", []):
        raise ValueError(f"{shot_id}: no {language} voices; it is voiced in {', '.join(spoken['languages'])}")
    dub = bool(language and language != primary and language in dialogue.dubbed_languages(spoken))
    for c in chars.values():
        gaze = gaze_curves(d, c, seen_by, props, cam["keys"][0]["loc"], frames, fps)
        c["face"] = face_track(p, cfg, d, c, frames, fps, gaze, shot_dir,
                               talking(spoken, c["name"], frames, language if dub else None))
        if d.camera.framing in ("ECU", "CU", "MCU") and cam["subject"] == c["name"]:
            follow_head(cam, c)
    tag = "preview" if preview else "final"
    dub_tag = f"-{language}" if dub else ""            # a dub's frames and guides: frames-preview-640x360-ja
    dub_suffix = f".{language}" if dub else ""         # its clip, plan and scene: 1-11.ja.mp4
    first, last = frames_only or (1, frames)
    return {
        "version": 1, "shot": shot_id, "chapter": s.chapter, "scene": d.scene, "description": s.shot,
        "status": d.status, "fps": fps, "frames": frames, "frame_range": [first, last],
        "resolution": res, "engine": cfg["engine"],
        "samples": cfg["preview_samples"] if preview else cfg["samples"],
        "outline_px": cfg["outline_px"], "blackout": d.blackout,
        "rig": rig, "pieces": pieces, "props": list(props.values()), "characters": list(chars.values()),
        "camera": cam, "timing": d.timing, "notes": d.notes, "cheated_out": hidden,
        "audio": (spoken or {}).get("audio", {}),
        "language": language if dub else primary,
        "output": {"dir": str(shot_dir), "frames": str(shot_dir / f"frames-{tag}-{res[0]}x{res[1]}{dub_tag}"),
                   "clip_suffix": dub_suffix,
                   "blend": str(shot_dir / f"{shot_id}{dub_suffix}.blend"),
                   "guides": {g: str(shot_dir / f"guide-{tag}-{res[0]}x{res[1]}{dub_tag}-{g}") for g in guides}},
        "guides": list(guides),
    }


def write(plan: dict) -> Path:
    out = Path(plan["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    name = "plan-preview" if "preview" in plan["output"]["frames"] else "plan"
    f = out / f"{name}{plan['output'].get('clip_suffix', '')}.json"       # a dub's plan beside the primary's
    f.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    return f
