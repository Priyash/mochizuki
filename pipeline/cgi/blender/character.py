"""Characters in a shot. Runs inside Blender.

With a model file (a VRM from VRoid Studio, through the VRM add-on), the model is placed on
its mark and its face shape keys play the plan's face channels. Without one, the character is
made from code (blender/anime.py): an anime body, face and hair for its age, playing the same
channels. The old maquette (a simple stand-in) remains for "stand_in": "maquette".
Akari is a lantern that acts with its light.
"""
from __future__ import annotations

import math

import bpy
from mathutils import Vector

from . import anime, props
from .keys import ch as _ch
from .keys import key_curve as _key_curve

PROXY_HEAD = {"standing": 0.9, "walking": 0.9, "kneeling": 0.54, "seiza": 0.52}   # the same as plan.POSTURE_HEAD


# ---------------------------------------------------------------- stand-in
def maquette(kit: props.Kit, c: dict, frames: int):
    h = c["height"]
    k = h / 1.45
    child = h < 1.3
    posture = c.get("posture", "standing")
    root = bpy.data.objects.new(c["name"], None)
    bpy.context.scene.collection.objects.link(root)
    body_mat = kit.mat(f"{c['name']}.clothes", c["colour"])
    skin = kit.mat("skin", [1.0, 0.86, 0.76])
    hair = kit.mat(f"{c['name']}.hair", c.get("hair", [0.1, 0.1, 0.1]))
    dark = kit.mat("eye", [0.04, 0.03, 0.05])
    head_z = h * PROXY_HEAD.get(posture, 0.93)
    torso_h = head_z - (0.12 * k if not child else 0.14 * k)
    if posture in ("kneeling", "seiza"):
        body = props.sphere(f"{c['name']}.body", 0.2 * k, [body_mat], scale=(1.0, 0.85, torso_h / (0.4 * k)))
        body.location = (0, 0, torso_h / 2)
        legs = props.box(f"{c['name']}.legs", (0.36 * k, 0.5 * k, 0.16 * k), [body_mat], bevel=0.03 * k)
        legs.location = (0, 0.08 * k, 0.08 * k)
        legs.parent = root
    else:
        body = props.sphere(f"{c['name']}.body", 0.19 * k, [body_mat], scale=(1.0, 0.8, torso_h / (0.38 * k)))
        body.location = (0, 0, torso_h / 2)
    body.parent = root
    pivot = bpy.data.objects.new(f"{c['name']}.neck", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.location = (0, 0, head_z - 0.06 * k)
    pivot.parent = root
    hr = (0.115 if child else 0.1) * k * (1.15 if child else 1.0)
    head = props.sphere(f"{c['name']}.head", hr, [skin], segments=32)
    head.location = (0, 0, 0.06 * k)
    head.parent = pivot
    cap = props.sphere(f"{c['name']}.hair", hr * 1.06, [hair], scale=(1, 1, 0.85), segments=24)
    cap.location = (0, hr * 0.32, hr * 0.38)
    cap.parent = head
    parts = {}
    for side, sx in (("L", 1), ("R", -1)):
        eye = props.sphere(f"{c['name']}.eye{side}", hr * 0.2, [dark], scale=(0.75, 0.35, 1.25), segments=16)
        eye.location = (sx * hr * 0.38, -hr * 0.9, hr * 0.05)
        eye.parent = head
        shine = props.sphere(f"{c['name']}.shine{side}", hr * 0.06, [kit.mat("shine", [1, 1, 1], emission=2.0)],
                             segments=8)
        shine.location = (sx * hr * 0.33, -hr * 0.99, hr * 0.14)
        shine.parent = head
        brow = props.box(f"{c['name']}.brow{side}", (hr * 0.32, hr * 0.05, hr * 0.05), [hair])
        brow.location = (sx * hr * 0.38, -hr * 0.9, hr * 0.32)
        brow.parent = head
        blush = props.sphere(f"{c['name']}.blush{side}", hr * 0.15, [kit.mat("blush", [1.0, 0.45, 0.5], emission=1.2)],
                             scale=(1.2, 0.2, 0.6), segments=10)
        blush.location = (sx * hr * 0.55, -hr * 0.82, -hr * 0.2)
        blush.parent = head
        tear = props.sphere(f"{c['name']}.tear{side}", hr * 0.1, [kit.mat("tear", [0.75, 0.9, 1.0], emission=1.5)],
                            scale=(1.4, 0.25, 0.35), segments=8)
        tear.location = (sx * hr * 0.38, -hr * 0.95, -hr * 0.18)
        tear.parent = head
        parts[side] = (eye, shine, brow, blush, tear)
    hands_z = h * {"standing": 0.55, "walking": 0.55, "kneeling": 0.36, "seiza": 0.34}.get(posture, 0.55)
    reach = 0.28 * h / 1.45
    for side, sx in (("L", 1), ("R", -1)):
        mitt = props.sphere(f"{c['name']}.hand{side}", 0.045 * k, [skin], scale=(0.8, 1.0, 0.6), segments=12)
        mitt.location = (sx * 0.09 * k, -reach, hands_z)
        mitt.parent = root
        arm = props.sphere(f"{c['name']}.arm{side}", 0.04 * k, [body_mat], scale=(1.0, 3.2, 1.0), segments=12)
        arm.location = (sx * 0.13 * k, -reach * 0.55, hands_z + 0.06 * k)
        arm.rotation_euler = (-0.5, 0, 0)
        arm.parent = root
    mouth = props.sphere(f"{c['name']}.mouth", hr * 0.16, [kit.mat("mouth", [0.45, 0.12, 0.14])],
                         scale=(1.4, 0.3, 0.25), segments=12)
    mouth.location = (0, -hr * 0.93, -hr * 0.42)
    mouth.parent = head
    face = c.get("face", {})
    fr = frames
    close = _ch(face, "Fcl_EYE_Close", fr)
    joy_eye = [max(a, b) for a, b in zip(_ch(face, "Fcl_EYE_Joy", fr),
                                         [(x + y) / 2 for x, y in zip(_ch(face, "Fcl_EYE_Joy_L", fr), _ch(face, "Fcl_EYE_Joy_R", fr))])]
    wide = _ch(face, "Fcl_EYE_Surprised", fr)
    for side in ("L", "R"):
        eye, shine, brow, blush, tear = parts[side]
        lid = [max(a, b) for a, b in zip(close, _ch(face, f"Fcl_EYE_Close_{side}", fr))]
        _key_curve(eye, "scale", [max(0.08, 1 - 0.92 * max(l, 0.7 * j)) * (1 + 0.25 * w)
                                  for l, j, w in zip(lid, joy_eye, wide)], index=2)
        _key_curve(shine, "scale", [(1 + 1.2 * x) * (1 - l) for x, l in zip(_ch(face, "anime.highlight", fr), lid)], index=0)
        _key_curve(shine, "scale", [(1 + 1.2 * x) * (1 - l) for x, l in zip(_ch(face, "anime.highlight", fr), lid)], index=2)
        sad = _ch(face, "Fcl_BRW_Sorrow", fr)
        ang = _ch(face, "Fcl_BRW_Angry", fr)
        up = _ch(face, "Fcl_BRW_Surprised", fr)
        sgn = 1 if side == "L" else -1
        _key_curve(brow, "rotation_euler", [sgn * (0.45 * s - 0.35 * a) for s, a in zip(sad, ang)], index=1)
        _key_curve(brow, "location", [hr * (0.32 + 0.1 * u + 0.05 * s - 0.06 * a) for u, s, a in zip(up, sad, ang)],
                   index=2)
        _key_curve(blush, "scale", [0.001 + 1.2 * x for x in _ch(face, "anime.blush", fr)], index=0)
        _key_curve(tear, "scale", [0.001 + 1.4 * x for x in _ch(face, "anime.tear", fr)], index=0)
    opened = _ch(face, "Fcl_MTH_A", fr)
    smile = _ch(face, "Fcl_MTH_Joy", fr)
    frown = _ch(face, "Fcl_MTH_Sorrow", fr)
    ooh = [max(a, b) for a, b in zip(_ch(face, "Fcl_MTH_O", fr), _ch(face, "Fcl_MTH_U", fr))]
    _key_curve(mouth, "scale", [1.4 * (1 + 0.6 * s - 0.4 * o) for s, o in zip(smile, ooh)], index=0)
    _key_curve(mouth, "scale", [0.25 + 1.6 * a + 0.6 * o for a, o in zip(opened, ooh)], index=2)
    _key_curve(mouth, "location", [-hr * (0.42 - 0.05 * s + 0.04 * f) for s, f in zip(smile, frown)], index=2)
    _key_curve(pivot, "rotation_euler", [-x for x in _ch(face, "head.pitch", fr)], index=0)   # +pitch looks up
    _key_curve(pivot, "rotation_euler", _ch(face, "head.yaw", fr), index=2)
    _key_curve(pivot, "rotation_euler", _ch(face, "head.roll", fr), index=1)
    return root


# ---------------------------------------------------------------- Akari
def akari(kit: props.Kit, c: dict, frames: int):
    ob = props.lantern(kit, glow=1.0)
    ob.name = c["name"]
    root = bpy.data.objects.new(f"{c['name']}.float", None)
    bpy.context.scene.collection.objects.link(root)
    ob.parent = root
    ob.location = (0, 0, 1.1)
    face = c.get("face", {})
    glow = _ch(face, "akari.glow", frames)
    mat = ob.data.materials[0]
    emit = next(n for n in mat.node_tree.nodes if n.type == "EMISSION")
    strength = emit.inputs["Strength"]
    for f, g in enumerate(glow):
        strength.default_value = 2.2 * g
        strength.keyframe_insert("default_value", frame=f + 1)
    light = bpy.data.lights.new(f"{c['name']}.light", "POINT")
    light.color = (1.0, 0.65, 0.32)
    lo = bpy.data.objects.new(f"{c['name']}.light", light)
    bpy.context.scene.collection.objects.link(lo)
    lo.parent = ob
    for f, g in enumerate(glow):
        light.energy = 40 * g
        light.keyframe_insert("energy", frame=f + 1)
    _key_curve(ob, "rotation_euler", [math.radians(t) for t in _ch(face, "akari.tilt", frames)], index=1)
    _key_curve(ob, "location", [1.1 + b for b in _ch(face, "akari.bob", frames)], index=2)
    return root


# ---------------------------------------------------------------- a real model
def _enable_vrm() -> bool:
    for mod in ("bl_ext.blender_org.vrm", "bl_ext.user_default.vrm", "io_scene_vrm", "VRM_Addon_for_Blender-release"):
        try:
            bpy.ops.preferences.addon_enable(module=mod)
            if hasattr(bpy.ops.import_scene, "vrm"):
                return True
        except Exception:  # noqa: BLE001 - not that name; try the next
            continue
    return hasattr(bpy.ops.import_scene, "vrm")


def _aim(arm_ob, pb, direction_arm: Vector) -> None:
    """Rotate pose bone `pb` so it points along `direction_arm` (armature space), from wherever its
    parents have already turned it, whatever the bone's roll."""
    bpy.context.view_layer.update()
    m = pb.matrix.copy()                                   # posed, in armature space
    cur = (m.to_3x3() @ Vector((0, 1, 0))).normalized()
    q = cur.rotation_difference(direction_arm.normalized())
    rot = (q.to_matrix() @ m.to_3x3()).to_4x4()
    rot.translation = m.translation
    local = arm_ob.convert_space(pose_bone=pb, matrix=rot, from_space="POSE", to_space="LOCAL")
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = local.to_quaternion()


HUMAN = {"hips": ("hips", "J_Bip_C_Hips"), "head": ("head", "J_Bip_C_Head"),
         "leftUpperArm": ("left_upper_arm", "J_Bip_L_UpperArm"), "rightUpperArm": ("right_upper_arm", "J_Bip_R_UpperArm"),
         "leftUpperLeg": ("left_upper_leg", "J_Bip_L_UpperLeg"), "rightUpperLeg": ("right_upper_leg", "J_Bip_R_UpperLeg"),
         "leftLowerLeg": ("left_lower_leg", "J_Bip_L_LowerLeg"), "rightLowerLeg": ("right_lower_leg", "J_Bip_R_LowerLeg")}


def human_bone(arm_ob, name: str):
    """The pose bone for a VRM humanoid bone (hips, head, leftUpperArm...), through the VRM add-on's
    humanoid map (VRM 1.0 or 0.x), else VRoid's bone names."""
    snake, vroid = HUMAN[name]
    ext = getattr(arm_ob.data, "vrm_addon_extension", None)
    bone = ""
    if ext is not None:
        try:
            bone = getattr(ext.vrm1.humanoid.human_bones, snake).node.bone_name
        except AttributeError:
            bone = ""
        if not bone:
            try:
                bone = next((hb.node.bone_name for hb in ext.vrm0.humanoid.human_bones if hb.bone == name), "")
            except AttributeError:
                bone = ""
    for cand in (bone, vroid):
        if cand and cand in arm_ob.pose.bones:
            return arm_ob.pose.bones[cand]
    return None


def pose(arm_ob, posture: str) -> None:
    """Arms down from the T-pose; and for kneeling, thighs forward and shins folded back (seiza)."""
    fwd = Vector((0, -1, 0))                 # a VRM faces -Y in its own space
    for side, sx in (("left", 1), ("right", -1)):
        ua = human_bone(arm_ob, f"{side}UpperArm")
        if ua:
            _aim(arm_ob, ua, Vector((sx * 0.35, -0.08, -1.0)))
        if posture in ("kneeling", "seiza"):
            ul, ll = human_bone(arm_ob, f"{side}UpperLeg"), human_bone(arm_ob, f"{side}LowerLeg")
            if ul and ll:
                _aim(arm_ob, ul, fwd + Vector((sx * 0.1, 0, -0.18)))      # thighs forward, a little down
                _aim(arm_ob, ll, -fwd + Vector((0, 0, -0.12)))             # shins folded back under them
    hips = human_bone(arm_ob, "hips")
    if hips and posture in ("kneeling", "seiza"):
        leg = human_bone(arm_ob, "leftUpperLeg")
        drop = (leg.bone.head_local.z * 0.74) if leg else 0.45       # sitting on the heels
        hips.location = (hips.bone.matrix_local.to_3x3().inverted() @ Vector((0, 0, -drop)))


# our channels -> VRM expression presets, for models without VRoid's Fcl_ shape keys
PRESETS = {
    "blink": ("Fcl_EYE_Close",), "blink_left": ("Fcl_EYE_Close_L",), "blink_right": ("Fcl_EYE_Close_R",),
    "aa": ("Fcl_MTH_A",), "ih": ("Fcl_MTH_I",), "ou": ("Fcl_MTH_U",), "ee": ("Fcl_MTH_E",), "oh": ("Fcl_MTH_O",),
    "happy": ("Fcl_MTH_Joy", "Fcl_EYE_Joy"), "sad": ("Fcl_MTH_Sorrow", "Fcl_BRW_Sorrow", "Fcl_EYE_Sorrow"),
    "angry": ("Fcl_BRW_Angry", "Fcl_MTH_Angry"), "surprised": ("Fcl_EYE_Surprised", "Fcl_BRW_Surprised"),
}


def drive_presets(arm_ob, face: dict, frames: int) -> list[str]:
    """Key the VRM add-on's expression previews (happy, sad, aa, blink...) from our channels."""
    try:
        preset = arm_ob.data.vrm_addon_extension.vrm1.expressions.preset
    except AttributeError:
        return []
    used = []
    for name, sources in PRESETS.items():
        ex = getattr(preset, name, None)
        if ex is None or not any(src in face for src in sources):
            continue
        vals = [max(_ch(face, src, frames)[f] for src in sources) for f in range(frames)]
        _key_curve(ex, "preview", vals)
        used.append(name)
    return used


def vrm(c: dict, frames: int):
    if not _enable_vrm():
        print("cgi.blender: the VRM add-on is not installed (Blender > Preferences > Get Extensions > VRM); "
              "using a stand-in", flush=True)
        return None
    before = set(bpy.data.objects)
    bpy.ops.import_scene.vrm(filepath=c["model"])
    new = [o for o in bpy.data.objects if o not in before]
    arm = next((o for o in new if o.type == "ARMATURE"), None)
    root = arm or (new[0] if new else None)
    if root is None:
        return None
    bpy.context.view_layer.update()
    tops = [(o.matrix_world @ Vector(corner)).z for o in new if o.type == "MESH" for corner in o.bound_box]
    if tops and max(tops) > 0.1:                            # the model is made the character's height
        k = c["height"] / max(tops)
        root.scale = (k, k, k)
    root.rotation_mode = "XYZ"                             # importers leave quaternions: euler is what we set
    root.location = c["at"]
    root.rotation_euler = (0, 0, math.radians(c["yaw"]))
    if arm:
        pose(arm, c.get("posture", "standing"))
    face = c.get("face", {})
    keys = {}
    for o in new:
        if o.type == "MESH" and o.data.shape_keys:
            for kb in o.data.shape_keys.key_blocks:
                keys.setdefault(kb.name.lower(), []).append(kb)
    wanted = [n for n in face if not n.startswith("_") and "." not in n]
    direct = [n for n in wanted if n.lower() in keys]
    if direct:                                             # a VRoid model: its own Fcl_ shapes, finest control
        for name in direct:
            for kb in keys[name.lower()]:
                _key_curve(kb, "value", face[name])
    elif arm:
        used = drive_presets(arm, face, frames)
        print(f"cgi.blender: {c['name']}: no VRoid shape keys; drove the VRM expressions {used}", flush=True)
    if arm:
        head = human_bone(arm, "head")
        if head:
            head.rotation_mode = "XYZ"
            _key_curve(head, "rotation_euler", _ch(face, "head.pitch", frames), index=0)
            _key_curve(head, "rotation_euler", _ch(face, "head.yaw", frames), index=1)
            _key_curve(head, "rotation_euler", _ch(face, "head.roll", frames), index=2)
    return root


def stand_in(kit: props.Kit, c: dict, frames: int):
    """Without a model: the procedural anime character when the plan has its figure, else the maquette."""
    if c.get("figure") and c.get("stand_in", "anime") == "anime":
        return anime.figure(kit, c, frames)
    return maquette(kit, c, frames)


def build(kit: props.Kit, c: dict, frames: int):
    if c.get("procedural") == "lantern":
        root = akari(kit, c, frames)
    elif c.get("model"):
        root = vrm(c, frames) or stand_in(kit, c, frames)
    else:
        root = stand_in(kit, c, frames)
    if c.get("procedural") == "lantern" or not c.get("model") or root.type != "ARMATURE":
        root.location = c["at"]
        root.rotation_euler = (0, 0, math.radians(c["yaw"]))
    return root


__all__ = ["build", "maquette", "stand_in", "akari", "vrm", "pose"]
