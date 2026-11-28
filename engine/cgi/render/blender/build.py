"""Build one shot from its plan, save the .blend, render its frames. Runs inside Blender:

    blender -b --python engine/cgi/render/blender/build.py -- --plan media/cgi/1-02/plan.json [--no-render]
    python engine/cgi/render/blender/build.py --plan ...          (a Python with the bpy module)

It decides nothing: every number comes from the plan (engine/cgi/plan.py). The saved .blend
is the shot, ready to open in Blender and adjust by hand.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bpy  # noqa: E402
from blender import character, guides, props, toon  # noqa: E402
from mathutils import Vector  # noqa: E402


def say(event: str, msg: str = "", **kw) -> None:
    """One line the runner (engine/cgi/render/runner.py) turns into a structured log event."""
    print("CGI " + json.dumps({"event": event, "msg": msg, **kw}), flush=True)


def clear() -> None:
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras,
                 bpy.data.curves, bpy.data.worlds):
        for item in list(coll):
            coll.remove(item)


def interpolation(kind: str) -> None:
    """New keys get this interpolation: CONSTANT for anything on twos, BEZIER for the camera."""
    try:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = kind
    except Exception:  # noqa: BLE001
        pass


def _fcurves(action):
    if getattr(action, "layers", None):                  # Blender 4.4+: layered (slotted) actions
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    yield from bag.fcurves
    else:
        yield from getattr(action, "fcurves", [])


def step_keys(skip: set) -> int:
    """Every key holds until the next one: the plan already says what each frame shows (twos, hit
    stops, blinks), so nothing may ease in between. The preference for new keys is not honoured
    everywhere (bpy as a module ignores it), so this sets it on the keys themselves. The camera,
    keyed on every frame, is left as it is."""
    cam_actions = {ob.animation_data.action for ob in bpy.data.objects
                   if ob.name in skip and ob.animation_data and ob.animation_data.action}
    n = 0
    for act in bpy.data.actions:
        if act in cam_actions:
            continue
        for fc in _fcurves(act):
            for kp in fc.keyframe_points:
                kp.interpolation = "CONSTANT"
                n += 1
    return n


def render_settings(plan: dict) -> None:
    sc = bpy.context.scene
    try:
        sc.render.engine = plan["engine"]
    except TypeError:                                     # Blender 4.2-4.4 called it EEVEE Next
        sc.render.engine = "BLENDER_EEVEE_NEXT" if plan["engine"] == "BLENDER_EEVEE" else plan["engine"]
    if sc.render.engine.startswith("BLENDER_EEVEE"):
        sc.eevee.taa_render_samples = int(plan["samples"])
        # cel shading wants clean, flat light: no screen-space GI or ray tracing (their noise flips pixels
        # between the toon bands as grain, and they cost time)
        for attr, val in (("use_fast_gi", False), ("use_raytracing", False), ("use_gtao", False)):
            if hasattr(sc.eevee, attr):
                setattr(sc.eevee, attr, val)
    elif sc.render.engine == "CYCLES":
        sc.cycles.samples = int(plan["samples"])
    sc.render.resolution_x, sc.render.resolution_y = plan["resolution"]
    sc.render.resolution_percentage = 100
    sc.render.fps = int(plan["fps"])
    sc.frame_start, sc.frame_end = plan["frame_range"]
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"           # cel colours exactly as chosen, no film curve
    try:
        sc.view_settings.look = "None"
    except TypeError:
        pass


def lights(rig: dict, blackout: bool) -> None:
    if blackout:
        return
    for name in ("key", "fill", "practical"):
        spec = rig.get(name)
        if not spec or float(spec.get("energy", 0)) <= 0:
            continue
        kind = spec.get("type", "SUN")
        data = bpy.data.lights.new(name, kind)
        data.color = spec.get("color", [1, 1, 1])[:3]
        data.energy = float(spec["energy"])
        # hard shadows: cel shading wants a clean line, and a soft light's shadow is sampled at random,
        # which few samples turn into grain where the light sits near a toon band
        if kind == "SUN":
            data.angle = math.radians(float(spec.get("angle", 0.0)))
        elif kind in ("POINT", "SPOT"):
            data.shadow_soft_size = float(spec.get("soft", 0.0))
        elif kind == "AREA":
            data.size = float(spec.get("size", 1.0))
        ob = bpy.data.objects.new(name, data)
        bpy.context.scene.collection.objects.link(ob)
        if "at" in spec:
            ob.location = spec["at"]
        az, el = math.radians(spec.get("azimuth", 0)), math.radians(spec.get("elevation", 45))
        if kind == "SUN" or "at" not in spec:
            d = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
            ob.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()


def animate(ob, motion: list, frame0: int = 1) -> None:
    """Per-frame location/rotation (degrees) from the plan; keys only where the pose changes."""
    last = None
    for m in motion:
        pose = (tuple(m["at"]), tuple(m.get("rot", (0, 0, 0))))
        if pose == last:
            continue
        ob.location = m["at"]
        ob.rotation_euler = [math.radians(r) for r in m.get("rot", (0, 0, 0))]
        ob.keyframe_insert("location", frame=m["frame"] + frame0 - 1)
        ob.keyframe_insert("rotation_euler", frame=m["frame"] + frame0 - 1)
        last = pose


def camera(plan: dict):
    c = plan["camera"]
    data = bpy.data.cameras.new("camera")
    data.lens = float(c["lens"])
    data.sensor_width = float(c["sensor_mm"])
    data.sensor_fit = "HORIZONTAL"
    data.clip_start, data.clip_end = 0.01, 2000.0
    data.dof.use_dof = True
    data.dof.aperture_fstop = float(c["fstop"])
    cam = bpy.data.objects.new("camera", data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    target = bpy.data.objects.new("camera.target", None)
    bpy.context.scene.collection.objects.link(target)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    focus = bpy.data.objects.new("camera.focus", None)
    bpy.context.scene.collection.objects.link(focus)
    focus.location = c["focus_point"]
    data.dof.focus_object = focus
    interpolation("BEZIER")
    keys = c["keys"]
    moving = len({tuple(k["loc"]) + tuple(k["target"]) for k in keys}) > 1
    for k in (keys if moving else keys[:1]):
        cam.location, target.location = k["loc"], k["target"]
        cam.keyframe_insert("location", frame=k["frame"])
        target.keyframe_insert("location", frame=k["frame"])
    return cam


def outlines(plan: dict, cam, skip: set) -> int:
    rig = plan["rig"]
    ink = toon.outline_material(rig.get("toon", {}).get("outline", [0.08, 0.06, 0.1]))
    cam_loc = tuple(cam.location)
    n = 0
    bpy.context.view_layer.update()
    for ob in list(bpy.data.objects):
        if ob.type != "MESH" or ob.name.split(".")[0] in skip or ob.get("no_ink"):
            continue                                       # drawn face parts have no ink line
        if any(m and m.node_tree and any(nd.type == "EMISSION" for nd in m.node_tree.nodes) and not
               any(nd.type == "BSDF_DIFFUSE" for nd in m.node_tree.nodes) for m in ob.data.materials):
            continue                                       # lamps and the moon have no ink line
        w = toon.line_width(tuple(ob.matrix_world.translation), cam_loc, plan["camera"]["vfov"],
                            plan["resolution"][1], plan["outline_px"])
        toon.add_outline(ob, ink, w)
        n += 1
    return n


def build(plan: dict) -> dict:
    t0 = time.time()
    clear()
    render_settings(plan)
    rig = plan["rig"]
    toon.world(rig, plan.get("blackout", False))
    lights(rig, plan.get("blackout", False))
    kit = props.Kit(rig.get("toon", {}))
    made = {"pieces": 0, "props": 0, "characters": 0}
    interpolation("CONSTANT")
    if plan.get("blackout"):                               # a black frame: a camera and nothing to see
        camera(plan)
        made["outlined"] = 0
        say("build.done", f"{plan['shot']}: black", seconds=round(time.time() - t0, 1), **made)
        return made
    for spec in plan["pieces"]:
        ob = props.build(kit, spec)
        if ob is not None:
            made["pieces"] += 1
            if spec.get("motion"):
                animate(ob, spec["motion"])
    for spec in plan["props"]:
        ob = props.build(kit, spec)
        if ob is not None:
            made["props"] += 1
            if spec.get("motion"):
                animate(ob, spec["motion"])
            if spec.get("hidden"):                         # cheated out of this shot: kept in the .blend, not rendered
                for o in [ob, *ob.children_recursive]:
                    o.hide_render = True
    for c in plan["characters"]:
        character.build(kit, c, plan["frames"])
        made["characters"] += 1
    cam = camera(plan)
    moon = bpy.data.objects.get("moon")
    if moon:                                               # the moon always faces the camera
        con = moon.constraints.new("TRACK_TO")
        con.target = cam
        con.track_axis = "TRACK_Z"
        con.up_axis = "UP_Y"
    made["outlined"] = outlines(plan, cam, {"sea", "moon", "floor"})
    made["stepped_keys"] = step_keys({"camera", "camera.target"})
    bpy.context.scene.frame_set(plan["frame_range"][0])
    say("build.done", f"{plan['shot']}: built in {time.time() - t0:.1f} s", seconds=round(time.time() - t0, 1), **made)
    return made


def render(plan: dict) -> None:
    out = plan["output"]["frames"]
    os.makedirs(out, exist_ok=True)
    sc = bpy.context.scene
    first, last = plan["frame_range"]
    for f in range(first, last + 1):
        t = time.time()
        sc.frame_set(f)
        sc.render.filepath = os.path.join(out, f"{f:04d}.png")
        bpy.ops.render.render(write_still=True)
        say("render.frame", "", frame=f, of=last, seconds=round(time.time() - t, 2))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    with open(a.plan, encoding="utf-8") as f:
        plan = json.load(f)
    build(plan)
    blend = plan["output"]["blend"]
    os.makedirs(os.path.dirname(blend), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
    say("blend.saved", blend, path=blend)
    if not a.no_render:
        render(plan)
        guides.render_guides(plan, say)              # the AI finish's depth and lines, when the plan asks
    say("done", plan["shot"])
    return 0


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    sys.exit(main(args))
