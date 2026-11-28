"""Guide passes for the AI finish, rendered after the shot's own frames, from the same scene:

    depth   every surface by its inverse distance from the camera (disparity, as depth estimators
            give it): the nearest surface the camera sees white, the far end of the shot black
    lines   the ink lines and the drawn face parts (eyes, brows, mouth), white on black

Both are sharp (no depth of field) on a black world, with the lights off: they are all emission.
What a pass changes (materials, the ink's colour, the world, colour management, focus, lights) is
put back afterwards, and the .blend is saved before any pass runs, so it never holds a guide's state.
Runs inside Blender, like the rest of this folder.
"""
from __future__ import annotations

import os
import time

import bpy
from mathutils import Vector

INK = "ink"                    # toon.outline_material's name (Blender may add .001 ...)
DEPTH_PERCENTILES = (0.02, 0.98)
DEPTH_SAMPLES = 3              # frames sampled (first, middle, last) to fit the depth range
DEPTH_GRID = (48, 27)          # rays across the frame at each sampled frame


def _emission(name: str, rgb) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*rgb, 1.0)
    mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _depth_material(near: float, far: float) -> bpy.types.Material:
    """Emission by inverse view depth (near / z), stretched so the nearest surface is 1 (white) and
    `far` is 0 (black): the disparity maps the finish model learned from. A straight ramp from near to
    far flattened a close-up face into one grey; in disparity the face keeps its relief."""
    mat = bpy.data.materials.new("guide.depth")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    cam = nodes.new("ShaderNodeCameraData")
    inv = nodes.new("ShaderNodeMath")
    inv.operation = "DIVIDE"
    inv.inputs[0].default_value = near
    rng = nodes.new("ShaderNodeMapRange")
    rng.clamp = True
    rng.inputs["From Min"].default_value = near / far
    rng.inputs["From Max"].default_value = 1.0
    rng.inputs["To Min"].default_value = 0.0
    rng.inputs["To Max"].default_value = 1.0
    links.new(cam.outputs["View Z Depth"], inv.inputs[1])
    links.new(inv.outputs["Value"], rng.inputs["Value"])
    links.new(rng.outputs["Result"], emit.inputs["Color"])
    links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


def _meshes() -> list:
    return [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.hide_render]


def _seen_depths(scene, cam) -> list[float]:
    """View depths of what the camera sees now: rays through a grid over the frame, where they hit."""
    dg = bpy.context.evaluated_depsgraph_get()
    m = cam.matrix_world
    origin = m.translation
    fwd = m.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    tr, _, bl, tl = cam.data.view_frame(scene=scene)      # the frame's corners, in the camera's space
    zs = []
    for i in range(DEPTH_GRID[0]):
        for j in range(DEPTH_GRID[1]):
            u, v = (i + 0.5) / DEPTH_GRID[0], (j + 0.5) / DEPTH_GRID[1]
            ray = (m @ (tl + (tr - tl) * u + (bl - tl) * v) - origin).normalized()
            hit, loc, *_ = scene.ray_cast(dg, origin, ray)
            if hit and (z := (loc - origin).dot(fwd)) > 0:
                zs.append(z)
    return zs


def depth_range(scene, first: int, last: int) -> tuple[float, float]:
    """Near and far, in metres along the camera's view, of what the camera actually sees in the shot
    (its first, middle and last frames): the nearest surface is the guide's white. Rays that hit
    nothing (the sky) are left out; with no hit at all, the meshes' bounding boxes decide."""
    cam = scene.camera
    zs: list[float] = []
    for f in sorted({first, (first + last) // 2, last})[:DEPTH_SAMPLES]:
        scene.frame_set(f)
        bpy.context.view_layer.update()
        seen = _seen_depths(scene, cam)
        if not seen:
            fwd = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
            loc = cam.matrix_world.translation
            seen = [z for ob in _meshes() for c in ob.bound_box
                    if (z := (ob.matrix_world @ Vector(c) - loc).dot(fwd)) > 0]
        zs += seen
    if not zs:
        return 0.1, 10.0
    zs.sort()
    lo = zs[int(DEPTH_PERCENTILES[0] * (len(zs) - 1))]
    hi = zs[int(DEPTH_PERCENTILES[1] * (len(zs) - 1))]
    near = max(0.05, lo * 0.95)
    return near, max(near * 1.5, hi * 1.05)


def _render_frames(out: str, first: int, last: int, say, name: str) -> None:
    os.makedirs(out, exist_ok=True)
    sc = bpy.context.scene
    for f in range(first, last + 1):
        t = time.time()
        sc.frame_set(f)
        sc.render.filepath = os.path.join(out, f"{f:04d}.png")
        bpy.ops.render.render(write_still=True)
        say("render.guide", "", guide=name, frame=f, of=last, seconds=round(time.time() - t, 2))


def render_guides(plan: dict, say) -> None:
    """Render every guide the plan asks for into its folder (plan["output"]["guides"][name])."""
    wanted = plan.get("guides") or []
    if not wanted:
        return
    sc = bpy.context.scene
    first, last = plan["frame_range"]
    folders = plan["output"]["guides"]
    view = sc.view_settings
    kept = {"world": sc.world, "view": view.view_transform, "look": view.look, "exposure": view.exposure,
            "gamma": view.gamma}
    dof = [(cam, cam.dof.use_dof) for cam in bpy.data.cameras]
    lights = [o for o in sc.objects if o.type == "LIGHT" and not o.hide_render]
    black_world = bpy.data.worlds.new("guide.world")
    black_world.color = (0.0, 0.0, 0.0)
    for node in (black_world.node_tree.nodes if black_world.node_tree else []):
        if node.type == "BACKGROUND":                   # Blender 5: a new world's nodes give a grey 0.05 sky
            node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
            node.inputs["Strength"].default_value = 0.0
    sc.world = black_world
    view.view_transform, view.look, view.exposure, view.gamma = "Standard", "None", 0.0, 1.0
    for cam, _ in dof:                     # a guide is sharp everywhere: the finish paints its own focus
        cam.dof.use_dof = False
    for o in lights:                       # a guide is all emission: lights would only cost shadow maps
        o.hide_render = True
    try:
        if "depth" in wanted:
            near, far = depth_range(sc, first, last)
            inks = [m for o in _meshes() for m in o.modifiers if m.type == "SOLIDIFY" and m.name == INK]
            for m in inks:                                  # the ink shell would only thicken each shape
                m.show_render = False
            bpy.context.view_layer.material_override = _depth_material(near, far)
            view.view_transform = "Raw"                     # a pixel's grey is the disparity itself
            try:
                say("guide.start", f"depth {near:.2f}-{far:.2f} m", guide="depth",
                    near=round(near, 3), far=round(far, 3))
                _render_frames(folders["depth"], first, last, say, "depth")
            finally:
                view.view_transform = "Standard"
                bpy.context.view_layer.material_override = None
                for m in inks:
                    m.show_render = True
        if "lines" in wanted:
            white, black = _emission("guide.white", (1.0, 1.0, 1.0)), _emission("guide.black", (0.0, 0.0, 0.0))
            swapped: list[tuple] = []
            for ob in _meshes():
                drawn = bool(ob.get("no_ink"))              # face parts are drawn, not modelled: they are lines
                for slot in ob.material_slots:
                    if slot.material is not None and slot.material.name.split(".")[0] == INK:
                        continue                            # recoloured below, keeping its back-face culling
                    swapped.append((slot, slot.material))
                    slot.material = white if drawn else black
            inks_m = [m for m in bpy.data.materials if m.name.split(".")[0] == INK and m.node_tree]
            colours = []
            for m in inks_m:
                emit = next(n for n in m.node_tree.nodes if n.type == "EMISSION")
                colours.append((emit, tuple(emit.inputs["Color"].default_value)))
                emit.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
            try:
                say("guide.start", "lines", guide="lines")
                _render_frames(folders["lines"], first, last, say, "lines")
            finally:
                for emit, c in colours:
                    emit.inputs["Color"].default_value = c
                for slot, mat in swapped:
                    slot.material = mat
    finally:
        sc.world = kept["world"]
        view.view_transform, view.look = kept["view"], kept["look"]
        view.exposure, view.gamma = kept["exposure"], kept["gamma"]
        for cam, use in dof:
            cam.dof.use_dof = use
        for o in lights:
            o.hide_render = False
