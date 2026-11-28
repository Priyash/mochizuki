"""Cel shading and ink lines for EEVEE. Runs inside Blender.

Toon material: the diffuse light is read back as a colour (Shader to RGB), cut into three
flat bands (shadow, mid, lit) by a constant colour ramp, multiplied with the base colour,
plus a rim of light on the edges facing away from the camera; emitted, so what the camera
sees is exactly those flat colours. Ink lines are an inverted hull: a slightly larger shell
with flipped normals and back faces culled, the technique 3D anime uses because it is fast
and stable from frame to frame. Line width is set per object from its distance to the camera,
so lines are the same thickness on screen everywhere.
"""
from __future__ import annotations

import math

import bpy


ALIASES = {"Fac": ("Fac", "Factor"), "Factor": ("Factor", "Fac")}     # Blender 5 renamed Fac to Factor


def _socket(sockets, name: str, kind: str | None = None):
    """A node socket by name (and type), skipping the hidden duplicates Blender keeps per data type."""
    names = ALIASES.get(name, (name,))
    for s in sockets:
        if s.name in names and (kind is None or s.type == kind) and getattr(s, "enabled", True):
            return s
    for s in sockets:
        if s.name in names and (kind is None or s.type == kind):
            return s
    raise KeyError(f"no socket {name} ({kind})")


def _mix(nodes, blend: str, fac: float = 1.0):
    m = nodes.new("ShaderNodeMix")
    m.data_type = "RGBA"
    m.blend_type = blend
    _socket(m.inputs, "Factor", "VALUE").default_value = fac
    return m, _socket(m.inputs, "A", "RGBA"), _socket(m.inputs, "B", "RGBA"), _socket(m.outputs, "Result", "RGBA")


def _rgba(c, a=1.0):
    return (float(c[0]), float(c[1]), float(c[2]), a)


def toon(name: str, base, toon_cfg: dict, emission: float = 0.0, texture: str = "",
         rim: bool = True) -> bpy.types.Material:
    """A cel-shaded material. `texture` adds a simple procedural pattern to the base colour
    ("wood", "tatami", "stone", "paper"). Floors and the sea get no rim: seen at a grazing angle,
    a rim would light the whole floor."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nodes, links = nt.nodes, nt.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    _socket(emit.inputs, "Strength").default_value = 1.0
    links.new(_socket(emit.outputs, "Emission"), _socket(out.inputs, "Surface"))
    base_rgb = nodes.new("ShaderNodeRGB")
    base_rgb.outputs[0].default_value = _rgba(base)
    base_out = base_rgb.outputs[0]
    if texture:
        base_out = _pattern(nodes, links, texture, base, base_out)
    if emission > 0:                                    # lamps, the moon, embers: they light, they are not lit
        links.new(base_out, _socket(emit.inputs, "Color"))
        _socket(emit.inputs, "Strength").default_value = emission
        return mat
    diff = nodes.new("ShaderNodeBsdfDiffuse")
    _socket(diff.inputs, "Color").default_value = (1, 1, 1, 1)
    s2r = nodes.new("ShaderNodeShaderToRGB")
    links.new(_socket(diff.outputs, "BSDF"), _socket(s2r.inputs, "Shader"))
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    bands = toon_cfg.get("bands", [0.42, 0.78])
    mid = float(toon_cfg.get("mid", 0.86))
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, _rgba(toon_cfg.get("shadow_tint", [0.55, 0.55, 0.75]))
    els[1].position, els[1].color = bands[0], (mid, mid, mid, 1)
    top = els.new(bands[1])
    top.color = (1, 1, 1, 1)
    links.new(_socket(s2r.outputs, "Color"), _socket(ramp.inputs, "Fac"))
    mul, a, b, res = _mix(nodes, "MULTIPLY")
    links.new(base_out, a)
    links.new(_socket(ramp.outputs, "Color"), b)
    # keep the light's colour: warm where the lantern reaches, cool where only the moon does
    lum = nodes.new("ShaderNodeRGBToBW")
    links.new(_socket(s2r.outputs, "Color"), _socket(lum.inputs, "Color"))
    div, a3, b3, hue = _mix(nodes, "DIVIDE")
    links.new(_socket(s2r.outputs, "Color"), a3)
    links.new(_socket(lum.outputs, "Val"), b3)
    tint, a4, b4, tinted = _mix(nodes, "MIX", float(toon_cfg.get("light_colour", 0.55)))
    a4.default_value = (1, 1, 1, 1)
    links.new(hue, b4)
    mul2, a5, b5, res5 = _mix(nodes, "MULTIPLY")
    links.new(res, a5)
    links.new(tinted, b5)
    res = res5
    rim_strength = float(toon_cfg.get("rim_strength", 0.25)) if rim else 0.0
    final = res
    if rim_strength > 0:
        lw = nodes.new("ShaderNodeLayerWeight")
        _socket(lw.inputs, "Blend").default_value = 0.35
        rr = nodes.new("ShaderNodeValToRGB")
        rr.color_ramp.interpolation = "CONSTANT"
        rr.color_ramp.elements[0].color = (0, 0, 0, 1)
        rr.color_ramp.elements[1].position = 0.72
        rc = toon_cfg.get("rim", [1, 1, 1])
        rr.color_ramp.elements[1].color = _rgba([x * rim_strength for x in rc])
        links.new(_socket(lw.outputs, "Facing"), _socket(rr.inputs, "Fac"))
        add, a2, b2, res2 = _mix(nodes, "ADD")
        links.new(res, a2)
        links.new(_socket(rr.outputs, "Color"), b2)
        final = res2
    links.new(final, _socket(emit.inputs, "Color"))
    return mat


def _pattern(nodes, links, kind: str, base, base_out):
    """Two tones of the base colour in a simple pattern: planks, tatami weave, stone grain, paper."""
    tc = nodes.new("ShaderNodeTexCoord")
    mp = nodes.new("ShaderNodeMapping")
    links.new(_socket(tc.outputs, "Object"), _socket(mp.inputs, "Vector"))
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    dark = [x * 0.8 for x in base]
    ramp.color_ramp.elements[0].color = _rgba(base)
    ramp.color_ramp.elements[1].position = 0.55
    ramp.color_ramp.elements[1].color = _rgba(dark)
    if kind in ("wood", "tatami"):
        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "X" if kind == "wood" else "Y"
        _socket(wave.inputs, "Scale").default_value = 2.2 if kind == "wood" else 9.0
        _socket(wave.inputs, "Distortion").default_value = 0.6 if kind == "wood" else 0.0
        links.new(_socket(mp.outputs, "Vector"), _socket(wave.inputs, "Vector"))
        links.new(_socket(wave.outputs, "Fac"), _socket(ramp.inputs, "Fac"))
    else:
        noise = nodes.new("ShaderNodeTexNoise")
        _socket(noise.inputs, "Scale").default_value = 6.0 if kind == "stone" else 2.0
        links.new(_socket(mp.outputs, "Vector"), _socket(noise.inputs, "Vector"))
        links.new(_socket(noise.outputs, "Fac"), _socket(ramp.inputs, "Fac"))
    return _socket(ramp.outputs, "Color")


def outline_material(color) -> bpy.types.Material:
    mat = bpy.data.materials.new("ink")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emit = nodes.new("ShaderNodeEmission")
    _socket(emit.inputs, "Color").default_value = _rgba(color)
    mat.node_tree.links.new(_socket(emit.outputs, "Emission"), _socket(out.inputs, "Surface"))
    mat.use_backface_culling = True
    return mat


def add_outline(obj, ink, width_m: float) -> None:
    """Inverted-hull ink line on a mesh object."""
    if obj.type != "MESH" or width_m <= 0:
        return
    obj.data.materials.append(ink)
    mod = obj.modifiers.new("ink", "SOLIDIFY")
    mod.thickness = width_m / max(1e-6, max(obj.matrix_world.to_scale()))
    mod.offset = 1.0
    mod.use_flip_normals = True
    mod.use_rim = False
    mod.material_offset = len(obj.data.materials) - 1
    mod.use_even_offset = False


def line_width(obj_loc, cam_loc, vfov: float, res_y: int, px: float) -> float:
    d = math.dist(obj_loc, cam_loc)
    return 2 * d * math.tan(vfov / 2) * px / max(1, res_y)


def world(rig: dict, blackout: bool = False) -> None:
    """The background the camera sees, and a dimmer version of it that lights the set."""
    w = bpy.data.worlds.new("sky")
    bpy.context.scene.world = w
    w.use_nodes = True
    nodes, links = w.node_tree.nodes, w.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputWorld")
    col = (0, 0, 0) if blackout else rig.get("world", [0.1, 0.1, 0.12])
    seen = nodes.new("ShaderNodeBackground")
    _socket(seen.inputs, "Color").default_value = _rgba(col)
    lit = nodes.new("ShaderNodeBackground")
    _socket(lit.inputs, "Color").default_value = _rgba(col)
    _socket(lit.inputs, "Strength").default_value = 0.35
    lp = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(_socket(lp.outputs, "Is Camera Ray"), mix.inputs[0])
    links.new(_socket(lit.outputs, "Background"), mix.inputs[1])
    links.new(_socket(seen.outputs, "Background"), mix.inputs[2])
    links.new(mix.outputs[0], _socket(out.inputs, "Surface"))
