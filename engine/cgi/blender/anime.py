"""Procedural anime characters: bodies, faces and hair made from code, no modelling. Runs inside Blender.

The plan gives a skeleton (engine/cgi/figure.py: joints for the character's age and posture)
and a look (age, hair style, colours). This builds, on those joints:

  * a body of clothes from the Skin modifier (a jacket with sleeves, work trousers, a sash, a
    collar), hands, socks, and a neck;
  * an anime head: a sphere with a narrowing jaw and a small chin, the face a little flat;
  * a drawn face on it, the way anime CG does it: flat unlit eyes, brows, mouth, blush and tears
    laid on the surface, never shaded or outlined. One shader draws both eyes: white, an iris
    that darkens towards the lid, a pupil, two catchlights, all clipped to the eye's shape, so
    the gaze moves inside the eye;
  * hair in clumps: a cap, bangs, side locks, and a bob, long hair or a bun.

Everything the face does is keyed from the plan's channels, the same ones a VRoid model plays
(Fcl_EYE_*, Fcl_BRW_*, Fcl_MTH_*, anime.*, head.*, eyes.*), so a character can be swapped for a
modelled one without changing a shot.
"""
from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Vector

from . import props, toon
from .keys import ch, key_curve, mix

SKIN = {"child": [1.0, 0.84, 0.74], "adult": [1.0, 0.83, 0.73], "old": [0.95, 0.8, 0.71]}
WHITE = [0.97, 0.97, 0.99]
LINE = [0.10, 0.06, 0.07]


def _band(age: float) -> str:
    return "child" if age < 13 else "old" if age >= 60 else "adult"


def look_for(c: dict) -> dict:
    """The character's look, with defaults by age."""
    lk = dict(c.get("look") or {})
    age = float(lk.get("age", c.get("figure", {}).get("age", 30)))
    band = _band(age)
    lk.setdefault("age", age)
    lk.setdefault("hair_style", {"child": "bob", "adult": "long", "old": "bun"}[band])
    lk.setdefault("hair_colour", c.get("hair", [0.08, 0.07, 0.09]))
    lk.setdefault("jacket", c.get("colour", [0.25, 0.3, 0.5]))
    lk.setdefault("trousers", [x * 0.55 for x in lk["jacket"]])
    lk.setdefault("sash", [0.62, 0.24, 0.18])
    lk.setdefault("eyes", [0.30, 0.16, 0.10])
    lk.setdefault("skin", SKIN[band])
    return lk


# ---------------------------------------------------------------- materials
def flat(name: str, colour, strength: float = 1.0) -> bpy.types.Material:
    """Unlit: drawn, never shaded, never outlined (blender/build.py skips emission-only materials)."""
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    em = nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*colour[:3], 1.0)
    em.inputs["Strength"].default_value = strength
    links.new(em.outputs[0], out.inputs["Surface"])
    return mat


class _N:
    """Small helper to wire shader math."""

    def __init__(self, mat):
        self.nodes, self.links = mat.node_tree.nodes, mat.node_tree.links

    def _in(self, sock, v):
        if isinstance(v, (int, float)):
            sock.default_value = float(v)
        else:
            self.links.new(v, sock)

    def math(self, op: str, a, b=0.0):
        n = self.nodes.new("ShaderNodeMath")
        n.operation = op
        self._in(n.inputs[0], a)
        self._in(n.inputs[1], b)
        return n.outputs[0]

    def value(self, v: float = 0.0):
        n = self.nodes.new("ShaderNodeValue")
        n.outputs[0].default_value = v
        return n

    def ellipse(self, u, v, cu, cv, ru, rv):
        """1 inside the ellipse, 0 outside."""
        du = self.math("DIVIDE", self.math("SUBTRACT", u, cu), ru)
        dv = self.math("DIVIDE", self.math("SUBTRACT", v, cv), rv)
        d2 = self.math("ADD", self.math("MULTIPLY", du, du), self.math("MULTIPLY", dv, dv))
        return self.math("LESS_THAN", d2, 1.0), d2

    def mix(self, fac, a, b):
        m = self.nodes.new("ShaderNodeMix")
        m.data_type = "RGBA"
        self._in(m.inputs[0], fac)
        sa = next(s for s in m.inputs if s.name == "A" and s.type == "RGBA")
        sb = next(s for s in m.inputs if s.name == "B" and s.type == "RGBA")
        for sock, val in ((sa, a), (sb, b)):
            if isinstance(val, (list, tuple)):
                sock.default_value = (*val[:3], 1.0)
            else:
                self.links.new(val, sock)
        return next(s for s in m.outputs if s.type == "RGBA")


def drawn(kit, key: str, colour) -> bpy.types.Material:
    """A drawn part of the face: lit like the skin around it (its shadow band too), no rim, no ink line."""
    return kit.mat(key, colour, rim=False)


def eye_material(kit, name: str, iris) -> tuple[bpy.types.Material, dict]:
    """Both eyes' shader. Returns the material and its animated values: gaze_u, gaze_v (iris
    offset, in the eye's UV), highlight (catchlight size), pupil (pupil size). The eye is lit
    like the face, so in the shadow band or at night it darkens with the skin."""
    mat = toon.toon(name, [1.0, 1.0, 1.0], kit.toon_cfg, rim=False)
    n = _N(mat)
    rgb = next(nd for nd in n.nodes if nd.type == "RGB")
    target = next(lk.to_socket for lk in n.links if lk.from_node == rgb)
    tc = n.nodes.new("ShaderNodeTexCoord")
    sep = n.nodes.new("ShaderNodeSeparateXYZ")
    n.links.new(tc.outputs["UV"], sep.inputs[0])
    u, v = sep.outputs[0], sep.outputs[1]
    vals = {k: n.value(d) for k, d in (("gaze_u", 0.0), ("gaze_v", 0.0), ("highlight", 0.0), ("pupil", 1.0))}
    cu = n.math("ADD", vals["gaze_u"].outputs[0], 0.5)
    cv = n.math("ADD", vals["gaze_v"].outputs[0], 0.44)
    iris_in, d2 = n.ellipse(u, v, cu, cv, 0.31, 0.43)
    # the iris: dark under the lid, lighter and warmer below, a dark rim
    dark = [x * 0.35 for x in iris]
    light = [min(1.0, x * 1.9 + 0.08) for x in iris]
    grad = n.math("MULTIPLY", n.math("SUBTRACT", v, n.math("SUBTRACT", cv, 0.43)), 1.0 / 0.86)
    col = n.mix(n.math("POWER", n.math("SUBTRACT", 1.0, n.math("MINIMUM", n.math("MAXIMUM", grad, 0.0), 1.0)), 1.6),
                dark, light)
    rim = n.math("GREATER_THAN", d2, 0.72)
    col = n.mix(rim, col, dark)
    pr = n.math("MULTIPLY", vals["pupil"].outputs[0], 0.13)
    pupil_in, _ = n.ellipse(u, v, cu, n.math("ADD", cv, 0.03), pr, n.math("MULTIPLY", pr, 1.45))
    col = n.mix(pupil_in, col, [x * 0.12 for x in iris])
    # the white, with the upper lid's shadow on it
    white = n.mix(n.math("GREATER_THAN", v, 0.76), WHITE, [0.72, 0.74, 0.86])
    col = n.mix(iris_in, white, col)
    # catchlights: one large, one small, both bigger when the eyes shine
    k = n.math("ADD", 0.75, n.math("MULTIPLY", vals["highlight"].outputs[0], 0.7))
    big, _ = n.ellipse(u, v, n.math("SUBTRACT", cu, 0.12), n.math("ADD", cv, 0.17),
                       n.math("MULTIPLY", k, 0.085), n.math("MULTIPLY", k, 0.11))
    small, _ = n.ellipse(u, v, n.math("ADD", cu, 0.12), n.math("SUBTRACT", cv, 0.2),
                         n.math("MULTIPLY", k, 0.04), n.math("MULTIPLY", k, 0.045))
    col = n.mix(n.math("MAXIMUM", big, small), col, [1.0, 1.0, 1.0])
    n.links.new(col, target)
    return mat, vals


# ---------------------------------------------------------------- geometry helpers
def _object(name: str, bm, mats, parent=None, smooth: bool = True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    for poly in me.polygons:
        poly.use_smooth = smooth
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def skin_body(name: str, verts: list, edges: list, radii: list, mat, parent, levels: int = 2):
    """A smooth, branching tube along joints (the Skin modifier), e.g. a jacket with sleeves."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], edges, [])
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    skin = ob.modifiers.new("skin", "SKIN")
    skin.use_smooth_shade = True
    for i, r in enumerate(radii):
        sv = me.skin_vertices[0].data[i]
        sv.radius = (float(r[0]), float(r[1]))
        sv.use_root = i == 0
    sub = ob.modifiers.new("smooth", "SUBSURF")
    sub.levels = levels
    sub.render_levels = levels
    return ob


class Head:
    """The skull as a function, so the face can be laid exactly on it."""

    def __init__(self, size, chin: float):
        self.rx, self.ry, self.rz = size[0] / 2, size[1] / 2, size[2] / 2
        self.chin = chin

    def shape(self, x: float, y: float, z: float) -> Vector:
        t = min(1.0, max(0.0, (-z / self.rz - 0.1) / 0.9))     # 0 down to the cheekbones, 1 at the chin
        front = max(0.0, -y / self.ry)
        k = 1.0 - self.chin * t ** 1.5                          # the jaw narrows
        return Vector((x * k, y * (1.0 - 0.1 * t) * (1.0 - 0.07 * front * (1 - t)),
                       z - self.rz * 0.22 * self.chin * t * t * front))

    def front(self, x: float, z: float, lift: float = 0.0, scale: float = 1.0) -> Vector:
        """The point of the face at (x, z) (as on the round skull before the jaw), lifted off the skin."""
        rx, ry, rz = self.rx * scale, self.ry * scale, self.rz * scale
        q = 1.0 - (x / rx) ** 2 - (z / rz) ** 2
        y = -ry * math.sqrt(max(0.0, q))
        p = self.shape(x, y, z)
        nrm = Vector((x / rx ** 2, y / ry ** 2, z / rz ** 2)).normalized()
        return p + nrm * lift

    def light_normal(self, x: float, y: float, z: float) -> Vector:
        """The normal the face is lit with: the round skull's, eased towards facing forward on the face,
        so light falls on it in one clean shape whatever the jaw does (anime CG's edited face normals)."""
        n = Vector((x / self.rx ** 2, y / self.ry ** 2, z / self.rz ** 2)).normalized()
        front = max(0.0, -y / self.ry)
        fwd = Vector((n.x * 0.6, -1.0, n.z * 0.5 + 0.1)).normalized()
        return n.lerp(fwd, 0.45 * front ** 2).normalized()

    def around(self, angle: float, z: float, scale: float = 1.0) -> Vector:
        """A point on the skull at `angle` around it (0 = the front, + towards its left) and height z."""
        rx, ry, rz = self.rx * scale, self.ry * scale, self.rz * scale
        r = math.sqrt(max(0.0, 1.0 - (z / rz) ** 2))
        return self.shape(rx * r * math.sin(angle), -ry * r * math.cos(angle), z)

    def mesh(self, name: str, mat, parent, scale: float = 1.0):
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=28, radius=1.0)
        normals = []
        for vt in bm.verts:
            x, y, z = vt.co.x * self.rx * scale, vt.co.y * self.ry * scale, vt.co.z * self.rz * scale
            normals.append(self.light_normal(x / scale, y / scale, z / scale))
            vt.co = self.shape(x, y, z)
        ob = _object(name, bm, [mat], parent)
        ob.data.normals_split_custom_set_from_vertices([tuple(n) for n in normals])
        return ob


def face_band(name: str, head: Head, upper: list, lower: list, mat, parent, lift: float, anchor=None):
    """A drawn stroke on the face between two edges of equal length (a lid line, a brow, a crease)."""
    return face_shape(name, head, upper + lower[::-1], mat, parent, lift, anchor=anchor, fan=True)


def face_shape(name: str, head: Head, pts: list, mat, parent, lift: float, anchor=None, uv_box=None,
               fan: bool = False):
    """A flat drawn shape on the face: `pts` is its outline (x, z) in head space, laid on the skin.
    `anchor` is the point it scales and turns about. With `fan`, vertex 0 is the shape's centre and
    1.. its outline (for shapes that change, like the mouth); otherwise the outline is triangulated."""
    cx, cz = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    ax, az = anchor if anchor else (cx, cz)
    origin = head.front(ax, az, lift)
    bm = bmesh.new()
    coords = {}
    rim = []
    for x, z in pts:
        vt = bm.verts.new(head.front(x, z, lift) - origin)
        coords[vt] = (x, z)
        rim.append(vt)
    if fan:                                                # a "ladder" across the shape: right for any shape
        n = len(rim)                                       # a line across it cuts twice, however it bends
        if n % 2:                                          # (a mouth that smiles, a lid line, an arch)
            raise ValueError(f"{name}: a ladder needs an even outline")
        half = n // 2
        for k in range(half - 1):
            bm.faces.new((rim[k], rim[k + 1], rim[n - k - 2], rim[n - k - 1]))
    else:
        face = bm.faces.new(rim)
        bmesh.ops.triangulate(bm, faces=[face], quad_method="BEAUTY", ngon_method="EAR_CLIP")
    if uv_box:
        x0, z0, x1, z1 = uv_box
        uvl = bm.loops.layers.uv.new("UVMap")
        for f in bm.faces:
            for lp in f.loops:
                x, z = coords[lp.vert]
                lp[uvl].uv = ((x - x0) / (x1 - x0), (z - z0) / (z1 - z0))
    for f in bm.faces:
        f.normal_update()
        if f.normal.y > 0:                                 # face outwards, off the skin
            f.normal_flip()
    order = [coords[v] for v in bm.verts]                  # vertex order of the mesh, before bmesh is freed
    ob = _object(name, bm, [mat], parent, smooth=True)
    ob.location = origin
    ob["no_ink"] = True
    nrm = []
    for x, z in order:
        q = 1.0 - (x / head.rx) ** 2 - (z / head.rz) ** 2
        nrm.append(tuple(head.light_normal(x, -head.ry * math.sqrt(max(0.0, q)), z)))
    ob.data.normals_split_custom_set_from_vertices(nrm)
    return ob


def lock(name: str, path: list, widths: list, flat_k: float, centre: Vector, mat, parent, ring: int = 8):
    """A clump of hair: a tube along `path`, wide along the skull and thin away from it, tapering to a point."""
    bm = bmesh.new()
    rings = []
    n = len(path)
    for i, p in enumerate(path):
        t = (path[min(n - 1, i + 1)] - path[max(0, i - 1)]).normalized()
        out = p - centre
        out = (out - t * out.dot(t)).normalized()
        b = t.cross(out).normalized()
        w = widths[i] / 2
        if w < 1e-5:
            rings.append([bm.verts.new(p)])
            continue
        rings.append([bm.verts.new(p + b * (w * math.cos(a)) + out * (w * flat_k * math.sin(a)))
                      for a in (math.tau * k / ring for k in range(ring))])
    for r0, r1 in zip(rings, rings[1:]):
        if len(r1) == 1:
            for k in range(len(r0)):
                bm.faces.new((r0[k], r0[(k + 1) % len(r0)], r1[0]))
        elif len(r0) == len(r1):
            for k in range(len(r0)):
                bm.faces.new((r0[k], r0[(k + 1) % len(r0)], r1[(k + 1) % len(r1)], r1[k]))
    if len(rings[0]) > 2:
        bm.faces.new(rings[0][::-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return _object(name, bm, [mat], parent)


def _taper(n: int, root: float, bulge: float = 0.25) -> list:
    return [root * max(0.0, (1 - (i / (n - 1)) ** 1.4)) * (1 + bulge * math.sin(math.pi * i / (n - 1)))
            for i in range(n)]


def _lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------- the character
FACE = {   # in head heights, from the head's centre; x towards the character's left
    "child": {"eye_x": 0.19, "eye_z": -0.1, "eye_w": 0.18, "eye_h": 0.21, "brow": 0.055, "mouth_z": -0.31,
              "mouth_w": 0.8, "nose_z": -0.2, "chin": 0.2},
    "adult": {"eye_x": 0.175, "eye_z": -0.07, "eye_w": 0.155, "eye_h": 0.15, "brow": 0.055, "mouth_z": -0.33,
              "mouth_w": 1.0, "nose_z": -0.19, "chin": 0.36},
    "old": {"eye_x": 0.175, "eye_z": -0.07, "eye_w": 0.145, "eye_h": 0.09, "brow": 0.05, "mouth_z": -0.32,
            "mouth_w": 1.0, "nose_z": -0.19, "chin": 0.2},
}
MOUTH = {  # half width, top, bottom, corner lift (in head heights)
    "basis": (0.045, 0.004, 0.005, 0.0), "Fcl_MTH_A": (0.040, 0.012, 0.056, 0.006), "Fcl_MTH_I": (0.055, 0.006, 0.02, 0.004),
    "Fcl_MTH_U": (0.022, 0.016, 0.02, 0.0), "Fcl_MTH_E": (0.048, 0.01, 0.032, 0.004), "Fcl_MTH_O": (0.03, 0.022, 0.034, 0.0),
    "Fcl_MTH_Joy": (0.056, 0.004, 0.007, 0.02), "Fcl_MTH_Sorrow": (0.042, 0.003, 0.004, -0.016),
    "Fcl_MTH_Surprised": (0.026, 0.02, 0.03, 0.0), "Fcl_MTH_Angry": (0.046, 0.004, 0.009, -0.009),
}


def _eye_outline(cx: float, cz: float, w: float, h: float, side: int, n: int = 28) -> list:
    pts = []
    for k in range(n):
        a = math.tau * k / n
        c, s = math.cos(a), math.sin(a)
        x = (w / 2) * math.copysign(abs(c) ** 0.78, c)
        z = (h / 2) * math.copysign(abs(s) ** 0.78, s)
        rel = x / (w / 2)                                    # -1 inner .. +1 outer, before mirroring
        if z > 0:
            z = z * 0.92 + 0.07 * h * rel                    # the top rises to the outer corner
        else:
            z = z * 0.82
        pts.append((cx + side * x, cz + z))
    return pts


def _edge(pts: list, top: bool) -> list:
    """The upper (or lower) edge of an outline, sorted along x."""
    cz = sum(p[1] for p in pts) / len(pts)
    e = [p for p in pts if (p[1] >= cz) == top]
    return sorted(e, key=lambda p: p[0])


def figure(kit: props.Kit, c: dict, frames: int):
    fig = c["figure"]
    lk = look_for(c)
    age = lk["age"]
    band = _band(age)
    H = fig["H"]
    j = {k: Vector(v) for k, v in fig["joints"].items()}
    name = c["name"]
    root = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(root)
    skin_m = kit.mat(f"{name}.skin", lk["skin"])
    jacket = kit.mat(f"{name}.jacket", lk["jacket"])
    trousers = kit.mat(f"{name}.trousers", lk["trousers"])
    sash = kit.mat(f"{name}.sash", lk["sash"])
    white = kit.mat("tabi", [0.93, 0.93, 0.9])
    hair_m = kit.mat(f"{name}.hair", lk["hair_colour"])
    kid = band == "child"

    # ---- the body: jacket with sleeves, trousers, sash, collar, neck, hands, feet
    r = (lambda a, b=None: (a * H, (b if b is not None else a) * H))
    hem = j["pelvis"] + Vector((0, 0, -0.28 * H))
    verts = [hem, j["pelvis"], j["waist"], j["chest"], j["neck"],
             j["shoulder_L"], j["elbow_L"], j["wrist_L"], j["shoulder_R"], j["elbow_R"], j["wrist_R"]]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7), (3, 8), (8, 9), (9, 10)]
    if kid:
        radii = [r(0.46, 0.36), r(0.4, 0.31), r(0.36, 0.28), r(0.4, 0.3), r(0.18, 0.16),
                 r(0.15), r(0.125), r(0.13), r(0.15), r(0.125), r(0.13)]
    elif band == "old":                                     # rounder, softer: a grandmother's work clothes
        radii = [r(0.6, 0.46), r(0.56, 0.42), r(0.5, 0.38), r(0.52, 0.4), r(0.17, 0.15),
                 r(0.17), r(0.15), r(0.17), r(0.17), r(0.15), r(0.17)]
    else:
        radii = [r(0.5, 0.38), r(0.46, 0.33), r(0.38, 0.28), r(0.47, 0.33), r(0.16, 0.14),
                 r(0.16), r(0.13), r(0.135), r(0.16), r(0.13), r(0.135)]
    skin_body(f"{name}.jacket", verts, edges, radii, jacket, root)
    legs = [j["pelvis"] + Vector((0, 0, -0.05 * H)), j["hip_L"], j["knee_L"], j["ankle_L"], j["hip_R"], j["knee_R"], j["ankle_R"]]
    lr = [r(0.36, 0.28), r(0.2), r(0.16), r(0.125), r(0.2), r(0.16), r(0.125)] if kid else \
        [r(0.42, 0.31), r(0.23), r(0.17), r(0.13), r(0.23), r(0.17), r(0.13)]
    skin_body(f"{name}.trousers", legs, [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6)], lr, trousers, root)
    if lk.get("apron") and fig["posture"] in ("standing", "walking"):   # a work apron, waist to knees
        top_z = j["waist"].z - 0.02 * H
        knee_z = (j["knee_L"].z + j["knee_R"].z) / 2
        drop = max(0.3 * H, top_z - knee_z - 0.1 * H) if fig["posture"] not in ("kneeling", "seiza") else 0.35 * H
        apron = props.box(f"{name}.apron", (radii[2][0] * 2.1, 0.02 * H, drop), [kit.mat(f"{name}.apron", lk["apron"])],
                          bevel=0.01 * H)
        apron.location = (0, j["waist"].y - radii[2][1] * 0.92, top_z - drop / 2)
        apron.parent = root
    belt = props.lathe(f"{name}.sash", [(radii[2][0] * 1.06, -0.07 * H), (radii[2][0] * 1.08, 0.0),
                                        (radii[2][0] * 1.06, 0.07 * H)], [sash], steps=32)
    belt.location = j["waist"] + Vector((0, 0, -0.04 * H))
    belt.scale = (1.0, radii[2][1] / radii[2][0], 1.0)
    belt.parent = root
    collar = props.lathe(f"{name}.collar", [(0.15 * H, -0.03 * H), (0.17 * H, 0.0), (0.15 * H, 0.04 * H)],
                         [kit.mat("collar", WHITE)], steps=28)
    collar.location = j["neck"] + Vector((0, -0.01 * H, -0.02 * H))
    collar.parent = root
    skin_body(f"{name}.neck", [j["neck"] + Vector((0, 0, -0.05 * H)), j["head_base"]], [(0, 1)],
              [r(0.12 if kid else 0.095), r(0.11 if kid else 0.09)], skin_m, root, levels=1)
    for side in ("L", "R"):
        wr, el = j[f"wrist_{side}"], j[f"elbow_{side}"]
        d = (wr - el).normalized()
        sx = 1 if side == "L" else -1
        across = d.cross(Vector((0, 0, 1)))
        across = across.normalized() if across.length > 1e-4 else Vector((sx, 0, 0))
        palm = wr + d * 0.16 * H
        tip = wr + d * 0.36 * H + Vector((0, 0, -0.03 * H))
        thumb = palm + across * (sx * 0.08 * H) + d * 0.06 * H
        skin_body(f"{name}.hand{side}", [wr - d * 0.04 * H, palm, tip, thumb], [(0, 1), (1, 2), (1, 3)],
                  [r(0.055, 0.045), r(0.075, 0.04), r(0.055, 0.03), r(0.028)], skin_m, root, levels=2)
        an, toe = j[f"ankle_{side}"], j[f"toe_{side}"]
        skin_body(f"{name}.foot{side}", [an + Vector((0, 0, 0.03 * H)), toe], [(0, 1)],
                  [r(0.085, 0.07), r(0.075, 0.05)], white, root, levels=2)

    # ---- the head on its neck
    pivot = bpy.data.objects.new(f"{name}.neckpivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.location = j["head_base"]
    pivot.parent = root
    hc = Vector(fig["head"]["center"])
    f = FACE[band]
    head = Head(fig["head"]["size"], f["chin"])
    head_ob = head.mesh(f"{name}.head", skin_m, pivot)
    head_ob.location = hc - j["head_base"]
    for side, sx in (("L", 1), ("R", -1)):
        ear = props.sphere(f"{name}.ear{side}", 0.07 * H, [skin_m], scale=(0.45, 0.8, 1.0), segments=12)
        ear.location = (sx * head.rx * 0.97, 0.05 * H, -0.06 * H)
        ear.parent = head_ob
    _face(kit, c, lk, head, head_ob, H, f, frames, band)
    _hair(kit, lk, head, head_ob, hair_m, H, band)
    face = c.get("face", {})
    key_curve(pivot, "rotation_euler", [-x for x in ch(face, "head.pitch", frames)], index=0)
    key_curve(pivot, "rotation_euler", ch(face, "head.yaw", frames), index=2)
    key_curve(pivot, "rotation_euler", ch(face, "head.roll", frames), index=1)
    for ob in root.children_recursive:                    # lit and shadowed by the set, but casting no shadow:
        if ob.type == "MESH":                             # a toon face shadowing itself turns to grain
            ob.visible_shadow = False
    return root


def _face(kit, c, lk, head: Head, head_ob, H: float, f: dict, frames: int, band: str):
    name = c["name"]
    face = c.get("face", {})
    fr = frames
    lift = 0.006 * H
    eye_mat, ev = eye_material(kit, f"{name}.eyes", lk["eyes"])
    line = drawn(kit, f"{name}.line", LINE)
    brow_col = [x * 0.6 for x in lk["hair_colour"]] if max(lk["hair_colour"]) < 0.5 else [0.42, 0.4, 0.42]
    brow_m = drawn(kit, f"{name}.brow", brow_col)
    w, h = f["eye_w"] * H, f["eye_h"] * H
    ex, ez = f["eye_x"] * H, f["eye_z"] * H
    close = mix(ch(face, "Fcl_EYE_Close", fr), [(a + b) / 2 for a, b in zip(ch(face, "Fcl_EYE_Close_L", fr),
                                                                          ch(face, "Fcl_EYE_Close_R", fr))])
    joy = ch(face, "Fcl_EYE_Joy", fr)
    wide = mix(ch(face, "Fcl_EYE_Surprised", fr), ch(face, "Fcl_EYE_Spread", fr))
    sad_e = ch(face, "Fcl_EYE_Sorrow", fr)
    ang_e = ch(face, "Fcl_EYE_Angry", fr)
    arch = [max(0.0, min(1.0, (jv - 0.45) / 0.3)) for jv in joy]
    opened = [max(0.0, min(1.0, 1.0 - cl - 0.25 * s - 0.15 * a)) * (1.0 - ar) for cl, s, a, ar in zip(close, sad_e, ang_e, arch)]
    for side, sx in (("L", 1), ("R", -1)):
        cx = sx * ex
        outline = _eye_outline(cx, ez, w, h, sx)
        x0, x1 = cx - w / 2, cx + w / 2
        anchor_z = ez - 0.18 * h
        white = face_shape(f"{name}.eye{side}", head, outline, eye_mat, head_ob, lift, anchor=(cx, anchor_z),
                           uv_box=(x0, ez - h / 2, x1, ez + h / 2))
        top = _edge(outline, True)
        thick = [(0.016 + 0.018 * max(0.0, sx * (x - cx) / (w / 2))) * H for x, _ in top]
        upper = [(x, z + t) for (x, z), t in zip(top, thick)]
        lower = [(x, z - 0.004 * H) for x, z in top]
        outer = top[-1] if sx > 0 else top[0]
        tip, base = (outer[0] + sx * 0.035 * H, outer[1] + 0.03 * H), (outer[0] + sx * 0.014 * H, outer[1] - 0.002 * H)
        if sx > 0:                                         # the lash flick at the outer corner
            upper, lower = upper + [tip], lower + [base]
        else:
            upper, lower = [tip] + upper, [base] + lower
        lid = face_band(f"{name}.lid{side}", head, upper, lower, line, head_ob, lift * 1.6, anchor=(cx, anchor_z))
        bottom = _edge(outline, False)
        part = [p for p in bottom if sx * (p[0] - cx) > 0.05 * w]
        lash = face_band(f"{name}.lash{side}", head, part, [(x, z - 0.01 * H) for x, z in part], line, head_ob, lift * 1.4,
                         anchor=(cx, anchor_z))
        arc = sorted([(cx + sx * w * 0.45 * t, ez - 0.05 * h + 0.32 * h * (1 - abs(t) ** 1.7))
                      for t in [i / 10 - 1 for i in range(21)]], key=lambda p: p[0])
        happy = face_band(f"{name}.happy{side}", head, [(x, z + 0.02 * H) for x, z in arc], arc, line, head_ob,
                          lift * 1.6, anchor=(cx, ez))
        # keys: the white opens and closes over its lower lid; the lid line follows the top; joy turns the eye to an arch
        k_open = [max(0.03, o) * (1 + 0.12 * wd) for o, wd in zip(opened, wide)]
        key_curve(white, "scale", [0.001 if ar > 0.6 else k for k, ar in zip(k_open, arch)], index=2)
        key_curve(white, "scale", [1 + 0.1 * wd for wd in wide], index=0)
        rise = (ez + h / 2) - anchor_z
        key_curve(lid, "location", [lid.location.z - rise * (1 - o) for o in k_open], index=2)
        key_curve(lid, "rotation_euler", [sx * (0.22 * s - 0.2 * a) for s, a in zip(sad_e, ang_e)], index=1)
        key_curve(lid, "scale", [max(0.001, 1.0 - ar) for ar in arch], index=0)
        key_curve(lid, "scale", [0.55 + 0.45 * min(1.0, o / 0.4) for o in k_open], index=2)
        key_curve(lash, "scale", [0.001 if o < 0.2 else min(1.0, o / 0.5) for o in k_open], index=0)
        key_curve(happy, "scale", [max(0.001, ar) for ar in arch], index=0)
        key_curve(happy, "scale", [max(0.001, ar) for ar in arch], index=2)
        # brows
        bz = ez + h / 2 + f["brow"] * H
        bpts = [(cx + sx * w * (t * 0.55 + 0.05), bz + 0.025 * H * (1 - t * t)) for t in [i / 8 - 1 for i in range(17)]]
        bpts = sorted(bpts, key=lambda p: p[0])
        bth = [(0.024 - 0.012 * max(0.0, sx * (x - cx) / w)) * H for x, _ in bpts]
        brow = face_band(f"{name}.brow{side}", head, [(x, z + t / 2) for (x, z), t in zip(bpts, bth)],
                         [(x, z - t / 2) for (x, z), t in zip(bpts, bth)], brow_m, head_ob, lift, anchor=(cx, bz))
        sad = ch(face, "Fcl_BRW_Sorrow", fr)
        angry = ch(face, "Fcl_BRW_Angry", fr)
        up = mix(ch(face, "Fcl_BRW_Surprised", fr), [0.4 * x for x in ch(face, "Fcl_BRW_Joy", fr)])
        key_curve(brow, "rotation_euler", [sx * (0.38 * s - 0.32 * a) for s, a in zip(sad, angry)], index=1)
        key_curve(brow, "location", [brow.location.z + H * (0.045 * u + 0.012 * s - 0.018 * a) for u, s, a in zip(up, sad, angry)],
                  index=2)
        # blush and tears
        blush = face_shape(f"{name}.blush{side}", head,
                           [(cx + sx * 0.01 * H + 0.05 * H * math.cos(a), ez - 0.16 * H + 0.022 * H * math.sin(a))
                            for a in (math.tau * k / 16 for k in range(16))],
                           drawn(kit, f"{name}.blushc", [1.0, 0.62, 0.64]), head_ob, lift * 0.8)
        base = 0.3 if band == "child" else 0.0
        key_curve(blush, "scale", [max(0.001, min(1.0, base + 1.2 * b)) for b in ch(face, "anime.blush", fr)], index=0)
        key_curve(blush, "scale", [max(0.001, min(1.0, base + 1.2 * b)) for b in ch(face, "anime.blush", fr)], index=2)
        tx, tz = cx + sx * 0.18 * w, ez - h * 0.48
        tear = face_shape(f"{name}.tear{side}", head,
                          [(tx + 0.014 * H * math.sin(a) * (1 - 0.6 * max(0.0, math.cos(a))), tz - 0.03 * H * (0.5 - 0.5 * math.cos(a)))
                           for a in (math.tau * k / 14 for k in range(14))],
                          drawn(kit, f"{name}.tearc", [0.78, 0.92, 1.0]), head_ob, lift * 2.0, anchor=(tx, tz))
        tears = ch(face, "anime.tear", fr)
        shown = [t if t > 0.2 else 0.0 for t in tears]      # a glint of feeling is in the eyes, not a drop
        key_curve(tear, "scale", [max(0.001, min(1.0, 1.4 * t)) for t in shown], index=0)
        key_curve(tear, "scale", [max(0.001, min(2.6, 3.0 * t)) for t in shown], index=2)
        if band == "old":                                    # the lines a long life leaves at the eyes
            for k, (dz, ang) in enumerate(((0.012, 0.35), (-0.012, -0.35))):
                ox, oz = cx + sx * (w / 2 + 0.02 * H), ez + dz * H
                seg = [(ox + sx * 0.03 * H * math.cos(ang) * t, oz + 0.03 * H * math.sin(ang) * t) for t in (0.0, 0.5, 1.0)]
                face_band(f"{name}.crow{side}{k}", head, seg, [(x, z - 0.006 * H) for x, z in seg],
                          drawn(kit, f"{name}.wrinkle", [x * 0.72 for x in lk["skin"]]), head_ob, lift)
    # gaze: the iris moves inside the eye
    gu = [max(-0.17, min(0.17, 0.42 * y)) for y in ch(face, "eyes.yaw", fr)]
    gv = [max(-0.12, min(0.12, 0.35 * p)) for p in ch(face, "eyes.pitch", fr)]
    hl = mix(ch(face, "anime.highlight", fr), [0.6 * t for t in ch(face, "anime.tear", fr)])
    key_curve(ev["gaze_u"].outputs[0], "default_value", gu)
    key_curve(ev["gaze_v"].outputs[0], "default_value", gv)
    key_curve(ev["highlight"].outputs[0], "default_value", hl)
    key_curve(ev["pupil"].outputs[0], "default_value", [1.0 - 0.35 * x for x in wide])
    # nose: a small tick of shadow
    nz = f["nose_z"] * H
    face_shape(f"{name}.nose", head, [(0.012 * H, nz), (0.0, nz - 0.014 * H), (-0.006 * H, nz - 0.012 * H)],
               drawn(kit, f"{name}.nosec", [x * 0.7 for x in lk["skin"]]), head_ob, lift)
    _mouth(kit, c, lk, head, head_ob, H, f, fr, lift, band)


def _mouth_pts(hw: float, top: float, bot: float, lift_c: float, cz: float, n: int = 13) -> list:
    """The mouth as a band: its upper edge left to right, then its lower edge right to left."""
    xs = [hw * math.cos(math.pi * (1 - k / (n - 1))) for k in range(n)]               # -hw .. hw
    curve = [lift_c * (x / hw) ** 2 if hw > 0 else 0.0 for x in xs]
    opening = [math.sqrt(max(0.0, 1 - (x / hw) ** 2)) if hw > 0 else 0.0 for x in xs]
    upper = [(x, cz + c + top * o) for x, c, o in zip(xs, curve, opening)]
    lower = [(x, cz + c - bot * o) for x, c, o in zip(xs, curve, opening)]
    return upper + lower[::-1]


def _mouth(kit, c, lk, head: Head, head_ob, H: float, f: dict, fr: int, lift: float, band: str):
    name = c["name"]
    face = c.get("face", {})
    cz = f["mouth_z"] * H
    k = f["mouth_w"] * H
    shapes = {key: _mouth_pts(hw * k, t * k, b * k, cl * k, cz) for key, (hw, t, b, cl) in MOUTH.items()}
    mat = drawn(kit, f"{name}.mouth", [0.42, 0.13, 0.14])
    ob = face_shape(f"{name}.mouth", head, shapes["basis"], mat, head_ob, lift, anchor=(0.0, cz), fan=True)
    origin = ob.location.copy()
    ob.shape_key_add(name="Basis", from_mix=False)
    for key, pts in shapes.items():
        if key == "basis":
            continue
        kb = ob.shape_key_add(name=key, from_mix=False)
        for i, (x, z) in enumerate(pts):
            kb.data[i].co = head.front(x, z, lift) - origin
        key_curve(kb, "value", [max(0.0, min(1.0, v)) for v in ch(face, key, fr)])
    if band == "old":                                         # smile lines beside the mouth
        for side, sx in (("L", 1), ("R", -1)):
            ox = sx * (MOUTH["basis"][0] * k + 0.035 * H)
            seg = [(ox + sx * 0.006 * H * t, cz + 0.05 * H - 0.055 * H * t) for t in (0.0, 0.5, 1.0)]
            face_band(f"{name}.smileline{side}", head, seg, [(x + sx * 0.005 * H, z) for x, z in seg],
                      drawn(kit, f"{name}.wrinkle", [x * 0.72 for x in lk["skin"]]), head_ob, lift)


# ---------------------------------------------------------------- hair
def _on_cap(head: Head, d: Vector, cap_s: float) -> Vector:
    """Where direction `d` from the head's centre meets the hair cap."""
    rx, ry, rz = head.rx * cap_s, head.ry * cap_s, head.rz * cap_s
    lam = 1.0 / math.sqrt((d.x / rx) ** 2 + (d.y / ry) ** 2 + (d.z / rz) ** 2)
    p = d * lam
    return head.shape(p.x, p.y, p.z)


def _cap(name: str, head: Head, cap_s: float, hairline, mat, parent, cols: int = 64, rows: int = 18):
    """The hair's shell: from the crown down to the hairline, the edge following it smoothly."""
    bm = bmesh.new()
    top = bm.verts.new(head.shape(0.0, 0.0, head.rz * cap_s))
    grid = []
    for i in range(cols):
        a = math.tau * i / cols - math.pi
        z_end = hairline(a) * head.rz * cap_s
        col = []
        for k in range(1, rows + 1):
            el = _lerp(math.pi / 2, math.asin(max(-0.999, min(0.999, z_end / (head.rz * cap_s)))), k / rows)
            col.append(bm.verts.new(head.around(a, head.rz * cap_s * math.sin(el), scale=cap_s)))
        grid.append(col)
    for i in range(cols):
        a, b = grid[i], grid[(i + 1) % cols]
        bm.faces.new((top, b[0], a[0]))
        for k in range(rows - 1):
            bm.faces.new((a[k], b[k], b[k + 1], a[k + 1]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    for f in bm.faces:
        f.normal_update()
        if f.normal.dot(f.calc_center_median()) < 0:
            f.normal_flip()
    return _object(name, bm, [mat], parent)


def _hair(kit, lk: dict, head: Head, head_ob, mat, H: float, band: str):
    style = lk["hair_style"]
    cap_s = 1.07
    front_line = 0.62 if style == "bun" else 0.5
    back_line = {"bob": -0.62, "long": -0.62, "bun": -0.45, "short": -0.5}.get(style, -0.6)

    def hairline(a: float) -> float:                     # height of the hairline (in skull radii) at angle a
        u = abs(a) / math.pi                               # 0 front .. 1 back
        return _lerp(front_line, -0.15, min(1.0, u / 0.5)) if u < 0.5 else _lerp(-0.15, back_line, (u - 0.5) / 0.5)
    _cap(f"{head_ob.name}.haircap", head, cap_s, hairline, mat, head_ob)
    name = head_ob.name.replace(".head", "")
    centre = Vector((0, 0, 0))

    def lifted(points, base: float, bulge: float):
        n = len(points)
        return [p + (p - centre).normalized() * (base + bulge * math.sin(math.pi * k / (n - 1))) for k, p in enumerate(points)]

    brow_z = (FACE[band]["eye_z"] + FACE[band]["eye_h"] / 2 + FACE[band]["brow"]) * H
    if style != "bun":
        # bangs: clumps from under the cap's edge down over the forehead, a curtain shorter at the sides
        n_b = 7 if band == "child" else 9
        for i in range(n_b):
            t = i / (n_b - 1) * 2 - 1
            ang0 = t * 0.85
            tip_ang = t * 0.72 + 0.07 * (1 if i % 2 else -1) * (1 - abs(t))
            z0 = 0.78 * head.rz
            z1 = brow_z + (0.005 + 0.07 * abs(t) ** 1.6 + (0.0 if i % 2 else 0.025)) * H
            pts = [head.around(_lerp(ang0, tip_ang, s), _lerp(z0, z1, s), scale=cap_s) for s in [k / 7 for k in range(8)]]
            lock(f"{name}.bang{i}", lifted(pts, 0.008 * H, 0.02 * H),
                 _taper(8, (0.21 if band == "child" else 0.17) * H, 0.3), 0.35, centre, mat, head_ob)
    else:
        # combed back from the forehead over the crown to the bun: narrow strands, so the gaps between
        # them read as combed hair rather than a cap
        for i in range(9):
            x0 = (i / 8 * 2 - 1) * 0.62 * head.rx
            d0 = _on_cap(head, Vector((x0, -head.ry, 0.6 * head.rz)).normalized(), cap_s) - centre
            pts = []
            for s_ in [k / 9 for k in range(10)]:
                b = math.radians(-115 * s_)
                d = Vector((d0.x * (1 - 0.55 * s_), d0.y * math.cos(b) - d0.z * math.sin(b),
                            d0.y * math.sin(b) + d0.z * math.cos(b)))
                pts.append(_on_cap(head, d.normalized(), cap_s))
            lock(f"{name}.combed{i}", lifted(pts, 0.004 * H, 0.012 * H), [0.075 * H * (1 - 0.6 * s_) for s_ in [k / 9 for k in range(10)]],
                 0.6, centre, mat, head_ob, ring=6)
    # side locks framing the face
    side_tip = {"bob": -0.95, "long": -1.7, "bun": -0.4, "short": -0.45}.get(style, -0.9) * head.rz
    for sx in (1, -1):
        pts = [head.around(sx * _lerp(1.22, 1.3, s), _lerp(0.35 * head.rz, side_tip, s), scale=cap_s) for s in [k / 7 for k in range(8)]]
        if side_tip < -head.rz * 0.9:
            pts = [p if p.z > -head.rz * 0.85 else Vector((p.x, p.y, p.z)) for p in pts]
        lock(f"{name}.side{'L' if sx > 0 else 'R'}", lifted(pts, 0.006 * H, 0.012 * H),
             _taper(8, (0.13 if style == "bun" else 0.17) * H, 0.2), 0.45, centre, mat, head_ob)
    # the back
    if style in ("bob", "long", "short"):
        tip = {"bob": -1.05, "long": -2.8, "short": -0.7}[style] * head.rz
        count = 7 if style != "long" else 9
        lo, hi = (112, 248) if style != "long" else (100, 260)
        for i in range(count):
            ang = math.radians(_lerp(lo, hi, i / (count - 1)))
            pts = []
            for s in [k / 9 for k in range(10)]:
                zz = _lerp(0.45 * head.rz, tip, s)
                if zz > -head.rz * 0.85:
                    p = head.around(ang, zz, scale=cap_s)
                else:                                          # below the skull, hair falls straight
                    p = head.around(ang, -head.rz * 0.85, scale=cap_s)
                    p = Vector((p.x, p.y, zz))
                flare = (0.02 if style == "bob" else 0.01) * H * s
                p = p + Vector((p.x, p.y, 0)).normalized() * flare
                pts.append(p)
            lock(f"{name}.back{i}", lifted(pts, 0.006 * H, 0.015 * H), _taper(10, 0.3 * H, 0.15), 0.32, centre, mat, head_ob)
    if style == "bun":
        bun = props.sphere(f"{name}.bun", 0.2 * H, [mat], scale=(1.0, 0.9, 0.85), segments=20)
        bun.location = (0, head.ry * 0.95, head.rz * 0.5)
        bun.parent = head_ob
        pin = props.box(f"{name}.pin", (0.4 * H, 0.014 * H, 0.014 * H), [kit.mat("kanzashi", [0.5, 0.12, 0.1])])
        pin.location = (0.0, head.ry * 1.02, head.rz * 0.58)
        pin.rotation_euler = (0, math.radians(-18), 0)
        pin.parent = head_ob
