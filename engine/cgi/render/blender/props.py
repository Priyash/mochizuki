"""Props and set pieces, made from code so they are identical in every shot. Runs inside Blender.

Each builder makes one object (its origin where a hand or the floor would hold it) and
returns it. Continuity lives in the arguments: the bowl's gold seams are a number, not a
drawing, so chapter 1's bowl has one seam in every shot and chapter 3's has two.
"""
from __future__ import annotations

import math
import random

import bmesh
import bpy
from mathutils import Matrix, Vector

from . import toon

COLOURS = {
    "stone": [0.42, 0.42, 0.44], "wood": [0.36, 0.22, 0.12], "dark_wood": [0.16, 0.1, 0.07],
    "rice": [0.95, 0.94, 0.9], "glaze": [0.07, 0.08, 0.14], "gold": [1.0, 0.74, 0.24],
    "paper": [1.0, 0.62, 0.3], "lacquer": [0.32, 0.05, 0.04], "indigo": [0.16, 0.2, 0.44],
    "tatami": [0.62, 0.6, 0.38], "snow": [0.88, 0.92, 1.0], "ground": [0.3, 0.32, 0.25],
    "sea": [0.02, 0.05, 0.12], "moon": [1.0, 0.95, 0.82], "ash": [0.12, 0.11, 0.1], "ember": [1.0, 0.42, 0.12],
    "grass": [0.85, 0.78, 0.6], "stem": [0.45, 0.42, 0.3], "shoji": [0.86, 0.88, 0.95],
}


class Kit:
    """Shared materials, so the same thing is the same colour everywhere."""

    def __init__(self, toon_cfg: dict):
        self.toon_cfg = toon_cfg
        self.cache: dict = {}

    def mat(self, key: str, base=None, emission: float = 0.0, texture: str = "", rim: bool = True):
        k = (key, emission, texture, rim)
        if k not in self.cache:
            self.cache[k] = toon.toon(key, base or COLOURS.get(key, [0.5, 0.5, 0.5]), self.toon_cfg, emission,
                                      texture, rim)
        return self.cache[k]


def _obj(name: str, bm: bmesh.types.BMesh, mats: list, smooth: bool = True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    if smooth:
        for poly in me.polygons:
            poly.use_smooth = True
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def lathe(name: str, profile: list, mats: list, steps: int = 48):
    """A round thing from its half-profile [(radius, height), ...] spun about Z."""
    bm = bmesh.new()
    verts = [bm.verts.new((r, 0.0, z)) for r, z in profile]
    edges = [bm.edges.new((verts[i], verts[i + 1])) for i in range(len(verts) - 1)]
    bmesh.ops.spin(bm, geom=verts + edges, cent=(0, 0, 0), axis=(0, 0, 1), angle=math.tau, steps=steps,
                   use_merge=True)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _obj(name, bm, mats)


def cylinder(bm, r: float, depth: float, matrix=None, segments: int = 24):
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments, radius1=r, radius2=r,
                          depth=depth, matrix=matrix or Matrix())


def box(name: str, size, mats, bevel: float = 0.0):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    ob = _obj(name, bm, mats, smooth=False)
    if bevel:
        m = ob.modifiers.new("bevel", "BEVEL")
        m.width = bevel
        m.segments = 2
    return ob


def sphere(name: str, r: float, mats, scale=(1, 1, 1), segments: int = 24):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=segments // 2, radius=r)
    bmesh.ops.scale(bm, vec=Vector(scale), verts=bm.verts)
    return _obj(name, bm, mats)


# ---------------------------------------------------------------- props
def mortar(kit: Kit, **_):
    prof = [(0.0, 0.0), (0.30, 0.0), (0.33, 0.02), (0.36, 0.14), (0.37, 0.32), (0.36, 0.40), (0.35, 0.42),
            (0.29, 0.42), (0.25, 0.385), (0.18, 0.335), (0.09, 0.305), (0.0, 0.295)]
    return lathe("mortar", prof, [kit.mat("stone", texture="stone")])


def mallet(kit: Kit, **_):
    bm = bmesh.new()
    cylinder(bm, 0.085, 0.32, Matrix.Rotation(math.pi / 2, 4, "Y"))           # the head, across
    cylinder(bm, 0.022, 1.0, Matrix.Translation((0, 0, 0.55)), segments=12)  # the handle, up
    return _obj("mallet", bm, [kit.mat("wood", texture="wood")])


def rice(kit: Kit, **_):
    return sphere("rice", 0.16, [kit.mat("rice")], scale=(1, 1, 0.42))


def bowl(kit: Kit, seams: int = 1, cracks: int = 0, seed: int = 7, **_):
    out_prof = [(0.0, 0.0), (0.035, 0.0), (0.05, 0.005), (0.068, 0.025), (0.077, 0.05), (0.078, 0.058)]
    in_prof = [(0.072, 0.058), (0.068, 0.046), (0.05, 0.023), (0.0, 0.016)]
    ob = lathe("bowl", out_prof + in_prof, [kit.mat("glaze")], steps=64)
    rng = random.Random(seed)

    def outer_r(z):
        for (r0, z0), (r1, z1) in zip(out_prof[1:], out_prof[2:]):
            if z0 <= z <= z1:
                return r0 + (r1 - r0) * (z - z0) / max(1e-6, z1 - z0)
        return out_prof[-1][0]

    def seam(name, theta0, colour, emission):
        pts = []
        for i in range(14):
            z = 0.058 - i * (0.05 / 13)
            th = theta0 + 0.09 * math.sin(i * 0.55 + rng.uniform(-0.15, 0.15)) + rng.uniform(-0.015, 0.015)
            r = outer_r(z) + 0.0009
            pts.append((r * math.cos(th), r * math.sin(th), z))
        cu = bpy.data.curves.new(name, "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0.0011
        cu.bevel_resolution = 2
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for p, c in zip(sp.points, pts):
            p.co = (c[0], c[1], c[2], 1.0)
        cob = bpy.data.objects.new(name, cu)
        cob.data.materials.append(kit.mat(colour, emission=emission))
        bpy.context.scene.collection.objects.link(cob)
        cob.parent = ob
        return cob

    for i in range(seams):
        seam(f"bowl.seam{i + 1}", -math.pi / 2 + i * 2.3, "gold", 1.6)       # gold catches light: it glows a little
    for i in range(cracks):
        seam(f"bowl.crack{i + 1}", -math.pi / 2 + (seams + i) * 2.3, "ash", 0.0)
    return ob


def lantern(kit: Kit, glow: float = 2.2, **_):
    prof = [(0.0, -0.18), (0.07, -0.172), (0.12, -0.14), (0.15, -0.08), (0.16, 0.0), (0.15, 0.08), (0.12, 0.14),
            (0.07, 0.172), (0.0, 0.18)]
    ob = lathe("lantern", prof, [kit.mat("paper", emission=glow)])
    caps = bmesh.new()
    cylinder(caps, 0.075, 0.03, Matrix.Translation((0, 0, 0.185)))
    cylinder(caps, 0.075, 0.03, Matrix.Translation((0, 0, -0.185)))
    cylinder(caps, 0.004, 0.45, Matrix.Translation((0, 0, 0.42)), segments=6)
    c = _obj("lantern.caps", caps, [kit.mat("dark_wood")], smooth=False)
    c.parent = ob
    for o in (ob, c):                                      # the lamp inside must shine through the paper
        o.visible_shadow = False
    return ob


def cushion(kit: Kit, **_):
    return box("cushion", (0.55, 0.55, 0.08), [kit.mat("indigo")], bevel=0.025)


def tray(kit: Kit, **_):
    ob = box("tray", (0.36, 0.26, 0.022), [kit.mat("lacquer")], bevel=0.006)
    ob.location.z = 0.011
    return ob


def dumplings(kit: Kit, **_):
    bm = bmesh.new()
    cylinder(bm, 0.11, 0.08, Matrix.Translation((0, 0, 0.04)), segments=20)
    stand = _obj("dumplings.stand", bm, [kit.mat("dark_wood")], smooth=False)
    r = 0.024
    layers = [(3, 0.08 + r), (2, 0.08 + r * 2.6), (1, 0.08 + r * 4.2)]
    dumpling_mat = [kit.mat("rice")]
    for n, z in layers:
        for i in range(n):
            for j in range(n):
                s = sphere(f"dango{n}{i}{j}", r, dumpling_mat, segments=12)
                s.location = ((i - (n - 1) / 2) * r * 2.05, (j - (n - 1) / 2) * r * 2.05, z)
                s.parent = stand
    return stand


def pampas(kit: Kit, seed: int = 3, **_):
    rng = random.Random(seed)
    bm = bmesh.new()
    cylinder(bm, 0.06, 0.22, Matrix.Translation((0, 0, 0.11)), segments=16)
    vase = _obj("pampas", bm, [kit.mat("glaze")], smooth=False)
    for i in range(7):
        tilt = Matrix.Rotation(rng.uniform(-0.35, 0.35), 4, "X") @ Matrix.Rotation(rng.uniform(-0.35, 0.35), 4, "Y")
        st = bmesh.new()
        cylinder(st, 0.003, 0.85, Matrix.Translation((0, 0, 0.42)), segments=5)
        stem = _obj(f"pampas.stem{i}", st, [kit.mat("stem")], smooth=False)
        stem.matrix_world = Matrix.Translation((0, 0, 0.2)) @ tilt
        plume = sphere(f"pampas.plume{i}", 0.03, [kit.mat("grass")], scale=(0.6, 0.6, 4.0), segments=10)
        plume.location = (0, 0, 0.88)
        plume.parent = stem
        stem.parent = vase
    return vase


# ---------------------------------------------------------------- set pieces
def floor(kit: Kit, material: str = "wood", size=(5, 3), **_):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    bmesh.ops.scale(bm, vec=Vector((size[0], size[1], 1)), verts=bm.verts)
    tex = {"wood": "wood", "tatami": "tatami", "ground": "stone"}.get(material, "")
    return _obj("floor", bm, [kit.mat(material, texture=tex, rim=False)], smooth=False)


def sea(kit: Kit, size: float = 400, **_):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=size / 2)
    return _obj("sea", bm, [kit.mat("sea", rim=False)], smooth=False)


def moon(kit: Kit, radius: float = 3.0, **_):
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, radius=radius, segments=48)
    ob = _obj("moon", bm, [kit.mat("moon", emission=3.0)], smooth=False)
    ob.visible_shadow = False
    ob.rotation_euler = (math.pi / 2, 0, 0)
    return ob


def eaves(kit: Kit, size=(5, 1), posts: bool = False, **_):
    """The roof's edge over the porch, and the two posts that hold it at the open side: they frame
    every shot that looks out."""
    ob = box("eaves", (size[0], size[1], 0.08), [kit.mat("dark_wood")])
    if posts:
        for i, x in enumerate((-size[0] / 2 + 0.12, size[0] / 2 - 0.12)):
            post = box(f"eaves.post{i}", (0.12, 0.12, 2.3), [kit.mat("dark_wood")])
            post.location = (x, size[1] / 2 - 0.1, -1.15)
            post.parent = ob
        beam = box("eaves.beam", (size[0], 0.14, 0.16), [kit.mat("dark_wood")])
        beam.location = (0, size[1] / 2 - 0.1, -0.1)
        beam.parent = ob
    return ob


def hearth(kit: Kit, size: float = 0.9, **_):
    bm = bmesh.new()
    t = 0.06
    for dx, dy, sx, sy in ((0, size / 2, size, t), (0, -size / 2, size, t), (size / 2, 0, t, size),
                           (-size / 2, 0, t, size)):
        bmesh.ops.create_cube(
            bm, size=1.0, matrix=Matrix.Translation((dx, dy, 0.03)) @ Matrix.Diagonal((sx, sy, 0.06, 1))
        )
    frame = _obj("hearth", bm, [kit.mat("dark_wood")], smooth=False)
    ash = box("hearth.ash", (size - t, size - t, 0.01), [kit.mat("ash")])
    ash.parent = frame
    embers = sphere("hearth.embers", 0.12, [kit.mat("ember", emission=4.0)], scale=(1.4, 1.0, 0.3), segments=12)
    embers.location.z = 0.03
    embers.parent = frame
    return frame


def shoji(kit: Kit, size=(3.6, 2.0), **_):
    w, h = size
    panel = box("shoji", (w, 0.02, h), [kit.mat("shoji", emission=0.55)])
    panel.location.z = h / 2
    bm = bmesh.new()
    step = 0.3
    x = -w / 2
    while x <= w / 2 + 1e-6:
        bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((x, 0, 0)) @ Matrix.Diagonal((0.025, 0.05, h, 1)))
        x += step
    z = -h / 2
    while z <= h / 2 + 1e-6:
        bmesh.ops.create_cube(bm, size=1.0, matrix=Matrix.Translation((0, 0, z)) @ Matrix.Diagonal((w, 0.05, 0.02, 1)))
        z += step * 1.3
    grid = _obj("shoji.grid", bm, [kit.mat("dark_wood")], smooth=False)
    grid.parent = panel
    return panel


BUILDERS = {"mortar": mortar, "mallet": mallet, "rice": rice, "bowl": bowl, "lantern": lantern,
            "cushion": cushion, "tray": tray, "dumplings": dumplings, "pampas": pampas,
            "floor": floor, "sea": sea, "moon": moon, "eaves": eaves, "hearth": hearth, "shoji": shoji}


def build(kit: Kit, spec: dict):
    """Build one prop or piece from its plan entry; None (and a note) for a kind not known yet."""
    fn = BUILDERS.get(spec["kind"])
    if fn is None:
        print(f"cgi.blender: no builder for {spec['kind']!r} yet; skipped", flush=True)
        return None
    args = {k: v for k, v in spec.items() if k not in ("kind", "at", "rot", "motion", "_what", "carried_by")}
    ob = fn(kit, **args)
    at, rot = spec.get("at", [0, 0, 0]), spec.get("rot", [0, 0, 0])
    ob.location = (ob.location.x + at[0], ob.location.y + at[1], ob.location.z + at[2])
    ob.rotation_euler = tuple(math.radians(r) + o for r, o in zip(rot, ob.rotation_euler))
    return ob
