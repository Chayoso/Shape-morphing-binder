"""blender_render.py -- the demo's Blender (Cycles) renderer of mesh_frames.py's per-frame surfaces (v2 look).

  blender -b -noaudio --factory-startup --python blender_render.py -- MESH_DIR OUT_DIR key=value ...

One light setup for everything: the bundled studio-light HDRI (studio.exr, CC0) for ambient light and reflections and
ONE large soft key area light; body, table / pedestal and wall are lit by the same lights (no background-only trick).
Real-world scale: the body (its last frame's largest extent) is BODY metres (0.18), the camera a 60 mm lens, mild
depth of field on the body.

  look=B  coloured plastic (roughness 0.3) on a wooden table (ambientCG Wood026, CC0: colour, roughness, normal)
          before a warm, low-contrast plaster wall (ambientCG Plaster001, CC0), warm key light.
  look=A  semi-gloss body on a white pedestal in a seamless bluish-grey studio sweep.

keys: look, color=r,g,b[;r,g,b] (one per body), rough=0.3, coat=0, frames=a-b, stride, res=3840x2160, samples=64,
az=35, el=16, fill=0.72 (share of the frame the clip's union box fills), lens=60, fstop=5.6, body=0.18, key (W per m^2
of scene size), hdri=studio, hdri_strength, floor_off=0 (pitches the floor plane lies below the scene's floor_y),
blend=PATH, only=K[,K], suffix=.png.

Geometry: the archive's world (y up) maps to Blender's (z up) by (x, y, z) -> (x, -z, y), scaled to BODY. Morph runs
(no gravity, no floor) are shown resting on the table: each frame is translated vertically so that its lowest point
(the 0.05 % lowest vertices' level) touches the surface -- a rigid display translation, no shape change. Scene runs
(gravity, the two-body throw) are not moved: the floor plane is the simulation's floor (floor_y, less floor_off
pitches). The shading normals are the paper display's (mesh_frames.py); the geometry is drawn as meshed."""
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MESH, OUT = Path(argv[0]), Path(argv[1])
opt = dict(a.split("=", 1) for a in argv[2:])
OUT.mkdir(parents=True, exist_ok=True)
meta = json.loads((MESH / "meta.json").read_text())
LOOK = opt.get("look", "B")
W, H = (int(v) for v in opt.get("res", "3840x2160").split("x"))
SAMPLES = int(opt.get("samples", 64))
AZ, EL = float(opt.get("az", 35)), float(opt.get("el", 16))
FILL = float(opt.get("fill", 0.72))
LENS = float(opt.get("lens", 60))
BODY = float(opt.get("body", 0.18))
TEX = Path("/data/relcfd/chayo/physmorph_v2/tools/textures")
nb = len(meta["rows"])
cols = [tuple(float(c) for c in s.split(",")) for s in opt.get("color", "0.55,0.04,0.03").split(";")]
cols = (cols * nb)[:nb]
keep = meta["keep"]
if "frames" in opt:
    f0, f1 = (int(v) for v in opt["frames"].split("-"))
    keep = [k for k in keep if f0 <= k <= f1]
keep = keep[::int(opt.get("stride", 1))]
only = [int(v) for v in opt["only"].split(",")] if "only" in opt else None
SCENE = meta.get("floor_y") is not None                    # a scene archive with a floor: nothing is moved


# ---------------------------------------------------------------- geometry: world -> blender, scale, union box
def to_b(V):
    return np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)


info = {(k, b): json.loads((MESH / f"k{k:04d}_b{b}.json").read_text()) for k in keep for b in range(nb)}
last = info[(keep[-1], 0)]
S = float(np.max(np.array(last["hi"]) - np.array(last["lo"])))
SC = BODY / S
lo_w = np.min([j["lo"] for j in info.values()], 0)
hi_w = np.max([j["hi"] for j in info.values()], 0)
C = to_b(0.5 * (lo_w + hi_w)[None])[0]                       # the union box's centre, blender axes
if SCENE:
    floor_w = meta["floor_y"] - float(opt.get("floor_off", 0.0)) * meta["pitches"][0]
    ground = (floor_w - C[2]) * SC
    blo = (to_b(lo_w[None])[0] - C) * SC
    bhi = (to_b(hi_w[None])[0] - C) * SC
    blo, bhi = np.minimum(blo, bhi), np.maximum(blo, bhi)
    shift = {k: 0.0 for k in keep}
else:
    ground = 0.0
    # each frame rests on the surface: its 0.05 % lowest vertices' level goes to z = 0
    shift = {k: -(info[(k, 0)]["ylow"] - C[2]) * SC for k in keep}
    blo = (to_b(lo_w[None])[0] - C) * SC
    bhi = (to_b(hi_w[None])[0] - C) * SC
    blo, bhi = np.minimum(blo, bhi), np.maximum(blo, bhi)
    blo[2] = 0.0
    bhi[2] = max((info[(k, 0)]["hi"][1] - info[(k, 0)]["ylow"]) * SC for k in keep)


def read_ply(p, dz):
    with open(p, "rb") as fh:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += fh.readline()
        h = head.decode().split("\n")
        nv = int([l for l in h if l.startswith("element vertex")][0].split()[-1])
        nf = int([l for l in h if l.startswith("element face")][0].split()[-1])
        VN = np.frombuffer(fh.read(nv * 24), "<f4").reshape(nv, 6)
        F = np.frombuffer(fh.read(nf * 13), dtype=[("n", "u1"), ("i", "<i4", (3,))])["i"]
    V = (to_b(VN[:, :3].astype(np.float64)) - C) * SC
    V[:, 2] += dz
    N = to_b(VN[:, 3:].astype(np.float64))
    return V.astype(np.float32), N.astype(np.float32), np.ascontiguousarray(F, np.int32)


def make_mesh(name, V, N, F):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set("co", V.ravel())
    me.loops.add(3 * len(F))
    me.loops.foreach_set("vertex_index", F.ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set("loop_start", np.arange(0, 3 * len(F), 3, dtype=np.int32))
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", np.ones(len(F), bool))
    me.normals_split_custom_set_from_vertices(N.reshape(-1, 3))
    return me


# ---------------------------------------------------------------- render settings
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "OPTIX"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "OPTIX"
sc.cycles.device = "GPU"
sc.cycles.samples = SAMPLES
sc.cycles.use_adaptive_sampling = True
sc.cycles.adaptive_threshold = 0.015
sc.cycles.use_denoising = True
sc.cycles.denoiser = "OPTIX"
sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
sc.cycles.max_bounces = 8
sc.cycles.diffuse_bounces = 4
sc.cycles.glossy_bounces = 4
sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
sc.cycles.blur_glossy = 1.0
sc.render.use_persistent_data = True
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"
sc.render.image_settings.color_depth = "8"
sc.render.image_settings.compression = 15
sc.view_settings.view_transform = "AgX"
sc.view_settings.look = opt.get("agx", "AgX - Medium High Contrast")
sc.view_settings.exposure = float(opt.get("exposure", 0.0))


# ---------------------------------------------------------------- materials
def node_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])
    return m, nt, bs


def body_mat(i, rgb):
    m, nt, bs = node_mat(f"body{i}")
    bs.inputs["Base Color"].default_value = (*rgb, 1.0)
    bs.inputs["Roughness"].default_value = float(opt.get("rough", 0.3))
    bs.inputs["Coat Weight"].default_value = float(opt.get("coat", 0.0))
    bs.inputs["Coat Roughness"].default_value = 0.1
    return m


def textured_mat(name, asset, tile, size, normal_strength, tint=(1, 1, 1), bright=1.0, rough_scale=1.0):
    """An ambientCG PBR set (colour, roughness, NormalGL) on a plane of `size` metres, one texture tile per `tile`
    metres (UV tiled)."""
    m, nt, bs = node_mat(name)
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (size / tile, size / tile, 1.0)
    L.new(tc.outputs["UV"], mp.inputs["Vector"])

    def img(kind, non_color):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = bpy.data.images.load(str(TEX / f"{asset}_2K-JPG" / f"{asset}_2K-JPG_{kind}.jpg"))
        if non_color:
            t.image.colorspace_settings.name = "Non-Color"
        t.interpolation = "Cubic"
        L.new(mp.outputs["Vector"], t.inputs["Vector"])
        return t

    col = img("Color", False)
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    mix.inputs[7].default_value = (tint[0] * bright, tint[1] * bright, tint[2] * bright, 1)
    L.new(col.outputs["Color"], mix.inputs[6])
    L.new(mix.outputs[2], bs.inputs["Base Color"])
    rou = img("Roughness", True)
    rs = nt.nodes.new("ShaderNodeMath")
    rs.operation = "MULTIPLY"
    rs.inputs[1].default_value = rough_scale
    L.new(rou.outputs["Color"], rs.inputs[0])
    L.new(rs.outputs[0], bs.inputs["Roughness"])
    nrm = img("NormalGL", True)
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = normal_strength
    L.new(nrm.outputs["Color"], nm.inputs["Color"])
    L.new(nm.outputs["Normal"], bs.inputs["Normal"])
    return m


def plain_mat(name, rgb, rough):
    m, nt, bs = node_mat(name)
    bs.inputs["Base Color"].default_value = (*rgb, 1)
    bs.inputs["Roughness"].default_value = rough
    return m


def add_obj(name, me, mat=None):
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    if mat is not None:
        ob.data.materials.append(mat)
    return ob


def plane(name, size_x, size_y, loc, rot, mat):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=loc)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = (size_x, size_y, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    ob.rotation_euler = rot
    ob.data.materials.append(mat)
    return ob


# ---------------------------------------------------------------- world: one HDRI (ambient + reflections)
wd = bpy.data.worlds.new("world")
sc.world = wd
wd.use_nodes = True
wn = wd.node_tree
for n in list(wn.nodes):
    wn.nodes.remove(n)
wout = wn.nodes.new("ShaderNodeOutputWorld")
bg = wn.nodes.new("ShaderNodeBackground")
env = wn.nodes.new("ShaderNodeTexEnvironment")
hdri_dir = Path(bpy.utils.system_resource("DATAFILES")) / "studiolights" / "world"
env.image = bpy.data.images.load(str(hdri_dir / (opt.get("hdri", "studio") + ".exr")))
wmap = wn.nodes.new("ShaderNodeMapping")
wmap.inputs["Rotation"].default_value = (0, 0, math.radians(float(opt.get("hdri_rot", 120))))
wtc = wn.nodes.new("ShaderNodeTexCoord")
wn.links.new(wtc.outputs["Generated"], wmap.inputs["Vector"])
wn.links.new(wmap.outputs["Vector"], env.inputs["Vector"])
wn.links.new(env.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = float(opt.get("hdri_strength", 0.35))
wn.links.new(bg.outputs["Background"], wout.inputs["Surface"])

# ---------------------------------------------------------------- set
ext = bhi - blo
R = max(float(max(ext[0], ext[1], ext[2])), BODY)          # the scene's size (m)
foot = 0.5 * max(ext[0], ext[1])
if LOOK == "A":
    # the pedestal (white, its top at the ground) inside a seamless bluish-grey sweep
    ped_h = 0.55 * BODY
    rad = max(0.8 * 2 * foot, 0.5 * BODY)
    lb = (to_b(np.array([last["lo"], last["hi"]])) - C) * SC          # under the end shape's footprint
    px, py = 0.5 * (lb[:, 0].min() + lb[:, 0].max()), 0.5 * (lb[:, 1].min() + lb[:, 1].max())
    bpy.ops.mesh.primitive_cylinder_add(vertices=256, radius=rad, depth=ped_h, location=(px, py, ground - ped_h / 2))
    ped = bpy.context.active_object
    ped.data.materials.append(plain_mat("pedestal", (0.80, 0.80, 0.78), 0.45))
    bpy.ops.object.shade_smooth()
    bev = ped.modifiers.new("bevel", "BEVEL")
    bev.width, bev.segments = 0.004, 5
    zf = ground - ped_h
    y0, rc, wdt, hh = bhi[1] + 0.6 * R, 0.6 * R, 12 * R, 6 * R
    prof = [(-8 * R, zf), (y0, zf)] + [(y0 + rc * math.sin(t), zf + rc - rc * math.cos(t))
                                        for t in np.linspace(0, math.pi / 2, 24)[1:]] + [(y0 + rc, zf + hh)]
    verts = [(x, y, z) for x in (-wdt / 2, wdt / 2) for (y, z) in prof]
    n = len(prof)
    faces = [(i, i + 1, n + i + 1, n + i) for i in range(n - 1)]
    me = bpy.data.meshes.new("sweep")
    me.from_pydata(verts, [], faces)
    me.update()
    me.polygons.foreach_set("use_smooth", np.ones(len(faces), bool))
    add_obj("sweep", me, plain_mat("sweep", tuple(float(v) for v in opt.get("sweep", "0.16,0.175,0.20").split(",")),
                                   0.75))
    key_dir = Vector((-1.0, -0.85, 1.25)).normalized()
else:
    T = max(4.0, 10 * R)
    wall_y = float(bhi[1] + max(0.35, 0.6 * R))
    plane("table", T, T, (0, wall_y - T / 2, ground), (0, 0, 0),
          textured_mat("wood", opt.get("wood", "Wood026"), 0.9, T, 0.5, rough_scale=0.85))
    plane("wall", 2 * T, T, (0, wall_y, ground + T / 2), (math.radians(90), 0, 0),
          textured_mat("plaster", "Plaster001", 2.0, 2 * T, 0.35, tint=(1.0, 0.90, 0.78), bright=0.95))
    key_dir = Vector((-1.0, -0.9, 1.15)).normalized()

# one large soft key light, warm for B
ld = bpy.data.lights.new("key", "AREA")
ld.shape = "SQUARE"
ld.size = float(opt.get("key_size", 1.2)) * max(R / BODY, 1.0) * BODY * 5
ld.energy = float(opt.get("key", 2600.0)) * (max(R / BODY, 1.0)) ** 2 * (BODY / 0.18) ** 2
ld.color = (1.0, 0.93, 0.84) if LOOK == "B" else (1.0, 0.97, 0.93)
key = bpy.data.objects.new("key", ld)
sc.collection.objects.link(key)
center = Vector((0.5 * (blo[0] + bhi[0]), 0.5 * (blo[1] + bhi[1]), 0.5 * (max(blo[2], ground) + bhi[2])))
key.location = center + key_dir * (7.0 * max(R, BODY))
key.rotation_euler = (center - key.location).to_track_quat("-Z", "Y").to_euler()

mats = [body_mat(i, c) for i, c in enumerate(cols)]
bodies = [add_obj(f"body{b}", bpy.data.meshes.new(f"empty{b}"), None) for b in range(nb)]

# ---------------------------------------------------------------- camera: three-quarter view, fit to the union box
cam_d = bpy.data.cameras.new("cam")
cam_d.lens = LENS
cam_d.sensor_width = 36
cam_d.clip_start, cam_d.clip_end = 0.01, 500
cam = bpy.data.objects.new("cam", cam_d)
sc.collection.objects.link(cam)
sc.camera = cam
az, el = math.radians(AZ), math.radians(EL)
dirv = Vector((math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)))
look_at = center.copy()
corners = [Vector((x, y, zz)) for x in (blo[0], bhi[0]) for y in (blo[1], bhi[1]) for zz in (max(blo[2], ground), bhi[2])]
tanx = 18.0 / LENS
tany = tanx * H / W


def fit(dist):
    p = look_at + dist * dirv
    rot = (look_at - p).to_track_quat("-Z", "Y").to_matrix()
    m = 0.0
    for c in corners:
        q = rot.transposed() @ (c - p)
        m = max(m, abs(q.x) / (-q.z * tanx), abs(q.y) / (-q.z * tany))
    return m


lo_d, hi_d = 0.01, 500.0
for _ in range(80):
    mid = 0.5 * (lo_d + hi_d)
    if fit(mid) > FILL:
        lo_d = mid
    else:
        hi_d = mid
dist = hi_d * float(opt.get("dist", 1.0))
cam.location = look_at + dist * dirv
cam.rotation_euler = (look_at - cam.location).to_track_quat("-Z", "Y").to_euler()
fstop = float(opt.get("fstop", 5.6))
if fstop > 0:
    cam_d.dof.use_dof = True
    cam_d.dof.focus_distance = dist
    cam_d.dof.aperture_fstop = fstop
print(f"scale {SC:.4f} m per world unit; box {blo} {bhi}; ground {ground:.4f}; cam dist {dist:.3f} m", flush=True)

if "blend" in opt:
    bpy.ops.wm.save_as_mainfile(filepath=opt["blend"])

# ---------------------------------------------------------------- render loop
for k in (only if only is not None else keep):
    outp = OUT / f"{k:04d}{opt.get('suffix', '.png')}"
    if outp.exists():
        continue
    t0 = time.time()
    for b in range(nb):
        V, N, F = read_ply(MESH / f"k{k:04d}_b{b}.ply", shift[k])
        me = make_mesh(f"m{k}_{b}", V, N, F)
        old = bodies[b].data
        bodies[b].data = me
        me.materials.append(mats[b])
        bpy.data.meshes.remove(old)
    t1 = time.time()
    sc.render.filepath = str(outp)
    bpy.ops.render.render(write_still=True)
    print(f"frame {k} load {t1 - t0:.1f}s render {time.time() - t1:.1f}s", flush=True)
