"""blender_render.py -- the demo's Blender (Cycles) renderer of mesh_frames.py's per-frame surfaces.

  blender -b -noaudio --factory-startup --python blender_render.py -- MESH_DIR OUT_DIR key=value ...

keys: look=A|B (A: glossy body on a white pedestal before a blurred HDRI; B: glossy plastic on a wooden table before a
mottled plaster wall; floor=1 makes B a wide wooden floor at the scene's floor height), color=r,g,b[;r,g,b] (one per
body), rough=0.15, coat=0.5, frames=a-b (kept-frame indices; default all meshed), stride=1, res=3840x2160, samples=64,
az=35, el=16 (the studio's three-quarter view: azimuth about the up axis from the front, elevation), fill=0.80 (the
share of the frame the clip's union box fills), lens=70, fstop (depth of field; default per look), hdri=interior,
blend=PATH (save the scene before rendering), only=K[,K] (render just these frames), suffix=.png.

The archive's world (y up) maps to Blender's (z up) by (x, y, z) -> (x, -z, y), scaled so that the body's own volume
sample is one unit across. The geometry is drawn as meshed (the field's gradient as vertex normals); nothing hidden."""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MESH, OUT = Path(argv[0]), Path(argv[1])
opt = dict(a.split("=", 1) for a in argv[2:])
OUT.mkdir(parents=True, exist_ok=True)
meta = json.loads((MESH / "meta.json").read_text())
LOOK = opt.get("look", "A")
FLOOR = opt.get("floor", "0") == "1"
W, H = (int(v) for v in opt.get("res", "3840x2160").split("x"))
SAMPLES = int(opt.get("samples", 64))
AZ, EL = float(opt.get("az", 35)), float(opt.get("el", 16))
FILL = float(opt.get("fill", 0.80))
LENS = float(opt.get("lens", 70))
nb = len(meta["rows"])
cols = [tuple(float(c) for c in s.split(",")) for s in opt.get("color", "0.02,0.02,0.025").split(";")]
cols = (cols * nb)[:nb]
keep = meta["keep"]
if "frames" in opt:
    f0, f1 = (int(v) for v in opt["frames"].split("-"))
    keep = [k for k in keep if f0 <= k <= f1]
keep = keep[::int(opt.get("stride", 1))]
only = [int(v) for v in opt["only"].split(",")] if "only" in opt else None


# ---------------------------------------------------------------- geometry: world -> blender, scale, union box
def to_b(V):
    return np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)


boxes = []
for k in keep:
    for b in range(nb):
        j = json.loads((MESH / f"k{k:04d}_b{b}.json").read_text())
        lo, hi = to_b(np.array([j["lo"]])), to_b(np.array([j["hi"]]))
        boxes.append(np.concatenate([np.minimum(lo, hi), np.maximum(lo, hi)], 0))
boxes = np.array(boxes)
ulo, uhi = boxes[:, 0].min(0), boxes[:, 1].max(0)
# unit: the body's size, the largest box extent of the clip's last frame (one body)
last = json.loads((MESH / f"k{keep[-1]:04d}_b0.json").read_text())
S = float(np.max(np.array(last["hi"]) - np.array(last["lo"])))
SC = 1.0 / S
C = 0.5 * (ulo + uhi)
if meta.get("floor_y") is not None:
    floor_z = (meta["floor_y"] * 1.0)                       # world y of the floor -> blender z
    ground = (floor_z - C[2]) * SC
else:
    ground = (ulo[2] - C[2]) * SC - 0.004                  # the pedestal / table just under the clip's lowest point
blo, bhi = (ulo - C) * SC, (uhi - C) * SC


def read_ply(p):
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


# ---------------------------------------------------------------- scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "OPTIX"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "OPTIX"
print("devices", [(d.name, d.type, d.use) for d in prefs.devices], flush=True)
sc.cycles.device = "GPU"
sc.cycles.samples = SAMPLES
sc.cycles.use_adaptive_sampling = True
sc.cycles.adaptive_threshold = 0.015
sc.cycles.use_denoising = True
sc.cycles.denoiser = "OPTIX"
sc.cycles.denoising_input_passes = "RGB_ALBEDO_NORMAL"
sc.cycles.max_bounces = 8
sc.cycles.diffuse_bounces = 3
sc.cycles.glossy_bounces = 4
sc.cycles.transmission_bounces = 2
sc.cycles.caustics_reflective = sc.cycles.caustics_refractive = False
sc.cycles.blur_glossy = 1.0
sc.render.use_persistent_data = True
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGB"
sc.render.image_settings.color_depth = "8"
sc.render.image_settings.compression = 15
try:
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast" if LOOK == "A" else "AgX - Base Contrast"
except TypeError:
    sc.view_settings.view_transform = "Filmic"
sc.view_settings.exposure = float(opt.get("exposure", 0.0))


def node_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bs = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(bs.outputs["BSDF"], out.inputs["Surface"])
    return m, nt, bs, out


def body_mat(i, rgb):
    m, nt, bs, _ = node_mat(f"body{i}")
    bs.inputs["Base Color"].default_value = (*rgb, 1.0)
    bs.inputs["Roughness"].default_value = float(opt.get("rough", 0.15 if LOOK == "A" else 0.22))
    bs.inputs["Specular IOR Level"].default_value = 0.5
    bs.inputs["Coat Weight"].default_value = float(opt.get("coat", 0.6 if LOOK == "A" else 0.35))
    bs.inputs["Coat Roughness"].default_value = 0.04
    return m


def wood_mat():
    """Procedural wood: a low-frequency plank tone (noise stretched along the grain) darkened by thin, warped grain
    lines (wave bands raised to a power), a faint fibre noise, a light satin coat."""
    m, nt, bs, out = node_mat("wood")
    L = nt.links
    tc = nt.nodes.new("ShaderNodeTexCoord")

    def node(kind, **inp):
        n = nt.nodes.new(kind)
        for k, v in inp.items():
            n.inputs[k].default_value = v
        return n

    mp = node("ShaderNodeMapping", Scale=(3.0, 0.12, 3.0))          # grain along blender y (depth)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])
    warp = node("ShaderNodeTexNoise", Scale=1.6, Detail=4.0)
    L.new(mp.outputs["Vector"], warp.inputs["Vector"])
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "VECTOR"
    mix.inputs["Factor"].default_value = 0.3
    L.new(mp.outputs["Vector"], mix.inputs[4])
    L.new(warp.outputs["Color"], mix.inputs[5])
    wv = nt.nodes.new("ShaderNodeTexWave")
    wv.wave_type, wv.bands_direction = "BANDS", "X"
    for k, v in dict(Scale=4.5, Distortion=1.5, Detail=2.0).items():
        wv.inputs[k].default_value = v
    L.new(mix.outputs[1], wv.inputs["Vector"])
    pw = nt.nodes.new("ShaderNodeMath")
    pw.operation = "POWER"
    pw.inputs[1].default_value = 7.0
    L.new(wv.outputs["Fac"], pw.inputs[0])                          # thin grain lines
    mp2 = node("ShaderNodeMapping", Scale=(0.9, 0.06, 0.9))
    L.new(tc.outputs["Object"], mp2.inputs["Vector"])
    big = node("ShaderNodeTexNoise", Scale=3.0, Detail=3.0)
    L.new(mp2.outputs["Vector"], big.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = 0.3, (0.16, 0.075, 0.032, 1)
    e[1].position, e[1].color = 0.7, (0.36, 0.19, 0.085, 1)
    L.new(big.outputs["Fac"], ramp.inputs["Fac"])
    mp3 = node("ShaderNodeMapping", Scale=(40.0, 1.5, 40.0))
    L.new(tc.outputs["Object"], mp3.inputs["Vector"])
    fib = node("ShaderNodeTexNoise", Scale=3.0, Detail=2.0)
    L.new(mp3.outputs["Vector"], fib.inputs["Vector"])
    dark = nt.nodes.new("ShaderNodeMath")                           # 1 - 0.45 line - 0.12 fibre
    dark.operation = "MULTIPLY_ADD"
    dark.inputs[1].default_value = -0.45
    dark.inputs[2].default_value = 1.0
    L.new(pw.outputs[0], dark.inputs[0])
    fd = nt.nodes.new("ShaderNodeMath")
    fd.operation = "MULTIPLY_ADD"
    fd.inputs[1].default_value = -0.14
    L.new(fib.outputs["Fac"], fd.inputs[0])
    L.new(dark.outputs[0], fd.inputs[2])
    col = nt.nodes.new("ShaderNodeMix")
    col.data_type = "RGBA"
    col.blend_type = "MULTIPLY"
    col.inputs["Factor"].default_value = 1.0
    L.new(ramp.outputs["Color"], col.inputs[6])
    L.new(fd.outputs[0], col.inputs[7])
    L.new(col.outputs[2], bs.inputs["Base Color"])
    bs.inputs["Roughness"].default_value = 0.42
    bump = node("ShaderNodeBump", Strength=0.06, Distance=0.002)
    L.new(pw.outputs[0], bump.inputs["Height"])
    L.new(bump.outputs["Normal"], bs.inputs["Normal"])
    bs.inputs["Coat Weight"].default_value = 0.12
    bs.inputs["Coat Roughness"].default_value = 0.2
    return m


def plaster_mat():
    m, nt, bs, out = node_mat("plaster")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = 1.6
    nz.inputs["Detail"].default_value = 10.0
    nz.inputs["Roughness"].default_value = 0.62
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    e = ramp.color_ramp.elements
    e[0].position, e[0].color = 0.30, (0.30, 0.275, 0.245, 1)
    e[1].position, e[1].color = 0.72, (0.62, 0.585, 0.54, 1)
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bs.inputs["Base Color"])
    bs.inputs["Roughness"].default_value = 0.92
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 14.0
    fine.inputs["Detail"].default_value = 8.0
    nt.links.new(tc.outputs["Object"], fine.inputs["Vector"])
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.25
    bump.inputs["Distance"].default_value = 0.01
    nt.links.new(fine.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bs.inputs["Normal"])
    return m


def white_mat():
    m, nt, bs, _ = node_mat("pedestal")
    bs.inputs["Base Color"].default_value = (0.82, 0.82, 0.80, 1)
    bs.inputs["Roughness"].default_value = 0.38
    return m


def add_obj(name, me, mat=None):
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    if mat is not None:
        ob.data.materials.append(mat)
    return ob


def area_light(name, loc, target, size, energy, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = "DISK"
    ld.size = size
    ld.energy = energy
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    sc.collection.objects.link(ob)
    ob.location = loc
    d = Vector(target) - Vector(loc)
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return ob


# world
wd = bpy.data.worlds.new("world")
sc.world = wd
wd.use_nodes = True
wn = wd.node_tree
for n in list(wn.nodes):
    wn.nodes.remove(n)
wout = wn.nodes.new("ShaderNodeOutputWorld")
bg = wn.nodes.new("ShaderNodeBackground")
hdri_dir = Path(bpy.utils.system_resource("DATAFILES")) / "studiolights" / "world"
env = wn.nodes.new("ShaderNodeTexEnvironment")
env.image = bpy.data.images.load(str(hdri_dir / (opt.get("hdri", "interior" if LOOK == "A" else "studio") + ".exr")))
wmap = wn.nodes.new("ShaderNodeMapping")
wmap.inputs["Rotation"].default_value = (0, 0, math.radians(float(opt.get("hdri_rot", 200))))
wtc = wn.nodes.new("ShaderNodeTexCoord")
wn.links.new(wtc.outputs["Generated"], wmap.inputs["Vector"])
wn.links.new(wmap.outputs["Vector"], env.inputs["Vector"])
wn.links.new(env.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = float(opt.get("hdri_strength", 0.9 if LOOK == "A" else 0.35))
wn.links.new(bg.outputs["Background"], wout.inputs["Surface"])

size = float(np.max(bhi - blo))                              # the clip's union box, in body units
cx, cy = 0.0, 0.0
if LOOK == "A" and not FLOOR:
    # white pedestal: a cylinder under the body, its top at the ground
    rad = 0.62 * max(bhi[0] - blo[0], bhi[1] - blo[1])
    bpy.ops.mesh.primitive_cylinder_add(vertices=256, radius=rad, depth=0.5, location=(cx, cy, ground - 0.25))
    ped = bpy.context.active_object
    ped.data.materials.append(white_mat())
    bpy.ops.object.shade_smooth()
    bev = ped.modifiers.new("bevel", "BEVEL")
    bev.width, bev.segments = 0.012, 6
    # a soft, wide floor far below the pedestal, so the blurred room reads as a room
    sun_t = (0, 0, 0.1)
    area_light("key", (-2.6, -2.8, 3.0), sun_t, 2.6, float(opt.get("key", 260)))
    area_light("rim", (1.8, 3.2, 2.4), sun_t, 1.8, float(opt.get("rim", 220)), (1.0, 0.97, 0.92))
    area_light("fill", (3.2, -2.2, 0.8), sun_t, 3.0, 60)
else:
    # wooden table (or wide floor) and a mottled plaster wall behind
    big = 80.0 if FLOOR else 30.0
    wall_y = float(opt.get("wall", bhi[1] + 1.2))
    # the table / floor: its far edge at the wall (object coordinates = world sizes: the scale applied)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, wall_y - big / 2, ground))
    tb = bpy.context.active_object
    tb.scale = (big, big, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    tb.data.materials.append(wood_mat())
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, wall_y, ground + big / 4))
    wl = bpy.context.active_object
    wl.scale = (2 * big, big / 2, 1)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    wl.rotation_euler = (math.radians(90), 0, 0)
    wl.data.materials.append(plaster_mat())
    tgt = (0, 0, 0.0)
    D = max(1.0, size)
    area_light("key", (-2.4 * D, -2.2 * D, 3.2 * D), tgt, 3.2 * D, float(opt.get("key", 420)) * D * D,
               (1.0, 0.96, 0.9))
    area_light("fill", (2.8 * D, -2.0 * D, 1.2 * D), tgt, 3.0 * D, 70 * D * D, (0.92, 0.96, 1.0))

mats = [body_mat(i, c) for i, c in enumerate(cols)]
bodies = [add_obj(f"body{b}", bpy.data.meshes.new(f"empty{b}"), None) for b in range(nb)]

# ---------------------------------------------------------------- camera: three-quarter view, fit to the union box
cam_d = bpy.data.cameras.new("cam")
cam_d.lens = LENS
cam_d.sensor_width = 36
cam = bpy.data.objects.new("cam", cam_d)
sc.collection.objects.link(cam)
sc.camera = cam
az, el = math.radians(AZ), math.radians(EL)
dirv = Vector((math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)))
look_at = Vector(((blo[0] + bhi[0]) / 2, (blo[1] + bhi[1]) / 2, (max(blo[2], ground) + bhi[2]) / 2))
corners = [Vector((x, y, zz)) for x in (blo[0], bhi[0]) for y in (blo[1], bhi[1]) for zz in (max(blo[2], ground), bhi[2])]
tanx = 18.0 / LENS
tany = tanx * H / W


def fit(dist):
    p = look_at + dist * dirv
    rot = (look_at - p).to_track_quat("-Z", "Y").to_matrix()
    m = 0.0
    for c in corners:
        q = rot.transposed() @ (c - p)
        zc = -q.z
        m = max(m, abs(q.x) / (zc * tanx), abs(q.y) / (zc * tany))
    return m


lo_d, hi_d = 0.2, 200.0
for _ in range(60):
    mid = 0.5 * (lo_d + hi_d)
    if fit(mid) > FILL:
        lo_d = mid
    else:
        hi_d = mid
dist = hi_d * float(opt.get("dist", 1.0))
cam.location = look_at + dist * dirv
cam.rotation_euler = (look_at - cam.location).to_track_quat("-Z", "Y").to_euler()
fstop = float(opt.get("fstop", 0))
if fstop > 0:
    cam_d.dof.use_dof = True
    cam_d.dof.focus_distance = dist
    cam_d.dof.aperture_fstop = fstop
print(f"unit {S:.4f} world; union box {blo} {bhi}; ground {ground:.3f}; cam dist {dist:.2f}", flush=True)

if "blend" in opt:
    bpy.ops.wm.save_as_mainfile(filepath=opt["blend"])

# ---------------------------------------------------------------- render loop
import time                                                 # noqa: E402
todo = only if only is not None else keep
for k in todo:
    outp = OUT / f"{k:04d}{opt.get('suffix', '.png')}"
    if outp.exists():
        continue
    t0 = time.time()
    for b in range(nb):
        V, N, F = read_ply(MESH / f"k{k:04d}_b{b}.ply")
        me = make_mesh(f"m{k}_{b}", V, N, F)
        old = bodies[b].data
        bodies[b].data = me
        me.materials.append(mats[b])
        bpy.data.meshes.remove(old)
    t1 = time.time()
    sc.render.filepath = str(outp)
    bpy.ops.render.render(write_still=True)
    print(f"frame {k} load {t1 - t0:.1f}s render {time.time() - t1:.1f}s", flush=True)
