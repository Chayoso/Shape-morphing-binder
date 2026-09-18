"""Photoreal morph video: isosurface mesh of the particle density, rendered with Open3D's
Filament path (PBR material, image-based + sun lighting, soft shadows, ground plane).

    render_photoreal.py --npz run.npz --out out.mp4 [--res 900 --stride 3 --views 35,215]
                        [--still <frame> --out still.png]

The density grid is the one render_iso_video.py uses (trilinear splat of the particle
masses + a Gaussian of --blur particle spacings, iso = --iso x the source bulk density), so
the surface is the same object the isosurface videos show; the mesh is extracted with
marching cubes and smoothed (Taubin). Nothing is hidden: every mesh component is rendered
(--largest_only exists for illustration and is OFF by default) and a sidecar
<out>.components.txt records, per video frame, the number of isosurface components and the
number of isolated particles (8-NN distance > 3 x median), the per-frame QA the videos are
judged by (no floating particles, no particle-looking blobs).
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as Fn

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=int, default=900, help="pixels per view (square)")
ap.add_argument("--stride", type=int, default=3, help="archived frames per video frame")
ap.add_argument("--views", default="35,215", help="azimuths in degrees, side by side")
ap.add_argument("--elev", type=float, default=18.0)
ap.add_argument("--fps", type=int, default=20)
ap.add_argument("--hold", type=int, default=20, help="repeat the last frame this many times")
ap.add_argument("--grid", type=int, default=160)
ap.add_argument("--iso", default="auto",
                help="isosurface level as a fraction of the source bulk density, or 'auto' = the level at which a "
                     "filament two particles across (a 2x2 bundle, the thinnest continuum the particles can carry) "
                     "still renders: 2 s^2 / (pi sigma^2) with s the particle spacing and sigma the blur, capped at "
                     "0.5. At 0.5 a thin neck (teat, whisker) falls below the level and its bulb renders as a "
                     "detached ball although the material is connected at the particle scale (cow at 150k).")
ap.add_argument("--blur", type=float, default=1.5)
ap.add_argument("--smooth", type=int, default=12, help="Taubin smoothing iterations")
ap.add_argument("--fov", type=float, default=30.0, help="vertical field of view (degrees)")
ap.add_argument("--fill", type=float, default=0.78, help="fraction of the frame height the box spans")
ap.add_argument("--color", default="0.86,0.80,0.72", help="base colour (linear RGB)")
ap.add_argument("--rough", type=float, default=0.32)
ap.add_argument("--metal", type=float, default=0.0)
ap.add_argument("--ground", type=int, default=1, help="shadow-catching ground plane")
ap.add_argument("--target_ghost", type=float, default=0.0, help="alpha of a translucent target isosurface (0 = off)")
ap.add_argument("--largest_only", type=int, default=0, help="render only the largest mesh component (illustration only)")
ap.add_argument("--min_cells", type=float, default=1.0,
                help="drop isosurface components whose volume is below this many MPM cells (dx^3; dx = source "
                     "bbox diagonal / cell_diag): material the grid cannot resolve is not a continuum element. "
                     "0 = draw everything. Dropped components are counted in the sidecar.")
ap.add_argument("--cell_diag", type=float, default=26.0)
ap.add_argument("--still", type=int, default=-1, help="render only this archived frame to --out (png)")
ap.add_argument("--max_frames", type=int, default=0)
ap.add_argument("--label", default="")
a = ap.parse_args()

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
import open3d as o3d  # noqa: E402
from skimage import measure  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
dev = "cuda" if torch.cuda.is_available() else "cpu"
z = np.load(a.npz, allow_pickle=True)
frames_np = z["frames"]
dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames_np)
tgt_np = np.asarray(z["tgt"], np.float32)
G = a.grid
tgt = torch.as_tensor(tgt_np, device=dev)
x0 = torch.as_tensor(np.asarray(frames_np[0], np.float32), device=dev)
N = x0.shape[0]

# ---- the box: every archived frame + the target, cube centred on the target ------------
lo = torch.minimum(torch.as_tensor(frames_np[:dn].reshape(-1, 3).min(0), device=dev), tgt.min(0).values)
hi = torch.maximum(torch.as_tensor(frames_np[:dn].reshape(-1, 3).max(0), device=dev), tgt.max(0).values)
ctr = 0.5 * (lo + hi)
half = float((hi - lo).max()) * 0.55
vox = 2 * half / G
n_sub = min(N, 20000)
sub = x0[torch.randperm(N, device=dev)[:n_sub]]
d8 = torch.cdist(sub, sub).topk(9, largest=False).values[:, -1]
spacing = float(d8.median()) * (n_sub / N) ** (1.0 / 3.0)
sig_vox = max(0.6, a.blur * spacing / vox)


def density(x):
    p = (x - (ctr - half)) / vox - 0.5
    i0 = torch.floor(p).long()
    f = p - i0.float()
    rho = torch.zeros(G * G * G, device=dev)
    for dz_ in (0, 1):
        for dy_ in (0, 1):
            for dx_ in (0, 1):
                w = ((f[:, 0] if dx_ else 1 - f[:, 0]) * (f[:, 1] if dy_ else 1 - f[:, 1]) * (f[:, 2] if dz_ else 1 - f[:, 2]))
                i = i0 + torch.tensor([dx_, dy_, dz_], device=dev)
                ok = ((i >= 0) & (i < G)).all(1)
                idx = (i[ok, 2] * G + i[ok, 1]) * G + i[ok, 0]
                rho.index_put_((idx,), w[ok], accumulate=True)
    rho = rho.view(1, 1, G, G, G)
    r = int(3 * sig_vox)
    k = torch.exp(-torch.arange(-r, r + 1, device=dev).float() ** 2 / (2 * sig_vox * sig_vox)); k = k / k.sum()
    rho = Fn.conv3d(rho, k.view(1, 1, 1, 1, -1), padding=(0, 0, r))
    rho = Fn.conv3d(rho, k.view(1, 1, 1, -1, 1), padding=(0, r, 0))
    rho = Fn.conv3d(rho, k.view(1, 1, -1, 1, 1), padding=(r, 0, 0))
    return rho[0, 0]                                       # (G,G,G) indexed [z,y,x]


rho0 = density(x0)
occ = rho0[rho0 > 0]
rho_bulk = float(occ.median()) if occ.numel() else 1.0
if str(a.iso).lower() == "auto":
    # a 2x2 bundle of particles (spacing s) blurred by a 3D Gaussian sigma has a line density
    # 4/s^2 per unit length -> peak 4 / (s^2 2 pi sigma^2) particles per volume; bulk = 1/s^3
    sig_wu = sig_vox * vox
    iso_frac = min(0.5, 2.0 * spacing ** 2 / (np.pi * sig_wu ** 2))
    print(f"[photoreal] iso auto: spacing {spacing:.4f} wu, blur sigma {sig_wu:.4f} wu -> iso {iso_frac:.3f} x bulk "
          f"(two-particle filament level; single particles peak at {spacing ** 3 / ((2 * np.pi) ** 1.5 * sig_wu ** 3):.3f})",
          flush=True)
else:
    iso_frac = float(a.iso)
iso = iso_frac * rho_bulk
origin = (ctr - half).cpu().numpy() + 0.5 * vox           # world position of voxel centre (0,0,0)


cell_wu = float(np.linalg.norm(np.asarray(frames_np[0], np.float32).max(0) - np.asarray(frames_np[0], np.float32).min(0))) / a.cell_diag
min_vol = a.min_cells * cell_wu ** 3


def mesh_of(x):
    """Isosurface mesh (Open3D) of the cloud x; returns (mesh, n_components_raw, n_dropped)."""
    rho = density(x).cpu().numpy()                         # [z,y,x]
    if float(rho.max()) <= iso:
        return None, 0, 0
    v, f, _, _ = measure.marching_cubes(rho, level=iso, spacing=(vox, vox, vox))
    v = v[:, ::-1] + origin                                # (z,y,x) -> (x,y,z) world
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)),
                                  o3d.utility.Vector3iVector(f[:, ::-1].astype(np.int32)))
    comp = np.asarray(m.cluster_connected_triangles()[0])
    n_comp = int(comp.max()) + 1 if len(comp) else 0
    n_drop = 0
    if n_comp > 1 and (a.largest_only or a.min_cells > 0):
        keep = np.ones(len(comp), bool)
        if a.largest_only:
            keep = comp == int(np.bincount(comp).argmax())
        else:
            # component volumes from the signed tetra sum (marching-cubes surfaces are closed)
            vv = v.astype(np.float64); ff = f[:, ::-1]
            tet = np.einsum("ij,ij->i", vv[ff[:, 0]], np.cross(vv[ff[:, 1]], vv[ff[:, 2]])) / 6.0
            vol = np.abs(np.bincount(comp, weights=tet, minlength=n_comp))
            small = vol < min_vol
            keep = ~small[comp]
        n_drop = n_comp - int(len(np.unique(comp[keep]))) if keep.any() else n_comp
        if not keep.all():
            m.remove_triangles_by_mask(~keep)
            m.remove_unreferenced_vertices()
    if a.smooth > 0:
        m = m.filter_smooth_taubin(number_of_iterations=a.smooth)
    m.compute_vertex_normals()
    return m, n_comp, n_drop


def isolated_count(x_np):
    d = cKDTree(x_np).query(x_np, k=9, workers=-1)[0][:, -1]
    return int((d > 3.0 * np.median(d)).sum())


# ---- scene ------------------------------------------------------------------------------
W = a.res
views = [float(s) for s in a.views.split(",")]
rend = o3d.visualization.rendering.OffscreenRenderer(W, W)
scene = rend.scene
scene.set_background([0.94, 0.94, 0.935, 1.0])
scene.set_lighting(o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, (0.35, -0.85, -0.4))
scene.scene.enable_indirect_light(True)
scene.scene.set_indirect_light_intensity(38000.0)
scene.scene.enable_sun_light(True)
scene.scene.set_sun_light((0.35, -0.85, -0.4), (1.0, 0.98, 0.94), 85000.0)
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultLit"
mat.base_color = [float(c) for c in a.color.split(",")] + [1.0]
mat.base_roughness = a.rough
mat.base_metallic = a.metal
mat.base_reflectance = 0.5
ghost = o3d.visualization.rendering.MaterialRecord()
ghost.shader = "defaultLitTransparency"
ghost.base_color = [0.45, 0.55, 0.75, a.target_ghost]
ghost.base_roughness = 0.6
gmat = o3d.visualization.rendering.MaterialRecord()
gmat.shader = "defaultLit"
gmat.base_color = [0.97, 0.97, 0.965, 1.0]
gmat.base_roughness = 0.9
floor_y = float(min(tgt_np[:, 1].min(), frames_np[:dn:max(1, dn // 40)][..., 1].min())) - 0.02 * half
if a.ground:
    ground = o3d.geometry.TriangleMesh.create_box(40 * half, 0.02 * half, 40 * half)
    ground.translate([-20 * half + float(ctr[0]), floor_y - 0.02 * half, -20 * half + float(ctr[2])])
    ground.compute_vertex_normals()
    scene.add_geometry("ground", ground, gmat)
if a.target_ghost > 0:
    tm, _, _ = mesh_of(tgt)
    if tm is not None:
        scene.add_geometry("target", tm, ghost)
c = ctr.cpu().numpy()
# the camera distance that makes the bounding cube span `fill` of the frame height
dist = half / (a.fill * math.tan(math.radians(a.fov) / 2.0))


def render_views(m):
    imgs = []
    if m is not None:
        scene.add_geometry("body", m, mat)
    for az in views:
        el = math.radians(a.elev); az_r = math.radians(az)
        eye = c + dist * np.array([math.cos(el) * math.sin(az_r), math.sin(el), math.cos(el) * math.cos(az_r)])
        rend.setup_camera(a.fov, c.tolist(), eye.tolist(), [0.0, 1.0, 0.0])
        imgs.append(np.asarray(rend.render_to_image()))
    if m is not None:
        scene.remove_geometry("body")
    return np.concatenate(imgs, axis=1)


def label(img, text):
    if not text:
        return img
    try:
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(img)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", max(14, img.shape[0] // 40))
        except Exception:
            font = ImageFont.load_default()
        ImageDraw.Draw(im).text((14, 10), text, fill=(50, 50, 50), font=font)
        return np.asarray(im)
    except Exception:
        return img


if a.still >= 0:
    fr = torch.as_tensor(np.asarray(frames_np[a.still], np.float32), device=dev)
    m, n_comp, n_drop = mesh_of(fr)
    img = label(render_views(m), f"{a.label} frame {a.still}  components {n_comp} (sub-cell dropped {n_drop})")
    o3d.io.write_image(a.out, o3d.geometry.Image(np.ascontiguousarray(img)))
    print(f"saved {a.out} (components {n_comp}, sub-cell dropped {n_drop})")
    sys.exit(0)

idx = list(range(0, dn, a.stride))
if idx[-1] != dn - 1:
    idx.append(dn - 1)
if a.max_frames > 0:
    idx = idx[: a.max_frames]
tmp = tempfile.mkdtemp(prefix="photoreal_")
qa = []
for k, i in enumerate(idx):
    x_np = np.asarray(frames_np[i], np.float32)
    m, n_comp, n_drop = mesh_of(torch.as_tensor(x_np, device=dev))
    n_iso = isolated_count(x_np)
    qa.append((i, n_comp, n_iso, n_drop))
    img = label(render_views(m), f"{a.label}  frame {i}/{dn - 1}")
    o3d.io.write_image(os.path.join(tmp, f"f{k:05d}.png"), o3d.geometry.Image(np.ascontiguousarray(img)))
    if k % 25 == 0:
        print(f"[photoreal] frame {k + 1}/{len(idx)} (archived {i}) components {n_comp} dropped {n_drop} isolated {n_iso}", flush=True)
n = len(idx)
for h in range(a.hold):
    os.link(os.path.join(tmp, f"f{n - 1:05d}.png"), os.path.join(tmp, f"f{n + h:05d}.png"))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f%05d.png"),
                "-movflags", "faststart", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", a.out], check=True)
with open(a.out + ".components.txt", "w") as fh:
    fh.write("archived_frame isosurface_components isolated_particles subcell_components_dropped\n")
    for i, n_comp, n_iso, n_drop in qa:
        fh.write(f"{i} {n_comp} {n_iso} {n_drop}\n")
    comps = np.array([q[1] for q in qa]); isos = np.array([q[2] for q in qa]); drops = np.array([q[3] for q in qa])
    drawn = comps - drops
    fh.write(f"# iso {iso_frac:.3f} x bulk ({'auto: two-particle filament level' if str(a.iso).lower() == 'auto' else 'fixed'}), "
             f"blur {a.blur} spacings, grid {a.grid}\n")
    fh.write(f"# frames {len(qa)}  raw components>1 in {(comps > 1).sum()} frames (max {comps.max()})  "
             f"drawn components>1 in {(drawn > 1).sum()} frames (max {drawn.max()})  "
             f"sub-cell components dropped in {(drops > 0).sum()} frames (cell {cell_wu:.3f} wu, min {a.min_cells:g} cells)  "
             f"isolated particles max {isos.max()} (frame {idx[int(isos.argmax())]})\n")
for fpath in os.listdir(tmp):
    os.remove(os.path.join(tmp, fpath))
os.rmdir(tmp)
print(f"saved {a.out} ({n} frames + {a.hold} hold; raw components>1 in {(comps > 1).sum()}/{len(qa)} frames, "
      f"drawn components>1 in {(drawn > 1).sum()}, max isolated particles {isos.max()})")
