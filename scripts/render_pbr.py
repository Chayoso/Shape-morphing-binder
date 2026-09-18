"""PBR still of a particle state: surface splatting + physically based shading. CPU, numpy.

Why not the 3DGS photoreal path for solids: `render_photoreal.py` was tuned on shell-era
layers; on a 40k solid the isotropic splats give a "rock" and the surfel mode needs a surface
extraction it does not have (docs/floaters.md, 2026-09-04). This renderer reconstructs the
visible surface in screen space instead — orthographic z-buffer of per-particle disks sized by
the local spacing (surface splatting), a bilateral-smoothed depth map, normals from depth,
Cook–Torrance/GGX shading with three lights, hemispherical ambient and a depth-based
ambient-occlusion term. The raw particle state is the only input; nothing is fitted.

Usage: python render_pbr.py --npz run.npz --out img.png [--frame -1|k] [--target]
       [--res 1200] [--azim 35] [--elev 18] [--albedo 0.80,0.78,0.74] [--rough 0.45]
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frame", type=int, default=None, help="frame index (default: delivered frame)")
ap.add_argument("--target", action="store_true", help="render the target cloud instead")
ap.add_argument("--res", type=int, default=1200)
ap.add_argument("--azim", type=float, default=35.0, help="camera azimuth, degrees")
ap.add_argument("--elev", type=float, default=18.0, help="camera elevation, degrees")
ap.add_argument("--albedo", default="0.80,0.78,0.74")
ap.add_argument("--rough", type=float, default=0.45)
ap.add_argument("--metal", type=float, default=0.0)
ap.add_argument("--splat_k", type=float, default=1.35, help="disk radius in local NN spacings")
ap.add_argument("--extent", type=float, default=0.0, help="half-width of the view in wu (0 = fit)")
ap.add_argument("--smooth", type=float, default=0.9, help="depth smoothing sigma in median splat radii")
a = ap.parse_args()

d = np.load(a.npz)
import sys as _sys, os as _os  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from physmorph.sampling.orientation import orient_archive  # noqa: E402
_fr, _tg, _sr, _orient = orient_archive(d, a.npz)              # y-up (physmorph/sampling/orientation.json)
d = {k: d[k] for k in d.files}; d["frames"], d["tgt"] = _fr, _tg
if _sr is not None:
    d["src"] = _sr
if a.target:
    x = d["tgt"].astype(np.float32)
else:
    n_frames = len(d["frames"])
    fi = a.frame
    if fi is None:
        fi = int(d["deliver_n"]) - 1 if "deliver_n" in d.files else n_frames - 1
    if fi < 0:
        fi = n_frames + fi
    x = d["frames"][fi].astype(np.float32)
N, RES = len(x), a.res

# ---- camera (orthographic) --------------------------------------------------------------
az, el = np.deg2rad(a.azim), np.deg2rad(a.elev)
fwd = np.array([-np.cos(el) * np.sin(az), -np.sin(el), -np.cos(el) * np.cos(az)], np.float32)  # view dir
fwd /= np.linalg.norm(fwd)
up0 = np.array([0.0, 1.0, 0.0], np.float32)
right = np.cross(fwd, up0); right /= np.linalg.norm(right)
up = np.cross(right, fwd)
ctr = 0.5 * (d["tgt"].max(0) + d["tgt"].min(0)).astype(np.float32)
ext = a.extent if a.extent > 0 else float(np.abs((d["tgt"] - ctr) @ np.stack([right, up]).T).max()) * 1.15
p = (x - ctr) @ np.stack([right, up]).T            # screen coords in wu
z = (x - ctr) @ fwd                                # depth along the view (larger = farther)
px = (p + ext) / (2 * ext) * RES                   # pixel coords (u right, v up)
pix_per_wu = RES / (2 * ext)

# ---- surface splatting: z-buffer of disks sized by the local spacing -------------------
sp = cKDTree(x).query(x, k=9, workers=-1)[0][:, 1:].mean(1)      # local NN spacing
rad = np.clip(a.splat_k * sp * pix_per_wu, 1.5, 14.0)             # disk radius in px
order = np.argsort(-z)                                            # far to near (painter for ties)
depth = np.full((RES, RES), np.inf, np.float32)
idmap = np.full((RES, RES), -1, np.int64)
rmax = int(np.ceil(rad.max()))
yy, xx = np.mgrid[-rmax:rmax + 1, -rmax:rmax + 1]
for i in order:
    r = rad[i]
    cx, cy = px[i, 0], px[i, 1]
    ix, iy = int(round(cx)), int(round(cy))
    if ix < -rmax or iy < -rmax or ix >= RES + rmax or iy >= RES + rmax:
        continue
    ri = int(np.ceil(r))
    sub_y, sub_x = yy[rmax - ri:rmax + ri + 1, rmax - ri:rmax + ri + 1], xx[rmax - ri:rmax + ri + 1, rmax - ri:rmax + ri + 1]
    dd = (sub_x + ix - cx) ** 2 + (sub_y + iy - cy) ** 2
    mask = dd <= r * r
    # a slightly curved splat: nearer at the centre (sphere cap of radius r in wu units)
    zc = z[i] - np.sqrt(np.maximum(r * r - dd, 0.0)) / pix_per_wu * 0.5
    ys = sub_y + iy; xs = sub_x + ix
    ok = mask & (ys >= 0) & (ys < RES) & (xs >= 0) & (xs < RES)
    ys, xs, zc = ys[ok], xs[ok], zc[ok]
    cur = depth[ys, xs]
    upd = zc < cur
    depth[ys[upd], xs[upd]] = zc[upd]
    idmap[ys[upd], xs[upd]] = i
cover = np.isfinite(depth)

# ---- depth smoothing (bilateral: keep silhouettes) and normals --------------------------
dz = depth.copy(); dz[~cover] = 0.0
w = cover.astype(np.float32)
sig = max(1.0, a.smooth * np.median(rad))
num = ndimage.gaussian_filter(dz * w, sig); den = ndimage.gaussian_filter(w, sig)
sm = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0)
# bilateral guard: fall back to the raw depth where smoothing jumps across a silhouette
jump = np.abs(sm - dz) > 0.6 * np.median(sp) * 4
sm = np.where(jump & cover, dz, sm)
# a second, gentler pass on the guarded field removes the remaining particle bumps
sm = np.where(cover, ndimage.gaussian_filter(sm * w, 0.5 * sig) / np.maximum(ndimage.gaussian_filter(w, 0.5 * sig), 1e-6), 0.0)
gy, gx = np.gradient(sm)                                  # in wu per px
gx *= pix_per_wu; gy *= pix_per_wu                        # -> dimensionless slopes
n = np.stack([-gx, gy, np.ones_like(gx)], -1)             # view-space normal (x right, y up, z toward camera)
n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
n[~cover] = 0.0

# ---- ambient occlusion (screen-space, depth based) ----------------------------------------
ao = np.ones((RES, RES), np.float32)
for rr in (3, 6, 12, 24):
    m = ndimage.uniform_filter(sm, rr * 2 + 1)
    occ = np.clip((sm - m) / (0.03 * ext), 0, 1)          # deeper than the neighbourhood -> occluded
    ao -= 0.18 * occ
ao = np.clip(ao, 0.35, 1.0)

# ---- Cook–Torrance / GGX shading -----------------------------------------------------------
albedo = np.array([float(c) for c in a.albedo.split(",")], np.float32)
rough, metal = a.rough, a.metal
V = np.array([0.0, 0.0, 1.0], np.float32)
lights = [((0.45, 0.65, 0.62), (1.0, 0.98, 0.94), 2.6),   # key
          ((-0.7, 0.25, 0.65), (0.80, 0.86, 1.0), 1.1),   # fill
          ((0.2, 0.5, -0.85), (1.0, 1.0, 1.0), 1.6)]      # rim (behind)
F0 = 0.04 * (1 - metal) + albedo * metal
NdV = np.clip(n @ V, 1e-4, 1.0)
col = np.zeros((RES, RES, 3), np.float32)
for L, c, inten in lights:
    L = np.array(L, np.float32); L /= np.linalg.norm(L)
    H = L + V; H /= np.linalg.norm(H)
    NdL = np.clip(n @ L, 0.0, 1.0)
    NdH = np.clip(n @ H, 0.0, 1.0)
    VdH = float(np.clip(V @ H, 0.0, 1.0))
    a2 = (rough * rough) ** 2
    Dg = a2 / (np.pi * (NdH * NdH * (a2 - 1) + 1) ** 2 + 1e-9)
    k = (rough + 1) ** 2 / 8
    G = (NdV / (NdV * (1 - k) + k)) * (NdL / (NdL * (1 - k) + k + 1e-9))
    Fr = F0 + (1 - F0) * (1 - VdH) ** 5
    spec = (Dg * G)[..., None] * Fr / (4 * NdV * NdL + 1e-4)[..., None]
    kd = (1 - Fr) * (1 - metal)
    col += (kd * albedo / np.pi + spec) * (NdL[..., None] * inten) * np.array(c, np.float32)
# hemispherical ambient (sky above, ground below) scaled by AO
sky, ground = np.array([0.62, 0.68, 0.78], np.float32), np.array([0.30, 0.27, 0.24], np.float32)
hemi = 0.5 * (1 + n[..., 1:2])
col += albedo * (hemi * sky + (1 - hemi) * ground) * 0.55 * ao[..., None]
# tone map + background
col = col / (1 + col)
col = np.clip(col, 0, 1) ** (1 / 2.2)
bg_t = np.linspace(0, 1, RES)[:, None]
bg = (0.97 - 0.06 * bg_t)[..., None] * np.array([1.0, 0.995, 0.985], np.float32)
bg = np.broadcast_to(bg, (RES, RES, 3)).copy()
# contact shadow on the background from the coverage silhouette
shadow = ndimage.gaussian_filter(cover.astype(np.float32), 14)
shifted = np.roll(np.roll(shadow, 22, axis=0), 8, axis=1)
bg *= (1 - 0.35 * shifted)[..., None]
img = np.where(cover[..., None], col, bg)
img = np.clip(img, 0, 1)[::-1]          # v up -> rows top-down
Image.fromarray((img * 255).astype(np.uint8)).save(a.out)
print(f"saved {a.out}  N={N} covered {cover.mean():.1%} spacing med {np.median(sp):.4f} wu splat radius med {np.median(rad):.1f} px")
