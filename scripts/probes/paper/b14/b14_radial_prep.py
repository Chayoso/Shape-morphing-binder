"""b14_radial_prep.py SRC TGT SAVE_DIR NAME -- D145 B14: prep_pair.py's CORR=radial data without trimesh's ray engine (env nsd has
no rtree, and ~/miniforge3 is not to be touched): each of 5k source samples is paired with the FIRST hit of the ray from
the pair's centre (the origin after prep_pair's common centring) through it, by a vectorised Moller-Trumbore test against
every target triangle; a ray that hits nothing falls back to the nearest point of a 400k-point surface sample of the
target (prep_pair.py's fallback is the exact closest point). Everything else as prep_pair.py (its own functions:
normalise, sample, make_dptc; 20k points per shape; extent 1; seed 0). Run from the Implicit-surf-Deformation root."""
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/baselines_prep")
import prep_pair as pp  # noqa: E402
from flax import serialization  # noqa: E402


def first_hit(V, F, d):
    v0 = V[F[:, 0]]
    e1 = V[F[:, 1]] - v0
    e2 = V[F[:, 2]] - v0
    n = len(d)
    tb = np.full(n, np.inf)
    fb = np.full(n, -1)
    b = max(1, int(2e6 // len(F)))
    for i in range(0, n, b):
        D = d[i:i + b][:, None, :]
        p = np.cross(D, e2[None])
        det = (e1[None] * p).sum(-1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = -v0[None]
        u = (s * p).sum(-1) * inv
        q = np.cross(s, e1[None])
        v = (D * q).sum(-1) * inv
        t = (e2[None] * q).sum(-1) * inv
        t = np.where(ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9), t, np.inf)
        j = t.argmin(1)
        tj = t[np.arange(len(j)), j]
        tb[i:i + b] = tj
        fb[i:i + b] = np.where(np.isfinite(tj), j, -1)
    return tb, fb


src, tgt, save, tname = sys.argv[1:5]
np.random.seed(0)
mx = trimesh.load(src, force="mesh", process=True)
my = trimesh.load(tgt, force="mesh", process=True)
info = pp.normalise(mx, my, "common", 1.0)
vx, nx = pp.sample(mx, 5000)
d = vx / np.linalg.norm(vx, axis=1, keepdims=True)
t, f = first_hit(np.asarray(my.vertices, np.float64), np.asarray(my.faces), d)
hit = np.isfinite(t)
vy = d * np.where(hit, t, 0.0)[:, None]
ny = np.asarray(my.face_normals)[np.maximum(f, 0)]
if (~hit).any():
    S, fid = trimesh.sample.sample_surface(my, 400000)
    _, j = cKDTree(S).query(vx[~hit])
    vy[~hit] = S[j]
    ny[~hit] = np.asarray(my.face_normals)[fid[j]]
print(f"radial: {hit.sum()}/{len(hit)} ray hits, {(~hit).sum()} nearest-sample fallbacks", flush=True)
px, npx = pp.sample(mx, 20000)
py, npy = pp.sample(my, 20000)
dx = pp.make_dptc(vx, nx, px, npx)
dy = pp.make_dptc(vy, ny, py, npy)
name_x, name_y = "00_sphere", "01_" + tname
for sub in ("mesh", "ptc", "train"):
    os.makedirs(os.path.join(save, sub), exist_ok=True)
with open(os.path.join(save, "train", f"{name_x}-{name_y}.flax"), "wb") as fh:
    fh.write(serialization.to_bytes([dx, dy]))
mx.export(os.path.join(save, "mesh", name_x + ".ply"))
my.export(os.path.join(save, "mesh", name_y + ".ply"))
trimesh.PointCloud(np.concatenate([vx, px])).export(os.path.join(save, "ptc", name_x + ".ply"))
trimesh.PointCloud(np.concatenate([vy, py])).export(os.path.join(save, "ptc", name_y + ".ply"))
info.update(src=os.path.abspath(src), tgt=os.path.abspath(tgt), corr="radial", n_points=20000, n_corr=5000, seed=0,
            radial_hits=int(hit.sum()), fallback="nearest point of a 400k surface sample (b14_radial_prep.py; no rtree)")
json.dump(info, open(os.path.join(save, "pair_info.json"), "w"), indent=2)
print(f"wrote {save}: src watertight {mx.is_watertight}, tgt watertight {my.is_watertight}", flush=True)
