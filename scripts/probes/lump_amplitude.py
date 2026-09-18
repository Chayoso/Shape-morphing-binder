"""Surface-lump amplitude: how much of the morph's surface error lives at the loss-cell scale.

The morph surface is the isosurface the photoreal renderer draws (same density: CIC + Gaussian
blur of --blur spacings on a --grid^3 box, level = the two-particle-filament level). For every
surface vertex, d = signed-free distance to the nearest TARGET point. That scalar field on the
surface is split by scale with two Laplacian smoothings over the mesh graph: a wide one
(radius ~ r_wide wu, the global shape error) and a narrow one (radius ~ r_narrow wu, the
particle noise). The LUMP band is the difference: lump_amp = RMS over vertices of
(d_narrow - d_wide). The target cloud rendered the same way gives the floor of the measure.

    lump_amplitude.py <out_dir> <run> [frame|-1|-2] [--grid 160] [--blur 1.5] [--r_wide 0.6] [--r_narrow 0.15]
-1 = delivered end frame, -2 = the target cloud itself (floor).
prints: run frame lump_amp_wu wide_rms_wu raw_mean_wu edge_wu n_vertices
"""
import argparse
import math
import sys

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import sparse
from scipy.spatial import cKDTree
from skimage import measure

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("run"); ap.add_argument("frame", type=int, nargs="?", default=-1)
ap.add_argument("--grid", type=int, default=160); ap.add_argument("--blur", type=float, default=1.5)
ap.add_argument("--r_wide", type=float, default=0.6); ap.add_argument("--r_narrow", type=float, default=0.15)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
z = np.load(f"{a.out}/{a.run}_render_full_dt_iso_nn.npz")
frames = z["frames"]; tgt = np.asarray(z["tgt"], np.float32)
dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames)
fi = dn - 1 if a.frame == -1 else a.frame
x_np = tgt if a.frame == -2 else np.asarray(frames[fi], np.float32)
G = a.grid
allp = np.concatenate([frames[:dn:max(1, dn // 40)].reshape(-1, 3), tgt], 0)
lo, hi = allp.min(0), allp.max(0)
ctr = torch.as_tensor((lo + hi) / 2, device=dev, dtype=torch.float32)
half = float((hi - lo).max()) * 0.55
vox = 2 * half / G
x0 = torch.as_tensor(frames[0], device=dev)
N = len(x0); n_sub = min(N, 20000)
sub = x0[torch.randperm(N, device=dev)[:n_sub]]
d8 = torch.cdist(sub, sub).topk(9, largest=False).values[:, -1]
spacing = float(d8.median()) * (n_sub / N) ** (1.0 / 3.0)
sig_vox = max(0.6, a.blur * spacing / vox)


def density(x):
    p = (x - (ctr - half)) / vox - 0.5
    i0 = torch.floor(p).long(); f = p - i0.float()
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
    return rho[0, 0]


rho0 = density(x0); occ = rho0[rho0 > 0]; bulk = float(occ.median())
iso = min(0.5, 2.0 * spacing ** 2 / (math.pi * (sig_vox * vox) ** 2)) * bulk
rho = density(torch.as_tensor(x_np, device=dev)).cpu().numpy()
v, f, _, _ = measure.marching_cubes(rho, level=iso, spacing=(vox, vox, vox))
v = v[:, ::-1] + ((ctr - half).cpu().numpy() + 0.5 * vox)
# largest component only (the body)
import open3d as o3d
m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v.astype(np.float64)), o3d.utility.Vector3iVector(f[:, ::-1].astype(np.int32)))
comp = np.asarray(m.cluster_connected_triangles()[0])
keep = comp == int(np.bincount(comp).argmax())
m.remove_triangles_by_mask(~keep); m.remove_unreferenced_vertices()
v = np.asarray(m.vertices); f = np.asarray(m.triangles)
# distance of every surface vertex to the nearest target point
d = cKDTree(tgt).query(v, workers=-1)[0].astype(np.float64)
# mesh-graph Laplacian smoothing: n Jacobi iterations spread over ~ edge * sqrt(n)
e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
edge = float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).mean())
nV = len(v)
A_ = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nV, nV)).tocsr()
A_ = A_ + A_.T
deg = np.asarray(A_.sum(1)).ravel().clip(min=1)
W = sparse.diags(1.0 / deg) @ A_                                   # neighbour average


def smooth(s, radius_wu):
    n = int(max(1, round((radius_wu / edge) ** 2)))
    out = s.copy()
    for _ in range(n):
        out = 0.5 * out + 0.5 * (W @ out)
    return out, n


d_wide, n_w = smooth(d, a.r_wide)
d_narrow, n_n = smooth(d, a.r_narrow)
lump = d_narrow - d_wide
print(f"{a.run} frame {a.frame}: lump_amp {np.sqrt((lump ** 2).mean()):.4f} wu | wide_rms {np.sqrt((d_wide ** 2).mean()):.4f} wu | "
      f"raw_mean {d.mean():.4f} wu | edge {edge:.4f} wu ({n_n}/{n_w} it) | vertices {nV}")
