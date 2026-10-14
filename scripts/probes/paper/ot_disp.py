"""ot_disp.py RUN_FRAMES_NPZ OUT_NPZ [K=21] -- D145 B6: OT displacement interpolation on the same 300k particles.

The source is the run's first kept frame (the pipeline's source sample at that seed), the target the run's own target
sample (key tgt): both are the samples our morph starts from and is scored against. The entropic OT plan between their
masses is solved on a grid whose cell is one sampling pitch (0.708 x the median 8th-neighbour distance of the target
sample), blur one cell (eps = cell^2), with the pipeline's separable log-domain Sinkhorn
(physmorph.losses.grid_ot.GridSinkhornLoss.solve, epsilon scaling from the grid's diameter). The barycentric map of
every grid node, T(i) = sum_j pi_ij y_j / sum_j pi_ij, is computed separably (the coordinate along each axis as an extra
log-weight), its displacement T(i) - y_i interpolated trilinearly to the particles, and the frames are
x(t) = x0 + t D(x0), t = 0, 1/(K-1), ..., 1: transport alone, no physics. Written in the pipeline's archive format
(frames, raws = 0..K-1, tgt, deliver_n)."""
import sys

import physmorph  # noqa: F401  (before torch)
import numpy as np
import torch
from scipy.spatial import cKDTree

from physmorph.losses.grid_ot import GridSinkhornLoss
from physmorph.losses.volumetric import rasterize_mass

src_path, out_path = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 21
dev = "cuda"
z = np.load(src_path, allow_pickle=True, mmap_mode="r")
x0 = np.asarray(z["frames"][0], np.float32)
tgt = np.asarray(z["tgt"], np.float32)
pitch = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9)[0][:, 8]))
dx = pitch
lo = np.minimum(x0.min(0), tgt.min(0)) - 4 * dx
hi = np.maximum(x0.max(0), tgt.max(0)) + 4 * dx
dims = tuple(int(np.ceil((h - l) / dx)) + 1 for l, h in zip(lo, hi))
grid_min = torch.tensor(lo, dtype=torch.float32, device=dev)
print(f"N {len(x0)}, pitch {pitch:.5f} wu, grid {dims} ({np.prod(dims) / 1e6:.2f} M nodes)", flush=True)


def mass(x):
    xt = torch.tensor(x, device=dev)
    return rasterize_mass(xt, torch.full((len(xt),), 1.0 / len(xt), device=dev), grid_min, dx, dims)


a, b = mass(x0), mass(tgt)
a, b = a / a.sum(), b / b.sum()
ot = GridSinkhornLoss(b, grid_min, dx, dims, eps=dx * dx, iters=20000, tol=1e-3)
f, g = ot.solve(a, b)
lb = b.log()
t0 = ot.transform(g, lb, ot.eps)                       # -eps LSE_j[(g_j - C_ij)/eps + log b_j]
axes = [torch.arange(n, device=dev, dtype=torch.float32) for n in dims]
D = torch.zeros((int(np.prod(dims)), 3), device=dev)
for d in range(3):
    shape = [1, 1, 1]
    shape[d] = dims[d]
    coord = (axes[d] + 1.0).reshape(shape).expand(dims).reshape(-1)          # node index + 1 > 0 (in cells)
    tw = ot.transform(g, lb + coord.log(), ot.eps)
    T_d = torch.exp((t0 - tw) / ot.eps)                # sum_j w_ij (k_j + 1) / sum_j w_ij, in cells
    D[:, d] = (T_d - coord) * dx                        # the node's barycentric displacement, wu
D = D.reshape(*dims, 3)
# trilinear gather of D at the source particles
x = torch.tensor(x0, device=dev)
rel = (x - grid_min) / dx
base = torch.floor(rel).long()
fr = rel - base.float()
disp = torch.zeros_like(x)
for ox in (0, 1):
    for oy in (0, 1):
        for oz in (0, 1):
            w = ((fr[:, 0] if ox else 1 - fr[:, 0]) * (fr[:, 1] if oy else 1 - fr[:, 1])
                 * (fr[:, 2] if oz else 1 - fr[:, 2]))
            disp += w[:, None] * D[base[:, 0] + ox, base[:, 1] + oy, base[:, 2] + oz]
disp = disp.cpu().numpy()
frames = np.stack([x0 + (k / (K - 1)) * disp for k in range(K)]).astype(np.float32)
end = frames[-1]
dist, _ = cKDTree(tgt).query(end)
print(f"end: distance to the nearest target point median {np.median(dist) / pitch:.3f}, p99 {np.quantile(dist, .99) / pitch:.3f}, "
      f"max {dist.max() / pitch:.2f} pitches; |D| median {np.median(np.linalg.norm(disp, axis=1)):.4f} wu", flush=True)
np.savez(out_path, frames=frames, raws=np.arange(K), tgt=tgt, deliver_n=np.int64(K))
print("wrote", out_path, flush=True)
