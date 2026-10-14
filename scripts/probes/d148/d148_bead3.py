"""d148_bead3.py FRAMES12 AZ K X0,Y0,X1,Y1 DX NGRID LOSS_RES [K0,K1,...] -- D148 forensic, the sides of a bead's particles
(CUDA): the bead's particles as d148_bead2.py selects them; at kept frames that start windows (raw a multiple of 40),
the run's own side rule recomputed on the frame (the debiased transport image of every particle on the gate's grid,
blur one cell, as the objective computes it at a window start); per frame, over the (particle, gap node) pairs in
reach: the share on side 1 by the label, by the particle's position, and by its position at the run's end; the share
where the label differs from the end; and the share of pairs whose particle's side has no other particle within the
node's reach (alone on its side, by the label)."""
import math
import os
import sys

import physmorph  # noqa: F401  (before torch)
import numpy as np
import torch
from scipy.spatial import cKDTree

from physmorph.losses.grid_ot import grid_transport_displacement
from physmorph.mpm.gap_split import GapPlanes
from physmorph.losses.volumetric import target_mass_grid

fr_path, AZd, K = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
x0b, y0b, x1b, y1b = [int(v) for v in sys.argv[4].split(",")]
dx, ngrid, lres = float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
z = np.load(fr_path)
fr, raws = z["frames"], np.asarray(z["raws"])
tgt = np.asarray(z["tgt"], np.float64)
ks = [int(v) for v in sys.argv[8].split(",")] if len(sys.argv) > 8 else \
    [k for k in range(len(raws)) if raws[k] % 40 == 0 and k <= K + 20]
a = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::10], 9)[0][:, 8]))
center = tgt.mean(0)
radius = float(np.linalg.norm(tgt - center, axis=1).max())
W, H, AZ, EL = 3840, 2160, math.radians(AZd), math.radians(18.0)
cam = center + 3.6 * radius * np.array([math.cos(EL) * math.sin(AZ), math.sin(EL), math.cos(EL) * math.cos(AZ)])
fwd = (center - cam) / np.linalg.norm(center - cam)
right = np.cross([0.0, 1.0, 0.0], fwd)
right /= np.linalg.norm(right)
up = np.cross(fwd, right)
tan_y = math.tan(math.radians(15.0))
tan_x = tan_y * W / H


def pixels(p):
    d = p - cam
    xc, yc, zc = d @ right, d @ up, d @ fwd
    return np.stack((((xc / zc) / tan_x + 1) / 2 * W - 0.5, (1 - (yc / zc) / tan_y) / 2 * H - 0.5), 1)


wq, hq = (x1b - x0b) / 4.0, (y1b - y0b) / 4.0
q = pixels(np.asarray(fr[K], np.float64))
sel = np.nonzero((q[:, 0] >= x0b + wq) & (q[:, 0] <= x1b - wq) & (q[:, 1] >= y0b + hq) & (q[:, 1] <= y1b - hq))[0]
print(f"bead at frame {K} ({raws[K] / 240:.2f} s): particles {sel.tolist()}", flush=True)
half = dx * ngrid / 2.0
gmin = np.full(3, -half)
gp = GapPlanes(tgt, dx, gmin, (ngrid,) * 3, 2.0 * half / lres)
dev = "cuda"
lg = torch.tensor(gmin, dtype=torch.float32, device=dev)
tg_t = torch.tensor(tgt, dtype=torch.float32, device=dev)
tgrid = target_mass_grid(tg_t, torch.ones(len(tgt), device=dev), lg, dx, (ngrid,) * 3)
xend = np.asarray(fr[-1], np.float64)


def pairs_of(xs):
    b = np.floor((xs - gmin) / dx).astype(int) - 1
    P, G = [], []
    for p, bb in enumerate(b):
        for oi in range(4):
            for oj in range(4):
                for ok in range(4):
                    nn = bb + [oi, oj, ok]
                    if (nn < 0).any() or (nn >= ngrid).any():
                        continue
                    g = int((nn[0] * ngrid + nn[1]) * ngrid + nn[2])
                    if gp.mask[g]:
                        P.append(p)
                        G.append(g)
    return np.asarray(P, int), np.asarray(G, int)


for k in ks:
    xa = np.asarray(fr[k], np.float64)
    xt = torch.tensor(xa, dtype=torch.float32, device=dev)
    disp = grid_transport_displacement(xt, torch.ones(len(xa), device=dev), tgrid, lg, dx, (ngrid,) * 3,
                                       eps=dx ** 2, iters=1600, tol=1e-2).double().cpu().numpy()
    qa = xa + disp
    P, G = pairs_of(xa[sel])
    if not len(P):
        print(f"t {raws[k] / 240:5.2f} s: no gap node in reach", flush=True)
        continue
    lab = gp.side(qa[sel][P], G)
    pos = gp.side(xa[sel][P], G)
    end = gp.side(xend[sel][P], G)
    # alone on its side: no other particle within the node's reach (|x - x_g|_inf < 2 dx) with the same label
    tree = cKDTree(xa)
    alone = 0
    for p, g, s in zip(P, G, lab):
        ii = np.stack(np.unravel_index(g, (ngrid,) * 3))
        xg = gmin + ii * dx
        nb = tree.query_ball_point(xg, 2 * dx * math.sqrt(3))
        nb = [j for j in nb if j != sel[p] and np.abs(xa[j] - xg).max() < 2 * dx]
        if not nb:
            alone += 1
            continue
        sn = gp.side(qa[nb], np.full(len(nb), g))
        alone += int(not (sn == s).any())
    print(f"t {raws[k] / 240:5.2f} s: {len(P)} pairs; side 1 by the label {lab.mean():.2f}, by position {pos.mean():.2f}, "
          f"by the end {end.mean():.2f}; label != end {np.mean(lab != end):.2f}; alone on its side {alone / len(P):.2f}; "
          f"|image - x| {np.median(np.linalg.norm(disp[sel], axis=1)) / a:.1f} pitches", flush=True)
