"""d148_bead_cover.py FRAMES12 AZ K X0,Y0,X1,Y1 DX NGRID LOSS_RES -- D148c: do the walls' gap nodes (pieces joined through
the box's one-cell ring, GapPlanes margin 1) cover a bead seen on a render? The bead's particles as d148_bead2.py selects
them (projecting into the middle of the box at kept frame K, camera as render_deliverable at azimuth AZ); every half
second up to K: how many of them have a gap node of D148's set (margin 0) in their 4^3 stencil, and how many of those
also have a node of the walls' set."""
import math
import sys

import numpy as np
from scipy.spatial import cKDTree

from physmorph.mpm.gap_split import GapPlanes

fr_path, AZd, K = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
x0b, y0b, x1b, y1b = [int(v) for v in sys.argv[4].split(",")]
dx, ngrid, lres = float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
z = np.load(fr_path)
fr, raws = z["frames"], np.asarray(z["raws"])
tgt = np.asarray(z["tgt"], np.float64)
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
half = dx * ngrid / 2.0
gmin = np.full(3, -half)
_d0 = np.full(3, -half).astype(np.float32)                 # the run's loss cell in float32 (target.py): the ring's pieces
cell = float((_d0 + dx * np.full(3, ngrid, np.float32) - _d0).max() / lres)   # move with its last digits (560 vs 540)
m0 = GapPlanes(tgt, dx, gmin, (ngrid,) * 3, cell).mask.reshape((ngrid,) * 3)
m1 = GapPlanes(tgt, dx, gmin, (ngrid,) * 3, cell, keep_boxes=True, margin=1).mask.reshape((ngrid,) * 3)
offs = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)


def hits(xs, mask3):
    b = np.floor((xs - gmin) / dx).astype(int) - 1
    h = np.zeros(len(xs), bool)
    for o in offs:
        n = b + o
        ok = ((n >= 0) & (n < ngrid)).all(1)
        h[ok] |= mask3[n[ok, 0], n[ok, 1], n[ok, 2]]
    return h


print(f"frame {K} ({raws[K] / 240:.2f} s): {len(sel)} particles in the bead's middle", flush=True)
if len(sel):
    ks = sorted({k for k in range(K + 1) if raws[k] % 120 == 0} | {K})
    for k in ks:
        xs = np.asarray(fr[k], np.float64)[sel]
        h0, h1 = hits(xs, m0), hits(xs, m1)
        print(f"  {raws[k] / 240:5.2f} s: near D148's nodes {int(h0.sum())}, of them near a walls node "
              f"{int((h0 & h1).sum())}; near a walls node only {int((h1 & ~h0).sum())}", flush=True)
