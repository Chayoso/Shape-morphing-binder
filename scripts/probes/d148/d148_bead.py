"""d148_bead.py FRAMES12 AZ K X0,Y0,X1,Y1 DX NGRID LOSS_RES -- D148 forensic of a detached bead seen on a rendered frame:
the particles apart from the body (components at 1.5 pitches, as d140_image.py) that project into the box at kept frame
K; then, over the kept frames, whether they are apart, their distance to the nearest body particle (pitches), their
distance to the target (pitches, + outside), and whether their stencil holds gap nodes (gap_split.GapPlanes on the
MPM grid, the loss cell) and on which side of those nodes' planes they lie."""
import math
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_split import GapPlanes  # noqa: E402

fr_path, AZd, K = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
bx = [int(v) for v in sys.argv[4].split(",")]
dx, ngrid, lres = float(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
z = np.load(fr_path)
fr, raws = z["frames"], np.asarray(z["raws"])
tgt = np.asarray(z["tgt"], np.float64)
ttree = cKDTree(tgt)
a = 0.708 * float(np.median(ttree.query(tgt[::10], 9)[0][:, 8]))
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


def comps(x):
    pairs = cKDTree(x).query_pairs(1.5 * a, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    _, cl = connected_components(g, directed=False)
    return cl, int(np.bincount(cl).argmax())


x = np.asarray(fr[K], np.float64)
cl, b = comps(x)
ap = np.nonzero(cl != b)[0]
q = pixels(x[ap])
sel = ap[(q[:, 0] >= bx[0] - 6) & (q[:, 0] <= bx[2] + 6) & (q[:, 1] >= bx[1] - 6) & (q[:, 1] <= bx[3] + 6)]
print(f"frame {K} ({raws[K] / 240:.2f} s): {len(sel)} apart particles in the box; their components "
      f"{np.unique(cl[sel]).size}; pitch {a:.4f}")
if not len(sel):
    sys.exit()
half = dx * ngrid / 2.0
gmin = np.full(3, -half)
gp = GapPlanes(tgt, dx, gmin, (ngrid,) * 3, 2.0 * half / lres)
for k in range(0, len(raws), max(1, len(raws) // 60)):
    xk = np.asarray(fr[k], np.float64)
    clk, bk = comps(xk)
    apart = clk[sel] != bk
    body = np.nonzero(clk == bk)[0]
    d_body = cKDTree(xk[body]).query(xk[sel])[0] / a
    d_tgt = ttree.query(xk[sel])[0] / a
    bse = np.floor((xk[sel] - gmin) / dx).astype(int) - 1
    ng, s1 = 0, 0
    for p, bb in enumerate(bse):
        for oi in range(4):
            for oj in range(4):
                for ok in range(4):
                    n = bb + [oi, oj, ok]
                    if (n < 0).any() or (n >= ngrid).any():
                        continue
                    g = (n[0] * ngrid + n[1]) * ngrid + n[2]
                    if gp.mask[g]:
                        ng += 1
                        s1 += int(np.dot(xk[sel][p] - gp.m[g], gp.n[g]) > 0)
    print(f"t {raws[k] / 240:6.2f} s: apart {int(apart.sum()):3d}/{len(sel)}, to the body {np.median(d_body):6.2f} "
          f"pitches (max {d_body.max():.2f}), to the target {np.median(d_tgt):5.2f}, centroid "
          f"{np.round(xk[sel].mean(0), 3).tolist()}, gap-node pairs {ng} (position side 1: {s1})", flush=True)
