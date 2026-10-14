"""d148_bead2.py FRAMES12 AZ K X0,Y0,X1,Y1 DX NGRID LOSS_RES -- D148 forensic of a bead on a rendered frame: the particles
projecting into the middle of its box (shrunk by a quarter each side) at kept frame K are the bead's (in a hole nothing
else projects there). For them: how many, their links (1.5 pitches) to the rest of the body (a thin neck or none), and,
every half second, where they are relative to the target (pitches; + outside), how many of their stencil's nodes are
gap nodes, the share of those pairs on each side of the node's plane by their position, and their centroid; at the end
which side of those planes they ended on."""
import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_split import GapPlanes  # noqa: E402

fr_path, AZd, K = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
x0b, y0b, x1b, y1b = [int(v) for v in sys.argv[4].split(",")]
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


wq, hq = (x1b - x0b) / 4.0, (y1b - y0b) / 4.0
x = np.asarray(fr[K], np.float64)
q = pixels(x)
sel = np.nonzero((q[:, 0] >= x0b + wq) & (q[:, 0] <= x1b - wq) & (q[:, 1] >= y0b + hq) & (q[:, 1] <= y1b - hq))[0]
tree = cKDTree(x)
nb = tree.query_ball_point(x[sel], 1.5 * a)
ss = set(sel.tolist())
out_links = sum(sum(1 for j in lst if j not in ss) for lst in nb)
in_links = sum(sum(1 for j in lst if j in ss) - 1 for lst in nb)
touch = len({j for lst in nb for j in lst if j not in ss})
print(f"frame {K} ({raws[K] / 240:.2f} s): {len(sel)} particles project into the bead's middle; links inside "
      f"{in_links // 2}, to the rest {out_links} (to {touch} other particles); depth spread "
      f"{np.ptp((x[sel] - cam) @ fwd) / a:.1f} pitches", flush=True)
if not len(sel):
    sys.exit()
half = dx * ngrid / 2.0
gmin = np.full(3, -half)
gp = GapPlanes(tgt, dx, gmin, (ngrid,) * 3, 2.0 * half / lres)


def gapstats(xs):
    b = np.floor((xs - gmin) / dx).astype(int) - 1
    ng = s1 = 0
    nodes = set()
    for p, bb in enumerate(b):
        for oi in range(4):
            for oj in range(4):
                for ok in range(4):
                    nn = bb + [oi, oj, ok]
                    if (nn < 0).any() or (nn >= ngrid).any():
                        continue
                    g = int((nn[0] * ngrid + nn[1]) * ngrid + nn[2])
                    if gp.mask[g]:
                        ng += 1
                        nodes.add(g)
                        s1 += int(np.dot(xs[p] - gp.m[g], gp.n[g]) > 0)
    return ng, s1, len(nodes)


step = max(1, int(round(10 / ((raws[1] - raws[0]) / 12.0)))) if len(raws) > 1 else 1
for k in list(range(0, len(raws), 10)) + [K, len(raws) - 1]:
    xs = np.asarray(fr[k], np.float64)[sel]
    dt_ = ttree.query(xs)[0] / a
    ng, s1, nn = gapstats(xs)
    print(f"t {raws[k] / 240:6.2f} s: to the target median {np.median(dt_):5.2f} max {dt_.max():5.2f} pitches; gap nodes "
          f"in reach {nn} ({ng} pairs, side 1 by position {s1 / max(ng, 1):.2f}); centroid "
          f"{np.round(xs.mean(0), 3).tolist()}", flush=True)
