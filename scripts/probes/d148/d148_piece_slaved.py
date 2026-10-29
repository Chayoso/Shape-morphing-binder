"""d148_piece_slaved.py FRAMES12 RUN_JSON T0:T1:P1,P2,... [...] -- does a visible floating piece move on its own or with
the body's grid? (CPU) For each piece (its time span and particles, from d148_why_floating.py's list) at every kept
frame from T0 - 0.1 s to T1 + 0.1 s: its distance to the body (the nearest particle outside its own connected set, in
pitches and in MPM cells), the body's share of the nodal mass in its 4^3 stencil (cubic B-spline weights at the run's MPM
grid: the share of the grid momentum the piece reads back that is not its own), its velocity (finite differences over a
kept frame) against the body's grid velocity there (the B-spline weighted mean of the other particles' velocities over
its stencil nodes, PIC) and that difference against the piece's own speed."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

fr_path, jpath = sys.argv[1], sys.argv[2]
z = np.load(fr_path)
fr, raws = z["frames"], np.asarray(z["raws"])
tgt = np.asarray(z["tgt"], np.float64)
a = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9)[0][:, 8]))
mpm = json.load(open(jpath))["provenance"]["mpm"]
DX, GMIN, NX = float(mpm["dx"]), np.asarray(mpm["grid_min"], np.float64), int(mpm["nx"])
OFFS = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)


def bspline(r):
    r = np.abs(r)
    return np.where(r < 1, 2 / 3 - r ** 2 + r ** 3 / 2, np.where(r < 2, (2 - r) ** 3 / 6, 0.0))


print(f"pitch {a:.4f} wu, MPM cell {DX:.4f} wu = {DX / a:.2f} pitches", flush=True)
for spec in sys.argv[3:]:
    t0, t1, ps = spec.split(":")
    t0, t1 = float(t0), float(t1)
    P = np.asarray([int(v) for v in ps.split(",")], np.int64)
    ks = [k for k in range(1, len(raws) - 1) if t0 - 0.1 <= raws[k] / 240 <= t1 + 0.1]
    ks = ks[::max(1, len(ks) // 12)]
    print(f"== piece {P.tolist()} {t0:.2f}-{t1:.2f} s", flush=True)
    for k in ks:
        x = np.asarray(fr[k], np.float64)
        v = (np.asarray(fr[k + 1], np.float64) - np.asarray(fr[k - 1], np.float64)) / ((raws[k + 1] - raws[k - 1]) / 240)
        mask = np.ones(len(x), bool)
        mask[P] = False
        others = np.nonzero(mask)[0]
        tree = cKDTree(x[others])
        dist, _ = tree.query(x[P])
        c = x[P].mean(0)
        vp = v[P].mean(0)
        b = np.floor((c - GMIN) / DX).astype(int) - 1
        nodes = GMIN + DX * (b[None] + OFFS)
        near = np.asarray(tree.query_ball_point(c, 3.0 * DX), np.int64)
        ids = others[near]
        mb, mvb = 0.0, np.zeros(3)
        mp = 0.0
        for xg in nodes:
            wb = bspline((x[ids] - xg) / DX).prod(1)
            wp = bspline((x[P] - xg) / DX).prod(1)
            mb += wb.sum()
            mp += wp.sum()
            mvb += (wb[:, None] * v[ids]).sum(0)
        vg = mvb / max(mb, 1e-12)
        print(f"  {raws[k] / 240:.2f} s: to the body {dist.min() / a:.2f} pitches ({dist.min() / DX:.2f} cells); body share "
              f"of the stencil's nodal mass {mb / (mb + mp):.4f}; |v| piece {np.linalg.norm(vp):.3f}, body grid "
              f"{np.linalg.norm(vg):.3f} wu/s, |v - v_grid| {np.linalg.norm(vp - vg):.3f} ({np.linalg.norm(vp - vg) / max(np.linalg.norm(vp), 1e-9):.2f} "
              f"of its speed)", flush=True)
