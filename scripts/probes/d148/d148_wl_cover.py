"""d148_wl_cover.py DX NGRID LOSS_RES T0 T1 LABEL=FRAMES12 [...] -- D148c (walls): does the walls' node set (the pieces
joined through the box's one-cell ring, gap_split.GapPlanes margin 1) still cover the places where the pieces fall off?
Per run: the particles apart from the body (as d148_cut_apart.py) at kept frames in [T0, T1) s whose 4^3 stencil holds a
gap node of D148's set (margin 0); the share of them whose stencil also holds a node of the walls' set (margin 1); and
the node counts of both sets on the run's own grid and loss cell."""
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from physmorph.mpm.gap_split import GapPlanes

dx, ngrid, lres, t0, t1 = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
half = dx * ngrid / 2.0
gmin = np.full(3, -half)
offs = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)


def stencil_hits(x, mask3):
    b = np.floor((x - gmin) / dx).astype(int) - 1
    hit = np.zeros(len(x), bool)
    for o in offs:
        n = b + o
        ok = ((n >= 0) & (n < ngrid)).all(1)
        hit[ok] |= mask3[n[ok, 0], n[ok, 1], n[ok, 2]]
    return hit


for spec in sys.argv[6:]:
    lab, path = spec.split("=", 1)
    z = np.load(path)
    fr, raws = z["frames"], np.asarray(z["raws"])
    tgt = np.asarray(z["tgt"], np.float64)
    _d0 = np.full(3, -half).astype(np.float32)             # the run's loss cell in float32 (target.py)
    cell = float((_d0 + dx * np.full(3, ngrid, np.float32) - _d0).max() / lres)
    g0 = GapPlanes(tgt, dx, gmin, (ngrid,) * 3, cell)
    g1 = GapPlanes(tgt, dx, gmin, (ngrid,) * 3, cell, keep_boxes=True, margin=1)
    m0, m1 = g0.mask.reshape((ngrid,) * 3), g1.mask.reshape((ngrid,) * 3)
    a = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::5], 9)[0][:, 8]))
    tot, cov, frames = 0, 0, 0
    for k in range(len(raws)):
        t = raws[k] / 240.0
        if not (t0 <= t < t1):
            continue
        x = np.asarray(fr[k], np.float64)
        pairs = cKDTree(x).query_pairs(1.5 * a, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
        _, cl = connected_components(g, directed=False)
        ap = np.nonzero(cl != np.bincount(cl).argmax())[0]
        frames += 1
        if not len(ap):
            continue
        h0 = stencil_hits(x[ap], m0)
        h1 = stencil_hits(x[ap], m1)
        tot += int(h0.sum())
        cov += int((h0 & h1).sum())
    print(f"{lab}: cell {cell:.4f} wu = {cell / dx:.2f} MPM cells; gap nodes {g0.n_gap} (margin 0), {g1.n_gap} with the "
          f"ring (margin 1, {g1.n_gap / max(g0.n_gap, 1):.2f}); frames in [{t0}, {t1}) s: {frames}; apart particle-frames "
          f"near D148's nodes {tot}, of them near a walls node {cov} ({cov / max(tot, 1):.2f})", flush=True)
