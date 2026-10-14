"""d148_cut_apart.py DX NGRID LOSS_RES LABEL=FRAMES12 [LABEL=FRAMES12 ...] -- D148's direct count of the pieces that
fall off at the cut (the coordinator's go / no-go item, 2026-10-14): per kept frame, the particles apart from the body
(outside the largest connected set at 1.5 pitches, as d120_apart.py) whose 4^3 stencil holds a gap node of the target
(gap_split.GapPlanes on the MPM grid and the loss cell: within two cells of a gap plane); per window (40 raw steps) the
union of those particles, and how many of them were not apart near a gap in the window before (new). Prints per run the
per-window counts, their sum over 1-3 s, over the last third of the run, and the unique particles over the run."""
import os
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gap_split import GapPlanes  # noqa: E402

dx, ngrid, lres = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
half = dx * ngrid / 2.0
gmin = np.full(3, -half)
gp = None
offs = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3)
for spec in sys.argv[4:]:
    lab, path = spec.split("=", 1)
    z = np.load(path)
    fr, raws = z["frames"], np.asarray(z["raws"])
    tgt = np.asarray(z["tgt"], np.float64)
    if gp is None:
        gp = GapPlanes(tgt, dx, gmin, (ngrid,) * 3, 2.0 * half / lres)
        mask3 = gp.mask.reshape((ngrid,) * 3)
    a = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::5], 9)[0][:, 8]))
    per_win = {}
    for k in range(len(raws)):
        x = np.asarray(fr[k], np.float64)
        pairs = cKDTree(x).query_pairs(1.5 * a, output_type="ndarray")
        g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
        _, cl = connected_components(g, directed=False)
        ap = np.nonzero(cl != np.bincount(cl).argmax())[0]
        if not len(ap):
            continue
        b = np.floor((x[ap] - gmin) / dx).astype(int) - 1
        near = np.zeros(len(ap), bool)
        for o in offs:
            n = b + o
            ok = ((n >= 0) & (n < ngrid)).all(1)
            near[ok] |= mask3[n[ok, 0], n[ok, 1], n[ok, 2]]
        w = int(raws[k] // 40)
        per_win.setdefault(w, set()).update(ap[near].tolist())
    wins = sorted(per_win)
    prev, rows, allp = set(), [], set()
    for w in range(max(wins) + 1 if wins else 0):
        s = per_win.get(w, set())
        rows.append((w, len(s), len(s - prev)))
        allp |= s
        prev = s
    t_end = raws[-1] / 240.0
    early = sum(r[1] for r in rows if 1.0 <= r[0] / 6.0 < 3.0)
    late = sum(r[1] for r in rows if r[0] / 6.0 >= 2 * t_end / 3)
    print(f"{lab}: gap nodes {gp.n_gap}; windows {len(rows)} (to {t_end:.1f} s); apart near a gap, summed over the "
          f"windows at 1-3 s {early}, over the last third {late}; unique particles over the run {len(allp)}")
    print("   per window (window: count / new): " + " ".join(f"{w}:{c}/{n}" for w, c, n in rows if c), flush=True)
