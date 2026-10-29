"""b13_endsets.py OUT_PNG NAME=END1_NPZ [...] -- B13 forensic (CPU only): where the pieces off the body sit at the end.

For each run's end frame: the connected sets at 1.5 pitches (pitch a = 0.708 x the target sample's median 8th-neighbour
distance, as pieces145.py / visible_apart.py); the largest set drawn as small grey points, every other set in red (size
by its particle count), in two orthographic views (x-y and z-y). A diagnostic scatter of particle positions, not a render.
Prints per run: sets, particles off the body, the ten largest pieces (size, gap to the body in pitches, centre)."""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

out = sys.argv[1]
specs = [s.split("=", 1) for s in sys.argv[2:]]
fig, axes = plt.subplots(len(specs), 2, figsize=(10, 4.6 * len(specs)), squeeze=False)
for row, (name, path) in enumerate(specs):
    z = np.load(path, allow_pickle=True)
    x = np.asarray(z["frames"][-1], np.float64)
    tgt = np.asarray(z["tgt"], np.float64)
    a = 0.708 * float(np.median(cKDTree(tgt).query(tgt, 9, workers=8)[0][:, 8]))
    pairs = cKDTree(x).query_pairs(1.5 * a, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    _, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    big = int(np.argmax(sizes))
    off = lab != big
    d = np.zeros(len(x))
    if off.any():
        d[off] = cKDTree(x[~off]).query(x[off], workers=8)[0] / a
    pcs = sorted(((int(sizes[c]), float(d[lab == c].min()), x[lab == c].mean(0)) for c in np.unique(lab[off])),
                 key=lambda p: -p[0])
    print(f"{name}: sets {len(sizes)}, off the body {int(off.sum())} particles in {len(pcs)} pieces; largest: "
          + "; ".join(f"{s} at {g:.2f} a ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})" for s, g, c in pcs[:10]))
    for col, (i, j, lab_ij) in enumerate(((0, 1, "x-y"), (2, 1, "z-y"))):
        ax = axes[row, col]
        ax.scatter(x[~off, i], x[~off, j], s=0.02, c="0.75", rasterized=True)
        ax.scatter(x[off, i], x[off, j], s=3, c="red", rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(f"{name} end, {lab_ij}: {len(pcs)} pieces, {int(off.sum())} particles (red)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
fig.tight_layout()
fig.savefig(out, dpi=110)
print("saved", out)
