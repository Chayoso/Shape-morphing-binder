"""Physical fragments per archived frame, by the grid's own criterion (no renderer).

A fragment is a cluster of particles that shares no MPM cell with the body once the occupancy is
dilated by one cell (the re-attachment net's criterion, docs/method.md 10.7/10.10). Clusters are
reported in cells of material (particles / ppc); a cluster below one cell is sub-cell material the
grid does not resolve. Written next to the photoreal sidecar so the report can separate "the
isosurface drew two pieces" (a threshold effect on a thin neck) from "the physics has two bodies".

    grid_fragments.py <out_dir> <run> <stride> <out_txt> [cell_diag]

per line: archived_frame  n_clusters_ge_1cell  largest_cluster_particles  n_clusters_ge_20
summary:  # frames N  fragments>=1cell in K frames (max count C, max size M particles = m cells)  clusters>=20 in K2 frames
"""
import sys
import numpy as np
from scipy import ndimage

out, run, stride, out_txt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
cell_diag = float(sys.argv[5]) if len(sys.argv) > 5 else 26.0
z = np.load(f"{out}/{run}_render_full_dt_iso_nn.npz")
X = z["frames"]; src = z["src"]
n_deliver = int(z["deliver_n"]) if "deliver_n" in z.files else len(X)
dx = float(np.linalg.norm(src.max(0) - src.min(0))) / cell_diag
N = X.shape[1]
vol = float(np.prod(src.max(0) - src.min(0))) * 0.5236          # sphere volume from its bbox
ppc = N * dx ** 3 / vol                                           # particles per cell (docs/method.md 10.9)
struct = np.ones((3, 3, 3), bool)


def clusters(x):
    gmin = x.min(0) - 2 * dx
    ijk = np.floor((x - gmin) / dx).astype(np.int64)
    occ = np.zeros(ijk.max(0) + 3, bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    occ = ndimage.binary_dilation(occ, structure=struct)
    lab, n = ndimage.label(occ, structure=struct)
    sizes = np.bincount(lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]])
    sizes = np.sort(sizes[1:])[::-1] if n > 1 else sizes[1:]
    return sizes                                                   # descending, body first


rows = []
for i in range(0, n_deliver, stride):
    s = clusters(X[i])
    others = s[1:] if len(s) > 1 else np.zeros(0, int)
    rows.append((i, int((others >= ppc).sum()), int(others.max()) if len(others) else 0, int((others >= 20).sum())))
k1 = [r for r in rows if r[1] > 0]; k2 = [r for r in rows if r[3] > 0]
with open(out_txt, "w") as fh:
    fh.write("archived_frame n_clusters_ge_1cell largest_cluster_particles n_clusters_ge_20\n")
    for r in rows:
        fh.write(f"{r[0]} {r[1]} {r[2]} {r[3]}\n")
    mx = max(r[2] for r in rows) if rows else 0
    fh.write(f"# frames {len(rows)}  fragments>=1cell in {len(k1)} frames (max count {max([r[1] for r in rows]) if rows else 0}, "
             f"max size {mx} particles = {mx / ppc:.2f} cells; ppc {ppc:.0f}, cell {dx:.3f} wu)  clusters>=20 in {len(k2)} frames\n")
print(f"{run}: frames {len(rows)}  fragments>=1cell in {len(k1)} frames  clusters>=20 in {len(k2)} frames  max size {mx} ({mx / ppc:.2f} cells)")
