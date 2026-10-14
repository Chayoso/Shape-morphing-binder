"""vol145.py OUT_JSONL MESH IND_NPZ LABEL=FRAMES_NPZ [...] -- D145: the enclosed volume over time, ONE routine for every
method on the shared representation (each method's frames as a volume sample: our particles; the baselines' frames sampled
by the pipeline's stratified sampler at the same N), against the target measured by the same routine on the independent
sample of the mesh (CPU only).

Routine: occupancy on a voxel grid of 1.5 sampling pitches (pitch = 0.708 x the independent sample's median 8th-neighbour
distance), one binary closing, interior holes filled; volume = voxels x voxel volume. Reported per method: the ratio to the
target at normalised times tau = raw / raw_end nearest 0, 0.25, 0.5, 0.75 and 1, and its minimum and maximum over every
kept frame. One JSON line per method; a table on stdout."""
import json
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

out, mesh, ind_path = sys.argv[1], sys.argv[2], sys.argv[3]
I = np.asarray(np.load(ind_path)["tgt"], np.float64)
pitch = 0.708 * float(np.median(cKDTree(I).query(I[::20], 9)[0][:, 8]))
vox = 1.5 * pitch


def volume(x):
    lo = x.min(0) - 3 * vox
    ijk = np.floor((x - lo) / vox).astype(np.int64)
    dims = ijk.max(0) + 4
    occ = np.zeros(dims, bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    occ = ndimage.binary_fill_holes(ndimage.binary_closing(occ, iterations=1))
    return float(occ.sum()) * vox ** 3


V_t = volume(I)
print(f"{mesh}: pitch {pitch:.5f} wu, voxel {vox:.5f} wu, target (independent sample) {V_t:.3f} wu^3", flush=True)
with open(out, "a") as fh:
    for spec in sys.argv[4:]:
        label, path = spec.split("=", 1)
        z = np.load(path, allow_pickle=True)
        F, raws = z["frames"], np.asarray(z["raws"]).astype(float)
        tau = raws / max(raws[-1], 1.0)
        ratios = np.array([volume(np.asarray(F[k], np.float64)) / V_t for k in range(len(F))])
        at = {f"{q:g}": float(ratios[int(np.argmin(np.abs(tau - q)))]) for q in (0.0, 0.25, 0.5, 0.75, 1.0)}
        row = dict(mesh=mesh, label=label, frames=path, n_frames=int(len(F)), at=at, min=float(ratios.min()),
                   max=float(ratios.max()), argmin_tau=float(tau[int(np.argmin(ratios))]),
                   argmax_tau=float(tau[int(np.argmax(ratios))]), target_volume=V_t, voxel=vox)
        fh.write(json.dumps(row) + "\n")
        print(f"  {label:8s} tau 0/.25/.5/.75/1: " + " ".join(f"{at[k]:.3f}" for k in at)
              + f" | min {row['min']:.3f} (tau {row['argmin_tau']:.2f}) max {row['max']:.3f} (tau {row['argmax_tau']:.2f})", flush=True)
