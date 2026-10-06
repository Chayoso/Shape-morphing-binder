"""d116_variants.py FRAMES12_NPZ OUT_PREFIX [TAIL_FRAC=0.3] -- D116 (2): the tail's kept frames with each part of the
surface motion alone. Outer particles and their normals as in d116_tail.py (on the last kept frame); every particle
within three median neighbour distances of an outer one takes that one's normal, deeper particles keep their own
motion. From the tail's first kept frame on, each step's displacement of those particles is split into its normal
and tangential parts and accumulated: OUT_PREFIX_{orig,normal,tangent}.npz with the frames file's keys."""
import sys, numpy as np
from scipy.spatial import cKDTree
z = np.load(sys.argv[1], allow_pickle=True)
out = sys.argv[2]
tail = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
raws, F = np.asarray(z["raws"]), z["frames"]
k0 = int(len(raws) * (1 - tail))
xe = np.asarray(F[-1], np.float32)
d, nb = cKDTree(xe).query(xe, 17)
off = xe[nb[:, 1:]].mean(1) - xe
outer = np.linalg.norm(off, axis=1) > 0.25 * np.median(d[:, 1:], axis=1)
nrm_o = -off[outer] / np.linalg.norm(off[outer], axis=1, keepdims=True)
dd, ii = cKDTree(xe[outer]).query(xe, 1)
near = dd < 3.0 * np.median(d[:, 1])
nrm = nrm_o[ii]
frames = [np.asarray(F[k], np.float32) for k in range(k0, len(raws))]
var = {"orig": frames, "normal": [frames[0].copy()], "tangent": [frames[0].copy()]}
for a, b in zip(frames, frames[1:]):
    dx = b - a
    dn = (dx * nrm).sum(1, keepdims=True) * nrm
    for name, part in (("normal", dn), ("tangent", dx - dn)):
        step = np.where(near[:, None], part, dx)
        var[name].append(var[name][-1] + step)
for name, fr in var.items():
    np.savez(f"{out}_{name}.npz", frames=np.stack(fr).astype(np.float32), raws=raws[k0:], tgt=z["tgt"],
             deliver_n=len(fr))
print(dict(frames=len(frames), outer=int(outer.sum()), near=int(near.sum()), k0=k0))
