"""d155b_spots.py MESH ARM_TAG ARM_NEW -- D155b frame-level forensic (CPU): the places where the switched run's end surface is
off the independent sample by more than half a pitch while no tag seed is (|offset| <= 0.5 in all three), joined into
spots (reference points within two pitches); per spot: its size, where it sits (height / depth / side as fractions of
the box), its sign, the offset there in every tag seed and in the new run, and how the new run's offset there evolved
from the ramp's end (every 3rd kept frame) -- whether the spot was there when the body arrived and stayed, or formed
after; and which particles carry it (the new run's outer-layer particles nearest the spot: their path from the source,
their arrival time at their end position within one pitch)."""
import json
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
O = f"{G}/d155/gate"
mesh, tag_arm, new_arm = sys.argv[1], sys.argv[2], sys.argv[3]          # e.g. homer V A3GL
SEEDS = (97, 99, 101)


def layer_normals(x, k_lay=33, k_n=25):
    tree = cKDTree(x)
    sp = float(np.median(tree.query(x[::15], 9)[0][:, 8]))
    _, nb = tree.query(x, k_lay, workers=2)
    off = x - x[nb[:, 1:]].mean(1)
    lay = np.linalg.norm(off, axis=1) >= 0.5 * sp
    off_n = x - x[nb[:, 1:k_n]].mean(1)
    return lay, off_n / np.maximum(np.linalg.norm(off_n, axis=1, keepdims=True), 1e-12)


def offsets(x, Y, NY, pitch):
    lay, _ = layer_normals(x)
    xl = x[lay]
    _, j = cKDTree(xl).query(Y, 8, workers=2)
    return ((xl[j] - Y[:, None, :]) * NY[:, None, :]).sum(2).mean(1) / pitch


ind = np.asarray(np.load(f"{G}/d92/ind_{mesh}.npz")["tgt"], np.float64)
pitch = 0.708 * float(np.median(cKDTree(ind).query(ind[::20], 9)[0][:, 8]))
lay, nrm = layer_normals(ind)
li = np.flatnonzero(lay)
Y, NY = ind[li], nrm[li]
lo, hi = Y.min(0), Y.max(0)
tag_off = []
for s in SEEDS:
    z = np.load(f"{O}/{mesh}40k_{tag_arm}{s}_frames12.npz")
    tag_off.append(offsets(np.asarray(z["frames"][-1], np.float64), Y, NY, pitch))
tag_off = np.stack(tag_off)
zn = np.load(f"{O}/{mesh}40k_{new_arm}97_frames12.npz")
frames, raws = zn["frames"], np.asarray(zn["raws"])
x_end = np.asarray(frames[-1], np.float64)
new_off = offsets(x_end, Y, NY, pitch)
bad = (np.abs(new_off) > 0.5) & (np.abs(tag_off).max(0) <= 0.5)
both = (np.abs(new_off) > 0.5) & (np.abs(tag_off).max(0) > 0.5)
fixed = (np.abs(new_off) <= 0.5) & (np.abs(tag_off).min(0) > 0.5)
print(f"{mesh} {new_arm} vs {tag_arm}: layer points {len(Y)}; off > 0.5 pitch in the new run only {int(bad.sum())} "
      f"({bad.mean() * 100:.1f} %), in it and in some tag seed {int(both.sum())}, in every tag seed but not the new run "
      f"{int(fixed.sum())}", flush=True)
idx = np.flatnonzero(bad)
if len(idx) == 0:
    sys.exit()
pairs = cKDTree(Y[idx]).query_pairs(2.0 * pitch, output_type="ndarray")
A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(idx), len(idx)))
nc, lab = connected_components(A, directed=False)
sizes = np.bincount(lab)
order = np.argsort(-sizes)
J = json.load(open(f"{O}/{mesh}40k_{new_arm}97.json"))
arm = next(iter(J["arms"].values()))
acc = [h for h in arm["history"] if h.get("frame_end") and not h.get("null_commit")]
t_ramp = next(h["frame_end"] / 240 for h in acc if (h.get("morph_s") or 0) >= 1.0)
sel = [i for i in range(0, len(raws), 3) if raws[i] / 240 >= t_ramp - 0.2] + [len(raws) - 1]
traj = {i: offsets(np.asarray(frames[i], np.float64), Y, NY, pitch) for i in sel}
x0 = np.asarray(frames[0], np.float64)
lay_end, _ = layer_normals(x_end)
le = np.flatnonzero(lay_end)
tree_end = cKDTree(x_end[le])
out = []
for c in order[:8]:
    m = idx[lab == c]
    cen = Y[m].mean(0)
    frac = (cen - lo) / (hi - lo)
    sgn = float(new_off[m].mean())
    hist = [(round(raws[i] / 240 - t_ramp, 2), round(float(traj[i][m].mean()), 3)) for i in sel]
    # the particles that carry the spot: the end outer-layer particles within one pitch of it
    near = np.unique(np.concatenate(tree_end.query_ball_point(Y[m], 1.0 * pitch)).astype(int)) if len(m) else []
    pids = le[near] if len(near) else np.array([], int)
    arr = []
    for p in pids[:400]:
        d = np.linalg.norm(np.asarray(frames[:, p], np.float64) - x_end[p], axis=1) / pitch
        k = next((i for i in range(len(d)) if np.all(d[i:] < 1.0)), len(d) - 1)
        arr.append(raws[k] / 240.0)
    path = float(np.median(np.linalg.norm(x_end[pids] - x0[pids], axis=1)) / pitch) if len(pids) else float("nan")
    row = dict(size=int(len(m)), where=[round(float(v), 2) for v in frac], sign=round(sgn, 3),
               tag=[round(float(tag_off[k][m].mean()), 3) for k in range(len(SEEDS))], particles=int(len(pids)),
               arrival_med=round(float(np.median(arr)), 2) if arr else None, path_med=round(path, 1), course=hist)
    out.append(row)
    print(f"  spot {len(m):4d} pts at x/y/z {row['where']} offset {sgn:+.2f} (tag {row['tag']}); {len(pids)} particles, "
          f"their end reached (within 1 pitch to stay) at a median {row['arrival_med']} s, path {path:.1f} pitches; "
          f"course after the ramp: {hist[:2] + hist[2::4] + hist[-1:]}", flush=True)
json.dump(dict(mesh=mesh, new=new_arm, tag=tag_arm, spots=out, n_bad=int(bad.sum()), n_layer=len(Y)),
          open(f"{O}/spots_{mesh}_{new_arm}.json", "w"))
