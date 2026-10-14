"""apart.py RUNS_DIR TAG [T0 T1] -- D147: the pieces of a body that have come apart from it, frame by frame (kept
frames between T0 and T1 s, default the whole run): each body's particles joined within two sample spacings, every
component but the largest reported (size, its centroid in the body's Buckens frame, its height above the floor, its
nearest distance to its body and to the other body). For each piece, its particles are followed back: the last kept
frame where they were still joined to the body, where on the body they sat then and at the morph's end (the body's
frame, the bunny's height fraction: the ears are the top), and whether they rejoin later. Read-only, CPU."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree


def frame_of(x, X):
    """The Buckens frame: y = (x - c) R, R the polar factor of sum (x - c)(X - Xbar)^T (scene/frame.body_frame)."""
    c = x.mean(0)
    A = (x - c).T @ (X - X.mean(0))
    U, _, Vt = np.linalg.svd(A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return (x - c) @ R


def pieces(x, r):
    """Components of the radius graph; returns (labels, the largest's label)."""
    pairs = cKDTree(x).query_pairs(r, output_type="ndarray")
    n = len(x)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab, int(np.bincount(lab).argmax())


runs, tag = Path(sys.argv[1]), sys.argv[2]
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
t1 = float(sys.argv[4]) if len(sys.argv) > 4 else 1e9
j = json.loads((runs / f"{tag}.json").read_text())
z = np.load(runs / f"{tag}.npz")
fr, rows, raws = z["frames"], z["rows"], z["raws"]
dt, R = j["mpm"]["dt"], j["scene"]["R"]
floor_y = float(z["floor_y"])
t_m = float(j["scene"]["spec"].get("morph_time", 0.0))
sp = (4.18879 * R ** 3 / (rows[0][1] - rows[0][0])) ** (1 / 3)
X = [np.asarray(z[f"src{b}"], np.float64) for b in range(len(rows))]
ts = raws * dt
k_end = int(np.argmin(np.abs(ts - max(t_m, 0.0))))          # the morph's end (the bunny's frame for "where")
seen = {}
for k in range(len(fr)):
    if not t0 <= ts[k] <= t1:
        continue
    for b, (s, e) in enumerate(rows):
        x = np.asarray(fr[k, s:e], np.float64)
        lab, main = pieces(x, 2.0 * sp)
        other = np.asarray(fr[k, rows[1 - b][0]:rows[1 - b][1]], np.float64) if len(rows) == 2 else None
        for c in np.unique(lab):
            if c == main:
                continue
            idx = np.nonzero(lab == c)[0]
            key = (b, int(idx.min()))
            y = frame_of(x, X[b])
            d_body = float(cKDTree(x[lab == main]).query(x[idx])[0].min())
            d_other = float(cKDTree(other).query(x[idx])[0].min()) if other is not None else None
            rec = {"t": round(float(ts[k]), 3), "body": b, "n": int(len(idx)), "first_index": int(idx.min()),
                   "centroid_body_frame": np.round(y[idx].mean(0), 3).tolist(),
                   "above_floor": round(float(x[idx, 1].min() - floor_y), 3),
                   "to_body_sp": round(d_body / sp, 2), "to_other_sp": None if d_other is None else round(d_other / sp, 2)}
            if key not in seen:                 # a new piece: where it came from
                back = None
                for kk in range(k - 1, max(-1, k - 121), -1):          # back at most 6 s
                    xb = np.asarray(fr[kk, s:e], np.float64)
                    lb, mb = pieces(xb, 2.0 * sp)
                    if (lb[idx] == mb).all():
                        back = kk
                        break
                xe = np.asarray(fr[k_end, s:e], np.float64)
                ye = frame_of(xe, X[b])
                h = (ye[idx, 1].mean() - ye[:, 1].min()) / max(np.ptp(ye[:, 1]), 1e-9)
                rec.update(joined_until=None if back is None else round(float(ts[back]), 3),
                           at_morph_end=np.round(ye[idx].mean(0), 3).tolist(), height_fraction=round(float(h), 3),
                           indices=idx[:20].tolist())
                seen[key] = idx
            print(json.dumps(rec), flush=True)
print(json.dumps({"tag": tag, "pieces": len(seen), "spacing": round(sp, 4), "window": [t0, t1]}))
