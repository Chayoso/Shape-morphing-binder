"""piece.py RUNS_DIR TAG BODY SEED T_FAR [T0 T1] -- D147, the post-landing ear-tip pieces, frame by frame from a run's
kept frames (positions every 12 steps; the coordinator's forensic of 2026-10-16). SEED: a body-local index of the piece
(apart.py's), T_FAR a kept frame where the piece is beyond 3 spacings. The piece's members: its component (joined
within two spacings) at T_FAR. Per kept frame in [T0, T1] (default the landing - 0.5 s to T_FAR + 0.5 s):
  - the link: the members' nearest distance to the rest of the body (spacings), and how many members still have a body
    particle within two spacings;
  - the members' velocity relative to their 8 nearest body neighbours (central differences of the kept frames), along
    the link (opening speed) and across;
  - the local stretch: the members' distances to their 8 nearest body neighbours of the morph's end, over those
    distances then (the material line stretch at the members);
  - the step between kept frames split by kind (wholly driven steps, wholly released, mixed): the members' displacement
    relative to their neighbours, to see whether the link opens while u and the relaxation act or while they do not;
  - their height above the floor, the body's lowest point above the floor.
Once: where the members sit at the morph's end (height fraction, distance to the target's surface, the number of body
particles and of target points within two spacings: the ear's local thickness against the bulk's), and the same for the
target there. Prints JSON lines. Read-only, CPU."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def frame_of(x, X):
    c = x.mean(0)
    A = (x - c).T @ (X - X.mean(0))
    U, _, Vt = np.linalg.svd(A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return (x - c) @ (U @ D @ Vt)


runs, tag, b, seed, t_far = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
j = json.loads((runs / f"{tag}.json").read_text())
z = np.load(runs / f"{tag}.npz")
fr, rows, raws = z["frames"], z["rows"], z["raws"]
dt, R = j["mpm"]["dt"], j["scene"]["R"]
T = int(j["config"]["T"])
s, e = rows[b]
n_b = e - s
sp = (4.18879 * R ** 3 / n_b) ** (1 / 3)
ts = raws * dt
floor_y = float(z["floor_y"])
X = np.asarray(z[f"src{b}"], np.float64)
tgt = np.asarray(z[f"tgt{b}"], np.float64)
t_m = float(j["scene"]["spec"].get("morph_time", 0.0))
k_far = int(np.argmin(np.abs(ts - t_far)))
k_m = int(np.argmin(np.abs(ts - t_m)))
xb = lambda k: np.asarray(fr[k, s:e], np.float64)        # noqa: E731
# the members at T_FAR
x = xb(k_far)
pairs = cKDTree(x).query_pairs(2.0 * sp, output_type="ndarray")
g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_b, n_b))
_, lab = connected_components(g, directed=False)
M = np.nonzero(lab == lab[seed])[0]
rest = np.setdiff1d(np.arange(n_b), M)
low = np.array([float(fr[k, s:e, 1].min()) for k in range(len(fr))]) - floor_y
hit = np.nonzero(low < sp)[0]
t_land = float(ts[hit[0]]) if len(hit) else None
t0 = float(sys.argv[6]) if len(sys.argv) > 6 else (t_land - 0.5 if t_land else t_far - 3.0)
t1 = float(sys.argv[7]) if len(sys.argv) > 7 else t_far + 0.5
# where they sit at the morph's end
ym = frame_of(xb(k_m), X)
h = (ym[:, 1] - ym[:, 1].min()) / np.ptp(ym[:, 1])
kt, kb = cKDTree(tgt), cKDTree(ym)
cm = ym[M].mean(0)
ri = np.random.default_rng(0).choice(n_b, 400, replace=False)
bulk = float(np.median([len(v) for v in kb.query_ball_point(ym[ri], 2.0 * sp)]))
tbulk = np.median([len(v) for v in kt.query_ball_point(tgt[np.random.default_rng(1).choice(len(tgt), 400, replace=False)],
                                                      2.0 * sp)])
print(json.dumps({"tag": tag, "body": b, "members": M.tolist(), "n": int(len(M)), "spacing": round(sp, 4),
                  "landing_s": t_land, "t_far": float(ts[k_far]),
                  "morph_end": {"centroid": np.round(cm, 3).tolist(), "height_fraction": np.round(h[M], 3).tolist(),
                                "to_target_sp": np.round(kt.query(ym[M])[0] / sp, 2).tolist(),
                                "body_within_2sp": [len(v) for v in kb.query_ball_point(ym[M], 2.0 * sp)],
                                "target_within_2sp": [len(v) for v in kt.query_ball_point(ym[M], 2.0 * sp)],
                                "bulk_body_within_2sp": float(bulk), "bulk_target_within_2sp": float(tbulk)}}))
# the members' 8 nearest body neighbours (not members) at the morph's end: the material lines of the link
nb = cKDTree(xb(k_m)[rest]).query(xb(k_m)[M], 8)[1]
nb = rest[nb]
L0 = np.linalg.norm(xb(k_m)[M][:, None] - xb(k_m)[nb], axis=2)
win = 2 * T


def kind(k):
    a0 = raws[k] % win
    a1 = raws[k + 1] - (raws[k] - a0)
    return "driven" if a1 <= T else ("released" if a0 >= T and a1 <= win else "mixed")


prev = None
for k in range(len(fr)):
    if not t0 <= ts[k] <= t1:
        continue
    xk = xb(k)
    tr = cKDTree(xk[rest])
    d, _ = tr.query(xk[M])
    near = sum(1 for v in tr.query_ball_point(xk[M], 2.0 * sp) if v)
    stretch = np.linalg.norm(xk[M][:, None] - xk[nb], axis=2) / L0
    rec = {"t": round(float(ts[k]), 3), "link_sp": round(float(d.min()) / sp, 2), "link_mean_sp": round(float(d.mean()) / sp, 2),
           "members_linked": int(near), "stretch_mean": round(float(stretch.mean()), 3),
           "stretch_max": round(float(stretch.max()), 3), "above_floor": round(float(xk[M, 1].min() - floor_y), 2),
           "body_lowest_above_floor_sp": round(float(low[k]) / sp, 2)}
    if 0 < k < len(fr) - 1:
        v = (xb(k + 1) - xb(k - 1)) / (ts[k + 1] - ts[k - 1])
        rel = v[M] - v[nb].mean(1)                                   # relative to the members' neighbours
        out = xk[M] - xk[nb].mean(1)
        out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-9)
        rec.update(open_speed_sp_s=round(float((rel * out).sum(1).mean()) / sp, 2),
                   rel_speed_sp_s=round(float(np.linalg.norm(rel, axis=1).mean()) / sp, 2),
                   speed_sp_s=round(float(np.linalg.norm(v[M], axis=1).mean()) / sp, 2))
    if k < len(fr) - 1:
        x1 = xb(k + 1)
        dd = (x1[M] - x1[nb].mean(1)) - (xk[M] - xk[nb].mean(1))      # the members' step relative to their neighbours
        rec.update(next_kind=kind(k), next_rel_step_sp=round(float(np.linalg.norm(dd, axis=1).mean()) / sp, 3))
    print(json.dumps(rec), flush=True)
