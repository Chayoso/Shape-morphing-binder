"""dump_forensic.py RUNS_DIR TAG BODY T_FAR|auto [SEED] -- D147, an ear-tip piece read step by step from a dump run
(scene_run --dump; scene/dump.py). The piece: the largest component (joined within two spacings) apart from body BODY
at the kept frame nearest T_FAR (or the one holding the body-local index SEED), beyond 3 spacings. Its link partners:
each member's nearest body particle (not a member) at the dump's first window. Per dumped window and step:
  - the link: the members' nearest distance to the rest of the body, in spacings, and the opening of each member-partner
    pair split by the update that moved it in the step: the grid (dt v_{t+1}, G2P's velocity), the update kernel's
    position projections (the minimum spacing's push and the bonds: xu_{t+1} - x_t - dt v_{t+1}), the layer step
    (x_{t+1} - xu_{t+1}), itself split into u's part (frac_u g u n, frac_u = 1/Tc in the driven steps) and the
    relaxation's (the rest along n);
  - the stored elastic F (its stretches, det) against the kinematic stretch the particles went through (Fg integrated by
    (1 + dt C) from the dump's first step): how much of the deformation the material resists;
  - the tracked J; the velocity gradient C (the grid's at the particle); the speeds;
  - the floor: the members' and the body's lowest height above it;
  - the damping: dt rate |v - (v_c + w x r)| at the members against |v| (the share of the deformation velocity it removes
    a step);
  - the controls: |dFc| against the cap (cfg.dfc_clip), u and its gate at the members; the objective's pull at the
    window's committed end (|grad transport|, |grad cleanup| at the members against the body's median, and their
    component along the link, outward positive).
Prints JSON lines (one per window, the steps summarised; the opening split summed over the window). Read-only. Run on
hyde06 (torch for the layer's normals, CPU is enough)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

runs, tag, b = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
if sys.argv[4] == "auto":                       # the largest piece beyond 3 spacings in apart_<tag>.txt (else the largest)
    recs = [json.loads(ln) for ln in open(runs / f"apart_{tag}.txt") if ln.startswith('{"t"')]
    recs = [r for r in recs if r["body"] == b] or [{"t": 0.0, "n": 0, "to_body_sp": 0.0}]
    t_far = max(recs, key=lambda r: (r["to_body_sp"] > 3.0, r["n"], r["to_body_sp"]))["t"]
else:
    t_far = float(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else None
j = json.loads((runs / f"{tag}.json").read_text())
z = np.load(runs / f"{tag}.npz")
fr, rows, raws = z["frames"], z["rows"], z["raws"]
dt, R = j["mpm"]["dt"], j["scene"]["R"]
s, e = rows[b]
n_b = e - s
sp = (4.18879 * R ** 3 / n_b) ** (1 / 3)
cfg = j["config"]
rate = float(j["scene"].get("deformation_drag", 0.0))
clip = float(cfg.get("dfc_clip", 0.0))
ts = raws * dt
k_far = int(np.argmin(np.abs(ts - t_far)))
x = np.asarray(fr[k_far, s:e], np.float64)
pairs = cKDTree(x).query_pairs(2.0 * sp, output_type="ndarray")
g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n_b, n_b))
_, lab = connected_components(g, directed=False)
main = int(np.bincount(lab).argmax())
if seed is not None:
    M = np.nonzero(lab == lab[seed])[0]
else:
    comps = [c for c in np.unique(lab) if c != main]
    far = [(np.sum(lab == c), c) for c in comps
           if cKDTree(x[lab == main]).query(x[lab == c])[0].min() > 3.0 * sp]
    if not far:
        far = [(np.sum(lab == c), c) for c in comps]
    M = np.nonzero(lab == max(far)[1])[0]
rest = np.setdiff1d(np.arange(n_b), M)
dumps = sorted((runs / f"{tag}_dump").glob("w*.npz"))
print(json.dumps({"tag": tag, "body": b, "t_far": float(ts[k_far]), "members": M.tolist(), "n": int(len(M)),
                  "spacing": round(sp, 5), "drag_rate": rate, "dfc_clip": clip, "dump_windows": len(dumps)}))
import torch                                                         # noqa: E402
from physmorph.pipeline.window.layer import layer_relax_data, layer_spacing     # noqa: E402

partner, Fg = None, None
for path in dumps:
    d = np.load(path)
    T, Tc = int(d["T"]), int(d["Tc"])
    X, V, C, F = (d[f"b{b}_{k}"].astype(np.float64) for k in ("x", "v", "C", "F"))
    XU = d[f"b{b}_xu"].astype(np.float64) if f"b{b}_xu" in d.files else None
    Jt = d[f"b{b}_J"] if f"b{b}_J" in d.files else None
    if partner is None:
        partner = rest[cKDTree(X[0][rest]).query(X[0][M])[1]]
        Fg = np.repeat(np.eye(3)[None], len(M), 0)
    # the layer's normals at the window start (window/layer.layer_relax_data, as the window computes them)
    with torch.no_grad():
        x0t = torch.tensor(X[0], dtype=torch.float32)
        lmask, lnrm, _, _ = layer_relax_data(x0t, layer_spacing(x0t), k=cfg["layer_k"], h_sp=cfg["layer_h_sp"])
    nrm = lnrm.double().numpy()
    u, gate = d[f"b{b}_u"].astype(np.float64), d[f"b{b}_gate"].astype(np.float64)
    split = {"grid": 0.0, "update": 0.0, "u": 0.0, "relax": 0.0}
    link0 = cKDTree(X[0][rest]).query(X[0][M])[0].min() / sp
    damp_share, speeds, Cn = [], [], []
    for t in range(T):
        dirv = X[t][M] - X[t][partner]
        dirv /= np.linalg.norm(dirv, axis=1, keepdims=True).clip(1e-12)
        grid = dt * V[t + 1]
        upd = (XU[t] - X[t] - grid) if XU is not None else np.zeros_like(grid)
        lay = (X[t + 1] - XU[t]) if XU is not None else np.zeros_like(grid)
        fu = (1.0 / Tc if t < Tc else 0.0) * gate * u
        up = fu[:, None] * nrm
        rel = lay - up
        for name, dx in (("grid", grid), ("update", upd), ("u", up), ("relax", rel)):
            split[name] += float(((dx[M] - dx[partner]) * dirv).sum(1).mean()) / sp
        Fg = np.einsum("pij,pjk->pik", np.eye(3)[None] + dt * C[t + 1][M], Fg)
        xb, vb = X[t], V[t]
        c = xb.mean(0)
        r = xb - c
        vc = vb.mean(0)
        Iner = (r * r).sum() * np.eye(3) - r.T @ r
        w = np.linalg.solve(Iner, np.cross(r, vb - vc).sum(0))
        vdef = vb - vc - np.cross(w, r)
        damp_share.append(float(np.linalg.norm(dt * rate * vdef[M], axis=1).mean() / max(np.linalg.norm(vb[M], axis=1).mean(), 1e-12)))
        speeds.append(float(np.linalg.norm(vdef[M], axis=1).mean()) / sp)
        Cn.append(float(np.linalg.norm(C[t + 1][M], axis=(1, 2)).mean()))
    link1 = cKDTree(X[T][rest]).query(X[T][M])[0].min() / sp
    sF = np.linalg.svd(F[T][M], compute_uv=False)
    sG = np.linalg.svd(Fg, compute_uv=False)
    sF0 = np.linalg.svd(F[T][rest], compute_uv=False)
    dfc = d[f"b{b}_dfc_norm"]
    rec = {"t_start": round(float(d["t_start"]), 3), "link_sp": [round(float(link0), 2), round(float(link1), 2)],
           "opening_sp_by_update": {k: round(v, 3) for k, v in split.items()},
           "F_stretch_max_mean": round(float(sF[:, 0].mean()), 3), "F_stretch_min_mean": round(float(sF[:, 2].mean()), 3),
           "F_det_mean": round(float(np.linalg.det(F[T][M]).mean()), 3),
           "body_F_stretch_max_median": round(float(np.median(sF0[:, 0])), 3),
           "kinematic_stretch_max_mean": round(float(sG[:, 0].mean()), 3),
           "kinematic_stretch_min_mean": round(float(sG[:, 2].mean()), 3),
           "J_mean": round(float(Jt[T][M].mean()), 3) if Jt is not None else None,
           "deformation_speed_sp_s": round(float(np.mean(speeds)), 2), "C_norm_mean": round(float(np.mean(Cn)), 2),
           "damping_share_per_step": round(float(np.mean(damp_share)), 5),
           "above_floor_members": round(float(X[T][M][:, 1].min() - d["floor_y"]), 3),
           "body_lowest_above_floor_sp": round(float((X[T][:, 1].min() - d["floor_y"]) / sp), 2),
           "dfc_at_cap_share": round(float((dfc[:, M] >= 0.999 * clip).mean()), 3) if clip > 0 else None,
           "dfc_members_mean": round(float(dfc[:, M].mean()), 5), "dfc_body_mean": round(float(dfc.mean()), 5),
           "u_members": round(float(np.abs(u[M]).mean()) / sp, 3), "gate_members": round(float(gate[M].mean()), 2),
           "layer_members": round(float(lmask.numpy()[M].mean()), 2)}
    ye = d[f"b{b}_y_end"].astype(np.float64)
    out_dir = ye[M] - ye[partner]
    out_dir /= np.linalg.norm(out_dir, axis=1, keepdims=True).clip(1e-12)
    for name in ("transport", "cleanup"):
        k = f"b{b}_grad_{name}"
        if k in d.files:
            gr = d[k].astype(np.float64)
            gn = np.linalg.norm(gr, axis=1)
            rec[f"pull_{name}_members_over_body_median"] = round(float(gn[M].mean() / max(np.median(gn), 1e-30)), 2)
            rec[f"pull_{name}_outward"] = round(float((-gr[M] * out_dir).sum(1).mean() / max(gn[M].mean(), 1e-30)), 3)
    print(json.dumps(rec), flush=True)
