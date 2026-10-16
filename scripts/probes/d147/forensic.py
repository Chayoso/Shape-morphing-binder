"""forensic.py RUNS_DIR TAG [T0] -- D147 (the coordinator, 2026-10-15 22:00: "why do the final runs look as they do"):
the window-by-window timeline of one scene run from T0 s (default 8.0, before the first landing) to the end, from the
run's records and its kept frames only (no new run). Per window and body: the contact impulse and the floor's impulse
(the body's momentum change minus gravity's and the contact's), the body's extent in its own (Buckens) frame per axis,
the particles outside the loss raster (|y| >= raster half-width - one loss cell, where the CIC deposit loses mass and the
transport's mass check returns inf), the transport and silhouette terms, the commit (accepted steps, line-search trials,
null reason, zero-control), the rigid and the deformation kinetic energy (central differences of the kept frames, the
least-squares rigid velocity removed), the angular momentum each mechanism changed (L_split: grid, u, spacing,
relaxation), and the head set's deformation: the particles in the top quarter of the body's height at the morph's end
(ears and the top of the head), their RMS displacement in the body frame from that shape, in sample spacings.
The head set's frame-to-frame non-rigid displacement is split by the part of the window the interval lies in: wholly
in the driven steps (grid, dFc, u, relaxation, spacing push) or wholly in the released steps (grid and spacing push;
the relaxation too when relax_driven is off): what moved the head when the controls could not act. Onset: the first
kept frame after the body's landing where the head set's RMS displacement exceeds its pre-landing maximum by one
spacing; the 12 particles of the head set displaced most then, with their displacement over the next second.
Prints JSON lines. Read-only, CPU."""
import json
import sys
from pathlib import Path

import numpy as np


def frame_R(x, X):
    c = x.mean(0)
    A = (x - c).T @ (X - X.mean(0))
    U, _, Vt = np.linalg.svd(A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return c, U @ D @ Vt


def body_y(x, X):
    c, R = frame_R(x, X)
    return (x - c) @ R


def rigid_free(x, v, m):
    """v minus its mass-weighted least-squares rigid part (v_c + w x r)."""
    c = x.mean(0)
    r = x - c
    vc = v.mean(0)
    L = np.cross(r, v - vc).sum(0)
    I = (r * r).sum() * np.eye(3) - r.T @ r
    w = np.linalg.solve(I, L)
    return v - vc - np.cross(w, r), vc, w


runs, tag = Path(sys.argv[1]), sys.argv[2]
T0 = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
j = json.loads((runs / f"{tag}.json").read_text())
z = np.load(runs / f"{tag}.npz")
fr, rows, raws = z["frames"], z["rows"], z["raws"]
dt, g = j["mpm"]["dt"], -j["mpm"]["f_ext"][1]
cfg = j["config"]
T = int(cfg["T"])
win = 2 * T
n_b = rows[0][1] - rows[0][0]
mass_p = float(cfg.get("mass_ref_n", 40000)) / n_b            # the dynamics mass of a particle
M = mass_p * n_b
sp = (4.18879 * j["scene"]["R"] ** 3 / n_b) ** (1 / 3)
t_m = float(j["scene"]["spec"].get("morph_time", 0.0))
ts = raws * dt
half = 5.48                                                    # each body's loss raster: the single-body domain cube
loss_res = int(cfg.get("loss_res") or 56)
edge = half - 2 * half / loss_res                               # the CIC deposit: mass lost at y >= edge or y < -half
print(json.dumps({"tag": tag, "spacing": round(sp, 4), "mass_body": M, "edge": round(edge, 3),
                  "relax_driven": cfg.get("relax_driven"), "u_adam_carry": cfg.get("u_adam_carry"),
                  "spacing_symmetric": cfg.get("spacing_symmetric")}))
X = [np.asarray(z[f"src{b}"], np.float64) for b in range(len(rows))]
k_m = int(np.argmin(np.abs(ts - t_m)))
Y = [np.stack([body_y(np.asarray(fr[k, s:e], np.float64), X[b]) for k in range(len(fr))])
     for b, (s, e) in enumerate(rows)]                         # every kept frame in each body's frame
head, base = [], []
for b in range(len(rows)):
    y = Y[b][k_m]
    hf = (y[:, 1] - y[:, 1].min()) / np.ptp(y[:, 1])
    head.append(np.nonzero(hf >= 0.75)[0])
    base.append(y)
hist = [r for r in j["history"] if "bodies" in r]
land = []
for b, (s, e) in enumerate(rows):
    low = np.array([float(fr[k, s:e, 1].min()) for k in range(len(fr))]) - float(z["floor_y"])
    hit = np.nonzero(low < sp)[0]
    land.append(float(ts[hit[0]]) if len(hit) else None)


def head_dev(b, k):
    d = np.linalg.norm(Y[b][k][head[b]] - base[b][head[b]], axis=1)
    return float(np.sqrt((d * d).mean()) / sp), float(d.max() / sp)


def kinetic(b, k):
    if k == 0 or k == len(fr) - 1:
        return None, None
    s, e = rows[b]
    x = np.asarray(fr[k, s:e], np.float64)
    v = (np.asarray(fr[k + 1, s:e], np.float64) - np.asarray(fr[k - 1, s:e], np.float64)) / (ts[k + 1] - ts[k - 1])
    vd, vc, w = rigid_free(x, v, mass_p)
    r = x - x.mean(0)
    vr = vc + np.cross(w, r)
    return 0.5 * mass_p * float((vr * vr).sum()), 0.5 * mass_p * float((vd * vd).sum())


# the head set's non-rigid displacement between consecutive kept frames, by the part of the window it lies in
steps = raws
kind = []
for k in range(len(fr) - 1):
    a0, a1 = steps[k] % win, steps[k + 1] - (steps[k] - steps[k] % win)
    if a1 <= T:
        kind.append("driven")
    elif a0 >= T and a1 <= win:
        kind.append("released")
    else:
        kind.append("mixed")
P_prev = None
for i, r in enumerate(hist):
    t = r["t"]
    if t < T0 - 1e-9:
        P_prev = [np.array(bd["P"]) for bd in r["bodies"]]
        continue
    k = int(np.argmin(np.abs(ts - t)))
    out = {"w": r["animation"] + 1, "t": round(t, 3), "contact_J": round(float(np.linalg.norm(r["contact_impulse"])), 1),
           "accepted": r.get("accepted"), "ls_trials": r.get("ls_trials"), "null": r.get("null_reason"),
           "zero": r.get("zero_commit"), "lambda": r.get("lambda"), "bodies": []}
    kprev = int(np.argmin(np.abs(ts - (t - win * dt))))
    for b, bd in enumerate(r["bodies"]):
        P = np.array(bd["P"])
        Jc = np.array(r["contact_impulse"]) * (1 if b == 0 else -1)
        floor = (P - P_prev[b] - Jc + np.array([0.0, M * g * win * dt, 0.0])) if P_prev is not None else None
        y = Y[b][k]
        ext = np.abs(y).max(0)
        hd, hmax = head_dev(b, k)
        ke = kinetic(b, k)
        ls = r["L_split"][b]["own"] if r.get("L_split") else None
        inc = {"driven": [], "released": [], "mixed": []}
        for kk in range(kprev, k):
            d = np.linalg.norm(Y[b][kk + 1][head[b]] - Y[b][kk][head[b]], axis=1)
            inc[kind[kk]].append(float(np.sqrt((d * d).mean()) / sp))
        out["bodies"].append({
            "floor_J": None if floor is None else round(float(np.linalg.norm(floor)), 1),
            "floor_Jy": None if floor is None else round(float(floor[1]), 1),
            "ext": np.round(ext, 2).tolist(), "outside": int(((y >= edge) | (y < -half)).any(1).sum()),
            "transport": bd["transport"], "d_sil": round(bd["d_sil"], 4),
            "KE_rigid": None if ke[0] is None else round(ke[0], 1), "KE_def": None if ke[1] is None else round(ke[1], 1),
            "head_rms_sp": round(hd, 2), "head_max_sp": round(hmax, 2),
            "head_step_driven": round(float(np.mean(inc["driven"])), 3) if inc["driven"] else None,
            "head_step_released": round(float(np.mean(inc["released"])), 3) if inc["released"] else None,
            "L_u": None if ls is None else round(float(np.linalg.norm(ls["L_u"])) * mass_p, 1),
            "L_space": None if ls is None else round(float(np.linalg.norm(ls["L_space"])) * mass_p, 1),
            "L_relax": None if ls is None else round(float(np.linalg.norm(ls["L_relax"])) * mass_p, 1),
            "L_grid": None if ls is None else round(float(np.linalg.norm(ls["L_grid"])) * mass_p, 1)})
        P_prev[b] = P
    print(json.dumps(out), flush=True)
# onset per body and the particles that move first
for b in range(len(rows)):
    if land[b] is None:
        continue
    kl = int(np.searchsorted(ts, land[b]))
    pre = max(head_dev(b, k)[0] for k in range(k_m, kl + 1))
    onset = next((k for k in range(kl, len(fr)) if head_dev(b, k)[0] > pre + 1.0), None)
    rec = {"body": b, "landing_s": land[b], "head_n": int(len(head[b])), "head_rms_pre_landing_max_sp": round(pre, 2),
           "onset_s": None if onset is None else round(float(ts[onset]), 3)}
    if onset is not None:
        d = np.linalg.norm(Y[b][onset][head[b]] - base[b][head[b]], axis=1) / sp
        top = head[b][np.argsort(-d)[:12]]
        k1 = min(len(fr) - 1, onset + 20)
        rec.update(top_indices=(top + rows[b][0]).tolist(), top_dev_sp=np.round(np.sort(d)[::-1][:12], 2).tolist(),
                   top_dev_1s_later_sp=np.round(np.linalg.norm(Y[b][k1][top] - base[b][top], axis=1) / sp, 2).tolist(),
                   top_height_frac=np.round((base[b][top, 1] - base[b][:, 1].min()) / np.ptp(base[b][:, 1]), 3).tolist())
        # the driven / released split of the head set's motion from the landing to the onset + 1 s
        acc = {"driven": [], "released": [], "mixed": []}
        for kk in range(kl, min(len(fr) - 1, onset + 20)):
            dd = np.linalg.norm(Y[b][kk + 1][head[b]] - Y[b][kk][head[b]], axis=1)
            acc[kind[kk]].append(float(np.sqrt((dd * dd).mean()) / sp))
        rec["head_step_mean_sp"] = {kk: (round(float(np.mean(v)), 3) if v else None) for kk, v in acc.items()}
    print(json.dumps(rec), flush=True)
