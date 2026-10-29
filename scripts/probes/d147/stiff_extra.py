"""stiff_extra.py RUNS_DIR TAGS REFS -- D147, the stiff-material gate's further items (pre-registered 2026-10-17): per
run, (8) the impacts' relative speed against the material's wave speed c (the scene info's): the pair's closing speed
along the line of centres at the window before the first contact impulse, and each body's centre's downward speed at
the window before its landing; (9) the ear tips' stretch through the landing's squash: each body's tip band (the top
tenth of its height at the morph's end, in its frame), each tip particle's distances to its 8 nearest band-or-body
neighbours of the morph's end, then over at the morph's end (the material-line stretch, piece.py's), its mean and its
95th percentile over the band, the largest over [landing + 2.1 s, landing + 3.4 s] (11.0-12.3 s for a body landing at
8.9 s: the forensic's window); (10) the bounce, the tumble and the parting: whether the pair parted in the air, each
body's lowest particle's largest height above the floor within 2 s after its landing (spacings), the tumble;
(11) rest at the end: each body's centre speed and angular speed at the last window, and the deformation's kinetic
energy per unit mass over the last second (central differences of the kept frames, the least-squares rigid velocity
removed). REFS: the runs it is read against (the previous gate's). Read-only, CPU."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def frame_of(x, X):
    c = x.mean(0)
    A = (x - c).T @ (X - X.mean(0))
    U, _, Vt = np.linalg.svd(A)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return (x - c) @ (U @ D @ Vt)


def rigid_free(x, v):
    c = x.mean(0)
    r = x - c
    vc = v.mean(0)
    Iner = (r * r).sum() * np.eye(3) - r.T @ r
    w = np.linalg.solve(Iner, np.cross(r, v - vc).sum(0))
    return v - vc - np.cross(w, r)


runs = Path(sys.argv[1])
for tag in sys.argv[2].split(",") + sys.argv[3].split(","):
    j = json.loads((runs / f"{tag}.json").read_text())
    z = np.load(runs / f"{tag}.npz")
    fr, rows, raws = z["frames"], z["rows"], z["raws"]
    dt, R, c = j["mpm"]["dt"], j["scene"]["R"], j["scene"]["c"]
    n_b = rows[0][1] - rows[0][0]
    sp = (4.18879 * R ** 3 / n_b) ** (1 / 3)
    ts = raws * dt
    floor_y = float(z["floor_y"])
    t_m = float(j["scene"]["spec"].get("morph_time", 0.0))
    k_m = int(np.argmin(np.abs(ts - t_m)))
    h = [r for r in j["history"] if "bodies" in r]
    hit = [bool(np.linalg.norm(r["contact_impulse"]) > 0) for r in h]
    k0 = hit.index(True) if any(hit) else None
    out = {"tag": tag, "c": round(c, 3), "young": j["config"].get("young")}
    if k0:
        pre = h[k0 - 1]["bodies"]
        c0, c1 = (np.array(b["com"]) for b in pre)
        n = (c1 - c0) / np.linalg.norm(c1 - c0)
        vrel = float(np.dot(np.array(pre[0]["v_com"]) - np.array(pre[1]["v_com"]), n))
        out["contact_closing_speed"] = round(vrel, 3)
        out["contact_v_over_c"] = round(vrel / c, 4)
    bodies = []
    for b, (s, e) in enumerate(rows):
        low = np.array([float(fr[k, s:e, 1].min()) for k in range(len(fr))]) - floor_y
        hits = np.nonzero(low < sp)[0]
        rec = {"body": b}
        if len(hits):
            kl = int(hits[0])
            tl = float(ts[kl])
            before = [r for r in h if r["t"] <= tl]
            vy = -float(before[-1]["bodies"][b]["v_com"][1]) if before else None
            after = [k for k in range(kl, len(fr)) if ts[k] <= tl + 2.0]
            rec.update(landing_s=tl, landing_v_over_c=round(vy / c, 4) if vy is not None else None,
                       landing_speed=round(vy, 3) if vy is not None else None,
                       bounce_height_sp=round(float(low[after].max()) / sp, 2))
            # the tip band's material-line stretch through the squash
            X = np.asarray(z[f"src{b}"], np.float64)
            ym = frame_of(np.asarray(fr[k_m, s:e], np.float64), X)
            hf = (ym[:, 1] - ym[:, 1].min()) / np.ptp(ym[:, 1])
            tip = np.nonzero(hf >= 0.9)[0]
            xm = np.asarray(fr[k_m, s:e], np.float64)
            nb = cKDTree(xm).query(xm[tip], 9)[1][:, 1:]
            L0 = np.linalg.norm(xm[tip][:, None] - xm[nb], axis=2)
            st_mean, st_p95 = 0.0, 0.0
            for k in range(len(fr)):
                if tl + 2.1 <= ts[k] <= tl + 3.4:
                    xk = np.asarray(fr[k, s:e], np.float64)
                    st = (np.linalg.norm(xk[tip][:, None] - xk[nb], axis=2) / L0).mean(1)
                    st_mean, st_p95 = max(st_mean, float(st.mean())), max(st_p95, float(np.percentile(st, 95)))
            rec.update(tip_n=int(len(tip)), tip_stretch_mean_max=round(st_mean, 3), tip_stretch_p95_max=round(st_p95, 3))
        last = h[-1]["bodies"][b]
        rec.update(end_v_com=round(float(np.linalg.norm(last["v_com"])), 4),
                   end_omega=round(float(np.linalg.norm(last["omega"])), 4))
        ke = []
        for k in range(1, len(fr) - 1):
            if ts[k] >= ts[-1] - 1.0:
                xk = np.asarray(fr[k, s:e], np.float64)
                v = (np.asarray(fr[k + 1, s:e], np.float64) - np.asarray(fr[k - 1, s:e], np.float64)) / (ts[k + 1] - ts[k - 1])
                ke.append(0.5 * float((rigid_free(xk, v) ** 2).sum(1).mean()))
        rec["end_deformation_ke_per_mass"] = round(float(np.mean(ke)), 5) if ke else None
        bodies.append(rec)
    out["bodies"] = bodies
    print(json.dumps(out), flush=True)
