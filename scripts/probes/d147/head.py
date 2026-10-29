"""head.py RUNS_DIR TAG[,TAG...] -- D147 (the fixes' gate, 2026-10-16): does each body's head (ears and the top of the
head) hold its shape after the landing? The head band is the top quarter of the target's height in its canonical frame
(height fraction >= 0.75: the ears and the crown). At every kept frame, in the body's Buckens frame (as progress.py and
the objective read it), the head band's two-sided mean distance between the target's points in the band and the body's
particles in the band (each to the other set's nearest), in sample spacings: 0 when the ears are where the bunny's are,
large when they are crumpled, bent or missing. Per body: the landing (its lowest particle within a spacing of the floor),
the curve every 0.25 s, and the recovery reading, the mean and the maximum over [landing + 1.5 s, cut] (cut.py's
rule; CUT_TXT RUNS_DIR/cut_<tag>.txt, else the whole run). (The forensic's earlier measure, the head particles'
displacement from their places at the morph's end, read the particles' rearrangement, not the shape: the interim's body
1 kept 13 spacings with its silhouette term at 0.003.) Prints one JSON line per run. Read-only, CPU."""
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


runs = Path(sys.argv[1])
for tag in sys.argv[2].split(","):
    j = json.loads((runs / f"{tag}.json").read_text())
    z = np.load(runs / f"{tag}.npz")
    fr, rows, raws = z["frames"], z["rows"], z["raws"]
    dt, R = j["mpm"]["dt"], j["scene"]["R"]
    n_b = rows[0][1] - rows[0][0]
    sp = (4.18879 * R ** 3 / n_b) ** (1 / 3)
    ts = raws * dt
    floor_y = float(z["floor_y"])
    t_cut = float(ts[-1])
    cf = runs / f"cut_{tag}.txt"
    if cf.exists():
        for ln in open(cf):
            if ln.startswith("{"):
                t_cut = json.loads(ln)["t_cut"]
    out = {"tag": tag, "spacing": round(sp, 5), "t_cut": t_cut, "bodies": []}
    step = max(1, int(round(0.25 / (dt * (raws[1] - raws[0])))))
    for b, (s, e) in enumerate(rows):
        X = np.asarray(z[f"src{b}"], np.float64)
        T = np.asarray(z[f"tgt{b}"], np.float64)
        h0, h1 = T[:, 1].min(), T[:, 1].max()
        cut_y = h0 + 0.75 * (h1 - h0)
        Th = T[T[:, 1] >= cut_y]
        kt = cKDTree(Th)
        low = np.array([float(fr[k, s:e, 1].min()) for k in range(len(fr))]) - floor_y
        hit = np.nonzero(low < sp)[0]
        t_land = float(ts[hit[0]]) if len(hit) else None

        def head(k):
            y = frame_of(np.asarray(fr[k, s:e], np.float64), X)
            yh = y[y[:, 1] >= cut_y]
            if len(yh) == 0:
                return float(cKDTree(y).query(Th)[0].mean() / sp)
            d1 = cKDTree(yh).query(Th)[0].mean()
            d2 = kt.query(yh)[0].mean()
            return float(0.5 * (d1 + d2) / sp)

        curve = [(round(float(ts[k]), 3), round(head(k), 3)) for k in range(0, len(fr), step)]
        rec = {"body": b, "landing_s": t_land, "head_n": int(len(Th)), "curve": curve}
        if t_land is not None:
            ks = [k for k in range(len(fr)) if t_land + 1.5 <= ts[k] <= t_cut + 1e-9]
            vals = [head(k) for k in ks]
            pre = [head(k) for k in range(len(fr)) if t_land - 1.0 <= ts[k] < t_land]
            rec.update(recovery_mean=round(float(np.mean(vals)), 3) if vals else None,
                       recovery_max=round(float(np.max(vals)), 3) if vals else None,
                       pre_landing_mean=round(float(np.mean(pre)), 3) if pre else None,
                       window=[round(t_land + 1.5, 3), t_cut])
        out["bodies"].append(rec)
    print(json.dumps(out), flush=True)
