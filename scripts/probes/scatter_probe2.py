"""Scatter-then-return probe v2: THIN-FEATURE region defined on the target by local
neighbour count (not a y-box), with a mask panel to verify it is the ears.
Usage: python scatter_probe2.py OUT_DIR run1 run2 ..."""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

OUT = sys.argv[1]
RUNS = sys.argv[2:]


def thin_mask(tgt, radius=0.9, q=0.22):
    tree = cKDTree(tgt)
    cnt = np.array([len(c) for c in tree.query_ball_point(tgt, radius, workers=-1)])
    thr = np.quantile(cnt, q)
    m = cnt <= thr
    # keep only the largest-extent side: thin features are FAR from the centroid too
    r = np.linalg.norm(tgt - tgt.mean(0), axis=1)
    return m & (r > np.quantile(r, 0.5)), cnt


rows_all = {}
figs = []
for name in RUNS:
    p = os.path.join(OUT, name + ".npz")
    if not os.path.exists(p):
        print("missing", p); continue
    z = np.load(p)
    fr, tgt, src = z["frames"], z["tgt"], z["src"]
    dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(fr)
    fr = fr[:dn]
    sp = float(np.median(cKDTree(tgt).query(tgt, k=2, workers=-1)[0][:, 1]))
    tm, cnt = thin_mask(tgt)
    thin_pts = tgt[tm]
    ttree = cKDTree(tgt); thin_tree = cKDTree(thin_pts)
    thin_tgt_frac = float(tm.mean())
    step = max(1, len(fr) // 120)
    rows = []
    for i in range(0, len(fr), step):
        x = fr[i]
        d_t = ttree.query(x, k=1, workers=-1)[0]
        d_nn = cKDTree(x).query(x, k=9, workers=-1)[0][:, 1:]
        d_thin = thin_tree.query(x, k=1, workers=-1)[0]
        in_thin = d_thin < 1.5 * sp                      # particle currently ON a thin-feature target point
        sparse = d_nn.mean(1) > 2.0 * sp
        rows.append(dict(frame=i, far2=float((d_t > 2 * sp).mean()),
                         thin_frac=float(in_thin.mean()),
                         thin_sparse=float(sparse[in_thin].mean()) if in_thin.any() else 0.0,
                         iso=float((d_nn.min(1) > 2 * sp).mean()),
                         thin_cover=float((thin_tree.query(x[in_thin], k=1)[0] < sp).sum() / max(len(thin_pts), 1)) if in_thin.any() else 0.0))
    peak = max(rows, key=lambda r: r["thin_sparse"]); end = rows[-1]
    rows_all[name] = dict(sp=sp, thin_tgt_frac=thin_tgt_frac, rows=rows, n_frames=len(fr))
    print(f"{name:45s} thin_tgt_frac={thin_tgt_frac:.3f} | thin-region sparse: peak {peak['thin_sparse']:.3f} @frame {peak['frame']} -> end {end['thin_sparse']:.3f} | "
          f"mass on thin targets: {rows[0]['thin_frac']:.3f} -> {end['thin_frac']:.3f} (target {thin_tgt_frac:.3f}) | far>2sp end {end['far2']:.4f} | iso end {end['iso']:.4f}")
    if not figs:
        figs.append((tgt, tm))

json.dump(rows_all, open(os.path.join(OUT, "scatter_probe2.json"), "w"))
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    tgt, tm = figs[0]
    def proj(x, az, el=0.18):
        right = np.array([np.cos(az), 0, -np.sin(az)]); up = np.array([-np.sin(el)*np.sin(az), np.cos(el), -np.sin(el)*np.cos(az)])
        return x @ right, x @ up
    u, v = proj(tgt, 0.6)
    ax[0].scatter(u[~tm], v[~tm], s=0.3, c="lightgray"); ax[0].scatter(u[tm], v[tm], s=0.5, c="crimson")
    ax[0].set_aspect("equal"); ax[0].set_title(f"thin-feature target region (red, {tm.mean()*100:.0f}% of mass), az 0.6")
    for name, r in rows_all.items():
        fx = [q["frame"] / r["n_frames"] for q in r["rows"]]
        ax[1].plot(fx, [q["thin_sparse"] for q in r["rows"]], label=name.replace("_render_full_dt_iso_nn", ""))
        ax[2].plot(fx, [q["thin_frac"] for q in r["rows"]], label=name.replace("_render_full_dt_iso_nn", ""))
    ax[1].set_title("particles ON thin targets that are locally sparse (8-NN mean > 2 sp)")
    ax[2].set_title("fraction of ALL particles on thin targets (dashed = target share)")
    for name, r in rows_all.items():
        ax[2].axhline(r["thin_tgt_frac"], ls="--", lw=0.8, color="gray")
    for a in ax[1:]:
        a.set_xlabel("delivered fraction"); a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "scatter_probe2.png"), dpi=110); print("figure saved")
except Exception as e:
    print("figure skipped:", e)
