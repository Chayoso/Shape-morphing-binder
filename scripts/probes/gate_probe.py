"""Gate-coverage probe: would / does the support gate (omega on the 3^3-cell count) cover the
thin-feature vanguard? For each archived frame: per-particle 3^3-cell count n_p on the MPM grid,
n0 = median over the SOURCE cloud, omega = smoothstep((n/n0 - lo)/(hi - lo)); reports the omega
statistics of the vanguard (particles ON thin targets AND locally sparse, same definition as
scatter_probe2.py) and of everything else. Raw state only, no renderer.
Usage: python gate_probe.py OUT_DIR dx lo hi run1 run2 ...   (grid_min -16, cubic domain)"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

OUT = sys.argv[1]
DX, LO, HI = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
RUNS = sys.argv[5:]
GMIN = -16.0
NG = int(round(32.0 / DX))


def thin_mask(tgt, radius=0.9, q=0.22):
    tree = cKDTree(tgt)
    cnt = np.array([len(c) for c in tree.query_ball_point(tgt, radius, workers=-1)])
    m = cnt <= np.quantile(cnt, q)
    r = np.linalg.norm(tgt - tgt.mean(0), axis=1)
    return m & (r > np.quantile(r, 0.5))


def counts27(x):
    """3^3-cell particle count around each particle's cell (the kernel's n_p)."""
    ijk = np.floor((x - GMIN) / DX).astype(np.int64)
    ok = ((ijk >= 0) & (ijk < NG)).all(1)
    grid = np.zeros((NG, NG, NG), np.int32)
    np.add.at(grid, (ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]), 1)
    pad = np.pad(grid, 1)
    n = np.zeros(len(x), np.float64)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                n[ok] += pad[ijk[ok, 0] + a, ijk[ok, 1] + b, ijk[ok, 2] + c]
    return n


def omega_of(n, n0):
    s = np.clip((n / n0 - LO) / (HI - LO), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


for name in RUNS:
    p = os.path.join(OUT, name + ".npz")
    if not os.path.exists(p):
        print("missing", p); continue
    z = np.load(p)
    fr, tgt, src = z["frames"], z["tgt"], z["src"]
    dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(fr)
    fr = fr[:dn]
    n_src = counts27(src)
    n0 = float(np.median(n_src[n_src > 0]))
    sp = float(np.median(cKDTree(tgt).query(tgt, k=2, workers=-1)[0][:, 1]))
    thin_tree = cKDTree(tgt[thin_mask(tgt)])
    step = max(1, len(fr) // 60)
    rows = []
    for i in range(0, len(fr), step):
        x = fr[i]
        n = counts27(x)
        om = omega_of(n, n0)
        d_nn = cKDTree(x).query(x, k=9, workers=-1)[0][:, 1:]
        in_thin = thin_tree.query(x, k=1, workers=-1)[0] < 1.5 * sp
        sparse = d_nn.mean(1) > 2.0 * sp
        van = in_thin & sparse
        body = ~sparse
        rows.append(dict(frame=i, gated_all=float((om < 1).mean()), zero_all=float((om == 0).mean()),
                         van_frac=float(van.mean()),
                         van_om=float(om[van].mean()) if van.any() else float("nan"),
                         van_zero=float((om[van] == 0).mean()) if van.any() else float("nan"),
                         van_n_over_n0=float(np.median(n[van]) / n0) if van.any() else float("nan"),
                         body_om=float(om[body].mean()), body_gated=float((om[body] < 1).mean()),
                         thin_om=float(om[in_thin].mean()) if in_thin.any() else float("nan")))
    pk = max(rows, key=lambda r: r["van_frac"])
    end = rows[-1]
    print(f"{name:44s} dx={DX} n0={n0:.0f} lo/hi={LO}/{HI} | source: gated {float((omega_of(n_src, n0) < 1).mean()):.2f} zero {float((omega_of(n_src, n0) == 0).mean()):.3f}")
    print(f"    vanguard @peak f{pk['frame']}: frac {pk['van_frac']:.3f}  median n/n0 {pk['van_n_over_n0']:.2f}  mean omega {pk['van_om']:.2f}  omega==0 {pk['van_zero']:.2f} | "
          f"body: mean omega {pk['body_om']:.2f} gated {pk['body_gated']:.2f} | on-thin mean omega {pk['thin_om']:.2f}")
    print(f"    end f{end['frame']}: vanguard frac {end['van_frac']:.3f} mean omega {end['van_om']:.2f} | body gated {end['body_gated']:.2f} | on-thin mean omega {end['thin_om']:.2f} | all gated {end['gated_all']:.2f} zero {end['zero_all']:.3f}")
