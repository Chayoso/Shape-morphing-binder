"""d155c_follow.py MESH ARM [ARM ...] -- D155c forensic (CPU): does the body take the local density the target on the way
asks? At fixed times, the body's kept frame and the window's own target Z_s (rebuilt from the dump: Y + (1 - s) D_Y,
rescaled about its centroid by the window's recorded morph_k when the run matched) are counted on one grid of cells two
gate cells wide. Interior cells only: the cell and its six face neighbours hold at least half the median interior count
of Y in both clouds (the rim's partial cells left out). There: the Spearman correlation of the body's count with the
target's, and the local volume (Y's median interior count / count) p01 / p50 / p99 of the body, of the target, and of Y
itself on its own interior (the final target's own spread, the reference). Equal totals (40k each)."""
import json
import sys

import numpy as np
from scipy.stats import spearmanr

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
mesh = sys.argv[1]
z = np.load(f"{O}/zdump_{mesh}40k.npz")
Y, D, gdx = z["Y"].astype(np.float64), z["D"].astype(np.float64), float(z["gdx"])
h = 2 * gdx


def interior(c, full):
    ok = c >= 0.5 * full
    m = ok.copy()
    for ax in range(3):
        for sh in (1, -1):
            m &= np.roll(ok, sh, axis=ax)
    return m


def q3(v):
    return "/".join(f"{x:.3f}" for x in np.quantile(v, [0.01, 0.5, 0.99]))


for arm in sys.argv[2:]:
    fz = np.load(f"{O}/{mesh}40k_{arm}_frames12.npz")
    fr, raws, dn = fz["frames"].astype(np.float64), fz["raws"], int(fz["deliver_n"])
    a = next(iter(json.load(open(f"{O}/{mesh}40k_{arm}.json"))["arms"].values()))
    acc = [r for r in a["history"] if r.get("frame_end") and not r.get("null_commit") and r["frame_end"] <= dn]
    print(f"{mesh} {arm}")
    for t_q in (0.17, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 99):
        i = int(np.argmin(np.abs(raws / 240.0 - t_q))) if t_q < 90 else int(np.argmin(np.abs(raws - dn)))
        rec = min(acc, key=lambda r: abs(r["frame_end"] - raws[i]))
        s = rec.get("morph_s", 1.0)
        Z = Y + (1 - s) * D
        if rec.get("morph_k") is not None and s < 1:
            c = Z.mean(0)
            Z = (Z - c) * rec["morph_k"] + c
        X = fr[i]
        allp = np.concatenate([X, Z, Y])
        lo = allp.min(0) - 2 * h
        dims = np.ceil((allp.max(0) + 2 * h - lo) / h).astype(int)

        def cnt(p):
            ijk = np.floor((p - lo) / h).astype(np.int64)
            return np.bincount(np.ravel_multi_index(ijk.T, dims), minlength=int(np.prod(dims))).reshape(dims)
        cy, cb, ct = cnt(Y), cnt(X), cnt(Z)
        full = np.median(cy[interior(cy, np.median(cy[cy > 0]))])
        iy = interior(cy, full)
        inside = interior(cb, full) & interior(ct, full)
        vb, vt = full / cb[inside], full / ct[inside]
        rho = spearmanr(cb[inside], ct[inside]).correlation if inside.sum() > 10 else float("nan")
        print(f"  t {raws[i] / 240:.2f}s s {s:.2f}: {inside.sum()} interior cells, corr {rho:+.3f}; local volume p01/p50/p99 "
              f"body {q3(vb)} | target {q3(vt)} | Y itself {q3(full / cy[iy])}; records J {rec.get('Jx_mean', 0):.4f} "
              f"p01 {rec.get('Jx_p01', 0):.3f} p99 {rec.get('Jx_p99', 0):.3f}", flush=True)
