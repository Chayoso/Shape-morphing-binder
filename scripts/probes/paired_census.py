"""Paired-at-equal-d_vol geometry census (REFUTE Opus 2026-09-04, the decisive measurement).

Compare two runs at states of EQUAL d_vol: run A at the first accepted commit whose d_vol
<= level (default: run B's best d_vol), run B at its best-d_vol commit. If the faster run
is geometrically equal or better at equal d_vol, its extra term is a genuine win; if it is
worse on out_nn / cluster ratio / CIC sub-cell histogram, it trades sub-cell quality for
cell-scale mass (the x3 failure with a new name).

Metrics at both states: chamfer, out_nn>2sp, far>3sp, uncovered@1.5/@2sp, band ratios,
ear frac, cluster ratio (particle/target kernel density at particle positions, h = 2 sp,
docs/floaters.md), and the histogram of particles' CIC sub-cell fractions per axis on the
loss grid (a peak at 0.5 = the density-blind lattice attractor of an uncorrected H^-1).

usage: python scripts/probes/paired_census.py A_PREFIX B_PREFIX [--level 30] [--loss_res 64]
"""
import argparse
import json

import numpy as np
from scipy.spatial import cKDTree


def _state(prefix, arm, pick):
    d = json.load(open(prefix + ".json"))
    h = [q for q in d["arms"][arm]["history"] if q.get("d_vol") is not None
         and not q.get("null_commit") and q.get("frame_end")]
    q = pick(h)
    z = np.load(f"{prefix}_{arm}.npz")
    return q, z["frames"][int(q["frame_end"]) - 1].astype(np.float64), z["tgt"].astype(np.float64), \
        z["frames"][0].astype(np.float64)


def _metrics(x, t, src, domain, loss_res):
    tt = cKDTree(t); sp = float(np.median(tt.query(t, k=2)[0][:, 1]))
    tx = cKDTree(x)
    dxt = tt.query(x)[0]; cov = tx.query(t)[0]
    chamfer = 0.5 * (dxt.mean() + cov.mean())
    ax = int(np.argmax(t.max(0) - t.min(0)))
    thr = np.quantile(t[:, ax], 0.9)
    lo, hi = t[:, ax].min(), t[:, ax].max()
    bands = [(lo, lo + 0.2 * (hi - lo)), (lo + 0.4 * (hi - lo), lo + 0.6 * (hi - lo)), (lo + 0.8 * (hi - lo), hi + 1e9)]
    br = [np.sum((x[:, ax] >= b0) & (x[:, ax] < b1)) / max(np.sum((t[:, ax] >= b0) & (t[:, ax] < b1)), 1) * len(t) / len(x)
          for b0, b1 in bands]
    # cluster ratio (docs/floaters.md): kernel density of particles vs of target points, at particles
    h = 2.0 * sp
    def kde(pts_tree, q, n_ref):
        d, _ = pts_tree.query(q, k=33)
        return np.exp(-(d / h) ** 2).sum(1) / n_ref
    rho_p = kde(tx, x, len(x)); rho_t = kde(tt, x, len(t))
    cr = np.median(rho_p / np.maximum(rho_t, 1e-12))
    # CIC sub-cell fractions on the loss grid
    dxl = domain / loss_res
    frac = ((x + domain / 2) / dxl) % 1.0
    hist = np.histogram(frac.reshape(-1), bins=10, range=(0, 1))[0] / frac.size
    centre_excess = hist[4:6].sum() / 0.2 - 1.0        # >0: pile-up at the cell centre
    return dict(chamfer=chamfer, out_nn=np.mean(dxt > 2 * sp) * 100, far=int(np.sum(dxt > 3 * sp)),
                unc15=np.mean(cov > 1.5 * sp) * 100, unc2=np.mean(cov > 2 * sp) * 100,
                ear=np.mean(x[:, ax] >= thr), bands=br, cluster=cr, centre_excess=centre_excess,
                hist=np.round(hist, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--arm", default="render_full_dt_iso_nn")
    ap.add_argument("--level", type=float, default=None)
    ap.add_argument("--domain", type=float, default=32.0)
    ap.add_argument("--loss_res", type=int, default=64)
    a = ap.parse_args()
    qb, xb, tb, sb = _state(a.b, a.arm, lambda h: min(h, key=lambda q: q["d_vol"]))
    level = a.level if a.level is not None else qb["d_vol"]
    def first_below(h):
        for q in h:
            if q["d_vol"] <= level:
                return q
        return h[-1]
    qa, xa, ta, sa = _state(a.a, a.arm, first_below)
    for name, q, x, t, s in (("A", qa, xa, ta, sa), ("B", qb, xb, tb, sb)):
        mtr = _metrics(x, t, s, a.domain, a.loss_res)
        print(f"{name} commit a{q['animation'] + 1} d_vol {q['d_vol']:.2f} frame {q['frame_end']} | chamfer {mtr['chamfer']:.4f} "
              f"out_nn {mtr['out_nn']:.2f}% far {mtr['far']} uncovered {mtr['unc15']:.1f}%/{mtr['unc2']:.1f}% | ear {mtr['ear']:.3f} "
              f"bands {mtr['bands'][0]:.2f}/{mtr['bands'][1]:.2f}/{mtr['bands'][2]:.2f} | cluster ratio {mtr['cluster']:.2f} | "
              f"CIC centre excess {mtr['centre_excess']:+.3f} hist {mtr['hist']}")


if __name__ == "__main__":
    main()
