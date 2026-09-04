"""Solid-target verdict census (docs/experiments.md): one line per metric, run on a finished
run's json + npz.

usage: python scripts/probes/solid_census.py OUT_PREFIX [--arm render_full_dt_iso_nn]
  reads OUT_PREFIX.json and OUT_PREFIX_<arm>.npz
Prints: run summary (last anim, accepted, rejects, best d_vol, anneal), guards, delivered
floaters (out_nn>2sp, far>3sp, uncovered@1.5sp/@2sp) plus the per-commit floater_frac
trajectory (REFUTE F6: the endpoint alone does not bound lateral ejection), ear-region
particle fraction, Lagrangian J_true (ears/body), y-band ratios (bottom / mid / top),
delivery cut vs the min-d_vol commit (REFUTE F2), tail zigzag as per-commit increments AND
as drift-removed peak-to-peak excursion (REFUTE F8: increments are not amplitude).
"""
import argparse
import json

import numpy as np
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--arm", default="render_full_dt_iso_nn")
    ap.add_argument("--ear_frac", type=float, default=0.10)
    ap.add_argument("--tail", type=int, default=40)
    a = ap.parse_args()
    d = json.load(open(a.prefix + ".json"))
    arm = d["arms"][a.arm]
    h = arm["history"]
    acc = [q for q in h if q.get("d_vol") is not None and not q.get("null_commit")]
    rej = [q for q in h if q.get("outer_rejected")]
    best = min(acc, key=lambda q: q["d_vol"]) if acc else None
    print(f"anims {len(h)} accepted {len(acc)} rejects {len(rej)} brake {sum(int(q.get('brake_reject', 0)) for q in rej)} "
          f"| best d_vol a{best['animation'] + 1} {best['d_vol']:.2f} | final d_vol {acc[-1]['d_vol']:.2f} "
          f"| anneal last {acc[-1].get('anneal')} | d_h1 first {acc[0].get('d_h1')} last {acc[-1].get('d_h1')}")
    g = {k: sum(int(q.get(k, 0) or 0) for q in h) for k in ("clamped", "nan_x", "nan_state", "F_reset", "F_flip", "F_invert_steps")}
    print("guards", g)
    ff = [q.get("floater_frac") for q in acc if q.get("floater_frac") is not None]
    if ff:
        print(f"floater_frac trajectory: first {ff[0]:.4f} max {max(ff):.4f} (a{acc[int(np.argmax(ff))]['animation'] + 1}) last {ff[-1]:.4f}")
    z = np.load(f"{a.prefix}_{a.arm}.npz")
    dn = int(z["deliver_n"]) if "deliver_n" in z else len(z["frames"])
    fr = z["frames"]; x = fr[dn - 1].astype(np.float64); t = z["tgt"].astype(np.float64)
    src = fr[0].astype(np.float64)
    tt = cKDTree(t); sp = float(np.median(tt.query(t, k=2)[0][:, 1]))
    dx_t = tt.query(x)[0]
    tx = cKDTree(x); cov = tx.query(t)[0]
    dl = [q for q in acc if q.get("frame_end") and q["frame_end"] <= dn]
    print(f"delivered frames {dn}/{len(fr)} (delivered commit a{dl[-1]['animation'] + 1 if dl else '?'}, d_vol {dl[-1]['d_vol'] if dl else '?'}; "
          f"min-d_vol commit a{best['animation'] + 1}) | out_nn>2sp {np.mean(dx_t > 2 * sp) * 100:.2f}%  far>3sp n={int(np.sum(dx_t > 3 * sp))}  "
          f"max_dt {dx_t.max() / sp:.1f}sp | uncovered@1.5sp {np.mean(cov > 1.5 * sp) * 100:.1f}% @2sp {np.mean(cov > 2 * sp) * 100:.1f}%")
    ext = t.max(0) - t.min(0); ax = int(np.argmax(ext))
    thr = np.quantile(t[:, ax], 1 - a.ear_frac)
    ear_p = x[:, ax] >= thr
    k = 8
    s0 = cKDTree(src).query(src, k=k + 1)[0][:, 1:].mean(1)
    s1 = tx.query(x, k=k + 1)[0][:, 1:].mean(1)
    J = (s1 / s0) ** 3
    print(f"ear-region particle frac {ear_p.mean():.3f} (target {a.ear_frac}) | J_true ears {np.median(J[ear_p]):.2f} body {np.median(J[~ear_p]):.2f}")
    lo, hi = t[:, ax].min(), t[:, ax].max()
    bands = [(lo, lo + 0.2 * (hi - lo)), (lo + 0.4 * (hi - lo), lo + 0.6 * (hi - lo)), (lo + 0.8 * (hi - lo), hi + 1e9)]
    out = []
    for b0, b1 in bands:
        nt = np.sum((t[:, ax] >= b0) & (t[:, ax] < b1)); nc = np.sum((x[:, ax] >= b0) & (x[:, ax] < b1))
        out.append(nc / max(nt, 1) * (len(t) / len(x)))
    print(f"band ratios (bottom20% / mid20% / top20%): {out[0]:.2f} / {out[1]:.2f} / {out[2]:.2f}")
    fe = [q["frame_end"] for q in dl]
    if len(fe) > 12:
        X = fr[np.array(fe[-(a.tail + 1):]) - 1].astype(np.float64)
        u = np.diff(X, axis=0); n = np.linalg.norm(u, axis=2); un = u / np.maximum(n, 1e-12)[..., None]
        cos = (un[1:] * un[:-1]).sum(2)
        # drift-removed peak-to-peak excursion per particle over the tail (amplitude, not increment)
        T = np.arange(len(X))[:, None, None]
        drift = X[0][None] + (X[-1] - X[0])[None] * (T / max(len(X) - 1, 1))
        dev = np.linalg.norm(X - drift, axis=2)          # (T, N)
        p2p = dev.max(0) / sp
        ear_tail = X[-1][:, ax] >= thr
        print(f"tail({a.tail} commits) rev-cos med {np.median(cos):+.3f} osc% {np.mean(cos < 0) * 100:.1f} | per-commit move med {np.median(n) / sp:.3f} sp, frac>0.5sp {np.mean(n > 0.5 * sp):.4f} "
              f"| drift-removed excursion p2p: med {np.median(p2p):.3f} sp p99 {np.percentile(p2p, 99):.3f} sp max {p2p.max():.2f} sp; ears med {np.median(p2p[ear_tail]):.3f} p99 {np.percentile(p2p[ear_tail], 99):.3f} sp")


if __name__ == "__main__":
    main()
