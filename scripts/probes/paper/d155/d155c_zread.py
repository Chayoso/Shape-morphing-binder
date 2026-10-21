"""d155c_zread.py MESH [...] -- D155c forensic (CPU): what volume and what local density the targets on the way ask,
from the dumped points (d155c_zdump.py). At each s, Z_s = Y + (1 - s) D_Y (before any match):
  med   the run's estimator: (median 8th-neighbour distance on the seeded 20k subsample / sp(Y))^3 = morph_vol_ratio;
  mean  the cloud's volume as the sum of its points' local volumes: mean d8(Z_s)^3 / mean d8(Y)^3 (kNN volume estimator,
        unbiased for 1 / density; its sum over the points estimates the occupied volume);
  r_i   each point's own local volume against itself on Y, (d32(Z_s, i) / d32(Y, i))^3 with d32 the mean distance to its
        32 nearest (smooth part; p01 / p50 / p99 after dividing by its mean, i.e. what a global match leaves);
  grid  the transport's own view: point counts per gate cell (CIC-free histogram at the gate's dx): the occupied cells'
        volume (cells holding >= half the median count of Y's interior cells) over Y's, and the p99 cell count over Y's.
The body's own volume against Y's (the source's mean d8^3) is printed once."""
import sys

import numpy as np
from scipy.spatial import cKDTree

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
S = (0.0, 0.008, 0.03, 0.06, 0.11, 0.16, 0.22, 0.3, 0.37, 0.44, 0.52, 0.6, 0.67, 0.74, 0.8, 0.86, 0.91, 0.95, 0.98, 1.0)


def dk(p, k, mean=False):
    d = cKDTree(p).query(p, k + 1, workers=4)[0][:, 1:]
    return d.mean(1) if mean else d[:, -1]


def med_est(p):
    n = len(p)
    idx = np.random.default_rng(0).choice(n, 20000, replace=False) if n > 20000 else np.arange(n)
    sub = p[idx]
    return float(np.median(dk(sub, 8))) * (len(sub) / n) ** (1 / 3)


def grid(p, h, lo):
    ijk = np.floor((p - lo) / h).astype(np.int64)
    _, c = np.unique(ijk, axis=0, return_counts=True)
    return c


for mesh in sys.argv[1:]:
    z = np.load(f"{O}/zdump_{mesh}40k.npz")
    Y, D, src, sp_y, gdx = z["Y"].astype(np.float64), z["D"].astype(np.float64), z["src"].astype(np.float64), float(z["sp_y"]), float(z["gdx"])
    d8Y, d32Y = dk(Y, 8), dk(Y, 32, True)
    vY = np.mean(d8Y ** 3)
    lo = np.minimum(Y.min(0), (Y + D).min(0)) - 2 * gdx
    cY = grid(Y, gdx, lo)
    full = np.median(cY)
    occY, p99Y = np.sum(cY >= 0.5 * full), np.quantile(cY, 0.99)
    print(f"{mesh}: sp(Y) {sp_y:.5f} (here {med_est(Y):.5f}); gate dx {gdx:.4f}; body (source) volume / Y's by mean d8^3 "
          f"{np.mean(dk(src, 8) ** 3) / vY:.4f}, by the median {(med_est(src) / sp_y) ** 3:.4f}")
    print("   s     med    mean  | r_i/mean p01   p50   p99  | grid occ  p99 count")
    for s in S:
        Zs = Y + (1 - s) * D
        med = (med_est(Zs) / sp_y) ** 3
        mean = np.mean(dk(Zs, 8) ** 3) / vY
        r = (dk(Zs, 32, True) / d32Y) ** 3
        rq = np.quantile(r / r.mean(), [0.01, 0.5, 0.99])
        c = grid(Zs, gdx, lo)
        print(f"  {s:.3f} {med:.4f} {mean:.4f} |      {rq[0]:.3f} {rq[1]:.3f} {rq[2]:.3f} |   {np.sum(c >= 0.5 * full) / occY:.4f} "
              f"{np.quantile(c, 0.99) / p99Y:.3f}", flush=True)
