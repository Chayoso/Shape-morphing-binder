"""jv145.py ARCHIVE_NPZ OUT_JSON -- D145: the body's physical volume over time, sum_i J_i V0_i from the carried J (the
archive's J_samples, every T frames; volume_exact carried makes det F = J). Every particle has the same rest volume V0 and
match_density makes N V0 the target sample's fill volume, so the physical volume over the target's is the mean of J.
Not comparable across methods: the baselines carry no physical volume. Writes per sample the frame, tau = frame /
deliver_n, mean / p01 / p99 / min / max of J, and the run's min / max / end of the mean."""
import json
import sys

import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
J = np.asarray(z["J_samples"], np.float64)
idx = np.asarray(z["F_sample_idx"]).astype(int)
dn = int(z["deliver_n"])
if J.size == 0:
    raise SystemExit("no J_samples in the archive (volume not tracked)")
keep = idx < dn
J, idx = J[keep], idx[keep]
rows = [dict(frame=int(f), tau=float(f / max(dn - 1, 1)), mean=float(j.mean()), p01=float(np.quantile(j, 0.01)),
             p99=float(np.quantile(j, 0.99)), min=float(j.min()), max=float(j.max())) for f, j in zip(idx, J)]
m = np.array([r["mean"] for r in rows])
out = dict(archive=sys.argv[1], n_samples=len(rows), mean_min=float(m.min()), mean_max=float(m.max()), mean_end=float(m[-1]),
           mean_first=float(m[0]), rows=rows)
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(f"physical volume / target (mean J): first {m[0]:.5f}, min {m.min():.5f}, max {m.max():.5f}, end {m[-1]:.5f} "
      f"over {len(rows)} samples; per-particle J p01-p99 at the end {rows[-1]['p01']:.3f}-{rows[-1]['p99']:.3f}")
