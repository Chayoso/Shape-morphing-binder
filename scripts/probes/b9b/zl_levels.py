"""zl_levels.py RUN_JSON [T0 T1] -- B9b step 1 reference (CPU): the assimilation's read p50 / p90 / p99 (median over the
committed windows starting in [T0, T1), default 4-6 s) of a run's window records."""
import json
import sys

import numpy as np

j = json.load(open(sys.argv[1]))
t0, t1 = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (4.0, 6.0)
h = [x for x in j["history"] if x.get("assim") and t0 <= x["t"] - 1.0 / 6.0 < t1]
print(f"{sys.argv[1].split('/')[-1]}: {len(h)} windows; read " + "/".join(
    f"{np.median([x['assim']['read'][q] for x in h]):.4f}" for q in ("p50", "p90", "p99")) + "; frozen read " + "/".join(
    f"{np.median([x['assim']['frozen_read'][q] for x in h]):.4f}" for q in ("p50", "p90", "p99")))
