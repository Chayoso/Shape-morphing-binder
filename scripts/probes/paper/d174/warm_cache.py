"""warm_cache.py [pipeline_run args] -- D174: the prepare stage and the thin set of a run, untimed, so that a timed run of the
same repo starts from the caches a repeated run has (the samples, the fill volume, the draws, the thin set)."""
import os
import sys
import time

import physmorph  # noqa: F401

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import pipeline_run as P  # noqa: E402
from physmorph.thin import thin_set_cached  # noqa: E402

args = P.parse_args()
t0 = time.time()
cfg0, prep = P.prepared(args, args.seed)
t1 = time.time()
ts = thin_set_cached(prep.tgt, prep.cell_shape or prep.prm.dx, cfg0.mass_ref_n)
print(f"[warm] prepare {t1 - t0:.1f} s, thin set {time.time() - t1:.1f} s ({len(ts.points)} points)", flush=True)
