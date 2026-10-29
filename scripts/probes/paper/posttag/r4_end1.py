"""r4_end1.py OUT_DIR NAME=ARCHIVE [...] -- D145 reading 4 (1): each archive's end frame as its own end1 archive (frames,
raws, tgt, deliver_n), the last frame that is not NaN (a baseline's unfillable frame is NaN), for the yardstick and the
display probes (as tmp/d145/endframe.py for ours). CPU, numpy only."""
import sys

import numpy as np

out = sys.argv[1]
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    z = np.load(path, allow_pickle=True)
    F, raws = z["frames"], np.asarray(z["raws"])
    k = len(F) - 1
    while k > 0 and not np.isfinite(np.asarray(F[k][:16])).all():
        k -= 1
    np.savez(f"{out}/{name}_end1.npz", frames=np.asarray(F[k:k + 1], np.float32), raws=raws[k:k + 1],
             tgt=np.asarray(z["tgt"], np.float32), deliver_n=np.int64(1))
    print(name, "frame", k, "of", len(F), "raw", int(raws[k]), flush=True)
