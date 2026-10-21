"""sc_mid.py OUT_JSON FRAMES12_NPZ -- SHOWCASE: the morph's progress per kept frame (CPU only), for the montage's middle
moment. d(k) = the symmetric mean nearest distance between kept frame k and the run's target sample (fixed 20k-particle
subsamples, seed 0); p(k) = 1 - d(k) / d(0). `mid` = the first kept frame with p >= 0.5, `t` = its time (k / 20 s:
a kept frame is every 12th of 240 per second)."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

z = np.load(sys.argv[2], allow_pickle=True)
fr, tgt = z["frames"], np.asarray(z["tgt"], np.float32)
rng = np.random.default_rng(0)
it = rng.choice(len(tgt), min(20000, len(tgt)), replace=False)
ip = rng.choice(fr.shape[1], min(20000, fr.shape[1]), replace=False)
tt = tgt[it]
kt = cKDTree(tt)
d = []
for k in range(len(fr)):
    x = np.asarray(fr[k][ip], np.float32)
    d.append(0.5 * (kt.query(x, workers=4)[0].mean() + cKDTree(x).query(tt, workers=4)[0].mean()))
d = np.asarray(d)
p = 1 - d / d[0]
mid = int(np.argmax(p >= 0.5)) if (p >= 0.5).any() else len(p) // 2
row = dict(frames=len(p), mid=mid, t_mid=mid / 20.0, p=[round(float(v), 4) for v in p], d0=float(d[0]), d_end=float(d[-1]))
json.dump(row, open(sys.argv[1], "w"))
print(json.dumps({k: row[k] for k in ("frames", "mid", "t_mid", "d0", "d_end")}))
