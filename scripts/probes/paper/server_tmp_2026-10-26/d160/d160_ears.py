"""d160_ears.py -- the bunny's ear base and head direction by the probe's slab rule (pieces of >= MIN points) on several
target samples (the frame is the mesh's: the same at every N and seed)."""
import glob, json, sys
import numpy as np
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d160")
from d160_probe import components, head_dir, pitch_of

def ear_base(tgt, a, mn):
    y = tgt[:, 1]
    two = False
    rows = []
    for h in np.arange(y.max() - 2 * a, y.min(), -.5 * a):
        sl = tgt[(y >= h) & (y < h + a)]
        n, lab = components(sl, 1.5 * a)
        big = int((np.bincount(lab) >= mn).sum()) if len(sl) else 0
        rows.append((round(float(h), 3), len(sl), big))
        if big >= 2:
            two = True
        if two and big == 1:
            return float(h + a), rows
    return None, rows

for p in sys.argv[1:]:
    tgt = np.asarray(np.load(p)["tgt"], np.float32)
    a = pitch_of(tgt)
    n = len(tgt)
    for mn in (20, int(round(20 * (n / 40000) ** (2 / 3)))):
        yb, rows = ear_base(tgt, a, mn)
        hd = head_dir(tgt, yb, a)[0].tolist() if yb else None
        print(json.dumps(dict(path=p.split("/")[-1], n=n, min_pts=mn, pitch=a, ear_base=yb, head_dir=hd,
                              above=float((tgt[:, 1] >= yb).mean()) if yb else None)), flush=True)
