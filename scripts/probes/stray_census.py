"""Mass-ejection census over archived runs (raw state): at the delivered frame, the number and
share of particles farther than 0.25 / 0.5 / 1.0 wu from ANY target point, the maximum such
distance, the first delivered commit at which any particle exceeds 0.5 wu (when the ejection
happens), and whether those particles ever come back (share of the 0.5-wu set at the end that
was already beyond 0.5 wu at the midpoint).
Usage: python stray_census.py OUT_DIR run1 run2 ...   (run = archive stem without .npz)
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

OUT = sys.argv[1]
for name in sys.argv[2:]:
    p = os.path.join(OUT, name + ".npz")
    if not os.path.exists(p):
        print(f"{name}: missing"); continue
    z = np.load(p)
    fr, tgt = z["frames"], z["tgt"]
    dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(fr)
    fr = fr[:dn]
    tree = cKDTree(tgt)
    d_end = tree.query(fr[-1], k=1, workers=-1)[0]
    n = len(d_end)
    c25, c50, c100 = int((d_end > 0.25).sum()), int((d_end > 0.5).sum()), int((d_end > 1.0).sum())
    # first frame with an ejected particle AFTER the body has arrived (median distance to
    # the target below 0.15 wu) — before arrival every particle is far by construction
    first = None
    step = max(1, len(fr) // 150)
    for i in range(0, len(fr), step):
        d = tree.query(fr[i], k=1, workers=-1)[0]
        if np.median(d) < 0.15 and (d > 0.5).any():
            first = i; break
    mid = tree.query(fr[len(fr) // 2], k=1, workers=-1)[0]
    far_end = d_end > 0.5
    persist = float((mid[far_end] > 0.5).mean()) if far_end.any() else float("nan")
    print(f"{name:44s} N={n} | end: >0.25 wu {c25} ({c25/n:.3%})  >0.5 wu {c50} ({c50/n:.3%})  >1.0 wu {c100}  max {d_end.max():.2f} wu | "
          f"first frame with >0.5 wu: {first if first is not None else 'never'} of {len(fr)} | of the end >0.5 set, already out at mid-run: {persist:.0%}")
