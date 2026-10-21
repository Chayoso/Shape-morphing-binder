"""b9tilt.py FRAMES12_NPZ RUN_JSON -- B9b reading (CPU): a single-body B9 floor run's tilt over its kept frames, read as the
scene reads its body (the Buckens frame against the kept source frame), its centroid height against the standing target
(the run's target is lifted onto the floor), the horizontal drift, in pitches (0.708 x the median 8th-neighbour distance
of the target, momentum_probe.py's)."""
import json, sys
import numpy as np
from scipy.spatial import cKDTree

def polar(A):
    U, _, Vt = np.linalg.svd(A)
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt

z = np.load(sys.argv[1], allow_pickle=True)
mpm = json.load(open(sys.argv[2]))["provenance"]["mpm"]
fr, raws, tgt = z["frames"], z["raws"], np.asarray(z["tgt"], np.float64)
a = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::50], k=9)[0][:, 8]))
X = np.asarray(fr[0], np.float64); X -= X.mean(0)
c0 = np.asarray(fr[0], np.float64).mean(0)
h_t = tgt[:, 1].mean() - float(mpm["floor_y"])
rows = []
for k in list(range(0, len(fr), 10)) + [len(fr) - 1]:
    x = np.asarray(fr[k], np.float64); c = x.mean(0)
    R = polar((x - c).T @ X)
    tilt = float(np.degrees(np.arccos(np.clip(R[1, 1], -1, 1))))
    turn = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    rows.append((raws[k] / 240.0, tilt, turn, (c[1] - float(mpm["floor_y"]) - h_t) / a, np.hypot(*(c - c0)[[0, 2]]) / a))
print(f"{sys.argv[1].split('/')[-1]}: pitch {a:.4f} wu; largest tilt {max(r[1] for r in rows):.2f} deg; end tilt {rows[-1][1]:.2f}, "
      f"turn {rows[-1][2]:.2f} deg, height vs the standing target {rows[-1][3]:+.2f} pitches, horizontal drift {rows[-1][4]:.2f} pitches")
for r in rows:
    print(f"  t {r[0]:5.2f}  tilt {r[1]:5.2f}  turn {r[2]:5.2f}  height {r[3]:+6.2f} sp  drift {r[4]:.2f} sp")
