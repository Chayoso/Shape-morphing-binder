"""b9body.py FRAMES12_NPZ RUN_JSON OUT_NPZ -- B9b: a single-body floor run of B9 (frames12, its lifted target) read as
the scene reads its bodies: the end in the run's own Buckens frame (X: the kept source frame), with the target about its
centroid (the lift taken off), so that endgen145t.py reads it against the unlifted independent sample as the scene's
runs are read; and its end in the world: the frame's turn, tilt and turn about the vertical, the horizontal drift of the
centre of mass (pitches: 0.708 x the median 8th-neighbour distance of the target, momentum_probe.py's), the height of
the centroid above the floor against the standing target's, the lowest centre against the floor. CPU only."""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from tobody import angle, polar  # noqa: E402

z = np.load(sys.argv[1], allow_pickle=True)
mpm = json.loads(open(sys.argv[2]).read())["provenance"]["mpm"]
fr, raws, tgt = z["frames"], z["raws"], np.asarray(z["tgt"], np.float64)
X = np.asarray(fr[0], np.float64)
X = X - X.mean(0)
x = np.asarray(fr[-1], np.float64)
c = x.mean(0)
R = polar((x - c).T @ X)
y = ((x - c) @ R).astype(np.float32)
tc = (tgt - tgt.mean(0)).astype(np.float32)
np.savez(sys.argv[3], frames=y[None], raws=raws[-1:], tgt=tc, deliver_n=z["deliver_n"])
a = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::50], k=9)[0][:, 8]))
floor = float(mpm["floor_y"])
c0 = np.asarray(fr[0], np.float64).mean(0)
yaw = float(np.degrees(np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2])))
print(json.dumps({"file": sys.argv[1], "t_end": float(raws[-1]) / 240.0, "pitch_a": a, "turn_deg": angle(R),
                  "tilt_deg": float(np.degrees(np.arccos(np.clip(R[1, 1], -1, 1)))), "yaw_deg": yaw,
                  "drift_xz_a": float(np.hypot(*(c - c0)[[0, 2]])) / a,
                  "dh_wu": float((c[1] - floor) - (tgt[:, 1].mean() - tgt[:, 1].min())),
                  "lowest_a": float((x[:, 1].min() - floor) / a)}))
