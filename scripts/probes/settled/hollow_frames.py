"""hollow_frames.py TAG FRAME[,FRAME...] — the vapour reading at EXPLICIT trajectory frames (equal physical time across
runs whose window counts differ; 19 frames per window, so frames 76 / 114 / 152 = windows 4 / 6 / 8): the top region
(y > 2.3) particle count, the share below half the target density (fewer than 4 within the target's shell radius),
and the region's mean count over the target's 8; plus the same for the whole cloud."""
import sys, numpy as np
from scipy.spatial import cKDTree
OUT = "/data/relcfd/chayo/physmorph_v2/output/"
tag = sys.argv[1]; frames = [int(f) for f in sys.argv[2].split(",")]
z = np.load(OUT + tag + "_render_full_dt_iso_nn.npz", allow_pickle=True); F = z["frames"]; n = len(F)
XE = np.asarray(F[-1], np.float32)
try:
    T = np.load(OUT + "bm300_bunny_render_full_dt_iso_nn.npz", allow_pickle=True)["frames"][-1]   # the target's own spacing proxy
except Exception:
    T = XE
X0 = np.asarray(F[0], np.float32); rcov = 0.0690
print(f"{tag}: {n} frames ({(n - 1) / 19:.1f} windows); top region y > 2.3, rcov {rcov}")
for f in frames:
    if f >= n:
        print(f"   frame {f}: beyond the run ({n} frames)"); continue
    X = np.asarray(F[f], np.float32); k = cKDTree(X)
    cnt = np.asarray(k.query_ball_point(X, r=rcov, return_length=True)) - 1
    top = X[:, 1] > 2.3
    print(f"   frame {f:4d} (window {f / 19:4.1f}): all below-half {100 * (cnt < 4).mean():5.1f} %   top {100 * (cnt[top] < 4).mean() if top.any() else 0:5.1f} % "
          f"(of {int(top.sum()):5d})   top mean n/8 = {cnt[top].mean() / 8 if top.any() else 0:.2f}")
