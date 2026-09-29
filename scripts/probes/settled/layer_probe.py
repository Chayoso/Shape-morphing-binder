"""layer_probe.py LABEL FILE REF PLAN — the delivered motion of the head-top-bound material by PARTICLE LAYER: the
outermost layer (the count-based outer set, depth 0), then depth bands in NATIVE spacings behind it (0-1, 1-2, 2-4,
4-8 sp). Median |x1-x0| in wu and the share along the paced step. A per-layer surface effect shows as the outermost
layer moving far more than the layers behind it at every N; a per-world-depth effect does not."""
import sys, numpy as np
from scipy.spatial import cKDTree
OUT = "/data/relcfd/chayo/physmorph_v2/output/"
label, fn, ref, plan = sys.argv[1:5]
z = np.load(fn); x0 = np.asarray(z["x0"], np.float32); x1 = np.asarray(z["x1"], np.float32)
zr = np.load(OUT + ref + "_render_full_dt_iso_nn.npz", allow_pickle=True); XE = np.asarray(zr["frames"][-1], np.float32)
p = np.load(OUT + "scratch/plandump/" + plan + "_plan.npz"); paced = (p["x_int"] - p["x0"]).astype(np.float32); pn = np.linalg.norm(paced, axis=1)
sp = float(np.median(cKDTree(x0).query(x0, k=2)[0][:, 1]))
k0 = cKDTree(x0); cnt = np.asarray(k0.query_ball_point(x0, r=2.0 * sp, return_length=True)); outer = cnt < 0.6 * np.median(cnt)
dep = cKDTree(x0[outer]).query(x0, k=1)[0] / sp; sel = XE[:, 1] > 2.3
d = x1 - x0; dn = np.linalg.norm(d, axis=1); along = (d * paced).sum(1) / np.maximum(pn ** 2, 1e-12)
cells = []
for name, m in [("outer", sel & outer)] + [(f"{lo}-{hi}", sel & ~outer & (dep > lo) & (dep <= hi)) for lo, hi in [(0, 1), (1, 2), (2, 4), (4, 8)]]:
    cells.append(f"{m.sum():5d} {np.median(dn[m]):.4f} {np.median(along[m]):5.2f}" if m.sum() >= 5 else "      -         ")
print(f"   {label:26s} sp {sp:.3f} | " + " | ".join(cells))
