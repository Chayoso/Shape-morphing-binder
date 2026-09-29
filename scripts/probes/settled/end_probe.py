"""End-frame audit of an archive (y-up frame): pieces off the body at 1.5 end spacings (components
other than the largest), the ear-tip mass (particles within 0.25 wu of the target's highest point,
in reference particles = count x ref_n / N, against the target's own count there) with their 8-NN
spacing, and the body under-fill (target points farther than one cell from any particle).
usage: end_probe.py NPZ [REF_N=40000] [REPO]"""
import sys, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else "/data/relcfd/chayo/physmorph_v2/repo")
from physmorph.sampling.orientation import orient_archive
path = sys.argv[1]; ref_n = int(sys.argv[2]) if len(sys.argv) > 2 else 40000
z = np.load(path, mmap_mode="r"); dn = int(z["deliver_n"])
frames, tgt, _, orient = orient_archive(z, path)
tgt = np.asarray(tgt, np.float32); N = len(tgt); dx = 0.306
x = np.asarray(frames[dn - 1], np.float32)
kx = cKDTree(x); kt = cKDTree(tgt)
sp = float(np.median(kx.query(x, k=9, workers=-1)[0][:, -1]))
sp_t = float(np.median(kt.query(tgt, k=9, workers=-1)[0][:, -1]))
pairs = kx.query_pairs(1.5 * sp, output_type="ndarray")
A = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(N, N))
nc, lab = connected_components(A, directed=False); sizes = np.bincount(lab); big = int(np.argmax(sizes))
off = sizes[np.arange(nc) != big]
print("%s (orient %s): end frame %d, edges at %.3f wu (1.5 end sp; end sp %.4f, target sp %.4f): %d pieces off the body, "
      "%d particles, largest piece %d" % (path.split("/")[-1], orient, dn - 1, 1.5 * sp, sp, sp_t, len(off), int(off.sum()),
                                           int(off.max()) if len(off) else 0))
tip = tgt[np.argmax(tgt[:, 1])]
idx = kx.query_ball_point(tip, 0.25); n_t = len(kt.query_ball_point(tip, 0.25))
nn8 = kx.query(x[idx], k=9, workers=-1)[0][:, -1] / sp if len(idx) else np.array([np.nan])
d_fill = kx.query(tgt, workers=-1)[0]
print("  ear tip at %s: %d particles within 0.25 wu = %.1f reference particles (target %d); their 8-NN %.2f sp (median) / "
      "%.2f (p90); target under-fill max %.3f wu, points farther than one cell from any particle: %d" %
      (np.round(tip, 2), len(idx), len(idx) * ref_n / N, n_t, float(np.median(nn8)), float(np.quantile(nn8, 0.9)),
       float(d_fill.max()), int((d_fill > dx).sum())))
