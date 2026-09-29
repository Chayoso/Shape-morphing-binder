"""Ear fill and thickness per height slab (the user: the ears grow too thin, as droplets that meet).
Ear extraction as in ear_probe.py (y-up frame, the ear base from the vertical profile, the largest
component of the target above it). Slabs of 0.3 wu from base + 0.3 (pure ear material) to the tip;
in each slab the target's two ears are its connected components at 2.5 spacings (sorted by x: L, R),
each with a point count and an xz cross-section thickness (4 sqrt(lambda_min) of the xz covariance).
Per archived frame the particles within 0.6 wu of the ear's target points and above base + 0.3 are
assigned to the nearest ear in their slab: count / target count (fill) and thickness / target
thickness (thk). The time series also gives the pieces this pure ear material forms at 1.5 native
spacings and at 1.5 reference spacings (spacing x (N/ref_n)^(1/3), what the reference-spacing render
shows) with the number of pieces of >= 20 particles.
usage: ear_slab.py NPZ T_LIST(comma-separated fractions for the slab matrices) [REPO] [REF_N]"""
import sys, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, sys.argv[3] if len(sys.argv) > 3 else "/data/relcfd/chayo/physmorph_v2/repo")
from physmorph.sampling.orientation import orient_archive
path = sys.argv[1]; t_list = [float(s) for s in sys.argv[2].split(",")]
ref_n = int(sys.argv[4]) if len(sys.argv) > 4 else 40000
z = np.load(path, mmap_mode="r"); dn = int(z["deliver_n"])
frames, tgt, _, orient = orient_archive(z, path)
tgt = np.asarray(tgt, np.float32); N = tgt.shape[0]
kt = cKDTree(tgt); sp = float(np.median(kt.query(tgt, k=9, workers=-1)[0][:, -1])); dx = 0.306
sp_ref = sp * max(1.0, N / ref_n) ** (1.0 / 3.0)


def comps(P, r):
    if len(P) < 2:
        return 1, np.zeros(len(P), int)
    pairs = cKDTree(P).query_pairs(r, output_type="ndarray")
    A = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(P), len(P)))
    return connected_components(A, directed=False)


def xz_extent(P):
    if len(P) < 10:
        return float("nan"), float("nan")
    Q = P[:, [0, 2]]; ev = np.linalg.eigvalsh(np.cov((Q - Q.mean(0)).T))
    return float(4 * np.sqrt(max(ev[0], 0))), float(4 * np.sqrt(max(ev[1], 0)))


# the ear base from the vertical profile (as ear_probe.py) and the ear = the largest component above it
ys = tgt[:, 1]; e = np.arange(ys.min(), ys.max() + 0.1, 0.1)
cnt, _ = np.histogram(ys, e); peak_i = int(np.argmax(cnt)); ref = cnt[peak_i]
y_head = next((float(e[i]) for i in range(peak_i, len(cnt)) if cnt[i] < 0.30 * ref), float(np.quantile(ys, 0.85)))
ear_t = tgt[ys > y_head]
nc_t, lab_t = comps(ear_t, 2.5 * sp); ear = ear_t[lab_t == np.argmax(np.bincount(lab_t))]
ke = cKDTree(ear)
y0 = y_head + 0.3; y1 = float(ear[:, 1].max()); edges = np.arange(y0, y1, 0.3)
edges = np.append(edges, y1 + 1e-3); nslab = len(edges) - 1
pure_t = ear[ear[:, 1] >= y0]
slabs = []
for s in range(nslab):
    P = ear[(ear[:, 1] >= edges[s]) & (ear[:, 1] < edges[s + 1])]
    nc, lab = comps(P, 2.5 * sp); sizes = np.bincount(lab)
    keep = [c for c in np.argsort(sizes)[::-1][:2] if sizes[c] >= 30]
    parts = sorted([P[lab == c] for c in keep], key=lambda Q: float(Q[:, 0].mean()))
    slabs.append([(Q, cKDTree(Q), xz_extent(Q)) for Q in parts])
print("%s (orient %s): N %d sp %.3f ref sp %.3f; ear base %.2f, pure ear (y >= %.2f) %d target points, %d slabs of 0.3 wu" %
      (path.split("/")[-1], orient, N, sp, sp_ref, y_head, y0, len(pure_t), nslab))
print("  target per slab [y: L n/thk | R n/thk]: " + "  ".join("%.1f: %s" % (edges[s], " | ".join("%d/%.2f" % (len(Q), tt) for Q, _, (tt, _) in slabs[s])) for s in range(nslab)))


def frame_stats(x):
    inside = ke.query(x, workers=-1)[0] <= 0.6
    P = x[inside]; P = P[P[:, 1] >= y0]
    fill = np.full((nslab, 2), np.nan); thk = np.full((nslab, 2), np.nan)
    for s in range(nslab):
        Q = P[(P[:, 1] >= edges[s]) & (P[:, 1] < edges[s + 1])]; parts = slabs[s]
        if len(parts) == 0 or len(Q) == 0:
            continue
        d = np.stack([kd.query(Q, workers=-1)[0] for _, kd, _ in parts], 1); a = d.argmin(1)
        for j, (T, _, (tt, _)) in enumerate(parts):
            R = Q[a == j]; fill[s, j] = len(R) / len(T)
            t2, _ = xz_extent(R); thk[s, j] = t2 / tt if tt > 0 else np.nan
    if len(P) < 2:
        return len(P) / len(pure_t), fill, thk, 1, int(len(P) >= 20), 1, int(len(P) >= 20)
    _, lab_n = comps(P, 1.5 * sp); _, lab_r = comps(P, 1.5 * sp_ref)
    bn = np.bincount(lab_n); br = np.bincount(lab_r)
    return len(P) / len(pure_t), fill, thk, len(bn), int((bn >= 20).sum()), len(br), int((br >= 20).sum())


print("%6s %5s | %6s | %5s %4s | %5s %4s | %6s %6s" % ("frame", "t", "n/nT", "pcsN", ">=20", "pcsR", ">=20", "fill~", "thk~"))
stride = max(1, dn // 30)
for i in list(range(0, dn, stride)) + [dn - 1]:
    r, fill, thk, pn, bn, pr, br = frame_stats(np.asarray(frames[i], np.float32))
    print("%6d %5.2f | %6.3f | %5d %4d | %5d %4d | %6.2f %6.2f" % (i, i / (dn - 1), r, pn, bn, pr, br, np.nanmedian(fill), np.nanmedian(thk)))
sel = [min(dn - 1, int(round(t * (dn - 1)))) for t in t_list]
res = [frame_stats(np.asarray(frames[i], np.float32)) for i in sel]
for j, name in enumerate(("L", "R")):
    for q, label in ((1, "fill"), (2, "thk")):
        print("ear %s %s per slab (rows y from the base) x frames t=%s" % (name, label, ",".join("%.2f" % t for t in t_list)))
        for s in range(nslab):
            print("   y %.2f | " % edges[s] + " ".join("%6.2f" % rr[q][s, j] for rr in res))
