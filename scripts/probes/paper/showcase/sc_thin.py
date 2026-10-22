"""sc_thin.py OUT_PREFIX CELL L_FRAMES12 P_NPZ [P_FRAMES12|-] -- SHOWCASE (the coordinator, 2026-10-22 00:00): where are the
uncovered thin points of the render arm L and of its lambda = 0 twin P, and when do they appear? CPU only (scipy).

The thin set as physmorph/thin.py defines it, recomputed on the CPU: the target sample's outer points (fewer than 0.6 of
the median count of points within two spacings) whose local thickness (Hildebrand & Ruegsegger max-ball thickness on a
voxel grid of the target spacing) is below two MPM cells (CELL, the run's cell). A thin point is uncovered when the
body's nearest particle is farther than 1.5 target spacings (thin_uncovered). The run's own record is printed beside
the recomputed share, as the check of the recomputation.

Writes OUT_PREFIX.json: the shares (L, P, L only, P only, both), the distance quantiles in spacings of the uncovered
points and of all thin points, the share of thin points in the band 1.4-1.6 spacings, the uncovered points' regions
(connected sets at 3 spacings, the largest ten: size, centroid, height in the target's extent, azimuth around the
vertical axis, L / P counts), and with frames the course over every 4th kept frame (share uncovered, distance
quantiles) and, for the points uncovered in L at the end, the time since when they stay uncovered. OUT_PREFIX.png:
the thin points from two cameras (the target's front az 180 and its side az 90; +y up), coloured by covered in both
(grey), uncovered in L only (red), in P only (blue), in both (purple)."""
import json
import math
import sys

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

out, cell, lpath, ppath = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
pframes = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
zl = np.load(lpath, allow_pickle=True)
tgt = np.asarray(zl["tgt"], np.float64)
tree_t = cKDTree(tgt)
sp = float(np.median(tree_t.query(tgt, 2, workers=8)[0][:, 1]))


def local_thickness(tgt, sp):
    lo = tgt.min(0) - 3 * sp
    dims = np.ceil((tgt.max(0) + 3 * sp - lo) / sp).astype(int)
    axes = [lo[i] + sp * (np.arange(dims[i]) + 0.5) for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d = np.concatenate([tree_t.query(c, 1, workers=8)[0] for c in np.array_split(g, max(1, len(g) // 2_000_000))])
    d8 = tree_t.query(tgt, 9, workers=8)[0][:, 8]
    occ = ndimage.binary_fill_holes((d <= 0.75 * np.median(d8)).reshape(dims))
    D = ndimage.distance_transform_edt(occ)
    thick = np.zeros(dims)
    for r in range(1, int(D.max()) + 1):
        core = D >= r
        if not core.any():
            break
        thick[(ndimage.distance_transform_edt(~core) <= r) & occ] = 2.0 * r * sp
    ijk = np.clip(((tgt - lo) / sp).astype(int), 0, dims - 1)
    h = thick[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    return np.where(h > 0, h, 2 * sp)


h = local_thickness(tgt, sp) / cell
dn = tree_t.query(tgt, 41, workers=8)[0]
c = (dn[:, 1:] < 2.0 * sp).sum(1)
outer = c < 0.6 * np.median(c)
sel = outer & (h < 2.0)
P = tgt[sel]
thick = h[sel]


def dist(x):
    return cKDTree(np.asarray(x, np.float64)).query(P, 1, workers=8)[0] / sp


xl = np.asarray(zl["frames"][-1])
zp = np.load(ppath, allow_pickle=True)
xp = np.asarray(zp["frames"][-1])
dl, dp = dist(xl), dist(xp)
ul, up = dl > 1.5, dp > 1.5
res = dict(spacing=sp, cell=cell, thin_n=int(sel.sum()), outer_n=int(outer.sum()),
           share_L=float(ul.mean()), share_P=float(up.mean()), L_only=int((ul & ~up).sum()), P_only=int((up & ~ul).sum()),
           both=int((ul & up).sum()),
           band_14_16_L=float(((dl > 1.4) & (dl <= 1.6)).mean()), band_14_16_P=float(((dp > 1.4) & (dp <= 1.6)).mean()),
           beyond_2_L=float((dl > 2).mean()), beyond_2_P=float((dp > 2).mean()),
           q_all_L=[float(v) for v in np.quantile(dl, [.5, .9, .99])], q_all_P=[float(v) for v in np.quantile(dp, [.5, .9, .99])],
           q_unc_L=[float(v) for v in np.quantile(dl[ul], [.5, .9, 1.])] if ul.any() else None,
           q_unc_P=[float(v) for v in np.quantile(dp[up], [.5, .9, 1.])] if up.any() else None,
           by_thickness={f"{a}-{b}": dict(n=int(((thick >= a) & (thick < b)).sum()),
                                          L=float(ul[(thick >= a) & (thick < b)].mean()) if ((thick >= a) & (thick < b)).any() else None,
                                          P=float(up[(thick >= a) & (thick < b)].mean()) if ((thick >= a) & (thick < b)).any() else None)
                         for a, b in ((0, 1), (1, 2))})
# regions of the uncovered points (either arm)
U = np.where(ul | up)[0]
lo, hi = tgt.min(0), tgt.max(0)
ctr = tgt.mean(0)
regions = []
if len(U):
    pairs = cKDTree(P[U]).query_pairs(3 * sp, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(U), len(U)))
    nc, lab = connected_components(g, directed=False)
    for k in np.argsort(-np.bincount(lab))[:10]:
        m = U[lab == k]
        cen = P[m].mean(0)
        regions.append(dict(size=int(len(m)), L=int(ul[m].sum()), P=int(up[m].sum()),
                            centroid=[round(float(v), 3) for v in cen],
                            height=round(float((cen[1] - lo[1]) / (hi[1] - lo[1])), 3),
                            azimuth_deg=round(float(math.degrees(math.atan2(cen[0] - ctr[0], cen[2] - ctr[2]))), 1),
                            thickness_cells=round(float(np.median(thick[m])), 2)))
res["regions"] = regions


def course(frames_path, label, end_unc):
    z = np.load(frames_path, allow_pickle=True)
    fr, raws = z["frames"], np.asarray(z["raws"])
    ks = sorted(set(list(range(0, len(fr), 4)) + [len(fr) - 1]))
    rows, since = [], np.full(len(P), -1.0)
    for k in ks:
        d = dist(fr[k])
        u = d > 1.5
        rows.append([round(raws[k] / 240.0, 3), round(float(u.mean()), 4)] + [round(float(v), 3) for v in np.quantile(d, [.5, .9])])
        since = np.where(u, np.where(since < 0, raws[k] / 240.0, since), -1.0)
    s = since[end_unc]
    res[f"course_{label}"] = rows
    if end_unc.any():
        res[f"uncovered_since_{label}_s"] = [float(v) for v in np.quantile(s[s >= 0], [.1, .5, .9])] if (s >= 0).any() else None


course(lpath, "L", ul)
if pframes:
    course(pframes, "P", up)
json.dump(res, open(out + ".json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if not k.startswith("course")}, indent=1))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

col = np.where(ul & up, 3, np.where(ul, 1, np.where(up, 2, 0)))
cmap = np.array([[.75, .75, .75], [.85, .15, .1], [.15, .35, .9], [.55, .1, .6]])
fig, axs = plt.subplots(1, 2, figsize=(14, 7))
for ax, az in zip(axs, (180, 90)):
    a = math.radians(az)
    right = np.array([math.cos(a), 0, -math.sin(a)])      # camera at (sin az, 0, cos az): image right, true handedness
    depth = np.array([math.sin(a), 0, math.cos(a)])
    o = np.argsort(P @ depth)                              # far first
    ax.scatter((P @ right)[o], P[o, 1], s=2, c=cmap[col[o]])
    ax.set_aspect("equal"); ax.set_title(f"thin points, camera azimuth {az} (grey: covered in both)")
    ax.set_xticks([]); ax.set_yticks([])
fig.legend(handles=[plt.Line2D([], [], ls="", marker="o", color=cmap[i], label=t) for i, t in
                    enumerate(("covered in both", "uncovered with the render loss only", "uncovered without it only",
                               "uncovered in both"))], loc="lower center", ncol=4)
fig.savefig(out + ".png", dpi=110, bbox_inches="tight")
print("wrote", out + ".png")
