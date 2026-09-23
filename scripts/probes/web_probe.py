"""Ear-web probe: is the mass between the ear and the head real (raw state), transient, or a
projection artefact? For a mid frame and the END frame: particles farther than 1/2/3 target
spacings from ANY target point (over-mass), clustered (radius 1.5 sp) and located relative to
the thin-feature (ear) target points; target points with no particle within 1 sp (under-mass).
Renders target / mid / end at four views with the over-mass particles in red.
Usage: python web_probe.py OUT_DIR run [mid_fraction]"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix

OUT, name = sys.argv[1], sys.argv[2]
MID = float(sys.argv[3]) if len(sys.argv) > 3 else 0.69
z = np.load(os.path.join(OUT, name + ".npz"))
fr, tgt, src = z["frames"], z["tgt"], z["src"]
dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(fr)
fr = fr[:dn]
ttree = cKDTree(tgt)
sp = float(np.median(ttree.query(tgt, k=2, workers=-1)[0][:, 1]))


def thin_mask(t, radius=0.9, q=0.22):
    cnt = np.array([len(c) for c in cKDTree(t).query_ball_point(t, radius, workers=-1)])
    r = np.linalg.norm(t - t.mean(0), axis=1)
    return (cnt <= np.quantile(cnt, q)) & (r > np.quantile(r, 0.5))


thin = tgt[thin_mask(tgt)]
thin_tree = cKDTree(thin)
print(f"{name}: frames {dn}, target spacing {sp:.4f} wu, ear/thin target points {len(thin)} ({len(thin)/len(tgt):.1%})")

views = [("az 0.6", 0.6, 0.18), ("az 1.4", 1.4, 0.18), ("az 2.2", 2.2, 0.18), ("top (el 1.3)", 0.6, 1.3)]


def proj(x, az, el):
    right = np.array([np.cos(az), 0, -np.sin(az)])
    up = np.array([-np.sin(el) * np.sin(az), np.cos(el), -np.sin(el) * np.cos(az)])
    return x @ right, x @ up


rows = []
for label, i in [("mid", int(MID * (dn - 1))), ("end", dn - 1)]:
    x = fr[i]
    d_t = ttree.query(x, k=1, workers=-1)[0]
    far1, far2, far3 = d_t > sp, d_t > 2 * sp, d_t > 3 * sp
    # under-mass: target points with no particle within 1 sp
    d_rev = cKDTree(x).query(tgt, k=1, workers=-1)[0]
    unc = d_rev > sp
    unc_thin = (cKDTree(x).query(thin, k=1, workers=-1)[0] > sp)
    line = (f"  {label} (frame {i}): over-mass >1sp {far1.mean():.3%}  >2sp {far2.mean():.3%}  >3sp {far3.mean():.3%} | "
            f"uncovered target >1sp {unc.mean():.2%} (thin/ear {unc_thin.mean():.2%})")
    print(line)
    # clusters of the >1sp particles
    if far1.sum() > 10:
        xf = x[far1]
        pairs = cKDTree(xf).query_pairs(1.5 * sp, output_type="ndarray")
        n = len(xf)
        A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _, lab = connected_components(A, directed=False)
        sizes = np.bincount(lab)
        order = np.argsort(sizes)[::-1][:4]
        for c in order:
            m = lab == c
            cen = xf[m].mean(0)
            d_ear = float(thin_tree.query(cen)[0]); d_any = float(ttree.query(cen)[0])
            print(f"     cluster {sizes[c]:5d} particles ({sizes[c]/len(x):.2%})  centroid {np.round(cen, 2)}  "
                  f"dist to ear target {d_ear:.2f} wu, to any target {d_any:.2f} wu, mean d_t {d_t[far1][m].mean()/sp:.1f} sp")
    rows.append((label, i, x, far1))

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(views), figsize=(4.2 * len(views), 12))
    for j, (vl, az, el) in enumerate(views):
        u, v = proj(tgt, az, el)
        ax = axes[0, j]; ax.scatter(u, v, s=0.6, c="0.55", lw=0); ax.set_title(f"target — {vl}"); ax.set_aspect("equal"); ax.axis("off")
        ut, vt = proj(thin, az, el); ax.scatter(ut, vt, s=0.8, c="crimson", lw=0)
        for r, (label, i, x, far1) in enumerate(rows, start=1):
            ax = axes[r, j]
            u, v = proj(x, az, el)
            ax.scatter(u, v, s=0.5, c="#3b6fb0", lw=0)
            ax.scatter(u[far1], v[far1], s=2.0, c="red", lw=0)
            uo, vo = proj(tgt, az, el)
            ax.scatter(uo, vo, s=0.15, c="0.75", lw=0, zorder=0)
            ax.set_title(f"{label} frame {i} — {vl} (red: >1 sp from target)"); ax.set_aspect("equal"); ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"web_probe_{name}.png"), dpi=110)
    print("figure saved")
except Exception as e:
    print("no figure:", e)
