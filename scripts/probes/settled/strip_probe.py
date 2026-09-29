"""strip_probe.py TAG — does the density objective move only the density-jump layer?
(1) Early transport: the particles that END in the head top (y > 2.3) grouped by their DEPTH in the source (distance to
    the source cloud's free surface, in spacings); their progress p = 1 - |x(t) - x(end)| / |x(0) - x(end)| at t. If the
    shallow layer is far ahead of the deep one, the front is stripped from the bulk (the vapour).
(2) Late sliding: at the end, the tangential vs normal share of the per-frame motion of the unpinned particles near the
    surface (normal = the density gradient's direction). Motion along the surface is invisible to a density objective."""
import sys, numpy as np
from scipy.spatial import cKDTree
OUT = "/data/relcfd/chayo/physmorph_v2/output/"
tag = sys.argv[1]
z = np.load(OUT + tag + "_render_full_dt_iso_nn.npz", allow_pickle=True)
F = z["frames"]; n = len(F); X0 = np.asarray(F[0], np.float32); XE = np.asarray(F[-1], np.float32)
sp = float(np.median(cKDTree(X0).query(X0, k=2)[0][:, 1]))
# depth in the source: distance to the nearest "outer" particle (fewer than 60 % of the median 2-spacing count)
k0 = cKDTree(X0); cnt = np.asarray(k0.query_ball_point(X0, r=2.0 * sp, return_length=True))
outer = cnt < 0.6 * np.median(cnt)
depth = cKDTree(X0[outer]).query(X0, k=1)[0] / sp
sel = XE[:, 1] > 2.3
tot = np.linalg.norm(XE - X0, axis=1)
print(f"{tag}: spacing {sp:.4f}; particles ending in the head top: {sel.sum()} (source-depth median {np.median(depth[sel]):.1f} sp)")
bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
print("   progress toward the end position by SOURCE DEPTH (spacings) at t:")
hdr = "   t     " + "  ".join(f"d{lo:g}-{hi if hi < 1e8 else 'inf'}" .ljust(9) for lo, hi in bands)
print(hdr)
for t in [0.05, 0.1, 0.14, 0.2, 0.3, 0.4]:
    x = np.asarray(F[int(round(t * (n - 1)))], np.float32)
    p = 1.0 - np.linalg.norm(x - XE, axis=1) / np.maximum(tot, 1e-6)
    row = []
    for lo, hi in bands:
        m = sel & (depth >= lo) & (depth < hi) & (tot > 4 * sp)
        row.append(f"{np.median(p[m]):.2f}({m.sum():5d})" if m.any() else "   -     ")
    print(f"   {t:.2f}  " + "  ".join(r.ljust(9) for r in row))
# (2) late sliding: tangential share of the motion near the surface
pin = np.asarray(z["pinned"], bool) if "pinned" in z.files and len(z["pinned"]) == len(XE) else np.zeros(len(XE), bool)
kE = cKDTree(XE); cntE = np.asarray(kE.query_ball_point(XE, r=2.0 * sp, return_length=True))
near_surf = cntE < 0.6 * np.median(cntE)
_, jn = kE.query(XE, k=33)
cen = XE[jn].mean(1); nrm = XE - cen; nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)   # outward
i0 = int(0.9 * (n - 1)); step = 12
for grp, m in [("surface, unpinned", near_surf & ~pin), ("surface, pinned", near_surf & pin), ("interior, unpinned", ~near_surf & ~pin)]:
    if m.sum() < 10:
        print(f"   {grp}: n {m.sum()}"); continue
    dd = []
    for i in range(i0, n - step, step):
        d = np.asarray(F[i + step], np.float32)[m] - np.asarray(F[i], np.float32)[m]
        dd.append(d)
    d = np.concatenate(dd, 0); nn = np.tile(nrm[m], (len(dd), 1))
    dn = np.abs((d * nn).sum(1)); dt = np.linalg.norm(d - (d * nn).sum(1, keepdims=True) * nn, axis=1)
    print(f"   late (t 0.9-1.0) {grp:20s} n {m.sum():6d}: move/frame normal {np.median(dn) / sp:.3f} sp, tangential {np.median(dt) / sp:.3f} sp")
