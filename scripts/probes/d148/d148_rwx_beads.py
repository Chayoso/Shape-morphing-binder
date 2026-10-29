"""d148_rwx_beads.py FRAMES12 DUMP_DIR K:X0,Y0,X1,Y1[:full] [...] -- D148's forensic of beads that strand with consistent
sides (the coordinator's list, 2026-10-14): the bead's particles as d148_bead2.py selects them on the side render
(azimuth 90) at kept frame K; per window from six before its stranding window to it, from D146's per-particle dump
(w_AAAA.npz, windows from D146_A0): whether they lie in the outer layer, dF_c's rms and cap share, u (times its gate),
the normal path by kind (the grid's advection and the jumps below it, driven and released halves, in spacings), the
speed at the window's start, the driven end and the end, and each term's pull (-gradient) along the outward direction
(from the mean of the particle's 16 nearest at the window's start to the particle), against the median of its 64
nearest neighbours (the body around it)."""
import math
import sys

import numpy as np
from scipy.spatial import cKDTree

fr_path, ddir = sys.argv[1], sys.argv[2]
z = np.load(fr_path)
fr, raws = z["frames"], np.asarray(z["raws"])
tgt = np.asarray(z["tgt"], np.float64)
a = 0.708 * float(np.median(cKDTree(tgt).query(tgt[::10], 9)[0][:, 8]))
center = tgt.mean(0)
radius = float(np.linalg.norm(tgt - center, axis=1).max())
W, H, AZ, EL = 3840, 2160, math.radians(90.0), math.radians(18.0)
cam = center + 3.6 * radius * np.array([math.cos(EL) * math.sin(AZ), math.sin(EL), math.cos(EL) * math.cos(AZ)])
fwd = (center - cam) / np.linalg.norm(center - cam)
right = np.cross([0.0, 1.0, 0.0], fwd)
right /= np.linalg.norm(right)
up = np.cross(fwd, right)
tan_y = math.tan(math.radians(15.0))
tan_x = tan_y * W / H


def pixels(p):
    d = p - cam
    xc, yc, zc = d @ right, d @ up, d @ fwd
    return np.stack((((xc / zc) / tan_x + 1) / 2 * W - 0.5, (1 - (yc / zc) / tan_y) / 2 * H - 0.5), 1)


def apart(x):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    pairs = cKDTree(x).query_pairs(1.5 * a, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs), np.int8), (pairs[:, 0], pairs[:, 1])), shape=(len(x), len(x)))
    _, cl = connected_components(g, directed=False)
    return cl != np.bincount(cl).argmax()


TERMS = ("ot", "surf", "near", "spray", "rend")
for spec in sys.argv[3:]:
    p = spec.split(":")
    K, box, full = int(p[0]), [int(v) for v in p[1].split(",")], len(p) > 2 and p[2] == "full"
    x0b, y0b, x1b, y1b = box
    wq, hq = (0, 0) if full else ((x1b - x0b) / 4.0, (y1b - y0b) / 4.0)
    q = pixels(np.asarray(fr[K], np.float64))
    sel = np.nonzero((q[:, 0] >= x0b + wq) & (q[:, 0] <= x1b - wq) & (q[:, 1] >= y0b + hq) & (q[:, 1] <= y1b - hq))[0]
    ap = apart(np.asarray(fr[K], np.float64))
    sel = sel[ap[sel]] if ap[sel].any() else sel
    wk = int(raws[K] // 40)
    print(f"\n== bead at frame {K} ({raws[K] / 240:.2f} s, window {wk + 1}), box {box}: particles {sel.tolist()[:12]}"
          f"{' ...' if len(sel) > 12 else ''} ({len(sel)}, apart at K: {int(ap[sel].sum())})", flush=True)
    if not len(sel):
        continue
    for w in range(max(1, wk - 5), wk + 2):
        try:
            d = np.load(f"{ddir}/w_{w:04d}.npz")
        except OSError:
            continue
        if "x0" not in d.files:
            continue
        x0 = d["x0"].astype(np.float64)
        tree = cKDTree(x0)
        _, nb16 = tree.query(x0[sel], 17)
        _, nb64 = tree.query(x0[sel], 65)
        out = x0[sel] - x0[nb16[:, 1:]].mean(1)
        out /= np.linalg.norm(out, axis=1)[:, None] + 1e-12
        G = {k: d["g_" + k].astype(np.float64) * float(d["s_" + k]) for k in TERMS if "g_" + k in d.files}
        T = max(int(k[1:]) for k in d.files if k.startswith("d") and k[1:].isdigit())
        Tc = T // 2
        lines = []
        for i, pidx in enumerate(sel):
            nbr = nb64[i, 1:]
            f = lambda key: float(d[key][pidx])                                  # noqa: E731
            med = lambda key: float(np.median(d[key][nbr].astype(np.float64)))  # noqa: E731
            pulls = []
            for k, g in G.items():
                pp = float(-g[pidx] @ out[i])
                pn = float(np.median(-(g[nbr] @ out[i])))
                pulls.append(f"{k} {pp:+.2e} (body {pn:+.2e})")
            u = f("u") * f("ug")
            disp = float(np.linalg.norm(d[f"d{T:02d}"][pidx].astype(np.float64))) / a
            dout = float(d[f"d{T:02d}"][pidx].astype(np.float64) @ out[i]) / a
            lines.append(
                f"p{pidx}: layer {int(f('lmask') > 0.5)}; dF_c rms {f('dfc_rms'):.3f} (body {med('dfc_rms'):.3f}), cap "
                f"share {f('dfc_capfrac') if 'dfc_capfrac' in d.files else float('nan'):.2f}; u x gate {u:+.3f} sp; "
                f"normal path adv drv {f('n_adv_drv'):+.2f} rel {f('n_adv_rel'):+.2f}, jumps drv {f('n_jmp_drv'):+.2f} rel "
                f"{f('n_jmp_rel'):+.2f} sp; speed start {f('spd00'):.3f} driven end {f(f'spd{Tc:02d}'):.3f} end "
                f"{f(f'spd{T:02d}'):.3f} (body {med(f'spd{T:02d}'):.3f}); moved {disp:.2f} pitches, outward {dout:+.2f}; "
                f"pulls outward: " + ", ".join(pulls))
        print(f"  window {w} (from {(w - 1) / 6:.2f} s):\n    " + "\n    ".join(lines[:6]), flush=True)
