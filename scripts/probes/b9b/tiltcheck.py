"""tiltcheck.py NPZ [NPZ ...] -- B9b (CPU): the shape tilt's measure checked at an archive's last kept frame.
(a) tobody.shape_rotation from three starts (identity = the target's own orientation; the Buckens rotation; the
    Buckens rotation then 400 iterations), each to its end, with its symmetric residual (the mean nearest distance both
    ways, pitches of the target sample); the lowest residual is the reading;
(b) a start-free check: the target turned by every tilt on a 0.25 deg grid to 8 deg about the two horizontal axes, the
    body about its centroid, the symmetric chamfer at each; the tilt of the best.
A scene archive (frames, src0, tgt0) or a frames12 (frames, tgt; the source its first kept frame)."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tobody import polar, shape_rotation  # noqa: E402

def resid(xs, R, t, tc, tree_t):
    y = xs @ R + t
    d1 = tree_t.query(y)[0]
    d2 = cKDTree(y).query(tc[::2])[0]
    return 0.5 * (d1.mean() + d2.mean())

def tilt(R):
    return float(np.degrees(np.arccos(np.clip(R[1, 1], -1, 1))))

def rot(ax, az):
    a, b = np.radians(ax), np.radians(az)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Rz = np.array([[np.cos(b), -np.sin(b), 0], [np.sin(b), np.cos(b), 0], [0, 0, 1]])
    return Rx @ Rz

for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    fr = z["frames"]
    X = np.asarray(z["src0"] if "src0" in z.files else fr[0], np.float64); X -= X.mean(0)
    tgt = np.asarray(z["tgt0"] if "tgt0" in z.files else z["tgt"], np.float64); tc = tgt - tgt.mean(0)
    pitch = 0.708 * float(np.median(cKDTree(tc).query(tc[::50], k=9)[0][:, 8]))
    tree_t = cKDTree(tc)
    x = np.asarray(fr[-1], np.float64); xc = x - x.mean(0)
    rng = np.random.default_rng(1)
    xs = xc[rng.choice(len(xc), min(20000, len(xc)), replace=False)]
    Rb = polar(xc.T @ X)
    out = []
    for name, R0, it in (("identity", np.eye(3), 40), ("Buckens", Rb, 40), ("Buckens x400", Rb, 400), ("identity x400", np.eye(3), 400)):
        R = shape_rotation(xc, tc, tree_t, R0, iters=it)
        # the shift that goes with R (one more pairing)
        y = xs @ R
        t = tc[tree_t.query(y)[1]].mean(0) - y.mean(0)
        out.append((name, tilt(R), resid(xs, R, t, tc, tree_t) / pitch))
    best = min(out, key=lambda r: r[2])
    # (b) the grid: the target turned by R_g (target frame), compare the body to it: equivalently the body turned by R_g^T
    grid = np.arange(-8.0, 8.01, 0.25)
    best_g = (None, np.inf)
    xs2 = xs[::4]
    for ax in grid:
        for az in grid:
            Rg = rot(ax, az)
            y = xs2 @ Rg
            r = 0.5 * (tree_t.query(y)[0].mean() + cKDTree(y).query(tc[::8])[0].mean())
            if r < best_g[1]:
                best_g = ((ax, az), r)
    (ax, az), rg = best_g
    gt = tilt(rot(ax, az))
    print(f"== {Path(path).name}: material tilt {tilt(Rb):.2f}; " + "; ".join(f"{n}: {tl:.2f} deg (resid {r:.4f} sp)" for n, tl, r in out)
          + f" | reading {best[1]:.2f} deg ({best[0]}) | grid best tilt {gt:.2f} deg (about x {ax:+.2f}, z {az:+.2f}; resid {rg / pitch:.4f} sp)", flush=True)
