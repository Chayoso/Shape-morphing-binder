"""dump_read.py DUMP_DIR KF.npz OUT.md -- D175 forensic (the next measurement of the 208 reading): where the render's pull
lands inside a keyframe segment. Per committed segment window (window/telemetry.write_term_dump: each term's position
gradient at the committed state, g_rend = lambda x d render / dx on the keyframe pictures):
  - the share of sum |g_rend|^2 on the particles by depth below the body's convex hull, in pitches (0-1, 1-3, > 3);
  - the cosine, over all particles and over the interior (> 3 pitches), between the render's descent -g_rend and the bulk
    stretch the keyframe asks for, (A - I)(x - c) (A the keyframe's map, c the source centroid);
  - the body's second-moment aspect sqrt(var_y / mean(var_x, var_z)) (the keyframe's: its own sample's).
CPU (numpy / scipy)."""
import glob
import sys

import numpy as np
from scipy.spatial import ConvexHull, cKDTree


def depth(x):
    h = ConvexHull(x)
    eq = h.equations                                      # n . x + d <= 0 inside, n unit
    return np.min(-(x @ eq[:, :3].T + eq[:, 3]), axis=1)


def aspect(x):
    v = ((x - x.mean(0)) ** 2).mean(0)
    return float(np.sqrt(v[1] / (0.5 * (v[0] + v[2]))))


def cos(a, b):
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def main():
    d, kf, out = sys.argv[1], np.load(sys.argv[2]), sys.argv[3]
    A, c = np.asarray(kf["A"]), np.asarray(kf["centre"])
    L = [f"# D175 forensic: where the render's pull lands in the segment ({d})", "",
         f"Keyframe aspect {aspect(np.asarray(kf['samples'][0], np.float64)):.3f}.", "",
         "| window | body aspect | share of |g_rend|^2 at depth 0-1 / 1-3 / > 3 pitches | particles 0-1 / 1-3 / > 3 | "
         "cos(-g_rend, stretch) all / > 3 pitches | |g_rend| rms all / > 3 pitches |", "|---|---|---|---|---|---|"]
    for f in sorted(glob.glob(f"{d}/terms_*.npz")):
        z = np.load(f)
        x, g = np.asarray(z["x"], np.float64), np.asarray(z["g_rend"], np.float64)
        a = 0.708 * float(np.median(cKDTree(x).query(x, 9)[0][:, 8]))
        dp = depth(x) / a
        bins = [dp < 1, (dp >= 1) & (dp < 3), dp >= 3]
        e = (g ** 2).sum(1)
        sh = [float(e[b].sum() / max(e.sum(), 1e-30)) for b in bins]
        st = (x - c) @ (A - np.eye(3)).T
        inn = bins[2]
        L.append(f"| {f[-8:-4]} | {aspect(x):.3f} | " + " / ".join(f"{s:.3f}" for s in sh) + " | "
                 + " / ".join(str(int(b.sum())) for b in bins) + f" | {cos(-g, st):+.3f} / {cos(-g[inn], st[inn]):+.3f} | "
                 f"{np.sqrt(e.mean()):.3e} / {np.sqrt(e[inn].mean()) if inn.any() else 0:.3e} |")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
