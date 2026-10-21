"""shapetilt.py NPZ [NPZ ...] -- B9b (CPU): the shape's tilt (tobody.shape_rotation: the rotation aligning the body's
shape to its target's, from the Buckens rotation) against the Buckens (material) frame's tilt, at five kept frames of
an archive: a scene run's (frames, src0, tgt0) or a single-body run's frames12 (frames, tgt; the source = its first kept
frame). The check of the upright item's measure before it is used: an upright body reads below 3 deg whatever its
material does, a turned one its turn."""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tobody import angle, polar, shape_rotation  # noqa: E402

for path in sys.argv[1:]:
    z = np.load(path, allow_pickle=True)
    fr = z["frames"]
    X = np.asarray(z["src0"] if "src0" in z.files else fr[0], np.float64)
    X = X - X.mean(0)
    tgt = np.asarray(z["tgt0"] if "tgt0" in z.files else z["tgt"], np.float64)
    tc = tgt - tgt.mean(0)
    tree = cKDTree(tc)
    n = len(fr)
    out = []
    for k in sorted({0, n // 4, n // 2, 3 * n // 4, n - 1}):
        x = np.asarray(fr[k], np.float64)
        xc = x - x.mean(0)
        R = polar(xc.T @ X)
        Rs = shape_rotation(xc, tc, tree, R)
        out.append(f"k {k}: material tilt {np.degrees(np.arccos(np.clip(R[1, 1], -1, 1))):6.2f} turn {angle(R):6.2f} | "
                   f"shape tilt {np.degrees(np.arccos(np.clip(Rs[1, 1], -1, 1))):6.2f} turn {angle(Rs):6.2f}")
    print(f"== {Path(path).name}:")
    print("\n".join("   " + s for s in out), flush=True)
