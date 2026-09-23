"""Outer-layer plane-residual RMS over an archive (mesh-free bump amplitude, spacings):
the acid-test number of the layer relaxation (docs/surface_gradient.md §6).

usage: layer_rms.py <npz> [stride] [--target]
Prints per sampled frame the RMS of the same-side plane residual over the outer layer
(surface_recon.layer_by_asymmetry + plane_residual), the mean over the morph, the end frame
and, with --target, the target cloud's own value (the floor of the sampling).
"""
import sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", ".."))
from physmorph.render.surface_recon import layer_by_asymmetry, plane_residual  # noqa: E402


def rms_of(x, sp):
    m, n = layer_by_asymmetry(x, sp)
    r, _ = plane_residual(x, m, n, sp)
    return float(np.sqrt((r ** 2).mean())) if len(r) else float("nan"), float(m.mean())


def main():
    path = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else 20
    z = np.load(path)
    fr = z["frames"]; dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(fr)
    x0 = np.asarray(fr[0], np.float32)
    sub = x0[np.random.default_rng(0).choice(len(x0), min(len(x0), 20000), replace=False)]
    sp = float(np.median(cKDTree(sub).query(sub, k=9, workers=-1)[0][:, -1])) * (min(len(x0), 20000) / len(x0)) ** (1 / 3)
    idx = list(range(0, dn, stride))
    if idx[-1] != dn - 1:
        idx.append(dn - 1)
    vals = []
    for i in idx:
        r, lf = rms_of(np.asarray(fr[i], np.float32), sp)
        vals.append(r)
        print(f"frame {i:5d}: rms {r:.3f} sp  (layer {lf * 100:.1f} %)")
    print(f"mean over the morph {np.nanmean(vals):.3f}  end {vals[-1]:.3f}  spacing {sp:.4f} wu  frames {dn}")
    if "--target" in sys.argv:
        r, lf = rms_of(np.asarray(z["tgt"], np.float32), sp)
        print(f"target cloud: rms {r:.3f} sp (layer {lf * 100:.1f} %)")


if __name__ == "__main__":
    main()
