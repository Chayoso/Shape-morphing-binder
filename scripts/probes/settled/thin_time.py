"""thin_time.py LABEL=ARCHIVE.npz ... — are thin features never reached, or reached and then lost?

Per window of the delivered morph, the uncovered share of the outer target (farther than 1.5 target spacings from
the body) in each thickness bin (MPM cells, as thin_regions.py), at the driven end (step T, controls on) and at the
released end (step 2T, the settled state the objective scores). Then, per bin: the share of the target points
uncovered at the end that were covered at some released end before (reached and lost), and the share covered at a
driven end but uncovered at the same window's released end, averaged over windows (lost in the release).
"""
import physmorph  # noqa: F401  (before torch: CuPy's CUDA 12 NVRTC)
import json
import sys

import numpy as np
import torch

from physmorph import gpu
from physmorph.mpm.state import MPMParams

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from thin_regions import BINS, local_thickness, npz_member, outer_mask  # noqa: E402


def covered(frames, i, tgt, thr):
    x = gpu.tensor(np.asarray(frames[i], np.float32))
    return gpu.KNN(x).query(tgt, 1)[0][:, 0] <= thr


def analyse(label, path):
    js = json.load(open(path.replace("_render_full_dt_iso_nn.npz", ".json")))
    arm = js["arms"]["render_full_dt_iso_nn"]
    prm = MPMParams(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in js["provenance"]["mpm"].items()
                       if k in MPMParams.__dataclass_fields__})
    T = int(arm["config"]["T"])
    z = np.load(path, allow_pickle=True)
    frames = npz_member(path, "frames")
    dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(frames)
    tgt = gpu.tensor(np.asarray(z["tgt"], np.float32))
    sp = gpu.median(gpu.knn(tgt, 2)[0][:, 1])
    h = local_thickness(tgt, sp) / prm.dx
    out_t = outer_mask(tgt, sp)
    thr = 1.5 * sp
    n_win = (dn - 1) // (2 * T)
    bins = [(h >= lo) & (h < hi) & out_t for lo, hi in zip(BINS[:-1], BINS[1:])]
    drv, rel = [], []
    for k in range(n_win):
        drv.append(covered(frames, k * 2 * T + T, tgt, thr))
        rel.append(covered(frames, (k + 1) * 2 * T, tgt, thr))
    end = rel[-1]
    ever = torch.stack(rel[:-1]).any(0) if n_win > 1 else torch.zeros_like(end)
    print(f"\n== {label}: {n_win} windows (T = {T}), outer target points per bin "
          f"{[int(b.sum()) for b in bins]}")
    print("win  uncovered at the driven end (<1 / 1-2 / 2-4 / >=4)   uncovered at the released end")
    f = lambda c, b: 100.0 * float((~c[b]).float().mean())  # noqa: E731
    for k in range(n_win):
        print(f"{k + 1:3d}  " + " / ".join(f"{f(drv[k], b):5.1f}" for b in bins) + "      "
              + " / ".join(f"{f(rel[k], b):5.1f}" for b in bins))
    lost_end = [float((ever[b] & ~end[b]).float().sum() / (~end[b]).float().sum().clamp_min(1)) for b in bins]
    lost_rel = [float(np.mean([float((drv[k][b] & ~rel[k][b]).float().sum() / drv[k][b].float().sum().clamp_min(1))
                               for k in range(n_win)])) for b in bins]
    print("of the points uncovered at the end, reached at an earlier released end: "
          + " / ".join(f"{100 * v:.1f} %" for v in lost_end))
    print("covered at the driven end, uncovered after the release (mean over windows): "
          + " / ".join(f"{100 * v:.1f} %" for v in lost_rel))
    return dict(label=label, lost_end=lost_end, lost_release=lost_rel,
                driven=[[f(d, b) for b in bins] for d in drv], released=[[f(r, b) for b in bins] for r in rel])


if __name__ == "__main__":
    res = [analyse(*a.split("=", 1)) for a in sys.argv[1:] if not a.startswith("--")]
    out = [a for a in sys.argv[1:] if a.startswith("--json=")]
    if out:
        json.dump(res, open(out[0].split("=", 1)[1], "w"), indent=1)
