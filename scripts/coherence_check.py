"""Is the morph SHAPE-PRESERVING (MPM-like) or registration-like (Chamfer-flow)?

Quantifies the user's question with three coherence metrics per saved run:

  nbr_overlap   mean fraction of each particle's kNN(16) set preserved frame0 -> final.
                Smooth material transport keeps neighbours; NN/Chamfer-style flows
                scramble them.
  disp_rough    mean_i |d_i - mean_{j in kNN(i)} d_j| / mean|d|  (total displacement field
                roughness; teleportation signature — high = particles moved independently)
  detF          distribution of det F at the final state (volume identity), when saved.

Run (hyde06):  python scripts/coherence_check.py output/v3_ab_render.npz output/v3_r3_vbd.npz ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scipy.spatial import cKDTree  # noqa: E402


def coherence(npz_path: str, k: int = 16) -> dict:
    d = np.load(npz_path)
    fr = d["frames"]
    x0, xT = np.ascontiguousarray(fr[0], np.float32), np.ascontiguousarray(fr[-1], np.float32)
    n0 = cKDTree(x0).query(x0, k=k + 1)[1][:, 1:]
    nT = cKDTree(xT).query(xT, k=k + 1)[1][:, 1:]
    overlap = np.array([len(np.intersect1d(a, b, assume_unique=True)) for a, b
                        in zip(n0, nT)], np.float32) / k
    disp = xT - x0
    dn = disp[n0].mean(1)
    rough = float(np.linalg.norm(disp - dn, axis=1).mean()
                  / max(np.linalg.norm(disp, axis=1).mean(), 1e-9))
    out = {"file": npz_path, "nbr_overlap_mean": float(overlap.mean()),
           "nbr_overlap_p10": float(np.percentile(overlap, 10)),
           "disp_rough": rough,
           "move_total": float(np.linalg.norm(disp, axis=1).mean())}
    if "F_samples" in d and len(d["F_samples"]):
        F = d["F_samples"][-1].reshape(-1, 3, 3)
        det = np.linalg.det(F)
        out.update({"detF_p05": float(np.percentile(det, 5)),
                    "detF_p50": float(np.percentile(det, 50)),
                    "detF_p95": float(np.percentile(det, 95))})
    return out


def main():
    rows = [coherence(p) for p in sys.argv[1:]]
    for r in rows:
        print(f"{Path(r['file']).stem:28s} nbr_overlap={r['nbr_overlap_mean']:.3f} "
              f"(p10 {r['nbr_overlap_p10']:.3f})  disp_rough={r['disp_rough']:.3f}  "
              f"move={r['move_total']:.3f}"
              + (f"  detF p05/p50/p95={r['detF_p05']:.2f}/{r['detF_p50']:.2f}/{r['detF_p95']:.2f}"
                 if "detF_p50" in r else ""), flush=True)
    Path("output").mkdir(exist_ok=True)
    Path("output/coherence.json").write_text(json.dumps(rows))


if __name__ == "__main__":
    main()
