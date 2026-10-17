"""w1_compare.py RUNS_DIR FROZEN_A FROZEN_B SCENE -- B9b: the first optimised window of the frozen single-body run (40k
bunny, no gravity, no floor; scripts/pipeline_run.py --animations 1, twice: the run-to-run spread) against the one-body
scene's (scripts/scene_run.py --windows 1, B9b's spec without gravity or floor: the Buckens frame, every window
committed, the world drag 0 and the deformation damped at 0.9 / s). The displacement fields x_end - x_start (the same
particle order), their difference against the frozen spread (RMS and largest, in sample pitches, and against the RMS
displacement), the window's merit, transport, silhouette and shading terms, lambda, g_share. CPU only."""
import json
import sys
from pathlib import Path

import numpy as np

ARM = "render_full_dt_iso_nn"


def frozen(R, tag):
    z = np.load(R / f"{tag}_{ARM}.npz", allow_pickle=True)
    x = np.asarray(z["frames"], np.float64)
    h = json.loads((R / f"{tag}.json").read_text())["arms"][ARM]["history"]
    w = next(r for r in h if r.get("outer_accepted") or r.get("accepted"))
    rec = {k: w.get(k) for k in ("loss", "selection_merit", "transport_energy", "d_sil", "d_pbr", "lambda", "g_share",
                                 "iters", "accepted", "rejected")}
    return x[0], x[min(40, len(x) - 1)], rec


def scene(R, tag):
    z = np.load(R / f"{tag}.npz")
    fr, raws = np.asarray(z["frames"], np.float64), z["raws"]
    k = int(np.nonzero(raws == 40)[0][0]) if (raws == 40).any() else len(fr) - 1
    h = json.loads((R / f"{tag}.json").read_text())["history"][0]
    b = h["bodies"][0]
    rec = {"loss": h.get("loss"), "selection_merit": h.get("selection_merit"), "transport_energy": b.get("transport"),
           "d_sil": b.get("d_sil"), "d_pbr": b.get("d_pbr"), "lambda": h.get("lambda"), "g_share": h.get("g_share"),
           "accepted": h.get("accepted"), "rejected": h.get("rejected")}
    return fr[0], fr[k], rec


def main():
    R = Path(sys.argv[1])
    a0, a1, ra = frozen(R, sys.argv[2])
    b0, b1, rb = frozen(R, sys.argv[3])
    s0, s1, rs = scene(R, sys.argv[4])
    # the pitch as momentum_probe.py reads it: 0.708 x the median 8th-neighbour distance of the source sample
    from scipy.spatial import cKDTree
    d = cKDTree(a0).query(a0[::40], k=9)[0][:, 8]
    sp = 0.708 * float(np.median(d))
    da, db, ds = a1 - a0, b1 - b0, s1 - s0
    rms = lambda u: float(np.sqrt((u * u).sum(1).mean()))                      # noqa: E731
    out = {"pitch": sp, "start_diff_max_wu": float(np.abs(a0 - s0).max()), "start_mean_diff_wu": (a0.mean(0) - s0.mean(0)).tolist(),
           "disp_rms_sp": {"frozen_a": rms(da) / sp, "frozen_b": rms(db) / sp, "scene": rms(ds) / sp},
           "frozen_a_vs_b": {"rms_sp": rms(da - db) / sp, "max_sp": float(np.linalg.norm(da - db, axis=1).max()) / sp},
           "scene_vs_frozen_a": {"rms_sp": rms(ds - da) / sp, "max_sp": float(np.linalg.norm(ds - da, axis=1).max()) / sp},
           "scene_vs_frozen_b": {"rms_sp": rms(ds - db) / sp, "max_sp": float(np.linalg.norm(ds - db, axis=1).max()) / sp},
           "cos_scene_frozen": float((ds * da).sum() / max(np.sqrt((ds * ds).sum() * (da * da).sum()), 1e-30)),
           "records": {"frozen_a": ra, "frozen_b": rb, "scene": rs}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
