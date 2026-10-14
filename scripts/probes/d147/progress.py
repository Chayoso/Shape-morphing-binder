"""progress.py RUNS_DIR TAG[,TAG...] [STEP] -- D147 (the bodies meet mid-morph): each body's morph progress, the
silhouette IoU with its final target in its Buckens frame (metrics.sil_iou on the archive's target, as the end metrics)
against its launch and end values, p(t) = (IoU(t) - IoU(0)) / (IoU(end) - IoU(0)), at the first kept frame where the
bodies come within one sample spacing of each other (pre-registered: the mean over the bodies in [0.35, 0.65]) and
every half second; with the schedule's s at those times. Read-only; run on hyde06 (CUDA for the frame and the IoU)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import physmorph  # noqa: E402,F401  (before torch)
from scipy.spatial import cKDTree  # noqa: E402

from physmorph import gpu, metrics  # noqa: E402
from physmorph.pipeline.scene.frame import body_frame  # noqa: E402
from physmorph.pipeline.scene.schedule import ramp  # noqa: E402

runs = Path(sys.argv[1])
STEP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5        # the curve's spacing, seconds
for tag in sys.argv[2].split(","):
    j = json.loads((runs / f"{tag}.json").read_text())
    z = np.load(runs / f"{tag}.npz")
    fr, rows, raws = z["frames"], z["rows"], z["raws"]
    dt, R = j["mpm"]["dt"], j["scene"]["R"]
    sp = (4.18879 * R ** 3 / (rows[0][1] - rows[0][0])) ** (1 / 3)
    t_m = float(j["scene"]["spec"].get("morph_time", 0.0))
    (s0, e0), (s1, e1) = rows
    k_c = None
    for k in range(len(fr)):
        a, b = np.asarray(fr[k, s0:e0]), np.asarray(fr[k, s1:e1])
        if (np.maximum(a.min(0) - b.max(0), b.min(0) - a.max(0)) > sp).any():
            continue                                  # the boxes more than a spacing apart
        if np.isfinite(cKDTree(b).query(a, distance_upper_bound=sp)[0]).any():
            k_c = k
            break
    tgts = [np.asarray(z[f"tgt{b}"]) for b in range(len(rows))]
    srcs = [gpu.tensor(np.asarray(z[f"src{b}"])) for b in range(len(rows))]

    def iou(k, b):
        s, e = rows[b]
        y = body_frame(gpu.tensor(np.asarray(fr[k, s:e])), srcs[b])[0].cpu().numpy()
        return metrics.sil_iou(y, tgts[b], extent=metrics.target_extent(tgts[b]))

    ends = [(iou(0, b), iou(len(fr) - 1, b)) for b in range(len(rows))]

    def prog(k):
        return [(iou(k, b) - i0) / max(i1 - i0, 1e-9) for b, (i0, i1) in enumerate(ends)]

    t_c = float(raws[k_c] * dt) if k_c is not None else None
    p_c = prog(k_c) if k_c is not None else None
    curve = []
    for k in range(0, len(fr), max(1, int(round(STEP / (dt * (raws[1] - raws[0])))))):
        curve.append((round(float(raws[k] * dt), 3), [round(p, 4) for p in prog(k)]))
    # the largest rise between neighbouring curve points per body, against the steady rate (the median rise over
    # 1 s to the morph time): a late snap shows as a jump many times the steady rate
    jumps = []
    for b in range(len(rows)):
        d = [(curve[i + 1][0], curve[i + 1][1][b] - curve[i][1][b]) for i in range(len(curve) - 1)]
        steady = float(np.median([v for t, v in d if 1.0 <= t <= max(t_m, 2.0)])) if d else 0.0
        early = [e for e in d if e[0] <= t_m + 1.0] or d          # to a second past the ramp's end (before landing)
        (t_e, v_e), (t_j, v_j) = max(early, key=lambda e: e[1]), max(d, key=lambda e: e[1])
        jumps.append({"steady": round(steady, 4), "early_t": t_e, "early_rise": round(v_e, 4),
                      "early_ratio": round(v_e / steady, 2) if steady > 0 else None,
                      "any_t": t_j, "any_rise": round(v_j, 4)})
    out = {"tag": tag, "morph_time": t_m, "t_first_contact": t_c,
           "s_at_contact": ramp(t_c, t_m) if (t_c is not None and t_m > 0) else None,
           "progress_at_contact": [round(p, 3) for p in p_c] if p_c else None,
           "progress_mean": round(float(np.mean(p_c)), 3) if p_c else None,
           "in_band": bool(p_c and 0.35 <= float(np.mean(p_c)) <= 0.65),
           "iou_launch_end": [[round(i0, 4), round(i1, 4)] for i0, i1 in ends], "jumps": jumps, "curve": curve}
    print(json.dumps(out), flush=True)
