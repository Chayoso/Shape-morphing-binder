"""d155b_steps.py -- D155b: the step after the arrival (records): the window's last step size, the merit gain, the
accepted / rejected line-search trials, over 0-1 s and 1-3 s after the arrival, the tag (seeds 97 / 99 / 101) and A3G."""
import json

import numpy as np

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"


def arrival(a):
    acc = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit")]
    t = next((h["frame_end"] / 240 for h in acc if (h.get("u_gate") or 0) >= 0.5), 0.0)
    if any("morph_s" in h for h in acc):
        t = max(t, next(h["frame_end"] / 240 for h in acc if (h.get("morph_s") or 0) >= 1.0))
    return t, acc


for m in ("bunny", "dragon", "teapot", "armadilo", "homer"):
    print(m)
    for arm in ("V97", "V99", "V101", "A3GL97", "W97", "A3GP97"):
        a = next(iter(json.load(open(f"{O}/{m}40k_{arm}.json"))["arms"].values()))
        t0, acc = arrival(a)

        def mean(lo, hi, k):
            v = [h.get(k) or 0 for h in acc if lo < h["frame_end"] / 240 - t0 <= hi]
            return float(np.mean(v)) if v else float("nan")
        print(f"  {arm:7s} alpha 0-1 s {mean(-0.01, 1, 'alpha_last'):.4f}, 1-3 s {mean(1, 3, 'alpha_last'):.4f}; gain "
              f"0-1 s {mean(-0.01, 1, 'outer_gain'):.3f}, 1-3 s {mean(1, 3, 'outer_gain'):.3f}; accepted / rejected "
              f"trials 0-3 s {mean(-0.01, 3, 'accepted'):.1f} / {mean(-0.01, 3, 'rejected'):.1f}")
