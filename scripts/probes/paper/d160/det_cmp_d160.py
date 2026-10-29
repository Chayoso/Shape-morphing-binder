"""det_cmp_d160.py MODE A_prefix B_prefix -- D160's bit-for-bit comparisons of two deterministic runs (tmp/rt/det_run.py),
as tmp/d145/det_cmp12.py, with D160's own records (src_albedo, tgt_albedo, appearance in the frames12 archive) left out.
MODE full: every frames12 array, every metric, every history field but the clocks (det_cmp12's reading);
MODE traj: the trajectory only (the kept frames, raws, deliver_n, tgt and every metric), for two runs whose render
records differ by construction (a lambda = 0 twin with and without a paint: its render terms are recorded, not used).
Prints the differences and SAME / DIFFERENT."""
import json
import sys

import numpy as np

CLOCK = ("t_", "seconds", "prof", "wall", "time")
D160 = ("src_albedo", "tgt_albedo", "appearance")
mode, pa, pb = sys.argv[1:4]
bad = 0
za, zb = (np.load(p + "_frames12.npz", allow_pickle=True) for p in (pa, pb))
fa, fb = (set(z.files) - set(D160) for z in (za, zb))
for k in sorted(fa | fb):
    if k not in fa or k not in fb:
        print("only in one:", k)
        bad += 1
    elif not (za[k].shape == zb[k].shape and np.array_equal(za[k], zb[k])):
        print("array differs:", k, za[k].shape, zb[k].shape)
        bad += 1
ja, jb = (list(json.load(open(p + ".json"))["arms"].values())[0] for p in (pa, pb))
for k in sorted(set(ja["metrics"]) | set(jb["metrics"])):
    if ja["metrics"].get(k) != jb["metrics"].get(k):
        print("metric differs:", k, ja["metrics"].get(k), jb["metrics"].get(k))
        bad += 1
n = 0
if ja.get("deliver_n") != jb.get("deliver_n"):
    print("deliver_n", ja.get("deliver_n"), jb.get("deliver_n"))
    bad += 1
if mode == "full":
    ha, hb = ja["history"], jb["history"]
    if len(ha) != len(hb):
        print("history rows", len(ha), len(hb))
        bad += 1
    for ra, rb in zip(ha, hb):
        for k in sorted(set(ra) & set(rb)):
            if k.startswith(CLOCK) or k in CLOCK:
                continue
            n += 1
            if json.dumps(ra[k], sort_keys=True) != json.dumps(rb[k], sort_keys=True):
                print("history differs:", k, ra[k], rb[k])
                bad += 1
print(f"[{mode}] frames12 arrays {len(fa)}, metrics {len(ja['metrics'])}, history fields compared {n}:",
      "SAME" if bad == 0 else f"DIFFERENT ({bad})")
