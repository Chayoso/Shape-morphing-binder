"""det_cmp12.py A_prefix B_prefix -- D145: two runs compared bit for bit on what the readings use: the kept frames
(<prefix>_frames12.npz: frames, raws, tgt, deliver_n) and the JSON (every history field except clocks, every metric,
deliver_n). Prints the differences and SAME / DIFFERENT."""
import json
import sys

import numpy as np

CLOCK = ("t_", "seconds", "prof", "wall", "time")
bad = 0
za, zb = (np.load(p + "_frames12.npz", allow_pickle=True) for p in sys.argv[1:3])
for k in sorted(set(za.files) | set(zb.files)):
    if k not in za.files or k not in zb.files:
        print("only in one:", k)
        bad += 1
    elif not (za[k].shape == zb[k].shape and np.array_equal(za[k], zb[k])):
        print("array differs:", k, za[k].shape, zb[k].shape)
        bad += 1
ja, jb = (list(json.load(open(p + ".json"))["arms"].values())[0] for p in sys.argv[1:3])
for k in sorted(set(ja["metrics"]) | set(jb["metrics"])):
    if ja["metrics"].get(k) != jb["metrics"].get(k):
        print("metric differs:", k, ja["metrics"].get(k), jb["metrics"].get(k))
        bad += 1
ha, hb = ja["history"], jb["history"]
if len(ha) != len(hb) or ja.get("deliver_n") != jb.get("deliver_n"):
    print("history rows", len(ha), len(hb), "deliver_n", ja.get("deliver_n"), jb.get("deliver_n"))
    bad += 1
n = 0
for ra, rb in zip(ha, hb):
    for k in sorted(set(ra) & set(rb)):
        if k.startswith(CLOCK) or k in CLOCK:
            continue
        n += 1
        if json.dumps(ra[k], sort_keys=True) != json.dumps(rb[k], sort_keys=True):
            print("history differs:", k, ra[k], rb[k])
            bad += 1
print(f"frames12 arrays {len(za.files)}, metrics {len(ja['metrics'])}, history fields compared {n}:",
      "SAME" if bad == 0 else f"DIFFERENT ({bad})")
