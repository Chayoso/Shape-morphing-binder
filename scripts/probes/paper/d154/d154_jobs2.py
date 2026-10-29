"""d154_jobs2.py OUT_PREFIX K MODE MESH SEED LABEL FRAMES12 -- D154 (2)'s reader jobs for one run's kept frames, split
over K files OUT_PREFIX_k.txt (appended). MODE yard: every kept frame from 4 to 12 s, every second outside, the last;
MODE exact: every 0.5 s from 5 to 12 s (raw a multiple of 120), every 2 s elsewhere, and the last (amended 17:10 from every
0.5 s over the run: CPU time)."""
import sys

import numpy as np

pre, K, mode, mesh, seed, label, path = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
raws = [int(r) for r in np.load(path)["raws"]]
if mode == "yard":
    pick = [r for r in raws if 960 <= r <= 2880 or r % 240 == 0 or r == raws[-1]]
else:
    pick = [r for r in raws if (r % 120 == 0 and 1200 <= r <= 2880) or r % 480 == 0 or r == raws[-1]]
for k in range(K):
    with open(f"{pre}_{k}.txt", "a") as f:
        for r in pick[k::K]:
            f.write(f"{mesh} {seed} {label} {path} {r}\n")
print(mode, label, len(pick))
