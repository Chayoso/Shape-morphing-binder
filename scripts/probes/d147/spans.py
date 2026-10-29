"""spans.py RUNS_DIR TAG TIMELINE_TXT CUT_TXT -- D147: the kept frames the renders draw, from the readings: the cut
(cut.py's amended rule, its kept frame) and the contact close-up's span (the coordinator's rule, 2026-10-15: the first
contact - 0.5 s to the first parting + 0.5 s, from read40k.py's timeline; on the interim s97 kept frames 67-136).
Prints "CUT K0 K1" (kept-frame indices of the archive). CPU."""
import json
import sys
from pathlib import Path

import numpy as np


def line(path, tag):
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith("{") and f'"tag": "{tag}"' in ln:
            try:
                return json.loads(ln)
            except json.JSONDecodeError:
                continue
    raise SystemExit(f"no line for {tag} in {path}")


runs, tag = Path(sys.argv[1]), sys.argv[2]
tl, cut = line(sys.argv[3], tag), line(sys.argv[4], tag)
dt = json.loads((runs / f"{tag}.json").read_text())["mpm"]["dt"]
ts = np.load(runs / f"{tag}.npz")["raws"] * dt
t0 = tl["t_contact"] - 0.5
t1 = (tl["t_part"] if tl["t_part"] is not None else float(ts[-1])) + 0.5
k0 = int(np.searchsorted(ts, t0 - 1e-9))
k1 = int(np.searchsorted(ts, t1 + 1e-9, side="right")) - 1
print(cut["cut"], k0, k1)
