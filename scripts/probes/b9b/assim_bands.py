"""assim_bands.py ROOT -- B9b assim_load forensic (CPU): what the assimilation reads, by time band. Per run, per band, the
median over its windows of the read's p99 (the largest |log sigma| of F_e) against the frozen read's p99 on the same
committed state, and the share of the frozen read that the definition removed, 1 - read / frozen. Runs: the definition
(load/ L), (1) alone (load/ A), the share (share/ L), the no-gravity assim_load run (load/ ZL97)."""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
runs = [("load", m, a) for m in ("bunny", "dragon") for a in ("L97", "L99", "L101", "P97", "A97", "A99", "A101")]
runs += [("share", m, a) for m in ("bunny", "dragon") for a in ("L97", "L99", "L101", "P97")]
runs += [("load", "bunny", "ZL97")]
for d, m, a in runs:
    p = root / d / f"{m}40k_{a}.json"
    if not p.exists():
        continue
    h = [x for x in json.loads(p.read_text())["history"] if x.get("assim")]
    te = h[-1]["t"]
    bands = [(0, 1), (1, 2), (2, 4), (4, 6), (6, te + 1)]
    out = []
    for lo, hi in bands:
        w = [x for x in h if lo <= x["t"] < hi]
        if not w:
            continue
        r = float(np.median([x["assim"]["read"]["p99"] for x in w]))
        f = float(np.median([x["assim"]["frozen_read"]["p99"] for x in w]))
        out.append(f"{lo}-{min(hi, te):.3g}s {r:.4f}/{f:.4f} ({100 * (1 - r / f):.0f}%)")
    print(f"{d}/{m}40k_{a}: read/frozen p99 (removed): " + " | ".join(out))
