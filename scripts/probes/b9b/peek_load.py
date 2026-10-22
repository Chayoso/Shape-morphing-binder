"""peek_load.py ROOT -- B9b assim_load, a quick look from the window records (CPU): the last 2 s of |omega| and |v_c|,
the assimilation's p99 read against the frozen read, the drag."""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
groups = [("load", ["ZA97", "ZB97", "ZC97", "ZL97", "L97", "L99", "L101", "P97", "A97", "A99", "A101"]),
          ("world", ["L97", "L99", "L101", "P97"]), ("share", ["L97", "L99", "L101", "P97"])]
for mesh in ("bunny", "dragon"):
    for d, arms in groups:
        for a in arms:
            p = root / d / f"{mesh}40k_{a}.json"
            if not p.exists():
                continue
            j = json.loads(p.read_text())
            h = [x for x in j["history"] if "bodies" in x]
            te = h[-1]["t"]
            late = [x for x in h if x["t"] >= te - 2.0]
            w = [float(np.linalg.norm(x["bodies"][0]["omega"])) for x in late]
            v = [float(np.linalg.norm(x["bodies"][0]["v_com"])) for x in late]
            ar = [x["assim"]["read"]["p99"] for x in h if x.get("assim")]
            af = [x["assim"]["frozen_read"]["p99"] for x in h if x.get("assim")]
            nan = float("nan")
            print(f"{d}/{mesh}40k_{a}: last 2 s |w| mean {np.mean(w):.4f} end {w[-1]:.4f} rad/s, |v_c| mean "
                  f"{np.mean(v):.4f} end {v[-1]:.4f} wu/s; assim p99 read {np.median(ar) if ar else nan:.4f} frozen "
                  f"{np.median(af) if af else nan:.4f}; drag {j['scene']['spec']['drag']}, deformation drag "
                  f"{j['scene'].get('deformation_drag')}; windows {len(h)}")
