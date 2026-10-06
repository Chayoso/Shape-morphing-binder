"""d100_rows.py IND_LOG RUN_DIR — D100's table: per mesh the silhouette IoU against the independent sample of O (the
code before D98, render arm), D (D98, render arm) and E (D98, physics-only twin), and from the records the body's net
translation over the run (the drift parts' body vectors summed; world units). Then the counts the criteria read."""
import json, os, sys
from collections import defaultdict
import numpy as np

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = defaultdict(dict)
for r in rows:
    mesh, arm = r["tag"].rsplit("_", 1)
    by[mesh][arm] = r


def drift(tag):
    p = os.path.join(sys.argv[2], tag + ".json")
    if not os.path.exists(p):
        return float("nan")
    h = [r for r in json.load(open(p))["arms"]["render_full_dt_iso_nn"]["history"]
         if r.get("body_vcom") is not None and r.get("frame_end") and not r.get("null_commit")]
    return float(np.linalg.norm(np.sum([r["body_vcom"] for r in h], 0))) if h else float("nan")


n = k_do = k_de = 0
dd, gain, dr_o, dr_d = [], [], [], []
for mesh in sorted(by):
    a = by[mesh]
    if not all(x in a for x in ("O", "D", "E")):
        continue
    n += 1
    s = {x: a[x]["ind"]["sil"] for x in ("O", "D", "E")}
    t = {x: drift(f"{mesh}_{x}") for x in ("O", "D", "E")}
    k_do += s["D"] >= s["O"]; k_de += s["D"] > s["E"]
    dd.append(s["D"] - s["O"]); gain.append((1 - s["D"]) / (1 - s["E"]) - 1)
    dr_o.append(t["O"]); dr_d.append(t["D"])
    print(f"{mesh:12s} sil (independent): O {s['O']:.4f} D {s['D']:.4f} E {s['E']:.4f} floor {a['D']['floor']['sil']:.4f} | "
          f"D - O {s['D'] - s['O']:+.4f} | 1-IoU D against E {100 * gain[-1]:+.0f} % | net translation O {t['O']:.2e} D {t['D']:.2e} E {t['E']:.2e}")
med = lambda v: float(np.nanmedian(v))  # noqa: E731
print(f"{n} meshes: D at or above O on {k_do}, D - O median {med(dd):+.4f} (min {min(dd):+.4f}, max {max(dd):+.4f}); "
      f"D above E on {k_de}, 1 - IoU against E median {100 * med(gain):+.0f} % (range {100 * min(gain):+.0f} to {100 * max(gain):+.0f} %); "
      f"net translation median O {med(dr_o):.2e}, D {med(dr_d):.2e}")
