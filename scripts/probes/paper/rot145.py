"""rot145.py OUT_JSONL DIR/NAME [...] -- D145: the run's rotation and angular momentum taken apart by the position update
that made them, from the runs' own per-window records (telemetry.below_grid_record, every committed window; CPU only).

Per run: summed over the committed windows, each part's rotation vector (grid = the grid's advection dt v, spacing = the
minimum spacing's push, u = the u channel's displacement, relax = the layer relaxation and the bonds; body = the whole),
in degrees: |sum| (the part's net rotation), its signed share along the body's net axis (the shares add up to the
body's net), and the path sum_w |part_w| (how much each part turns the body window by window, which is what the
frame-to-frame net / gross angular reading sees). The angular momentum per unit mass changed by each update
(L_grid, L_space, L_u, L_relax): |sum| and sum_w |.|. The measured end rotation (m_<run>.log, mom.sh) beside the
records' body sum, as a closure check."""
import json
import re
import sys

import numpy as np

G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
PARTS = ("grid", "spacing", "u", "relax", "body")
LP = ("L_grid", "L_space", "L_u", "L_relax")
out = open(sys.argv[1], "a")
for spec in sys.argv[2:]:
    d, name = spec.split("/")
    js = json.load(open(f"{G}/{d}/{name}.json"))
    hist = next(iter(js["arms"].values()))["history"]
    rows = [h for h in hist if h.get("body_vrot") is not None]
    rec = dict(name=name, windows=len(rows))
    vr = {p: np.array([h[p + "_vrot"] for h in rows], np.float64) for p in PARTS}
    tot = {p: vr[p].sum(0) for p in PARTS}
    axis = tot["body"] / max(np.linalg.norm(tot["body"]), 1e-300)
    deg = 180.0 / np.pi
    for p in PARTS:
        rec[p + "_net_deg"] = float(np.linalg.norm(tot[p]) * deg)
        rec[p + "_along_deg"] = float(tot[p] @ axis * deg)
        rec[p + "_path_deg"] = float(np.linalg.norm(vr[p], axis=1).sum() * deg)
    for k in LP:
        a = np.array([h[k] for h in rows if h.get(k) is not None], np.float64)
        rec[k + "_net"] = float(np.linalg.norm(a.sum(0))) if len(a) else None
        rec[k + "_path"] = float(np.linalg.norm(a, axis=1).sum()) if len(a) else None
    try:
        s = open(f"{G}/{d}/m_{name}.log").read()
        m = re.search(r"net rotation over the run ([\d.e+-]+) deg", s)
        rec["measured_rot_deg"] = float(m.group(1)) if m else None
        m = re.search(r"net / gross angular move: largest ([\d.e+-]+), mean ([\d.e+-]+)", s)
        rec["ang_net_gross_mean"] = float(m.group(2)) if m else None
    except OSError:
        rec["measured_rot_deg"] = rec["ang_net_gross_mean"] = None
    out.write(json.dumps(rec) + "\n")
    print(f"{name:20s} w={rec['windows']:3d} measured {rec['measured_rot_deg']} deg, ratio {rec['ang_net_gross_mean']} | "
          + " | ".join(f"{p} net {rec[p + '_net_deg']:.2e} along {rec[p + '_along_deg']:+.2e} path {rec[p + '_path_deg']:.2e}"
                       for p in PARTS)
          + " | " + " ".join(f"{k} {rec[k + '_net']:.2e}/{rec[k + '_path']:.2e}" for k in LP if rec[k + "_net"] is not None),
          flush=True)
