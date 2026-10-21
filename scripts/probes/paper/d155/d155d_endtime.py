"""d155d_endtime.py -- D155d forensic (CPU, records): does the end's shortfall follow the refinement the body gets after it
arrives? Per run, render arm, every ramp variant (A3, A3G, A3C, A3D at seeds 97 / 99 / 101) and the tag's V seeds: the
committed windows and the simulated time between the arrival (u's gate over half the layer; on the ramp not before s =
1) and the delivered end, against the end measures (IoU against the independent sample, yardstick silhouette and
shading); the Spearman correlation over the ramp runs of each mesh, and over all ramp runs of the excess over the tag's
mean against the windows after arrival."""
import json

import numpy as np
from scipy.stats import spearmanr

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
R = {r["name"]: r for r in (json.loads(line) for line in open(f"{O}/gate_all.jsonl"))}


def after(name):
    a = next(iter(json.load(open(f"{O}/{name}.json"))["arms"].values()))
    dn = int(a["deliver_n"])
    acc = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit")]
    t_u = next((h["frame_end"] / 240 for h in acc if (h.get("u_gate") or 0) >= 0.5), 0.0)
    t_s = next((h["frame_end"] / 240 for h in acc if (h.get("morph_s") if h.get("morph_s") is not None else 1.0) >= 1.0), 0.0)
    ta = max(t_u, t_s)
    return len([h for h in acc if ta - 1e-9 < h["frame_end"] / 240 <= dn / 240 + 1e-9]), dn / 240 - ta


pool = []
for m in ("bunny", "dragon", "teapot", "armadilo", "homer"):
    print(m)
    tag = [R[f"{m}40k_V{s}"] for s in (97, 99, 101)]
    tm = {k: np.mean([t["end"][k] for t in tag]) for k in ("ind_sil", "yard_sil", "yard_pbr")}
    rows = []
    for arm in ("V97", "V99", "V101", "A3L97", "A3GL97", "A3CL97", "A3DL97", "A3DL99", "A3DL101"):
        r = R.get(f"{m}40k_{arm}")
        if r is None:
            continue
        w, t = after(f"{m}40k_{arm}")
        e = r["end"]
        print(f"  {arm:8s} after arrival {w:3d} windows {t:5.2f} s | IoU {e['ind_sil']:.4f} yard sil {e['yard_sil'] * 1e3:.3f} "
              f"shading {e['yard_pbr'] * 1e3:.3f}")
        if not arm.startswith("V"):
            rows.append((w, e["ind_sil"], e["yard_sil"], e["yard_pbr"]))
            pool.append((w, tm["ind_sil"] - e["ind_sil"], e["yard_sil"] / tm["yard_sil"] - 1, e["yard_pbr"] / tm["yard_pbr"] - 1))
    if len(rows) >= 4:
        w = [x[0] for x in rows]
        print(f"  ramp runs ({len(rows)}): windows after arrival vs IoU rho {spearmanr(w, [x[1] for x in rows]).correlation:+.2f}, "
              f"vs yard sil {spearmanr(w, [x[2] for x in rows]).correlation:+.2f}, vs shading "
              f"{spearmanr(w, [x[3] for x in rows]).correlation:+.2f}")
w = [x[0] for x in pool]
print(f"all ramp runs ({len(pool)}), excess over the tag's mean against the windows after arrival: IoU deficit rho "
      f"{spearmanr(w, [x[1] for x in pool]).correlation:+.2f}, yard sil excess {spearmanr(w, [x[2] for x in pool]).correlation:+.2f}, "
      f"shading excess {spearmanr(w, [x[3] for x in pool]).correlation:+.2f}")
