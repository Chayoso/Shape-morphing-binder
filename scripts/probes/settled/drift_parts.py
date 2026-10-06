"""drift_parts.py LABEL=RUN_JSON [...] — D97: the run's net drift by position update. Over the committed windows the
record's vectors are summed per part (grid, minimum spacing, u, relaxation; body = their sum): the net translation
(world units) and net rotation (degrees) of each, the share of the body's net along it (part . body / |body|^2),
and the path (the sum of the per-window sizes). Checked: the parts sum to the body."""
import json, sys
import numpy as np

PARTS = ("grid", "spacing", "u", "relax", "body")
for spec in sys.argv[1:]:
    name, p = spec.split("=")
    h = [r for r in json.load(open(p))["arms"]["render_full_dt_iso_nn"]["history"]
         if r.get("frame_end") and not r.get("null_commit") and r.get("body_vcom") is not None]
    t = {k: np.sum([r[k + "_vcom"] for r in h], 0) for k in PARTS}
    w = {k: np.degrees(np.sum([r[k + "_vrot"] for r in h], 0)) for k in PARTS}
    path = {k: sum(r[k + "_com"] for r in h) for k in PARTS}
    gap = np.linalg.norm(sum(t[k] for k in PARTS[:-1]) - t["body"])
    print(f"{name}: {len(h)} windows | parts - body {gap:.1e} wu")
    for k in PARTS:
        st = float(t[k] @ t["body"] / max(t["body"] @ t["body"], 1e-30))
        sr = float(w[k] @ w["body"] / max(w["body"] @ w["body"], 1e-30))
        print(f"   {k:8s} net translation {np.linalg.norm(t[k]):.3e} wu (share of the body's {st:+.2f}), path {path[k]:.3e} | "
              f"net rotation {np.linalg.norm(w[k]):.4f} deg (share {sr:+.2f})")
