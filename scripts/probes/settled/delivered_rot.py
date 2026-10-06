"""delivered_rot.py LABEL=RUN_JSON [...] — the record's net rotation (degrees) over the delivered windows only (up to the
truncation's best window) against over every committed window, by part (grid, spacing, u, relaxation; body their
sum), and the angular momentum the jumps put in over the delivered windows by update."""
import json, sys
import numpy as np
for spec in sys.argv[1:]:
    name, p = spec.split("=")
    a = json.load(open(p))["arms"]["render_full_dt_iso_nn"]
    h = [r for r in a["history"] if r.get("frame_end") and not r.get("null_commit") and r.get("body_vrot") is not None]
    end = (a.get("truncation") or {}).get("frames_kept", a["deliver_n"])
    d = [r for r in h if r["frame_end"] <= end]
    n = lambda v: float(np.linalg.norm(v))  # noqa: E731
    rot = lambda rows, k: n(np.degrees(np.sum([r[k + "_vrot"] for r in rows], 0))) if rows else 0.  # noqa: E731
    parts = " ".join(f"{k} {rot(d, k):.4f}" for k in ("grid", "spacing", "u", "relax"))
    Ls = {k: n(np.sum([r[k] for r in d], 0)) for k in ("L_jump", "L_space", "L_u", "L_relax") if d and d[0].get(k) is not None}
    print(f"{name:8s} delivered {len(d)} of {len(h)} windows | body rotation delivered {rot(d, 'body'):.4f} deg, all {rot(h, 'body'):.4f} | "
          f"delivered parts: {parts} | jumps' L delivered: " + " ".join(f"{k} {v:.1f}" for k, v in Ls.items()))
