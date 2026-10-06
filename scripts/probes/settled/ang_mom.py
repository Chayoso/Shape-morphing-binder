"""ang_mom.py LABEL=RUN_JSON [...] — the body's angular momentum (per unit mass, about the window's starting centre of
mass, APIC's affine part included) at each committed window's start, the conserved quantity a rotation estimate
only approximates: its size at the end, its largest, its mean over the run and over the last ten windows, and the
kinetic scale it is set against (sum of |r| |v| at those windows is not recorded; the size alone, both arms alike)."""
import json, sys
import numpy as np
for spec in sys.argv[1:]:
    name, p = spec.split("=")
    h = [r for r in json.load(open(p))["arms"]["render_full_dt_iso_nn"]["history"]
         if r.get("frame_end") and not r.get("null_commit") and r.get("L_start") is not None]
    L = np.array([np.linalg.norm(r["L_start"]) for r in h])
    print(f"{name:8s} {len(h)} windows | |L| at the window starts: last {L[-1]:.3f}, largest {L.max():.3f} (window {h[int(L.argmax())]['animation']}), "
          f"mean {L.mean():.3f}, last ten {L[-10:].mean():.3f}")
