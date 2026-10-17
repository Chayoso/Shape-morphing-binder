"""fx_read.py RUNS_DIR TAG [TAG ...] -- B9b forensic reading (CPU): from tobody.py's tracks and the runs' records, each
run's tilt (the body's up axis from the vertical), the turn about the vertical, the centroid's height against the
standing target (pitches) and the rigid angular speed, every 0.5 s; then the first time the tilt passes 3 deg, the
largest tilt and the end; and from the window records the angular momentum's change summed over the run by update
(the grid's steps: stress with the control, gravity, the floor; u; the minimum spacing; the relaxation and the rest
below the grid), horizontal components (the tilt's) and vertical."""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1])
for tag in sys.argv[2:]:
    try:
        tr = json.loads((R / f"{tag}_track.json").read_text())
        j = json.loads((R / f"{tag}.json").read_text())
    except (OSError, ValueError) as e:
        print(f"{tag}: missing ({e})")
        continue
    t = tr["track"]
    first = next((p["t"] for p in t if p["tilt_deg"] > 3.0), None)
    big = max(t, key=lambda p: p["tilt_deg"])
    print(f"== {tag}: tilt first above 3 deg at {first} s; largest {big['tilt_deg']:.1f} deg at {big['t']:.2f} s; end tilt "
          f"{t[-1]['tilt_deg']:.1f}, about the vertical {t[-1]['yaw_deg']:+.1f} deg, height vs the standing target "
          f"{t[-1]['dh_sp']:+.2f} pitches; zero-control windows "
          f"{sum(1 for h in j['history'] if h.get('zero_commit'))} of {len(j['history'])}")
    step = 0.5
    nxt = 0.0
    for p in t:
        if p["t"] + 1e-9 >= nxt:
            print(f"   t {p['t']:5.2f}  tilt {p['tilt_deg']:5.1f}  vertical {p['yaw_deg']:+5.1f}  height {p['dh_sp']:+6.2f} sp  "
                  f"|w| {p.get('omega', float('nan')):.3f} rad/s  deformation speed {p.get('v_def_rms', float('nan')):.3f} wu/s")
            nxt += step
    acc = {k: np.zeros(3) for k in ("L_grid", "L_u", "L_space", "L_relax")}
    for h in j["history"]:
        s = (h.get("L_split") or [None])[0]
        if not s:
            continue
        for k in acc:
            acc[k] += np.asarray(s["own"][k], np.float64)
    tot = sum(np.hypot(v[0], v[2]) for v in acc.values()) or 1.0
    print("   angular momentum's change over the run by update (horizontal |sum| / its share; vertical): " + "; ".join(
        f"{k[2:]} {np.hypot(v[0], v[2]):.3g} ({np.hypot(v[0], v[2]) / tot:.2f}), {v[1]:+.3g}" for k, v in acc.items()))
