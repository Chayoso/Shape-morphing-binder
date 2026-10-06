"""ang_split.py LABEL=RUN_JSON [...] — D101: the angular momentum the position updates below the grid put in, per update
(minimum spacing, u, relaxation), summed over the committed windows: size, path (sum of the per-window sizes), and
the share of the jumps' sum along it; with the grid's change and the run's whole change."""
import json, sys
import numpy as np
for spec in sys.argv[1:]:
    name, p = spec.split("=")
    h = [r for r in json.load(open(p))["arms"]["render_full_dt_iso_nn"]["history"]
         if r.get("L_relax") is not None and r.get("frame_end") and not r.get("null_commit")]
    n = lambda v: float(np.linalg.norm(v))  # noqa: E731
    S = {k: np.sum([r[k] for r in h], 0) for k in ("L_grid", "L_jump", "L_space", "L_u", "L_relax")}
    P = {k: sum(n(r[k]) for r in h) for k in S}
    J = S["L_jump"]
    print(f"{name}: {len(h)} windows | grid {n(S['L_grid']):.3e} (path {P['L_grid']:.3e}) | jumps {n(J):.3e} (path {P['L_jump']:.3e}) | "
          f"whole {n(S['L_grid'] + J):.3e}")
    for k in ("L_space", "L_u", "L_relax"):
        print(f"   {k:8s} {n(S[k]):.3e} (path {P[k]:.3e}), share of the jumps' sum along it {float(S[k] @ J / max(J @ J, 1e-30)):+.2f}")
