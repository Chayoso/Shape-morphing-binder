"""b10_curves.py OUT_JSONL DIR/NAME [...] -- D145 B10: the per-window loss terms of each run, from its own record (no
new run). Per committed attempt (the record's history): the window (animation), its simulated end time (frame_end / 240
s), the selection merit (total) and its physics part (outer_merit, outer_merit_phys), the transport energy, the released
motion (stab) and the end drift (stab_end), the silhouette and shading terms (d_sil, d_pbr), the render weight lambda
and share g_share, the kinetic record; with --telemetry also the proximity (sup_B) and transport (sup_E) split, and the
near band (merit_far, recorded always). One JSON line per run."""
import json
import sys
from pathlib import Path

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
KEYS = ("animation", "frame_end", "outer_accepted", "outer_merit", "outer_merit_phys", "selection_merit",
        "transport_energy", "stab", "stab_end", "d_sil", "d_pbr", "d_render", "lambda", "g_share", "kin", "merit_far",
        "sup_E", "sup_B", "n_spray", "n_near_active")
with open(sys.argv[1], "a") as out:
    for spec in sys.argv[2:]:
        d, name = spec.split("/")
        arm = json.loads((G / d / f"{name}.json").read_text())["arms"]["render_full_dt_iso_nn"]
        rows = [{k: h.get(k) for k in KEYS} for h in arm["history"]]
        for r in rows:
            r["t"] = None if r["frame_end"] is None else r["frame_end"] / 240.0
        out.write(json.dumps({"dir": d, "name": name, "deliver_n": arm.get("deliver_n"), "windows": rows}) + "\n")
        print(name, len(rows), "attempts", flush=True)
