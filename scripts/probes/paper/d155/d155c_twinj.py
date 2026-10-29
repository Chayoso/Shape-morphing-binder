"""d155c_twinj.py -- D155c: the volume at the end and the ramp's largest mean J, render arm and its lambda = 0 twin."""
import json

R = {}
for line in open("/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate/gate_all.jsonl"):
    r = json.loads(line)
    R[r["name"]] = r
for m in ("bunny", "dragon", "teapot", "armadilo", "homer"):
    out = []
    for a in ("V97", "W97", "A3GL97", "A3GP97", "A3CL97", "A3CP97"):
        r = R.get(f"{m}40k_{a}")
        if r and r.get("J_end") and r["J_end"].get("mean") is not None:
            je = r["J_end"]
            out.append(f"{a} J {je['mean']:.4f} p99 {je['p99']:.3f} ramp max {r.get('J_ramp_max') or 0:.4f}")
    print(m, " | ".join(out))
