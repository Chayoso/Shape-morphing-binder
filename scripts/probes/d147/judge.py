"""judge.py TIMELINE_TXT PROGRESS_TXT TAG[,TAG...] -- D147: the throw's rule as pre-registered (docs/experiments.md,
D147, the continuous target): per run, the morph's progress at the first contact in [0.35, 0.65] (mean over the
bodies), the pair parted in the air, both landed, the end shape error <= 0.05, and no rise in shape progress between
neighbouring curve points (0.15 s) up to a second past the ramp larger than 3x the steady rate, per body; the tumble
reported. The JSON lines of read40k.py's timeline and of progress.py, read as they are. Prints one line per run and
PASS or FAIL (every run must pass). CPU."""
import json
import sys


def lines(path):
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith("{") and '"tag"' in ln:
            try:
                j = json.loads(ln)
            except json.JSONDecodeError:
                continue
            out.setdefault(j["tag"], {}).update(j)
    return out


tl, pg = lines(sys.argv[1]), lines(sys.argv[2])
ok_all = True
for tag in sys.argv[3].split(","):
    t, p = tl.get(tag, {}), pg.get(tag, {})
    jumps = p.get("jumps") or []
    checks = {
        "progress_band": bool(p.get("in_band")),
        "parted_in_air": bool(t.get("parted_in_air")),
        "landed": bool(t.get("landing_s")) and all(v is not None for v in t["landing_s"]),
        "end_error": t.get("end_shape_error") is not None and t["end_shape_error"] <= 0.05,
        "no_snap": bool(jumps) and all(j.get("early_ratio") is not None and j["early_ratio"] <= 3.0 for j in jumps),
    }
    ok = all(checks.values())
    ok_all &= ok
    print(json.dumps({"tag": tag, "pass": ok, **checks, "progress": p.get("progress_mean"),
                      "early_ratio": [j.get("early_ratio") for j in jumps], "end_error": t.get("end_shape_error"),
                      "landing_s": t.get("landing_s"), "tumble_deg": t.get("tumble_deg")}))
print("PASS" if ok_all else "FAIL")
