import json, glob
for p in sorted(glob.glob("/data/relcfd/chayo/physmorph_v2/output/gpu/d157/read300_*.jsonl")):
    for ln in open(p):
        r = json.loads(ln)
        if r["mesh"] == "bunny" and r["label"] == "L":
            print(p, r["seed"], list(r.keys()))
            print(" d3:", {k: list(v.keys()) for k, v in r["d3"].items()})
            print(" d3 zb cor:", list(r["d3"]["zb"]["cor"].keys()))
            print(" img:", {k: list(v.keys()) for k, v in r["img"].items()})
            print(" img cor:", {k: list(v.keys()) for k, v in r["img"]["cor"].items()})
            print(" img cor shaded held:", r["img"]["cor"]["shaded"]["held"])
            raise SystemExit
