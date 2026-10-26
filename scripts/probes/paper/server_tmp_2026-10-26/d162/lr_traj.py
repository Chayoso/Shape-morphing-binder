import json
for name in ("coin40k_full_97", "coin40k_twin_97", "coin40k_full_97_lh", "bunny40k_full_97", "bunny40k_twin_97", "bunny40k_full_97_lh"):
    js = json.load(open(name + ".json")); arm = list(js["arms"].values())[0]
    h = [r for r in arm["history"] if r.get("frame_end") and not r.get("null_commit")]
    sel = [i for i in (0, 2, 5, 10, 15, 20, 25, len(h) - 1) if i < len(h)]
    print(name, " ".join("w%d:%.2e" % (h[i]["animation"], h[i]["d_sil"] + h[i]["d_pbr"]) for i in sel))
