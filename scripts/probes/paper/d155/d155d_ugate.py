"""d155d_ugate.py -- D155d forensic (CPU, records): how much sub-cell surface work (u, the outer layer's control) the body
gets, render arm and twin: u's gate (the share of the layer it acts on) summed over the committed windows of the whole
delivered run and of the time after the arrival (window-equivalents of a fully open gate), its first window over half,
and the end's yardstick silhouette; the tag's seeds against A3G / A3D."""
import json

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
R = {r["name"]: r for r in (json.loads(line) for line in open(f"{O}/gate_all.jsonl"))}


def ug(name):
    a = next(iter(json.load(open(f"{O}/{name}.json"))["arms"].values()))
    dn = int(a["deliver_n"])
    acc = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit") and h["frame_end"] <= dn]
    t_s = next((h["frame_end"] / 240 for h in acc if (h.get("morph_s") if h.get("morph_s") is not None else 1.0) >= 1.0), 0.0)
    tot = sum(h.get("u_gate") or 0 for h in acc)
    post = sum(h.get("u_gate") or 0 for h in acc if h["frame_end"] / 240 > t_s - 1e-9)
    half = next((h["frame_end"] / 240 for h in acc if (h.get("u_gate") or 0) >= 0.5), None)
    return tot, post, half, len(acc)


for m in ("bunny", "dragon", "teapot", "armadilo", "homer"):
    print(m)
    for arm in ("V97", "V99", "V101", "A3GL97", "A3DL97", "A3DL99", "A3DL101", "W97", "W99", "W101", "A3GP97", "A3DP97"):
        if f"{m}40k_{arm}" not in R:
            continue
        tot, post, half, n = ug(f"{m}40k_{arm}")
        ys = R[f"{m}40k_{arm}"]["end"].get("yard_sil")
        print(f"  {arm:8s} u gate summed {tot:5.1f} window-equivalents over {n} windows, after s = 1 {post:5.1f}; first over half "
              f"{'-' if half is None else f'{half:.2f} s'}; yard sil {'-' if ys is None else f'{ys * 1e3:.3f}'}")
