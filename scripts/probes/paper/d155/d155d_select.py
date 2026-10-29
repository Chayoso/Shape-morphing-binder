"""d155d_select.py -- D155d forensic (CPU, records): how the selection judges the windows on the ramp. Every window that
reads a new pack opens a selection epoch (D155: costs compare within one target), so on the ramp each window is the
first of its epoch: no reference merit, no rejection, "improved" by definition. Per run, render arm: over the windows
ending by 3.3 s and after it, the committed windows, the outer-rejected ones, the latched ones, the improved share, the
mean step scale (anneal) and the epochs; the tag's seeds against A3G / A3D."""
import json

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"


def tab(name):
    a = next(iter(json.load(open(f"{O}/{name}.json"))["arms"].values()))
    h = [x for x in a["history"] if x.get("frame_end") or x.get("null_commit")]
    out = {}
    for lab, sel in (("by 3.3 s", lambda x: (x.get("frame_end") or 0) / 240 <= 3.3 + 1e-9),
                     ("after", lambda x: (x.get("frame_end") or 1e9) / 240 > 3.3 + 1e-9)):
        w = [x for x in h if sel(x) and x.get("frame_end")]
        rej = sum(1 for x in w if x.get("outer_accepted") == 0)
        cand = sum((x.get("rejected") or 0) for x in w)
        lat = sum(1 for x in w if x.get("outer_gate_latched"))
        imp = sum(1 for x in w if x.get("improved"))
        ann = [x.get("anneal") for x in w if x.get("anneal") is not None]
        ep = sorted(set(x.get("selection_epoch") for x in w if x.get("selection_epoch") is not None))
        out[lab] = (len(w), rej, cand, lat, imp, sum(ann) / len(ann) if ann else float("nan"), len(ep))
    return out


for m in ("bunny", "dragon", "homer"):
    print(m)
    for arm in ("V97", "V99", "V101", "A3GL97", "A3DL97", "A3DL99", "A3DL101"):
        try:
            t = tab(f"{m}40k_{arm}")
        except OSError:
            continue
        print(f"  {arm:8s} " + " | ".join(
            f"{k}: {n} windows, outer-rejected {r}, rejected candidates {c}, latched {l}, improved {i}, anneal {a:.2f}, "
            f"epochs {e}" for k, (n, r, c, l, i, a, e) in t.items()))
