"""ext_yard145.py OUT_MD -- D145's methods table, the side table: the yardstick (render_terms_probe: the exterior's
silhouette and shading against the independent sample) and the display (surface_layer_probe: 1 - IoU front) at the end,
where the pool's end readings exist (t3end_ / e3end_ logs: ours on the tag, Xu et al., Buonomo et al. seed 1, OT, ARAP,
ISD and 4Deform seed 1 and radial on bunny and dragon). CPU only; reads logs, computes nothing."""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
SRC = [("ours (tag)", "d145t_b1", r"L(97|99|101|103|105)"), ("Xu et al.", "d145_b4", r"XU\d+"),
       ("Buonomo et al.", "d145_b5", r"BUO"), ("ISD", "d145_b14", r"ISDn1"), ("ISD radial", "d145_b14", r"ISDr1"),
       ("4Deform", "d145_b14", r"FDn1"), ("4Deform radial", "d145_b14", r"FDr1"), ("ARAP", "d145_b11", r"ARAP"),
       ("OT", "d145_b6", r"OT")]


def last(path, pred):
    try:
        rows = [json.loads(s) for s in open(path) if s.startswith("{")]
    except OSError:
        return None
    rows = [r for r in rows if pred(r)]
    return rows[-1] if rows else None


out = {}
for name, d, pat in SRC:
    for f in glob.glob(str(G / d / "t3end_*300k_*.log")) + glob.glob(str(G / d / "e3end_*300k_*.log")):
        m = re.match(rf"(t3end|e3end)_(\w+?)300k_({pat})\.log$", Path(f).name)
        if not m:
            continue
        kind, mesh = m.group(1), m.group(2)
        slot = out.setdefault((mesh, name), {"sil": [], "pbr": [], "front": []})
        if kind == "t3end":
            r = last(f, lambda r: "exterior" in r)
            if r:
                slot["sil"].append(r["exterior"]["sil"])
                slot["pbr"].append(r["exterior"]["pbr"])
        else:
            r = last(f, lambda r: r.get("state") not in ("target", "own") and "to_target" in r)
            if r:
                slot["front"].append(1 - r["to_target"]["front"]["iou"])
f = lambda v, s: "-" if not v else (f"{min(v) * s:.3g}" if len(v) == 1 else f"{min(v) * s:.3g}-{max(v) * s:.3g}")
L = ["### The end on our own exterior measures (where the pool read them): yardstick and display against the independent sample", "",
     "| mesh | method | runs | yardstick silhouette e-4 | yardstick shading e-4 | display front 1 - IoU % |", "|---|---|---|---|---|---|"]
for mesh in ("bunny", "dragon", "teapot", "bob", "armadilo", "homer"):
    for name, _, _ in SRC:
        v = out.get((mesh, name))
        if v and (v["sil"] or v["front"]):
            L.append(f"| {mesh} | {name} | {max(len(v['sil']), len(v['front']))} | {f(v['sil'], 1e4)} | {f(v['pbr'], 1e4)} | "
                     f"{f(v['front'], 100)} |")
text = "\n".join(L) + "\n"
Path(sys.argv[1]).write_text(text)
print(text)
