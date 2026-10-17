"""b9_40k.py [OUT_MD] -- D145 B9 (iii) post-tag: the 40k gravity runs with the floor (G2: f_ext -0.65, G = 0.02; G5:
f_ext -1.6, G = 0.05; render arm, seed 97) on the tag (d145t_b9) and before it (d145_b9), beside the no-gravity render
arm V's three-seed range (d145t_b3 / d145_b3). CPU only: gallery_ind.py's ind.log (the independent sample lifted onto
the floor as the run's target was), the runs' JSON (windows, the last kinetic record, the stop reason)."""
import json
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")


def ind(d):
    out = {}
    try:
        for s in open(d / "ind.log"):
            if s.startswith("{"):
                r = json.loads(s)
                out[r["tag"]] = r
    except OSError:
        pass
    return out


def rec(d, tag):
    try:
        arm = json.loads((d / f"{tag}.json").read_text())["arms"]["render_full_dt_iso_nn"]
    except (OSError, KeyError, ValueError):
        return {}
    h = arm["history"]
    kin = [x["kin"] for x in h if x.get("kin") is not None]
    com = [x["com"] for x in h if x.get("outer_accepted") and isinstance(x.get("com"), list)]
    return dict(windows=sum(1 for x in h if x.get("outer_accepted")), kin=kin[-1] if kin else None,
                dy=com[-1][1] if com else None, stop=h[-1].get("commit_reason") if h else None)


lines = ["### B9 (iii): 40k with the floor and gravity (render arm, seed 97), tag and pre-tag", "",
         "| mesh | G | tag sil IoU ind | tag windows | tag end kin | tag centre-of-mass height change (wu) | "
         "pre-tag sil IoU ind | pre-tag windows | pre-tag end kin | no-gravity V on the tag (3 seeds) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
it, ip, iv = ind(G / "d145t_b9"), ind(G / "d145_b9"), ind(G / "d145t_b3")
for mesh in ("bunny", "dragon", "armadilo", "bob"):
    vv = [iv[f"{mesh}40k_V{s}"]["ind"]["sil"] for s in (97, 99, 101) if f"{mesh}40k_V{s}" in iv]
    for g in ("G2", "G5"):
        tag = f"{mesh}40k_{g}L97"
        a, b = it.get(tag), ip.get(tag)
        ra, rb = rec(G / "d145t_b9", tag), rec(G / "d145_b9", tag)
        fm = lambda v, spec: "-" if v is None else format(v, spec)
        cells = [mesh, g, fm(a["ind"]["sil"] if a else None, ".4f"), str(ra.get("windows", "-")),
                 fm(ra.get("kin"), ".2e"), fm(ra.get("dy"), "+.3f"), fm(b["ind"]["sil"] if b else None, ".4f"),
                 str(rb.get("windows", "-")), fm(rb.get("kin"), ".2e"),
                 "-" if not vv else f"{min(vv):.4f} / {np.median(vv):.4f} / {max(vv):.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
text = "\n".join(lines) + "\n"
print(text)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(text)
