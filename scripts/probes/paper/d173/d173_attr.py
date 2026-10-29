"""d173_attr.py OUT_DIR SEED [ARM2_NAME] [TITLE] -- D173: the attribution proof at 300k (the 3D target the smoothed dragon
everywhere): (1) the twin, (2) Hu with pictures of the relief (the stylized scales E, or option C's magnified real scales),
(3) Hu with pictures of the smoothed dragon (render on, the same weights, no scale content), and, when OUT_DIR/scene.json
lists it, (4) Hu with u off (--u_off: the render reaches dF_c alone, at the rule's weight). Reads OUT_DIR/read.json
(tmp/d172/d172_detail_fast.py on the runs: share = (A - floor) / (ceiling - floor) of d170_read.detail) and the runs'
records: share and corr at the stop and the end, kept, the render's share of each block's step at the start / arrival /
stop (tmp/d173/d173_shares.py: u and dF_c, after PCGrad, with the weights as applied), the layer's share with a non-zero
render gradient on u, rest and shape. Writes OUT_DIR/summary.md and verdict.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d170")
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d173")
import d170_read as R7  # noqa: E402
import d173_shares as SH  # noqa: E402

D, s = Path(sys.argv[1]), int(sys.argv[2])
ARM2 = sys.argv[3] if len(sys.argv) > 3 else "(2) Hu, pictures of the SCALED dragon"
TITLE = sys.argv[4] if len(sys.argv) > 4 else "the stylized scales E"
runs = json.load(open(D / "scene.json"))["runs"]
rd = json.load(open(D / "read.json"))
cA = float(np.mean([rd["refs"][f"detail_s1{s}"]["relief"]["A"], rd["refs"][f"detail_s2{s}"]["relief"]["A"]]))
fA = float(rd["refs"][f"smooth_s2{s}"]["relief"]["A"])
NAMES = {f"T_{s}": "(1) twin (lambda = 0)", f"Hs_{s}": ARM2, f"Hm_{s}": "(3) Hu, pictures of the SMOOTH dragon",
         f"Hu0_{s}": "(4) Hu, u off (the render reaches dF_c alone)"}
t = R7.run_rec(runs[f"T_{s}"])
rows = ["| arm | share stop | share end | kept | corr stop | corr end | " + " | ".join(SH.HEAD) + " | "
        "layer with a render gradient on u (mean) | v end / twin | sil_iou / twin | thin / twin |", "|" + "---|" * 12]
V = {}
for lab in [k for k in (f"T_{s}", f"Hs_{s}", f"Hm_{s}", f"Hu0_{s}") if k in runs]:
    r = R7.run_rec(runs[lab])
    ser = [q for q in rd["runs"][lab]["series"] if q["relief"].get("A") is not None]
    raws = [q["raw"] for q in ser]
    vals = [(q["relief"]["A"] - fA) / (cA - fA) for q in ser]
    cors = [q["relief"]["corr"] for q in ser]
    st, _ = R7.sample_at(raws, vals, r["stop_fe"] - 1)
    cs, _ = R7.sample_at(raws, cors, r["stop_fe"] - 1)
    en, ce = vals[-1], cors[-1]
    kept = en / st if st and st > 0.05 else None
    sh = SH.block_shares(runs[lab] + ".json", twin=lab.startswith("T_"))
    nz = [h.get("u_render_nonzero") for h in SH.opt_windows(runs[lab] + ".json") if h.get("u_render_nonzero") is not None]
    ratio = R7.vx(r, t)
    V[lab] = dict(share_stop=st, share_end=en, kept=kept, corr_stop=cs, corr_end=ce, shares=sh,
                  u_nonzero=float(np.mean(nz)) if nz else None, vx=ratio, sil=(r["sil"], t["sil"]), thin=(r["thin"], t["thin"]),
                  windows=r["windows"])
    v = V[lab]
    rows.append(f"| {NAMES[lab]} | {R7.fmt(st)} | {R7.fmt(en)} | {R7.fmt(kept)} | {R7.fmt(cs)} | {R7.fmt(ce)} | "
                f"{SH.fmt3(sh, 'u')} | {SH.fmt3(sh, 'dfc')} | {R7.fmt(v['u_nonzero'])} | {R7.fmt(ratio, '.2f')}x | "
                f"{r['sil']:.4f} / {t['sil']:.4f} | {r['thin']:.3f} / {t['thin']:.3f} |")
a2, a1, a3 = V[f"Hs_{s}"], V[f"T_{s}"], V[f"Hm_{s}"]
ok = bool(a2["share_end"] >= 0.40 and a1["share_end"] < 0.10 and a3["share_end"] < 0.10)
verdict = (("ATTRIBUTED" if ok else "NOT ATTRIBUTED") + f": the scales appear only with the relief's pictures (share end "
           f"{R7.fmt(a2['share_end'])} vs twin {R7.fmt(a1['share_end'])} and smooth pictures {R7.fmt(a3['share_end'])}; corr "
           f"{R7.fmt(a2['corr_end'])} / {R7.fmt(a1['corr_end'])} / {R7.fmt(a3['corr_end'])})")
if f"Hu0_{s}" in V:
    a4 = V[f"Hu0_{s}"]
    frac = a4["share_end"] / a2["share_end"] if a2["share_end"] and a2["share_end"] > 0.05 else None
    verdict += (f"; u off (dF_c alone): share end {R7.fmt(a4['share_end'])}, corr {R7.fmt(a4['corr_end'])}"
                + (f", {frac:.2f} of arm 2's" if frac is not None else ""))
L = [f"# D173 attribution: {TITLE} at 300k, seed {s}", "", *rows, "",
     "Share = (A - floor) / (ceiling - floor) (D172's detail reading inside the relief's pattern, the fast port "
     "d172_detail_fast.py; ceiling = the relief's own 300k samples, floor = the smoothed dragon's own sample); the 3D target "
     "is the smoothed dragon in every arm, so the physics objective, the proximity term and the relaxation never see the "
     "scales. " + SH.NOTE + " Criterion (set before the runs): share end >= 0.40 in (2), < 0.10 in (1) and (3).", "",
     "VERDICT: " + verdict]
(D / "summary.md").write_text("\n".join(L) + "\n")
json.dump(dict(arms=V, verdict=verdict, ceiling_A=cA, floor_A=fA), open(D / "verdict.json", "w"), indent=1, default=float)
print(verdict)
