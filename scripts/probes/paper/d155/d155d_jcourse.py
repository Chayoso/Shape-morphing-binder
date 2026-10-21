"""d155d_jcourse.py -- D155d (the tag and A3G re-run with the elastic record, A3D): the course of the body's volume against the targets' on the way: the tracked J's mean /
p99, the rest volume det Fp's p99, and for the moving target its volume before the match (morph_vol_ratio) and k, at
fixed times; the tag (V97) against D155b (A3GL97) and D155c (A3CL97), render arm. argv: meshes."""
import json
import os
import sys

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
for m in sys.argv[1:]:
    print(m)
    for arm in ("VE97", "A3GE97", "A3DL97"):
        p = f"{O}/{m}40k_{arm}.json"
        if not os.path.exists(p):
            continue
        a = next(iter(json.load(open(p))["arms"].values()))
        acc = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit")
               and h["frame_end"] <= a["deliver_n"]]
        rows = []
        for t_q in (0.5, 1.0, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 99):
            h = min(acc, key=lambda r: abs(r["frame_end"] / 240 - t_q))
            s = h.get("morph_s")
            rows.append(f"{h['frame_end'] / 240:.2f}s"
                        + (f"(s={s:.2f} V/Vb {h['morph_vol_ratio']:.3f})" if s is not None and s < 1
                           and h.get("morph_vol_ratio") is not None else (f"(s={s:.2f})" if s is not None and s < 1 else ""))
                        + f" J {h.get('Jx_mean', 0):.4f} p01 {h.get('Jx_p01', 0):.3f} p99 {h.get('Jx_p99', 0):.3f} Fp99 {h.get('Jp_p99', 0):.3f} Je {h.get('Je_p01', 0):.3f}..{h.get('Je_p99', 0):.3f} E {h.get('E_el', 0):.1f} Evol {h.get('E_el_vol', 0):.1f} kin {h.get('kin', 0):.4f}")
        print(f"  {arm:7s}\n    " + "\n    ".join(dict.fromkeys(rows)))
