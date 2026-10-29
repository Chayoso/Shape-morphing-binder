"""d155_b7sum.py OUT_DIR -- D155: the B7 window decomposition per phase (b7_{bunny,dragon}.json from d155_b7.py)."""
import json
import sys

import numpy as np

O = sys.argv[1]
BINS = (("0-1 s", 0, 1), ("1-2 s", 1, 2), ("2-3 s", 2, 3), ("3-6 s", 3, 6), ("6 s-end", 6, 1e9))
THIRDS = (("1st third", 0, 1 / 3), ("2nd third", 1 / 3, 2 / 3), ("3rd third", 2 / 3, 1.01))
out = open(f"{O}/tables_b7.md", "w")
summ = {}


def p(s=""):
    print(s)
    out.write(s + "\n")


for mesh in ("bunny", "dragon"):
    r = json.load(open(f"{O}/b7_{mesh}.json"))
    rows = [w for w in r["rows"] if w["chained"] and w["t"] is not None]
    tot_D = sum(w["layer"]["D"] for w in rows)
    tot_Dn = sum(w["layer"]["Dn"] for w in rows)
    summ[mesh] = {"windows": len(rows), "phases": {}}
    p(f"## {mesh} RG97 ({len(rows)} chained windows; pitch {r['pitch']:.4f} wu; outer layer {r['layer']} particles)")
    p("| phase | windows | share of the summed surface motion (all / normal) | surface motion a window, rms pitch: actual / lambda=0 / render part | normal: actual / render part | render's share of the actual motion: projection (layer / normal / all particles) | abs(d)^2 / abs(D)^2 (layer) | own silhouette with / without render (median ratio) | shading ratio | kinetic ratio |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    for lab, lo, hi in BINS + THIRDS:
        key = "t" if "s" in lab.split()[-1] or lab.endswith("end") else "f"
        sel = [w for w in rows if lo < w[key] <= hi] if lo > 0 else [w for w in rows if w[key] <= hi]
        if not sel:
            continue
        L = [w["layer"] for w in sel]
        A = [w["all"] for w in sel]
        sh = sum(x["D"] for x in L) / tot_D
        shn = sum(x["Dn"] for x in L) / tot_Dn
        proj = sum(x["dD"] for x in L) / sum(x["DD"] for x in L)
        projn = sum(x["dDn"] for x in L) / sum(x["DDn"] for x in L)
        proja = sum(x["dD"] for x in A) / sum(x["DD"] for x in A)
        e2 = sum(x["dd"] for x in L) / sum(x["DD"] for x in L)
        med = lambda k: float(np.median([x[k] for x in L]))   # noqa: E731
        rat = lambda a: float(np.median([w[f"{a}_r"] / w[f"{a}_p"] for w in sel  # noqa: E731
                                         if w.get(f"{a}_r") is not None and w.get(f"{a}_p")]))
        row = dict(windows=len(sel), share=sh, share_n=shn, D=med("D"), D0=med("D0"), d=med("d"), Dn=med("Dn"),
                   dn=med("dn"), proj=proj, projn=projn, proj_all=proja, e2=e2, sil=rat("d_sil"), pbr=rat("d_pbr"),
                   kin=rat("kin"))
        summ[mesh]["phases"][lab] = row
        p(f"| {lab} | {len(sel)} | {sh:.3f} / {shn:.3f} | {row['D']:.4f} / {row['D0']:.4f} / {row['d']:.4f} | "
          f"{row['Dn']:.4f} / {row['dn']:.4f} | {proj:+.3f} / {projn:+.3f} / {proja:+.3f} | {e2:.3f} | {row['sil']:.3f} | "
          f"{row['pbr']:.3f} | {row['kin']:.3f} |")
    p()
json.dump(summ, open(f"{O}/b7_phases.json", "w"), indent=1)
out.close()
