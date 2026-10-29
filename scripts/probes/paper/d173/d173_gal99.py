"""d173_gal99.py OUT_DIR -- (for D172) the Hu gallery at seed 99 (job 191: output/gpu/d172/runs/gal_{mesh}_Hu_99) read on its own
row: per mesh sil_iou / chamfer / thin against the tag's rule at seed 99 (V 99, d155 gate / d154) and, when they exist, the
rule's own rerun at seed 99 on repo_d166 (R 99: output/gpu/d173/runs/gal_{mesh}_R_99, the defaults, as d167 job 114's R 97).
Holds = sil_iou >= V - 0.003, chamfer <= V + 0.002, thin <= V + 0.02 (d170_read.SH_TOL); two-baseline = within the same
tolerance of the worse of R 99 / V 99 (as the seed-97 reading takes R 97 / V 97); rest = end v_mean <= 2 x the twin W 99.
The readings are d170_read.run_rec's (the runs' own metrics) and d162_gallery.pieces (pieces >= 10 particles, end frame).
Writes OUT_DIR/summary.md (a VERDICT line) and OUT_DIR/verdict.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d170")
import d170_read as R7  # noqa: E402

D = Path(sys.argv[1])
D.mkdir(parents=True, exist_ok=True)
G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
GL = R7.R6.gl()
TOL = R7.SH_TOL
REST = getattr(R7, "REST_X", 2.0)


def rd(prefix):
    r = R7.run_rec(prefix)
    if r is not None:
        fr = Path(f"{prefix}_frames12.npz")
        fr = fr if fr.exists() else Path(f"{prefix}_end1.npz")
        if fr.exists():
            r["p10"] = GL.pieces(np.asarray(np.load(fr, allow_pickle=True)["frames"][-1], np.float64))[1]
    return r


def tag(m, a):
    jp, _ = GL.paths(m, a, 99)
    return rd(jp[:-5]) if jp else None


f4 = lambda r: "-" if r is None else f"{r['sil']:.4f} / {r['ch']:.4f} / {r['thin']:.3f} / {r['v_end'] * 1e3:.1f} / {r.get('p10', '-')}"  # noqa: E731
rows = ["| mesh | Hu 99 | V 99 (tag) | R 99 (repo_d166, defaults) | twin W 99 | d sil / chamfer / thin vs V 99 | holds (V 99) | two-baseline (R 99 / V 99) |",
        "|" + "---|" * 8]
v = dict(held=0, held_two=0, n=0, miss=[], miss_two=[], rest=True, have_r=0)
for m in R7.MESHES:
    x, v99, w99 = rd(str(G / f"d172/runs/gal_{m}_Hu_99")), tag(m, "V"), tag(m, "W")
    r99 = rd(str(G / f"d173/runs/gal_{m}_R_99"))
    if x is None or v99 is None:
        rows.append(f"| {m} | {f4(x)} | {f4(v99)} | {f4(r99)} | {f4(w99)} | - | - | - |")
        continue
    d = (x["sil"] - v99["sil"], x["ch"] - v99["ch"], x["thin"] - v99["thin"])
    ok = d[0] >= -TOL[0] and d[1] <= TOL[1] and d[2] <= TOL[2]
    refs = [q for q in (v99, r99) if q is not None]
    ok2 = (x["sil"] >= min(q["sil"] for q in refs) - TOL[0] and x["ch"] <= max(q["ch"] for q in refs) + TOL[1]
           and x["thin"] <= max(q["thin"] for q in refs) + TOL[2])
    if w99 is None or x["v_end"] > REST * w99["v_end"]:
        v["rest"] = False
    v["n"] += 1
    v["held"] += int(ok)
    v["held_two"] += int(ok2)
    v["have_r"] += int(r99 is not None)
    if not ok:
        v["miss"].append(m)
    if not ok2:
        v["miss_two"].append(m)
    rows.append(f"| {m} | {f4(x)} | {f4(v99)} | {f4(r99)} | {f4(w99)} | {d[0]:+.4f} / {d[1]:+.4f} / {d[2]:+.3f} | "
                f"{'yes' if ok else 'NO'} | {'yes' if ok2 else 'NO'}{'' if r99 is not None else ' (V only)'} |")
two = (f"two-baseline {v['held_two']}/{v['n']}" + (f", misses {','.join(v['miss_two'])}" if v["miss_two"] else "")
       + ("" if v["have_r"] == v["n"] else f" (R 99 read on {v['have_r']}/{v['n']} meshes; the rest against V 99 alone)"))
verdict = (f"Hu seed 99 {'PASS' if v['held'] == v['n'] == 6 and v['rest'] else 'FAIL'} (holds {v['held']}/{v['n']} vs V 99"
           + (f", misses {','.join(v['miss'])}" if v["miss"] else "") + f"; {two}; rest {'ok' if v['rest'] else 'no'})")
L = ["# D172 B: Hu on the 40k gallery, seed 99 (job 191)", "", *rows, "",
     "Cells: sil_iou / chamfer / thin / end v_mean x1e3 / pieces >= 10. Hu = dF_c + layer u + render loss (full), held on u, rule "
     "on dF_c; V 99 = the tag's rule at seed 99 (d155 gate / d154); R 99 = the rule rerun on repo_d166 at seed 99 (the defaults; "
     "d167 job 114 ran R at seed 97 only); W 99 = the twin, dF_c + layer u (twin). Holds = within 0.003 / 0.002 / 0.02 of V 99; "
     "two-baseline = within the same tolerance of the worse of R 99 / V 99; rest = end v_mean <= 2x W 99. (tmp/d173/d173_gal99.py)",
     "", "VERDICT: " + verdict]
(D / "summary.md").write_text("\n".join(L) + "\n")
json.dump(dict(arms={"Hu_99": dict(v, status="PASS" if v["held"] == v["n"] == 6 and v["rest"] else "FAIL")}, verdict=verdict),
          open(D / "verdict.json", "w"), indent=1, default=float)
print(verdict)
