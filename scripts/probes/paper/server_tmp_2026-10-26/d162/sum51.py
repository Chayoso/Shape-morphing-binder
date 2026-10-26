"""Job 51 table: E x3 / x7 (settled 30 windows) vs job 50 (E x1, d162h); D161 reader share of the ceiling."""
import json, sys
import numpy as np
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d162")
from d162_auto import G, end_v
L = ["# Job 51: stiffer material under the held weight, settled 30 windows (40k, seed 97)", "",
     "| scene | E | relief share stop | settled | kept | corr end | v stop | v settled | twin v end | chamfer |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for sc in ("coin", "bunny"):
    vt, _, _ = end_v(f"{G}/d161/runs40/{sc}40k_twin_97")
    for k, rdp, lab, p in ((1, G / "d162h/after" / f"read_{sc}_Hs.json", "full97", f"{G}/d162h/runs/{sc}40k_Hs_97"),
                           (3, G / "d162s/after" / f"read_{sc}.json", "full3", f"{G}/d162s/runs/{sc}40k_E3_97"),
                           (7, G / "d162s/after" / f"read_{sc}.json", "full7", f"{G}/d162s/runs/{sc}40k_E7_97")):
        rd = json.load(open(rdp))
        ceil = np.mean([v["relief"]["A"] for kk, v in rd["refs"].items() if kk.startswith("relief_s")])
        v_end, com, _ = end_v(p)
        stop = [h for h in com if not h.get("settle")][-1]
        ser = rd["runs"][lab]["series"]
        raws = np.array([q["raw"] for q in ser]); A = np.array([q["relief"]["A"] for q in ser]) / ceil
        a0 = float(np.interp(stop["frame_end"], raws, A))
        ch = list(json.load(open(f"{p}.json"))["arms"].values())[0]["metrics"]["chamfer"]
        L.append(f"| {sc} | x{k} | {a0:.2f} | {A[-1]:.2f} | {A[-1] / a0:.2f} | {ser[-1][relief][corr]:.2f} | "
                 f"{stop[v_mean]:.2e} | {v_end:.2e} | {vt:.2e} | {ch:.4f} |")
print("\n".join(L))
