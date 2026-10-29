"""d173_band300sum.py OUT_DIR RAW -- D173: the band at 300k, seed 99 summary (after_band300s99.sh): per arm (Rx, Rxn) the
optimisation windows and the stop that ended them (the run log: the physics convergence or the delivery-merit plateau),
colour r at the end (D160's reading, read_colour.jsonl), the particle edge (d173_edge2.py, logistic, pitches) at the end
and at RAW (the shorter run's stop), the block shares; seed 97's numbers beside them. Writes OUT_DIR/summary.md."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/d173")
import d173_shares as SH  # noqa: E402

D, st = Path(sys.argv[1]), sys.argv[2]
O = Path("/data/relcfd/chayo/physmorph_v2/output/gpu/d173")
col = {json.loads(x)["name"]: json.loads(x) for x in open(D / "read_colour.jsonl") if x.strip()}
edg = {json.loads(x)["name"]: json.loads(x) for x in open(D / "edge.jsonl") if x.strip()}
L = ["# D173: the band at 300k, seed 99 (twin, Rx, Rx + nearest colour)", "",
     f"| arm | optimisation windows / the stop | colour r end | particle edge end (pitches) | particle edge at raw {st} | "
     + " | ".join(SH.HEAD) + " |", "|---|---|---|---|---|---|---|"]
mt = col["band300k_T_99"]["end"]["mis"]
V = {}
for a in ("Rx", "Rxn"):
    nm = f"band300k_{a}_99"
    c, sh = col[nm], SH.block_shares(str(O / "runs" / f"{nm}.json"))
    log = open(O / "runs" / f"{nm}.log").read()
    why = "physics convergence" if "converged at anim" in log else ("delivery-merit plateau" if "merit plateau" in log else "other")
    r = (mt - c["end"]["mis"]) / max(mt - c["floor"], 1e-12)
    e_end, e_st = edg[nm]["particles"]["logistic"], edg[f"{nm}@{st}"]["particles"]["logistic"]
    V[a] = dict(windows=sh["windows"], stop=why, r=r, edge_end=e_end, edge_at=e_st)
    L.append(f"| {a} | {sh['windows']} / {why} | {r:.2f} | {e_end:.2f} | {e_st:.2f} | {SH.fmt3(sh, 'u')} | {SH.fmt3(sh, 'dfc')} |")
v = V["Rxn"]
verdict = (f"Rxn s99: edge {v['edge_end']:.2f} pitches at the end (Rx {V['Rx']['edge_end']:.2f}), at the same time {v['edge_at']:.2f} "
           f"(Rx {V['Rx']['edge_at']:.2f}); r {v['r']:.2f} (Rx {V['Rx']['r']:.2f}); stops: Rxn {v['stop']} at {v['windows']}, "
           f"Rx {V['Rx']['stop']} at {V['Rx']['windows']}")
L += ["", "Seed 97 (jobs 205 / 209 and d170): Rx 82 windows (physics convergence), edge 2.68, r 0.92; Rxn 42 windows "
      "(delivery-merit plateau), edge 2.93, r 0.83; at raw 1680 Rx 4.28, Rxn 2.94. " + SH.NOTE, "", "VERDICT: " + verdict]
(D / "summary.md").write_text("\n".join(L) + "\n")
print(verdict)
