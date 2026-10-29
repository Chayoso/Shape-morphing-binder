"""d154_tables.py OUT_DIR -- D154 (1)'s tables from output/gpu/d154/read/end_*.jsonl (CPU, read-only).

Per mesh and measure, over its seeds: the render arm L, the twin P, the run's own target sample (a perfect placement,
`own`) and D90's independent sample (`ind`), all read against the exact mesh; L / P, (P - L) / P and the share of the
twin's excess over the own sample that the arm removes, (P - L) / (P - own); medians and ranges over the seeds. The
yardstick as it is (against the independent sample) beside it, with D152's floor (own against ind). Mid-run: bunny
and dragon 97 at t = 1, 2, 3, 4, 5, 6, 8, 10 s. Writes table1.json, midrun.json and prints the lines the entry quotes."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

O = Path("/data/relcfd/chayo/physmorph_v2/output/gpu/d154")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(s) for f in sorted(glob.glob(str(O / "read/end_*.jsonl"))) for s in open(f) if s.startswith("{")]
MESHES = ["bunny", "dragon", "teapot", "bob", "armadilo", "homer"]
# (key, label, getter); every one an error (smaller = nearer the mesh) but `dilation`
MEAS = [
    ("yard_ind_sil", "yardstick exterior silhouette vs independent sample (96 px)", lambda r: r["yard_ind"]["sil"]),
    ("yard_ind_pbr", "yardstick exterior shading vs independent sample (96 px)", lambda r: r["yard_ind"]["pbr"]),
    ("yard_mesh_sil", "yardstick exterior silhouette vs exact-mesh discs (96 px)", lambda r: r["yard_mesh"]["sil"]),
    ("yard_mesh_pbr", "yardstick exterior shading vs exact-mesh discs (96 px)", lambda r: r["yard_mesh"]["pbr"]),
    ("iou96", "1 - IoU vs the mesh, exact, 96 px", lambda r: r["exact_96"]["one_minus_iou"]),
    ("iou1024", "1 - IoU vs the mesh, exact, 1024 px", lambda r: r["exact_1024"]["one_minus_iou"]),
    ("outline_sd", "outline offset spread (state's outline on the mesh's), 1024 px, pitches",
     lambda r: r["exact_1024"]["outline_state_sd_a"]),
    ("outline_mesh_sd", "outline offset spread (mesh's outline on the state's), 1024 px, pitches",
     lambda r: r["exact_1024"]["outline_mesh_sd_a"]),
    ("normal_px", "normal angle at the same pixel, 1024 px, deg", lambda r: r["exact_1024"]["normal_deg"]),
    ("shading", "headlight shading error (mean square), 1024 px", lambda r: r["exact_1024"]["shading_mse"]),
    ("dist_sd", "surface distance spread, state to mesh, pitches", lambda r: r["dist"]["state_to_mesh_signed_sd"]),
    ("dist_mesh_sd", "surface distance spread, mesh to state, pitches", lambda r: r["dist"]["mesh_to_state_signed_sd"]),
    ("normal_cl", "normal angle at the nearest mesh point, deg", lambda r: r["dist"]["normal_closest_deg"]),
    ("dilation", "mean offset outside the mesh (state's visible surface), pitches",
     lambda r: r["dist"]["state_to_mesh_signed_mean"]),
    ("outline_dilation", "mean outline offset outside the mesh's, 1024 px, pitches",
     lambda r: r["exact_1024"]["outline_state_mean_a"]),
]
end = {(r["mesh"], r["seed"], r["label"]): r for r in rows if r["raw"] in ("end", "tgt", "ind")}
mid = {(r["mesh"], r["seed"], r["label"]): r for r in rows if "@" in r["label"]}


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return (float(np.median(v)), float(np.min(v)), float(np.max(v))) if v else None


table = {}
for mesh in MESHES:
    seeds = sorted({s for (m, s, _) in end if m == mesh})
    table[mesh] = {}
    for key, label, get in MEAS:
        per = []
        for s in seeds:
            try:
                L, P, own, ind = (get(end[(mesh, s, a)]) for a in ("L", "P", "own", "ind"))
            except KeyError:
                continue
            row = dict(seed=s, L=L, P=P, own=own, ind=ind, ratio=L / P if P else None, rel=(P - L) / P if P else None,
                       share=(P - L) / (P - own) if P != own else None, L_over_own=L / own if own else None,
                       P_over_own=P / own if own else None)
            if key.startswith("yard_ind"):             # D152's floor: the run's own sample against the independent one
                row.update(share=(P - L) / (P - own) if P != own else None)
            per.append(row)
        table[mesh][key] = dict(label=label, n=len(per), **{k: med([p[k] for p in per]) for k in
                                ("L", "P", "own", "ind", "ratio", "rel", "share", "L_over_own", "P_over_own")},
                                per_seed=per)
json.dump(table, open(out / "table1.json", "w"), indent=1)

midrun = {}
for mesh in ("bunny", "dragon"):
    midrun[mesh] = {}
    for t in (1, 2, 3, 4, 5, 6, 8, 10):
        try:
            L, P = mid[(mesh, 97, f"L@{t}")], mid[(mesh, 97, f"P@{t}")]
        except KeyError:
            continue
        own = end.get((mesh, 97, "own"))
        midrun[mesh][t] = {key: dict(L=get(L), P=get(P), own=get(own) if own else None,
                                     rel=(get(P) - get(L)) / get(P) if get(P) else None) for key, _, get in MEAS}
    end97 = {key: dict(L=get(end[(mesh, 97, "L")]), P=get(end[(mesh, 97, "P")]), own=get(end[(mesh, 97, "own")]))
             for key, _, get in MEAS} if (mesh, 97, "L") in end else {}
    for v in end97.values():
        v["rel"] = (v["P"] - v["L"]) / v["P"] if v["P"] else None
    midrun[mesh]["end"] = end97
json.dump(midrun, open(out / "midrun.json", "w"), indent=1)


def f(v, p=4):
    return "nan" if v is None else f"{v:.{p}g}"


for key, label, _ in MEAS:
    print(f"== {key}: {label}")
    for mesh in MESHES:
        d = table[mesh].get(key)
        if not d or not d["n"]:
            continue
        print(f"  {mesh:9s} n{d['n']} L {f(d['L'][0])} P {f(d['P'][0])} own {f(d['own'][0])} ind {f(d['ind'][0])} | "
              f"L/P {f(d['ratio'][0], 3)} ({f(d['ratio'][1], 3)}-{f(d['ratio'][2], 3)}) rel {f(d['rel'][0], 3)} | share "
              f"{f(d['share'][0] if d['share'] else None, 3)} ({f(d['share'][1] if d['share'] else None, 3)}.."
              f"{f(d['share'][2] if d['share'] else None, 3)}) | L/own {f(d['L_over_own'][0], 3)} P/own {f(d['P_over_own'][0], 3)}")
for mesh, d in midrun.items():
    for t, v in d.items():
        print(f"MID {mesh} t={t}: " + "  ".join(f"{k} {f(v[k]['rel'], 3)}" for k in
                                               ("yard_ind_sil", "yard_ind_pbr", "iou1024", "outline_sd", "normal_cl",
                                                "shading", "dist_sd") if k in v))
