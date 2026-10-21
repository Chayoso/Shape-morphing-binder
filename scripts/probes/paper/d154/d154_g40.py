"""d154_g40.py OUT_DIR -- D154 follow-ups (3) / (3b): the 40k gallery against the exact mesh, 19 meshes (the five of
D155's gate re-runs, read/g40_*.jsonl, and the 14 of (3b)'s re-runs on repo_t1s, read/g40b_*.jsonl; both re-runs of B3's
V / W on the tag's default path; CPU, read-only). Per mesh and measure: V / W over the seeds (median, range), the share
of W's excess over the perfect sample that V removes, (W - V) / (W - own), and the paired tests: the sign count and the
two-sided Wilcoxon signed-rank test over the mesh-seed pairs and over the per-mesh medians; the count of meshes at
V / W <= 0.90 and > 0.95 (the pre-registered rule). Writes g40_table.json and prints the lines the entry quotes."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

O = Path("/data/relcfd/chayo/physmorph_v2/output/gpu/d154")
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(s) for f in sorted(glob.glob(str(O / "read/g40_*.jsonl")) + glob.glob(str(O / "read/g40b_*.jsonl"))) for s in open(f) if s.startswith("{")]
R = {(r["mesh"], r["seed"], r["label"]): r for r in rows}
MESHES = ["A", "armadilo", "beast", "bimba", "bob", "bunny", "C", "cheburashka", "cow", "dragon", "fandisk", "heart", "homer",
          "maxplanck", "nefertiti", "ogre", "spot", "teapot", "V"]          # (3): the five of D155's gate; (3b): the other 14
SEEDS = (97, 99, 101)
MEAS = [("outline_sd", "outline-shape error (outline offset spread, 1024 px), pitches", lambda r: r["exact_1024"]["outline_state_sd_a"]),
        ("iou1024", "1 - IoU, exact binary silhouettes, 1024 px", lambda r: r["exact_1024"]["one_minus_iou"]),
        ("iou96", "1 - IoU, exact binary silhouettes, 96 px", lambda r: r["exact_96"]["one_minus_iou"]),
        ("normal_cl", "normal angle at the nearest mesh point, deg", lambda r: r["dist"]["normal_closest_deg"]),
        ("yard_mesh_sil", "yardstick silhouette against exact-mesh discs, 96 px", lambda r: r["yard_mesh"]["sil"]),
        ("yard_mesh_pbr", "yardstick shading against exact-mesh discs, 96 px", lambda r: r["yard_mesh"]["pbr"]),
        ("shading", "headlight shading error, exact, 1024 px", lambda r: r["exact_1024"]["shading_mse"]),
        ("dilation", "mean offset outside the mesh, pitches (context)", lambda r: r["dist"]["state_to_mesh_signed_mean"])]


def rng(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(np.median(v)), float(np.min(v)), float(np.max(v))] if v else None


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    if len(d) < 2 or np.all(d == 0):
        return None
    return float(wilcoxon(a, b, alternative="two-sided").pvalue)


table = {}
for key, label, get in MEAS:
    per_mesh, pairs_v, pairs_w, med_v, med_w = {}, [], [], [], []
    for mesh in MESHES:
        per = []
        for s in SEEDS:
            try:
                V, W, own = (get(R[(mesh, s, a)]) for a in ("V", "W", "own"))
            except KeyError:
                continue
            per.append(dict(seed=s, V=V, W=W, own=own, ratio=V / W if W else None,
                            share=(W - V) / (W - own) if W != own else None))
            pairs_v.append(V)
            pairs_w.append(W)
        if not per:
            continue
        per_mesh[mesh] = dict(n=len(per), V=rng([p["V"] for p in per]), W=rng([p["W"] for p in per]),
                              own=rng([p["own"] for p in per]), ratio=rng([p["ratio"] for p in per]),
                              share=rng([p["share"] for p in per]), per_seed=per)
        med_v.append(per_mesh[mesh]["V"][0])
        med_w.append(per_mesh[mesh]["W"][0])
    table[key] = dict(label=label, meshes=per_mesh,
                      pairs=dict(n=len(pairs_v), v_lower=int(sum(v < w for v, w in zip(pairs_v, pairs_w))),
                                 wilcoxon_p=wil(pairs_v, pairs_w)),
                      mesh_medians=dict(n=len(med_v), v_lower=int(sum(v < w for v, w in zip(med_v, med_w))),
                                        wilcoxon_p=wil(med_v, med_w)),
                      ratio_le_090=int(sum(m["ratio"][0] <= 0.90 for m in per_mesh.values())),
                      ratio_gt_095=int(sum(m["ratio"][0] > 0.95 for m in per_mesh.values())))
json.dump(table, open(out / "g40_table.json", "w"), indent=1)
for key, d in table.items():
    print(f"== {key}: {d['label']}")
    for mesh, m in d["meshes"].items():
        print(f"  {mesh:9s} n{m['n']} V {m['V'][0]:.4g} W {m['W'][0]:.4g} own {m['own'][0]:.4g} | V/W {m['ratio'][0]:.3f} "
              f"({m['ratio'][1]:.3f}-{m['ratio'][2]:.3f}) | share {m['share'][0] if m['share'] else float('nan'):.2f} "
              f"({m['share'][1] if m['share'] else float('nan'):.2f}..{m['share'][2] if m['share'] else float('nan'):.2f})")
    p, q = d["pairs"], d["mesh_medians"]
    print(f"  pairs: V lower in {p['v_lower']} / {p['n']}, Wilcoxon p {p['wilcoxon_p']}; mesh medians: V lower in "
          f"{q['v_lower']} / {q['n']}, p {q['wilcoxon_p']}; meshes with V / W <= 0.90: {d['ratio_le_090']}, > 0.95: "
          f"{d['ratio_gt_095']}")
