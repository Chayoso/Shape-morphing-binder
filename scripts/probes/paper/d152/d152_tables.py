"""d152_tables.py OUT_DIR -- D152's tables from the tag's B1 readings (CPU only, read-only; measurement only).
H1  error forms against floors: every end measure of B1 (output/gpu/d145t_evidence/tab_tx.jsonl) for L (render arm) and
    P (lambda = 0 twin) with its floor: silhouette IoU against the independent sample (endgen's floor: the run's own
    target sample against the independent one), the yardstick (d152_yard_floor.py's `own` rows), the display
    (e3end_*.log's `own` row), chamfer (own against independent, here); per seed: error ratio L / P, relative change
    (P - L) / P and the share of the twin's excess over the floor that the arm removes, (P - L) / (P - floor).
H2  the render term's own floor (d152_own_floor.py: the independent sample against the run's pictures) against the
    ends of L and P and the windows' records (d_sil, d_pbr of every committed window of L and P).
H5  PCGrad from the records: windows with g_raw_cos < 0, the removed share cos^2, the cosine early / late, the two
    gradients' norms (first committed window against the last ten).
40k the pre-tag 40k gallery (output/gpu/d145_b3/gal_V_W.json) in error form."""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
B1 = G / "d145t_b1"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
MESHES = ["bunny", "dragon", "teapot", "bob", "armadilo", "homer"]


def jl(path):
    try:
        return [json.loads(s) for s in open(path) if s.startswith("{")]
    except OSError:
        return []


tab = {r["name"]: r for r in jl(G / "d145t_evidence/tab_tx.jsonl") if r.get("dir") == "d145t_b1"}
yard = {(r["mesh"], r["seed"], r["state"]): r for r in jl(G / "d152/yard_floor.jsonl")}
own = {(r["mesh"], r["seed"], r["state"]): r for r in jl(G / "d152/own_floor.jsonl")}


def disp_own(name):
    for r in jl(B1 / f"e3end_{name}.log"):
        if r.get("state") == "own":
            return r["to_target"]
    return None


def seeds_of(mesh):
    return sorted({int(re.search(r"[LP](\d+)$", n).group(1)) for n in tab if n.startswith(mesh + "300k_")})


chamf_floor = {}


def chamfer_floor(mesh, seed):
    if (mesh, seed) not in chamf_floor:
        z = np.load(B1 / f"{mesh}300k_L{seed}_end1.npz")
        a = np.asarray(z["tgt"], np.float64)
        b = np.asarray(np.load(G / f"d90/{mesh}_ind300k.npz")["tgt"], np.float64)
        chamf_floor[(mesh, seed)] = float(cKDTree(b).query(a, 1)[0].mean() + cKDTree(a).query(b, 1)[0].mean())
    return chamf_floor[(mesh, seed)]


# ---------- H1 ----------
MEAS = [  # key, label, how to get (arm row -> error), floor getter
    ("sil_iou_ind", "silhouette IoU vs independent sample (as 1 - IoU)"),
    ("chamfer_ind", "chamfer vs independent sample"),
    ("yard_ext_sil", "yardstick exterior silhouette (96 px)"),
    ("yard_ext_pbr", "yardstick exterior shading (96 px)"),
    ("yard_par_sil", "yardstick particle silhouette (96 px)"),
    ("disp_front", "display front 1 - IoU"),
    ("disp_back", "display back 1 - IoU"),
    ("disp_front_diff", "display front image difference"),
]


def err(name, key):
    r = tab.get(name)
    if r is None:
        return None
    return {"sil_iou_ind": (1 - r["sil_ind"]) if r.get("sil_ind") is not None else None,
            "chamfer_ind": r.get("chamfer_ind"), "yard_ext_sil": r.get("yard_ext_sil"),
            "yard_ext_pbr": r.get("yard_ext_pbr"), "yard_par_sil": r.get("yard_par_sil"),
            "disp_front": r.get("disp_front_1miou"), "disp_back": r.get("disp_back_1miou"),
            "disp_front_diff": r.get("disp_front_diff")}[key]


def floor(mesh, seed, key):
    r = tab.get(f"{mesh}300k_L{seed}")
    if key == "sil_iou_ind":
        return (1 - r["floor_sil"]) if r and r.get("floor_sil") is not None else None
    if key == "chamfer_ind":
        return chamfer_floor(mesh, seed)
    if key.startswith("yard_"):
        y = yard.get((mesh, seed, "own"))
        if not y:
            return None
        return {"yard_ext_sil": y["exterior"]["sil"], "yard_ext_pbr": y["exterior"]["pbr"],
                "yard_par_sil": y["particles"]["sil"]}[key]
    o = disp_own(f"{mesh}300k_L{seed}")
    if not o:
        return None
    return {"disp_front": 1 - o["front"]["iou"], "disp_back": 1 - o["back"]["iou"],
            "disp_front_diff": o["front"]["difference"]}[key]


h1 = {}
for mesh in MESHES:
    h1[mesh] = {}
    for key, label in MEAS:
        per = []
        for s in seeds_of(mesh):
            eL, eP, f = err(f"{mesh}300k_L{s}", key), err(f"{mesh}300k_P{s}", key), floor(mesh, s, key)
            if eL is None or eP is None:
                continue
            row = {"seed": s, "L": eL, "P": eP, "floor": f, "ratio": eL / eP, "rel": (eP - eL) / eP}
            if f is not None and eP - f != 0:
                row["gap_share"] = (eP - eL) / (eP - f)
                row["P_over_floor"] = eP / f
                row["L_over_floor"] = eL / f
            per.append(row)
        if not per:
            continue

        def med(k):
            v = [p[k] for p in per if p.get(k) is not None]
            return (float(np.median(v)), float(np.min(v)), float(np.max(v))) if v else None

        h1[mesh][key] = {"label": label, "n": len(per), "L": med("L"), "P": med("P"), "floor": med("floor"),
                         "ratio": med("ratio"), "rel": med("rel"), "gap_share": med("gap_share"),
                         "P_over_floor": med("P_over_floor"), "L_over_floor": med("L_over_floor"), "per_seed": per}
json.dump(h1, open(out / "h1_table.json", "w"), indent=1)

# ---------- H2 + the windows' render terms ----------
h2 = {}
series = {}
for mesh in MESHES:
    for s in seeds_of(mesh):
        rec = {}
        for arm in "LP":
            try:
                J = json.loads((B1 / f"{mesh}300k_{arm}{s}.json").read_text())
            except OSError:
                continue
            hist = next(iter(J["arms"].values()))["history"]
            com = [h for h in hist if h.get("outer_accepted") and h.get("d_sil") is not None and h.get("frame_end")]
            series[f"{mesh}_{arm}{s}"] = {"t": [h["frame_end"] / 240.0 for h in com], "k": [h["animation"] for h in com],
                                          "d_sil": [h["d_sil"] for h in com], "d_pbr": [h["d_pbr"] for h in com],
                                          "g_raw_cos": [h.get("g_raw_cos") for h in com],
                                          "g_cos": [h.get("g_cos") for h in com],
                                          "g_phys_norm": [h.get("g_phys_norm") for h in com],
                                          "g_rend_norm": [h.get("g_rend_norm") for h in com],
                                          "lambda": [h.get("lambda") for h in com], "g_share": [h.get("g_share") for h in com],
                                          "dfc_absmax": [h.get("dfc_absmax") for h in com],
                                          "transport_energy": [h.get("transport_energy") for h in com]}
            rec[arm] = com
        fl = own.get((mesh, s, "ind"))
        if not fl:
            continue
        row = {"floor_sil": fl["sil"], "floor_pbr": fl["pbr"], "floor_render": fl["render"], "floor_TE": fl.get("TE")}
        for st in ("own", "L", "P"):
            r = own.get((mesh, s, st))
            if r:
                row[f"{st}_sil"], row[f"{st}_pbr"], row[f"{st}_render"], row[f"{st}_TE"] = r["sil"], r["pbr"], r["render"], r.get("TE")
                if st in "LP":
                    row[f"{st}_rec_d_sil"] = r.get("rec_d_sil")
        for t in ("sil", "pbr", "render"):
            if f"L_{t}" in row and f"P_{t}" in row:
                row[f"share_removed_{t}"] = (row[f"P_{t}"] - row[f"L_{t}"]) / (row[f"P_{t}"] - row[f"floor_{t}"])
                row[f"L_over_floor_{t}"] = row[f"L_{t}"] / row[f"floor_{t}"]
                row[f"P_over_floor_{t}"] = row[f"P_{t}"] / row[f"floor_{t}"]
        for arm in "LP":
            com = rec.get(arm, [])
            below = [h for h in com if h["d_sil"] <= row["floor_sil"]]
            row[f"{arm}_first_below_floor_t"] = below[0]["frame_end"] / 240.0 if below else None
            row[f"{arm}_first_below_floor_k"] = below[0]["animation"] if below else None
            row[f"{arm}_share_windows_below"] = len(below) / max(len(com), 1)
            row[f"{arm}_end_t"] = com[-1]["frame_end"] / 240.0 if com else None
        h2[f"{mesh}_{s}"] = row
json.dump(h2, open(out / "h2_table.json", "w"), indent=1)
json.dump(series, open(out / "window_series.json", "w"))

# ---------- H5 ----------
h5 = {}
for name, sr in series.items():
    if "_L" not in name:
        continue
    c = np.array([v for v in sr["g_raw_cos"] if v is not None], float)
    gp = np.array([v for v in sr["g_phys_norm"] if v is not None], float)
    gr = np.array([v for v in sr["g_rend_norm"] if v is not None], float)
    lam = np.array([v for v in sr["lambda"] if v is not None], float)
    t = np.array(sr["t"], float)
    neg = c < 0
    h5[name] = {"windows": int(len(c)), "pcgrad_active": int(neg.sum()), "pcgrad_active_share": float(neg.mean()),
                "removed_sq_share_median_active": float(np.median(c[neg] ** 2)) if neg.any() else 0.0,
                "removed_sq_share_max": float((c[neg] ** 2).max()) if neg.any() else 0.0,
                "removed_sq_share_mean_all": float(np.where(neg, c ** 2, 0).mean()),
                "cos_median_first10": float(np.median(c[:10])), "cos_median_last_half": float(np.median(c[len(c) // 2:])),
                "cos_min": float(c.min()),
                "gphys_first": float(gp[0]), "gphys_last10": float(np.median(gp[-10:])),
                "grend_first": float(gr[0]), "grend_last10": float(np.median(gr[-10:])),
                "gphys_decay": float(gp[0] / np.median(gp[-10:])), "grend_decay": float(gr[0] / np.median(gr[-10:])),
                "lam_first": float(lam[0]), "lam_last10": float(np.median(lam[-10:])),
                "dfc_absmax_at_cap_share": float(np.mean([v >= 0.0199 for v in sr["dfc_absmax"] if v is not None])),
                "t_end": float(t[-1])}
json.dump(h5, open(out / "h5_table.json", "w"), indent=1)

# ---------- the 40k gallery (pre-tag) in error form ----------
g40 = []
try:
    for r in json.load(open(G / "d145_b3/gal_V_W.json")):
        eV = 1 - np.mean(r["A"])
        eW = 1 - np.mean(r["B"])
        ef = 1 - r["floor"]
        g40.append({"mesh": r["mesh"], "err_V": eV, "err_W": eW, "err_floor": ef, "rel": (eW - eV) / eW,
                    "gap_share": (eW - eV) / (eW - ef), "W_excess_over_floor_share_of_err": (eW - ef) / eW})
except OSError:
    pass
json.dump(g40, open(out / "g40_error_form.json", "w"), indent=1)
if g40:
    rel = [r["rel"] for r in g40]
    gs = [r["gap_share"] for r in g40]
    ex = [r["W_excess_over_floor_share_of_err"] for r in g40]
    print(f"40k: relative error reduction median {np.median(rel):.3f} ({min(rel):.3f}-{max(rel):.3f}); gap share median "
          f"{np.median(gs):.3f}; the twin's excess over the floor is {np.median(ex):.3f} of its error (median)")

# ---------- print ----------
for mesh, d in h1.items():
    for key, v in d.items():
        f = v["floor"]
        print(f"H1 {mesh:9s} {key:16s} L {v['L'][0]:.4g} P {v['P'][0]:.4g} floor {f[0] if f else float('nan'):.4g} "
              f"L/P {v['ratio'][0]:.3f} rel {v['rel'][0]:+.3f} gap share "
              f"{v['gap_share'][0] if v['gap_share'] else float('nan'):+.3f} "
              f"({v['gap_share'][1] if v['gap_share'] else float('nan'):+.2f}..{v['gap_share'][2] if v['gap_share'] else float('nan'):+.2f}) n {v['n']}")
for k, r in h2.items():
    print("H2", k, {a: (round(b, 6) if isinstance(b, float) else b) for a, b in r.items()})
for k, r in h5.items():
    print("H5", k, {a: (round(b, 5) if isinstance(b, float) else b) for a, b in r.items()})
