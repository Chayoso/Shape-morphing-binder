"""d157_table.py OUT_DIR -- D157's reading: per metric and N the arm / twin / own values, the paired tests over the
mesh-seed pairs, the per-mesh consistency and the pre-registered rule; (4a) the offset by region; (4b) the bias.

Error form e: 1 - score (IoU, BIoU, F, NC, SSIM, recall), the MSE for PSNR, the value otherwise. r = (e_twin - e_arm) /
e_twin per pair (> 0: the arm better). Sees the render at an N: Wilcoxon p < 0.01 over the pairs, the per-mesh median r
of one sign on >= 5 of 6 meshes (300k) / >= 16 of 19 (40k), |median r| >= 0.05. Writes OUT_DIR/table.json, table.txt."""
import glob
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, wilcoxon

D = "/data/relcfd/chayo/physmorph_v2/output/gpu/d157"
G = "/data/relcfd/chayo/physmorph_v2/output/gpu"
out_dir = sys.argv[1]
ARM, TWIN = {"300k": "L", "40k": "V"}, {"300k": "P", "40k": "W"}
NEED = {"300k": 5, "40k": 16}


def heldout_far(n=64, cap=12.0, far=25.0):
    def dvec(th, ph):
        return np.array([np.sin(th) * np.cos(ph), np.sin(ph), np.cos(th) * np.cos(ph)])
    tr = []
    for i, ph in enumerate((0.0, 0.5, -0.5)):
        off = (i / 3) * (2 * np.pi / 6)
        for t in np.linspace(0, 2 * np.pi, 6, endpoint=False):
            tr.append(dvec(t + off, ph))
    tr = np.array(tr)
    i = np.arange(n) + 0.5
    pol = np.arccos(1 - 2 * i / n)
    az = np.pi * (1 + 5 ** 0.5) * i
    P = np.stack([np.cos(az) * np.sin(pol), np.cos(pol), np.sin(az) * np.sin(pol)], 1)
    ang = np.degrees(np.arccos(np.clip(np.abs(P @ tr.T), 0, 1))).min(1)
    keep = ang > cap
    return ang[keep] > far


FAR = heldout_far()
rows = defaultdict(dict)
for f in sorted(glob.glob(f"{D}/read300_*.jsonl") + glob.glob(f"{D}/read40_*.jsonl")):
    for ln in open(f):
        r = json.loads(ln)
        rows[(r["n"], r["mesh"], r["seed"])][r["label"]] = r


def run_json(r):
    npz = r["npz"]
    if r["n"] == "300k":
        p = f"{G}/d145t_b1/{npz}"
    elif "frames12" in npz:
        p = f"{G}/d155/gate/{npz}"
    else:
        p = f"{G}/d154/g40runs/{npz}"
    js = json.load(open(p.replace("_end1.npz", ".json").replace("_frames12.npz", ".json")))
    return js["arms"][next(iter(js["arms"]))]["metrics"]


_RUN = {}


def runm(r, k):
    key = (r["n"], r["mesh"], r["seed"], r["label"])
    if key not in _RUN:
        _RUN[key] = run_json(r)
    return _RUN[key][k]


def metric_defs():
    M = {}
    sil = ("iou_err", "biou2pct_err", "biou1a_err", "bfdavis_err", "bf05a_err", "bf1a_err", "chamfer_a", "hd_a",
           "hd95_a")
    for surf in ("zb", "zbs"):
        for ref in ("raw", "cor"):
            for res in ("256", "512", "1024"):
                for vs in ("train", "held", "far"):
                    for m in sil:
                        M[f"sil/{surf}/{ref}/{res}/{vs}/{m}"] = (lambda r, s=surf, f=ref, x=res, v=vs, k=m:
                                                                  r["sil"][s][f][x][v][k])
            for m in ("f0.25", "f0.5", "f1", "f2", "nc", "thin_rec1", "thin_rec05"):
                M[f"d3/{surf}/{ref}/{m}"] = (lambda r, s=surf, f=ref, k=m: 1 - r["d3"][s][f][k])
            for m in ("chamfer_l1_a", "acc_a", "comp_a"):
                M[f"d3/{surf}/{ref}/{m}"] = (lambda r, s=surf, f=ref, k=m: r["d3"][s][f][k])
    for ref in ("raw", "cor"):
        for kind in ("shaded", "normal"):
            for vs, sel in (("held", np.ones(len(FAR), bool)), ("far", FAR)):
                M[f"img/{ref}/{kind}/{vs}/mse"] = (lambda r, f=ref, k=kind, s=sel:
                                                   float(np.mean(np.array(r["img"][f][k]["mse"])[s])))
                M[f"img/{ref}/{kind}/{vs}/ssim"] = (lambda r, f=ref, k=kind, s=sel:
                                                    1 - float(np.mean(np.array(r["img"][f][k]["ssim"])[s])))
                M[f"img/{ref}/{kind}/{vs}/lpips"] = (lambda r, f=ref, k=kind, s=sel:
                                                     float(np.mean(np.array(r["img"][f][k]["lpips"])[s])))
        M[f"img/{ref}/angle/held/deg"] = (lambda r, f=ref: float(np.nanmean(r["img"][f]["normal_angle_deg"])))
    M["run/sil_iou"] = lambda r: 1 - runm(r, "sil_iou")
    M["run/thin_uncovered"] = lambda r: runm(r, "thin_uncovered")
    M["run/chamfer"] = lambda r: runm(r, "chamfer")
    M["d154/pooled_iou_train1024_raw"] = (lambda r: 1 - sum(r["sil_views"]["raw"]["inter_px"][:18])
                                          / sum(r["sil_views"]["raw"]["union_px"][:18]))
    return M


DECISION = ["sil/zb/cor/1024/held/biou2pct_err", "sil/zb/cor/1024/held/bfdavis_err", "sil/zb/cor/1024/held/chamfer_a",
            "sil/zb/cor/1024/held/hd95_a", "d3/zb/cor/f0.5", "d3/zb/cor/f1", "d3/zb/cor/nc", "d3/zb/cor/chamfer_l1_a",
            "img/cor/shaded/held/mse", "img/cor/shaded/held/ssim", "img/cor/shaded/held/lpips",
            "img/cor/normal/held/mse", "img/cor/normal/held/ssim", "img/cor/normal/held/lpips"]


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def read(M, name, N):
    keys = sorted(k for k in rows if k[0] == N and ARM[N] in rows[k] and TWIN[N] in rows[k])
    if not keys:
        return None
    fn = M[name]
    ea, et, eo, rr, meshes = [], [], [], [], []
    for k in keys:
        try:
            a, t = fn(rows[k][ARM[N]]), fn(rows[k][TWIN[N]])
        except (KeyError, TypeError, ZeroDivisionError):
            continue
        o = fn(rows[k]["own"]) if "own" in rows[k] else np.nan
        ea.append(a); et.append(t); eo.append(o); meshes.append(k[1])
        rr.append((t - a) / t if t != 0 else np.nan)
    if len(ea) < 3:
        return None
    ea, et, eo, rr = map(np.array, (ea, et, eo, rr))
    d = et - ea
    try:
        p = float(wilcoxon(ea, et, alternative="two-sided").pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    better = int((d > 0).sum())
    ps = float(binomtest(better, int((d != 0).sum()), 0.5).pvalue) if np.any(d != 0) else 1.0
    pm = defaultdict(list)
    for m, x in zip(meshes, rr):
        pm[m].append(x)
    med = {m: float(np.median(v)) for m, v in pm.items()}
    mv = np.array(list(med.values()))
    pos, neg = int((mv > 0).sum()), int((mv < 0).sum())
    medr = float(np.nanmedian(rr))
    total = {"300k": 6, "40k": 19}[N]                         # partial data (while reading): the rule pro rata
    need = NEED[N] if len(med) == total else int(np.ceil(len(med) * NEED[N] / total))
    one_sign = max(pos, neg) >= need
    gap = (et - ea) / np.where(np.abs(et - eo) > 0, et - eo, np.nan)
    sees = bool(p < 0.01 and one_sign and abs(medr) >= 0.05 and np.sign(medr) == np.sign(pos - neg))
    extra = {}
    if name.endswith("/mse"):                                  # PSNR: the paired gain in dB, and the means
        db = 10 * np.log10(et / ea)
        extra = dict(psnr_gain_db_med=float(np.median(db)), psnr_gain_db_q25=float(np.percentile(db, 25)),
                     psnr_gain_db_q75=float(np.percentile(db, 75)), psnr_arm_med=float(np.median(-10 * np.log10(ea))),
                     psnr_twin_med=float(np.median(-10 * np.log10(et))))
    return dict(metric=name, n=N, pairs=len(ea), meshes=len(med), **extra, arm_med=float(np.median(ea)),
                arm_range=[float(ea.min()), float(ea.max())], twin_med=float(np.median(et)),
                twin_range=[float(et.min()), float(et.max())], own_med=float(np.nanmedian(eo)),
                r_med=medr, r_q25=float(np.nanpercentile(rr, 25)), r_q75=float(np.nanpercentile(rr, 75)),
                arm_better=better, wilcoxon_p=p, sign_p=ps, mesh_pos=pos, mesh_neg=neg, per_mesh_r=med,
                gap_closed_med=float(np.nanmedian(gap)), sees=sees,
                direction=("gain" if medr > 0 else "loss") if sees else None)


def main():
    M = metric_defs()
    res = {}
    for N in ("300k", "40k"):
        for name in M:
            x = read(M, name, N)
            if x is not None:
                res[f"{N}:{name}"] = x
        dec = [f"{N}:{m}" for m in DECISION if f"{N}:{m}" in res]
        for k, h in zip(dec, holm([res[k]["wilcoxon_p"] for k in dec])):
            res[k]["holm_p_decision"] = float(h)
    # (4a) the offset by region, arm - twin, pitches and the run's render pixels
    reg = {}
    for N in ("300k", "40k"):
        keys = sorted(k for k in rows if k[0] == N and ARM[N] in rows[k] and TWIN[N] in rows[k])
        names = ["crevice", "rim", "headon"] + [f"bin{i}" for i in range(6)] + ["all"]
        per = defaultdict(list)
        for k in keys:
            A, T, O = rows[k][ARM[N]], rows[k][TWIN[N]], rows[k].get("own")
            px_a = (2 * A["ctx"]["extent"] / {"300k": 188, "40k": 96}[N]) / A["ctx"]["pitch"]
            for nm in names:
                if nm == "all":
                    va, vt = A["d3"]["zb"]["raw"]["signed_mean_a"], T["d3"]["zb"]["raw"]["signed_mean_a"]
                    vo = O["d3"]["zb"]["raw"]["signed_mean_a"] if O else np.nan
                    sa = st = 1.0
                else:
                    ra, rt = A["region"]["zb"].get(nm), T["region"]["zb"].get(nm)
                    if ra is None or rt is None or ra["offset_a"] is None or rt["offset_a"] is None:
                        continue
                    va, vt, sa, st = ra["offset_a"], rt["offset_a"], ra["share"], rt["share"]
                    ro = O["region"]["zb"].get(nm) if O else None
                    vo = ro["offset_a"] if ro and ro["offset_a"] is not None else np.nan
                per[nm].append(dict(mesh=k[1], seed=k[2], arm=va, twin=vt, own=vo, d=va - vt, d_px=(va - vt) / px_a,
                                    share_arm=sa, share_twin=st))
        out = {}
        for nm, L in per.items():
            d = np.array([x["d"] for x in L])
            p = float(wilcoxon(d).pvalue) if len(d) > 2 and np.any(d != 0) else 1.0
            out[nm] = dict(pairs=len(d), d_med=float(np.median(d)), d_q25=float(np.percentile(d, 25)),
                           d_q75=float(np.percentile(d, 75)), outward=int((d > 0).sum()), p=p,
                           d_px_med=float(np.median([x["d_px"] for x in L])),
                           arm_med=float(np.median([x["arm"] for x in L])), twin_med=float(np.median([x["twin"] for x in L])),
                           own_med=float(np.nanmedian([x["own"] for x in L])),
                           share_med=float(np.median([x["share_arm"] for x in L])), rows=L)
        # rim minus head-on, per pair
        if "rim" in per and "headon" in per:
            dr = {(x["mesh"], x["seed"]): x["d"] for x in per["rim"]}
            dh = {(x["mesh"], x["seed"]): x["d"] for x in per["headon"]}
            both = [k for k in dr if k in dh]
            diff = np.array([dr[k] - dh[k] for k in both])
            out["rim_minus_headon"] = dict(pairs=len(diff), med=float(np.median(diff)), pos=int((diff > 0).sum()),
                                           p=float(wilcoxon(diff).pvalue) if len(diff) > 2 else 1.0)
        reg[N] = out
    # (4b) the bias
    bias = []
    for f in sorted(glob.glob(f"{D}/bias*.jsonl")):
        for ln in open(f):
            b = json.loads(ln)
            k = (b["n"], b["mesh"], b["seed"])
            obs = None
            if k in rows and ARM[b["n"]] in rows[k] and TWIN[b["n"]] in rows[k]:
                obs = (rows[k][ARM[b["n"]]]["d3"]["zb"]["raw"]["signed_mean_a"]
                       - rows[k][TWIN[b["n"]]]["d3"]["zb"]["raw"]["signed_mean_a"])
            for res_, rr in b["res"].items():
                st = rr["states"]
                bias.append(dict(n=b["n"], mesh=b["mesh"], seed=b["seed"], res=int(res_), run_res=b["run_res"],
                                 draws=b["draws"], h_a=rr["h_a"], px_a=rr["px_a"], edge_mean_a=rr["edge_mean_a"],
                                 edge_single_a=rr["edge_single_a"], obs_arm_minus_twin=obs,
                                 **{f"{s}_{q}": v[q] for s, v in st.items() for q in ("argmin21", "argmin11", "bias",
                                                                                     "at_edge")},
                                 tgt_reproduced=b["tgt_reproduced_maxdiff"]))
    json.dump(dict(metrics=res, region=reg, bias=bias), open(f"{out_dir}/table.json", "w"), indent=1)
    with open(f"{out_dir}/table.txt", "w") as t:
        for N in ("300k", "40k"):
            t.write(f"=== {N}: decision set (held-out views at 1024 px, corrected referee; 3D on the exterior)\n")
            for m in DECISION:
                x = res.get(f"{N}:{m}")
                if x:
                    t.write(fmt(x) + "\n")
            t.write(f"--- {N}: all metrics\n")
            for k, x in res.items():
                if x["n"] == N:
                    t.write(fmt(x) + "\n")
        for N, out in reg.items():
            t.write(f"=== (4a) {N}: arm - twin signed offset (pitches) by region\n")
            for nm, x in out.items():
                t.write(f"{nm:18s} " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                   for k, v in x.items() if k != "rows"}) + "\n")
        t.write("=== (4b) bias rows\n")
        for b in bias:
            t.write(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()}) + "\n")


def fmt(x):
    return (f"{x['metric']:42s} {x['n']:4s} pairs {x['pairs']:2d} arm {x['arm_med']:.5g} twin {x['twin_med']:.5g} "
            f"own {x['own_med']:.5g} r {x['r_med']:+.3f} [{x['r_q25']:+.3f},{x['r_q75']:+.3f}] better "
            f"{x['arm_better']}/{x['pairs']} pW {x['wilcoxon_p']:.2g} mesh +{x['mesh_pos']}/-{x['mesh_neg']} "
            f"gap {x['gap_closed_med']:+.2f} {'SEES ' + x['direction'] if x['sees'] else ''}"
            + (f" holm {x['holm_p_decision']:.2g}" if "holm_p_decision" in x else ""))


if __name__ == "__main__":
    main()
