"""d159_table.py OUT_DIR -- D159's reading (CPU; reads the readers' files). Per mesh (bunny, dragon):
  R1  read/relief_MESH.jsonl: the carried share k by band (below 2.7 / 2.7-5.4 / 5.4-10.8 pitches and the whole band
      below 10.8) in the region, the contrast and the not-thin surface, the exterior and the outer layer, for D159 M97,
      the tag's L97 (and its five seeds L97-L105: the range), P97, the P99 re-run, the own sample; M97 and L97 at 4 s;
  R2  read/course_MESH.jsonl (M97, P99 on P97's layer) and read/course_tag_MESH.jsonl (L97): per kept frame the
      displacement of the arm's surface from the twin's, its detail band (below 10.8 pitches), in the region;
  R3  D157's decision set (held-out views at 1024 px, the corrected referee; 3D on the exterior; studio images) for
      M97 (read/d157_MESH.jsonl) against the tag's L97 and its five seeds (output/gpu/d157/read300_*.jsonl);
  R4  tab145.py's row of d159/MESH300k_M97 (the yardstick, the display, thin, rest, momentum, the render's share) with
      the visible pieces and the end pieces, against B1's rows (d145t_evidence/tab_tx.jsonl, L97-L105);
  40k d154_mesh_read rows of the M97 gallery (read/g40_m97.jsonl) against the tag's V97 / V99 (d154/read).
  D159c (C97: the mesh-relief pictures with the render weight held, --lambda_ema 0) beside M97 in R1-R5, and R6 (per
      window before / after arrival, read/window.jsonl + window_c.jsonl) for the twin, the tag, D159 and D159c.
Writes OUT_DIR/table.json, table.txt and the figures course.png, relief.png, holds.png."""
import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

G = Path("/data/relcfd/chayo/physmorph_v2/output/gpu")
O = G / "d159"
R = O / "read"
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
SEEDS = (97, 99, 101, 103, 105)
BANDS = ("<2.7", "2.7-5.4", "5.4-10.8")
DEC = [("BIoU 2% (err)", lambda r: r["sil"]["zb"]["cor"]["1024"]["held"]["biou2pct_err"]),
       ("boundary F DAVIS (err)", lambda r: r["sil"]["zb"]["cor"]["1024"]["held"]["bfdavis_err"]),
       ("contour Chamfer (a)", lambda r: r["sil"]["zb"]["cor"]["1024"]["held"]["chamfer_a"]),
       ("HD95 (a)", lambda r: r["sil"]["zb"]["cor"]["1024"]["held"]["hd95_a"]),
       ("3D F@0.5 (err)", lambda r: 1 - r["d3"]["zb"]["cor"]["f0.5"]),
       ("3D F@1 (err)", lambda r: 1 - r["d3"]["zb"]["cor"]["f1"]),
       ("3D normal consistency (err)", lambda r: 1 - r["d3"]["zb"]["cor"]["nc"]),
       ("3D Chamfer-L1 (a)", lambda r: r["d3"]["zb"]["cor"]["chamfer_l1_a"]),
       ("shaded MSE", lambda r: float(np.mean(r["img"]["cor"]["shaded"]["mse"]))),
       ("shaded 1-SSIM", lambda r: 1 - float(np.mean(r["img"]["cor"]["shaded"]["ssim"]))),
       ("shaded LPIPS", lambda r: float(np.mean(r["img"]["cor"]["shaded"]["lpips"]))),
       ("normal MSE", lambda r: float(np.mean(r["img"]["cor"]["normal"]["mse"]))),
       ("normal 1-SSIM", lambda r: 1 - float(np.mean(r["img"]["cor"]["normal"]["ssim"]))),
       ("normal LPIPS", lambda r: float(np.mean(r["img"]["cor"]["normal"]["lpips"])))]
THREE_D = {"3D F@0.5 (err)", "3D F@1 (err)", "3D normal consistency (err)", "3D Chamfer-L1 (a)"}
R4K = ["yard_ext_sil", "yard_ext_pbr", "disp_front_1miou", "disp_back_1miou", "disp_crop_1miou", "thin", "kin_last",
       "pc_end_ge10", "pc_end_pieces", "vis_run_blob", "vis_end_blob", "com_end_a", "ang_net_gross_mean", "minutes",
       "windows", "share_median", "lam_median"]


def jl(p):
    try:
        return [json.loads(s) for s in open(p) if s.startswith("{")]
    except OSError:
        return []


def rng(vals):
    v = [x for x in vals if x is not None]
    return (min(v), max(v)) if v else (None, None)


res, txt = {}, []
for mesh in ("bunny", "dragon"):
    M = res[mesh] = {}
    # ---- R1
    rel = {r["label"]: r for f in ("relief", "relief_c") for r in jl(R / f"{f}_{mesh}.jsonl")}
    M["R1"] = {}
    for lab, r in rel.items():
        M["R1"][lab] = {s: {reg: dict(k=[b["k"] for b in r["sets"][s][reg]["bands"]], k_all=r["sets"][s][reg]["k_below_10_8"],
                                      t_rms=[b["t_rms_a"] for b in r["sets"][s][reg]["bands"]])
                            for reg in ("region", "contrast", "not_thin")} for s in ("exterior", "layer")}
    txt.append(f"=== {mesh}: R1 carried share k (1 = follows the mesh's relief, 0 = smooth over it), bands {BANDS}, all < 10.8")
    for s in ("layer", "exterior"):
        for reg in ("region", "contrast"):
            line = f"  {s:8s} {reg:8s}"
            for lab in ("own", "P97", "P99re", "L97", "M97", "C97", "L97_4s", "M97_4s", "C97_4s"):
                if lab in M["R1"]:
                    v = M["R1"][lab][s][reg]
                    line += f" | {lab} " + "/".join(f"{x:.2f}" for x in v["k"]) + f" ({v['k_all']:.2f})"
            tag = [M["R1"][f"L{sd}"][s][reg]["k"] for sd in SEEDS if f"L{sd}" in M["R1"]]
            if tag:
                lo, hi = np.min(tag, 0), np.max(tag, 0)
                line += " | tag L97-L105 range " + "/".join(f"{a:.2f}-{b:.2f}" for a, b in zip(lo, hi))
                M["R1"].setdefault("_tag_range", {})[f"{s}/{reg}"] = [list(lo), list(hi)]
            txt.append(line)
    # ---- R2
    co = {}
    for f, labs in ((R / f"course_{mesh}.jsonl", None), (R / f"course_tag_{mesh}.jsonl", None), (R / f"course_c_{mesh}.jsonl", None)):
        for r in jl(f):
            for lab, v in r["arms"].items():
                co.setdefault(lab, []).append(dict(t=r["t_s"], kept=r["kept"], **v))
    M["R2"] = co
    txt.append(f"=== {mesh}: R2 the arm's surface from the twin's (P97 layer): rms / detail band rms / region detail rms (pitches)")
    for t in (0.5, 1.0, 2.0, 4.0, 8.0, None):
        line = f"  t {('end' if t is None else t):>4}"
        for lab in ("M97", "C97", "L97", "P99"):
            rows = [x for x in co.get(lab, []) if (x["t"] is None) == (t is None) and (t is None or abs(x["t"] - t) < 1e-6)]
            if rows:
                x = rows[-1]
                line += f" | {lab} {x['rms_a']:.3f} / {x['detail_rms_a']:.3f} / {x['region_detail_rms_a']:.3f}" + \
                        (f" (particles {x['particle_rms_a']:.2f})" if "particle_rms_a" in x else "")
        txt.append(line)
    # ---- R3
    d157 = {}
    for f in glob.glob(str(G / "d157/read300_*.jsonl")):
        for r in jl(f):
            if r["mesh"] == mesh and r["label"] == "L":
                d157[r["seed"]] = r
    M["R3"], M["R3_worse"] = {}, {}
    for run, lab, fname in (("M97", "M", f"d157_{mesh}.jsonl"), ("C97", "C", f"d157c_{mesh}.jsonl")):
        mine = [r for r in jl(R / fname) if r["label"] == lab]
        if not mine:
            continue
        txt.append(f"=== {mesh}: R3 D157's decision set (lower is better): {run} | tag L97 | tag range L97-L105 | outside on the worse side")
        worse = []
        for name, fn in DEC:
            m, t = fn(mine[-1]), fn(d157[97]) if 97 in d157 else None
            lo, hi = rng([fn(d157[s]) for s in SEEDS if s in d157])
            out_w = hi is not None and m > hi
            if out_w:
                worse.append(name)
            M["R3"].setdefault(name, dict(L97=t, lo=lo, hi=hi))[run] = dict(v=m, worse=out_w, better=lo is not None and m < lo)
            txt.append(f"  {name:28s} {m:.4g} | {t:.4g} | {lo:.4g}-{hi:.4g} | {'WORSE' if out_w else ('better' if m < lo else 'within')}")
        M["R3_worse"][run] = worse
        txt.append(f"  {run} outside on the worse side: {len(worse)} of {len(DEC)} ({', '.join(worse) or '-'}); 3D among them: {[w for w in worse if w in THREE_D]}")
    # ---- R4
    tag = {r["name"]: r for r in jl(G / "d145t_evidence/tab_tx.jsonl") if r.get("dir") == "d145t_b1"}
    M["R4"] = {}
    rows4 = {}
    for run in ("M97", "C97"):
        tmp = O / f"tab_{mesh}_{run}.jsonl"
        tmp.unlink(missing_ok=True)
        subprocess.run([sys.executable, "/data/relcfd/chayo/physmorph_v2/tmp/d145/tab145.py", str(tmp), f"d159/{mesh}300k_{run}"],
                       capture_output=True)
        rr = jl(tmp)
        if not rr:
            continue
        m = rr[-1]
        vf = jl(O / f"vis_full_{mesh}300k_{run}.jsonl")
        rgv = [v for v in vf if v.get("window") == "range"]
        win = [v for v in vf if v.get("window") != "range"]
        if rgv:
            m.update(vis_run_blob=rgv[-1]["blob"])
        if win:
            m.update(vis_end_blob=win[-1]["blob"])
        pc = [p for p in jl(O / "pieces.jsonl") if p.get("run") == f"{mesh}300k_{run}"]
        if pc:
            m.update(pc_end_pieces=pc[-1]["end_pieces"], pc_end_ge10=pc[-1]["end_ge10"], pc_end_ge5a=pc[-1].get("end_ge5a"),
                     pc_end_largest=pc[-1].get("end_largest"), pc_end_s=pc[-1].get("end_s"))
        rows4[run] = m
    if rows4:
        runs = [r for r in ("M97", "C97") if r in rows4]
        txt.append(f"=== {mesh}: R4 {' | '.join(runs)} | tag L97 | tag range L97-L105")
        for k in R4K + ["pc_end_ge5a", "pc_end_largest", "pc_end_s"]:
            vals = [tag.get(f"{mesh}300k_L{s}", {}).get(k) for s in SEEDS]
            lo, hi = rng([v for v in vals if isinstance(v, (int, float))])
            M["R4"][k] = dict(L97=vals[0], lo=lo, hi=hi, **{r: rows4[r].get(k) for r in runs})
            txt.append(f"  {k:20s} " + " | ".join(str(rows4[r].get(k)) for r in runs) + f" | {vals[0]} | {lo}-{hi}")
# ---- R5
for mesh in ("bunny", "dragon"):
    rows = jl(R / f"terms_{mesh}.jsonl")
    if not rows:
        continue
    res[mesh]["R5"] = {}
    txt.append(f"=== {mesh}: R5 the detail band (< 10.8 pitches) of each term's normal gradient on the outer layer: rms, its share of the"
               f" term's energy, the cosine with the render's detail part; mean over the last 10 windows and at the end")
    for name in ("M97", "T97", "C97"):
        rr = [r for r in rows if r["name"] == name]
        if not rr:
            continue
        last10 = rr[-10:]
        S = {}
        for t in ("ot", "surf", "near", "spray", "rend", "physics"):
            det = np.mean([r["terms"][t]["detail_rms"] for r in last10])
            rest = np.mean([r["terms"][t]["rest_rms"] for r in last10])
            cs = [r["terms"][t]["cos_with_render"] for r in last10 if r["terms"][t]["cos_with_render"] is not None]
            S[t] = dict(detail_rms=float(det), detail_share=float(det ** 2 / max(det ** 2 + rest ** 2, 1e-60)),
                        cos_with_render=float(np.mean(cs)) if cs else None,
                        detail_rms_end=rr[-1]["terms"][t]["detail_rms"], detail_rms_region_end=rr[-1]["terms"][t]["detail_rms_region"])
        S["relaxation"] = rr[-1].get("relaxation")
        S["course"] = [(r["window"], r["terms"]["rend"]["detail_rms"], r["terms"]["physics"]["detail_rms"]) for r in rr]
        res[mesh]["R5"][name] = S
        txt.append(f"  {name}: " + " | ".join(f"{t} {v['detail_rms']:.2e} (share {v['detail_share']:.2f}, cos {v['cos_with_render'] if v['cos_with_render'] is None else round(v['cos_with_render'], 2)})"
                                          for t, v in S.items() if t not in ("relaxation", "course")))
        if S["relaxation"]:
            rx = S["relaxation"]
            txt.append(f"     relaxation at the end: removes {rx['residual_detail_rms_a']:.3f} a a step in the band ({rx['residual_detail_rms_region_a']:.3f} in the region),"
                       f" cos with the render's push {rx['cos_removal_with_render']:+.2f}; its reference rms {rx['ref_rms_a']:.3f} a")
# ---- R6
W6 = {r["label"]: r for f in ("window.jsonl", "window_c.jsonl") for r in jl(R / f)}
for mesh in ("bunny", "dragon"):
    labs = [l for l in (f"{mesh}_twin_P97", f"{mesh}_tag_L97", f"{mesh}_D159_M97", f"{mesh}_D159c_C97") if l in W6]
    if not labs:
        continue
    res[mesh]["R6"] = {l: {k: W6[l][k] for k in ("windows", "arrival_window", "before", "after", "last10", "end_reason", "stop")} for l in labs}
    txt.append(f"=== {mesh}: R6 per window, before / after arrival (90 % of the cumulative move): lambda, lambda|g_r| vs |g_p|, the"
               f" render's share, the step alpha|g|, the move per window (spacings), null windows, outer rejects")
    for l in labs:
        w = W6[l]
        for ph in ("before", "after", "last10"):
            b = w[ph]
            txt.append(f"  {l:20s} {ph:6s} n {b['windows']:3d} lam {b['lam']:.4g} lam|g_r| {b['lam_gr']:.3g} |g_p| {b['gp']:.3g}"
                       f" share {b['share']:.2f} step {b['step']:.3g} move {b['move_sp']:.4f} (sum {b['move_sp_sum']:.2f})"
                       f" null {b['null']} outer_rej {b['outer_rej']} it {b['accepted_iters']}/{b['rejected_iters']}")
        txt.append(f"  {l:20s} windows {w['windows']} arrival {w['arrival_window']} end {w['end_reason']} stop {w['stop']}")
# ---- 40k
g40 = jl(R / "g40_m97.jsonl")
if g40:
    tagrows = [json.loads(s) for f in glob.glob(str(G / "d154/read/g40*_*.jsonl")) for s in open(f) if s.startswith("{")]
    T = {(r["mesh"], r["seed"], r["label"]): r for r in tagrows}
    meas = [("outline_sd", lambda r: r["exact_1024"]["outline_state_sd_a"]), ("iou1024", lambda r: r["exact_1024"]["one_minus_iou"]),
            ("normal_cl", lambda r: r["dist"]["normal_closest_deg"])]
    res["g40"], bad = {}, []
    txt.append("=== 40k gallery: M97 against the tag's V97 / V99; worse = M97 > max(V97, V99) + |V97 - V99|")
    for r in g40:
        mesh = r["mesh"]
        row = {}
        for nm, fn in meas:
            try:
                a, b, m = fn(T[(mesh, 97, "V")]), fn(T[(mesh, 99, "V")]), fn(r)
            except KeyError:
                continue
            w = m > max(a, b) + abs(a - b)
            row[nm] = dict(M97=m, V97=a, V99=b, worse=bool(w))
        try:
            js = lambda p: json.load(open(p))["arms"]                                    # noqa: E731
            tm = [next(iter(js(G / f"d154/g40runs/{mesh}40k_V{s}.json").values()))["metrics"]["thin_uncovered"]
                  if (G / f"d154/g40runs/{mesh}40k_V{s}.json").exists() else
                  next(iter(js(G / f"d155/gate/{mesh}40k_V{s}.json").values()))["metrics"]["thin_uncovered"] for s in (97, 99)]
            mm = next(iter(js(O / "g40" / f"{mesh}40k_M97.json").values()))["metrics"]["thin_uncovered"]
            row["thin"] = dict(M97=mm, V97=tm[0], V99=tm[1], worse=bool(mm > max(tm) + abs(tm[0] - tm[1])))
        except (OSError, StopIteration, KeyError):
            pass
        res["g40"][mesh] = row
        ws = [k for k, v in row.items() if v["worse"]]
        if ws:
            bad.append((mesh, ws))
        txt.append(f"  {mesh:12s} " + " ".join(f"{k} {v['M97']:.4g} ({v['V97']:.4g}/{v['V99']:.4g}){'*' if v['worse'] else ''}" for k, v in row.items()))
    res["g40_worse"] = bad
    txt.append(f"  meshes worse: {len(bad)} {bad}")
json.dump(res, open(out / "table.json", "w"), indent=1, default=float)
open(out / "table.txt", "w").write("\n".join(txt) + "\n")
print("\n".join(txt))

# ---- figures
import matplotlib                                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                      # noqa: E402
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
COL = {"M97": "#2a78d6", "C97": "#7a4fc4", "L97": "#eb6834", "P99": "#898781"}
LAB = {"M97": "D159 full arm (target with mesh relief) vs twin", "C97": "D159c full arm (mesh relief, render weight held) vs twin", "L97": "tag full arm vs twin",
       "P99": "twin seed 99 vs twin seed 97 (noise floor)"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), sharex="col")
for j, mesh in enumerate(("bunny", "dragon")):
    co = res[mesh].get("R2", {})
    for i, (key, ttl) in enumerate((("detail_rms_a", "how far the surface is from the twin's, detail band (< 10.8 pitches)"),
                                    ("region_detail_rms_a", "the same in the scales / wrinkles region"))):
        a_ = ax[i, j]
        for lab in ("P99", "L97", "M97", "C97"):
            rows = [x for x in co.get(lab, []) if x["t"] is not None]
            if rows:
                a_.plot([x["t"] for x in rows], [x[key] for x in rows], color=COL[lab], lw=2, label=LAB[lab])
        a_.set_title(f"{mesh} 300k: {ttl}", fontsize=10)
        a_.grid(axis="y", color=GRID, lw=0.8)
        a_.set_ylabel("rms (pitches)")
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
        if i == 1:
            a_.set_xlabel("time in the morph (s)")
        if i == 0 and j == 0:
            a_.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(out / "course.png", dpi=130, bbox_inches="tight")
plt.close(fig)
fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), sharey=True)
for j, mesh in enumerate(("bunny", "dragon")):
    R1 = res[mesh].get("R1", {})
    for i, s in enumerate(("layer", "exterior")):
        a_ = ax[i, j]
        labs = [l for l in ("own", "P97", "L97", "M97", "C97") if l in R1]
        names = {"own": "the run's own 300k sample", "P97": "dF_c + layer u (twin)", "L97": "tag full arm",
                 "M97": "D159 full arm", "C97": "D159c full arm (weight held)"}
        cols = {"own": "#c3c2b7", "P97": "#898781", "L97": "#eb6834", "M97": "#2a78d6", "C97": "#7a4fc4"}
        w = 0.8 / max(len(labs), 1)
        for q, l in enumerate(labs):
            ks = R1[l][s]["region"]["k"]
            a_.bar(np.arange(3) + (q - (len(labs) - 1) / 2) * w, ks, width=w * 0.9, color=cols[l], label=names[l])
        rg = R1.get("_tag_range", {}).get(f"{s}/region")
        if rg:
            q = labs.index("L97") if "L97" in labs else 0
            x = np.arange(3) + (q - (len(labs) - 1) / 2) * w
            a_.vlines(x, rg[0], rg[1], color=INK, lw=1.5)
        a_.set_xticks(range(3))
        a_.set_xticklabels([f"below 2.7", "2.7-5.4", "5.4-10.8"])
        a_.axhline(1, color="#c3c2b7", lw=1)
        a_.axhline(0, color="#c3c2b7", lw=1)
        what = "the outer-layer particles" if s == "layer" else "the loss's surface (exterior, r = 3)"
        a_.set_title(f"{mesh} 300k, region: mesh relief carried by {what}", fontsize=9)
        a_.set_ylabel("carried share (1 = all of it)")
        a_.set_xlabel("relief wavelength band (pitches)")
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        if i == 0 and j == 0:
            a_.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out / "relief.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote", out)
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for j, mesh in enumerate(("bunny", "dragon")):
    a_ = ax[j]
    for name, col, lab in (("M97", "#2a78d6", "D159 full arm"), ("T97", "#eb6834", "tag full arm (re-run)"),
                           ("C97", "#7a4fc4", "D159c full arm (weight held)")):
        S = res.get(mesh, {}).get("R5", {}).get(name)
        if not S:
            continue
        w = [c[0] for c in S["course"]]
        share = [c[1] ** 2 / max(c[1] ** 2 + c[2] ** 2, 1e-60) for c in S["course"]]
        a_.plot(w, share, color=col, lw=2, label=lab)
    a_.set_ylim(0, 1)
    a_.set_title(f"{mesh} 300k: render's share of the detail-band push (outer layer)", fontsize=10)
    a_.set_xlabel("window")
    a_.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        a_.spines[sp].set_visible(False)
    if j == 0:
        a_.set_ylabel("render / (render + physics), energy")
        a_.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(out / "holds.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote holds.png")
