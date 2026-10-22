"""d160_gate.py OUT_DIR -- D160's gate, read by the pre-registered criteria C1-C4 (CPU, read-only), and its figures.

Sources (output/gpu/d160 unless noted): read_colour.jsonl (d160_read.py on every run), gate_d160.jsonl (D155's
d155_gate_read.py on every run: windows, minutes, the kinetic record, the end readings), runs40/NAME.json (G3),
pieces.jsonl (tmp/d145r/pieces145.py: pieces of >= 10 particles at the end, the tag's V / W too), d157_L.jsonl
(D157's reader on the L runs); the tag's references: d155/gate/gate_all.jsonl (V / W 97 / 99 / 101), d157/read40_*.jsonl
(V), tmp/d160/read_tag.jsonl (the tag's colour layout). Writes OUT_DIR/gate.json, OUT_DIR/table.md, OUT_DIR/fig_*.png."""
import glob
import json
import sys
from pathlib import Path

import numpy as np

B = "/data/relcfd/chayo/physmorph_v2"
G, O = f"{B}/output/gpu", f"{B}/output/gpu/d160"
SCENES = [("bunny", "half"), ("bunny", "band"), ("bunny", "ears"), ("bunny", "turn"), ("dragon", "band")]
SEEDS = (97, 99)
out = Path(sys.argv[1])
FULL = sys.argv[2] if len(sys.argv) > 2 else "L"           # the full arm's label: L (D160, the rule's weight), H (D160c, held)
FULL_LABEL = "dF_c + layer u + render loss (full)" + (", lambda held" if FULL == "H" else "")
out.mkdir(parents=True, exist_ok=True)


def jl(p):
    rows = []
    for f in glob.glob(p):
        rows += [json.loads(s) for s in open(f) if s.startswith("{")]
    return rows


col = {r["name"]: r for r in jl(f"{O}/read_colour*.jsonl")}
gate = {r["name"]: r for r in jl(f"{O}/gate_d160*.jsonl")}
tagg = {r["name"]: r for r in jl(f"{G}/d155/gate/gate_all.jsonl")}
pieces = {r["run"]: r for r in jl(f"{O}/pieces.jsonl")}
d157 = {}
for r in jl(f"{G}/d157/read40_*.jsonl") + jl(f"{O}/d157_*.jsonl"):
    if r.get("label") in ("V",) or r.get("label", "").endswith(f"_{FULL}"):
        d157[(r["mesh"], r["seed"], r["label"])] = dict(
            biou=r["sil"]["zb"]["cor"]["1024"]["held"]["biou2pct_err"], f05=r["d3"]["zb"]["cor"]["f0.5"],
            psnr=float(np.mean(r["img"]["cor"]["shaded"]["psnr"])))


def name(mesh, sc, arm, seed):
    return f"{mesh}40k_{sc}_{arm}{seed}"


def g3(n):
    try:
        a = next(iter(json.load(open(f"{O}/runs40/{n}.json"))["arms"].values()))
        return bool(a["gates"]["G3_rest"]), a["gates"].get("drift_rel"), a["metrics"].get("jitter_rel")
    except (OSError, StopIteration, KeyError):
        return None, None, None


# ---- C1: colour ----
c1 = {}
for mesh, sc in SCENES:
    per = []
    for s in SEEDS:
        L, P = col.get(name(mesh, sc, FULL, s)), col.get(name(mesh, sc, "P", s))
        Lr = col.get(name(mesh, sc, "L", s))
        if not (L and P):
            per.append(None)
            continue
        mP, mL, fl = P["end"]["mis"], L["end"]["mis"], L["floor"]
        hP, hL = P["img"]["held"]["colour"], L["img"]["held"]["colour"]
        per.append(dict(seed=s, mis_P=mP, mis_L=mL, floor=fl, r=(mP - mL) / max(mP - fl, 1e-9),
                        psnr_P=hP["psnr"], psnr_L=hL["psnr"], lpips_P=hP["lpips"], lpips_L=hL["lpips"],
                        de_P=hP["de"], de_L=hL["de"], travel_P=P["end"]["travel_med"], travel_L=L["end"]["travel_med"],
                        recall_P=P["end"]["recall"], recall_L=L["end"]["recall"], false_P=P["end"]["false"],
                        false_L=L["end"]["false"], vs_twin=L.get("vs_twin"), mis_Lrule=Lr["end"]["mis"] if Lr else None,
                        tl_P=P.get("timeline"), tl_L=L.get("timeline")))
    ok = [p for p in per if p]
    if sc == "half" and mesh == "bunny":
        passed = len(ok) == len(SEEDS) and all(p["mis_L"] <= p["mis_P"] + 0.01 for p in ok)
    else:
        passed = len(ok) == len(SEEDS) and all(p["r"] >= 0.5 and p["psnr_L"] > p["psnr_P"] and p["lpips_L"] < p["lpips_P"]
                                               for p in ok)
    c1[f"{mesh}_{sc}"] = dict(passed=passed, seeds=per)
rearr = [k for k in c1 if k != "bunny_half"]
C1 = sum(c1[k]["passed"] for k in rearr) >= 2 and c1["bunny_half"]["passed"]

# ---- C2: geometry against the tag's V97 / V99 / V101 (worse side by more than its spread) ----
LOWER = {"ind_sil": False, "yard_sil": True, "yard_pbr": True, "display": True, "thin": True,
         "biou": True, "f05": False, "psnr": False}          # True: lower is better
c2, cells, miss = {}, 0, 0
for mesh, sc in SCENES:
    for s in SEEDS:
        n = name(mesh, sc, FULL, s)
        if n not in gate:
            continue
        row = {}
        for k, low in LOWER.items():
            if k in ("biou", "f05", "psnr"):
                ref = [d157[(mesh, t, "V")][k] for t in (97, 99, 101) if (mesh, t, "V") in d157]
                v = d157.get((mesh, s, f"{sc}_{FULL}"), {}).get(k)
            else:
                ref = [tagg[f"{mesh}40k_V{t}"]["end"].get(k) for t in (97, 99, 101) if f"{mesh}40k_V{t}" in tagg]
                ref = [x for x in ref if x is not None]
                v = gate[n]["end"].get(k)
            if v is None or len(ref) < 2:
                row[k] = None
                continue
            spread = max(ref) - min(ref)
            worst = max(ref) if low else min(ref)
            bad = v > worst + spread if low else v < worst - spread
            row[k] = dict(v=v, ref=[min(ref), max(ref)], miss=bool(bad))
            cells += 1
            miss += int(bad)
        c2[n] = row
C2 = cells > 0 and miss <= 0.1 * cells

# ---- C3: at rest; C4: pieces ----
kin_ref = {m: max(tagg[f"{m}40k_V{t}"]["kin_end"] for t in (97, 99, 101) if f"{m}40k_V{t}" in tagg)
           for m in ("bunny", "dragon")}
c3, c4 = {}, {}
for mesh, sc in SCENES:
    for s in SEEDS:
        n = name(mesh, sc, FULL, s)
        if n not in gate:
            continue
        ok3, drift, jit = g3(n)
        c3[n] = dict(G3=ok3, drift_rel=drift, jitter_rel=jit, kin_end=gate[n]["kin_end"], kin_ref=1.25 * kin_ref[mesh],
                     passed=bool(ok3) and gate[n]["kin_end"] <= 1.25 * kin_ref[mesh])
        pc = pieces.get(n)
        tag_ge10 = [pieces[f"{mesh}40k_V{t}"]["end_ge10"] for t in (97, 99, 101) if f"{mesh}40k_V{t}" in pieces]
        c4[n] = dict(end_ge10=pc["end_ge10"] if pc else None, end_ge5a=pc["end_ge5a"] if pc else None,
                     tag_V_end_ge10=tag_ge10, passed=bool(pc) and pc["end_ge10"] == 0)
C3 = bool(c3) and all(v["passed"] for v in c3.values())
C4 = bool(c4) and all(v["passed"] for v in c4.values())

runs = {}
for mesh, sc in SCENES:
    for s in SEEDS:
        for arm in (FULL, "P"):
            n = name(mesh, sc, arm, s)
            if n in gate:
                g = gate[n]
                runs[n] = dict(windows=g["windows"], minutes=g["minutes"], kin_end=g["kin_end"], end=g["end"],
                               g_share=g["ramp"].get("g_share"))
res = dict(C1=C1, C2=C2, C3=C3, C4=C4, PASS=bool(C1 and C2 and C3 and C4), c1=c1, c2=c2, c2_cells=cells,
           c2_miss=miss, c3=c3, c4=c4, runs=runs)
(out / "gate.json").write_text(json.dumps(res, indent=1, default=str))

# ---- the table ----
L_ = ["| scene | seed | mis twin → full (floor) | recovered r | held-out colour PSNR twin → full | LPIPS twin → full | "
      "ΔE twin → full | travel twin → full (pitches) | red outer shift full − twin (pitches) | outer |Δ| med / p90 | "
      "D160 full (the rule's λ) mis |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for k, v in c1.items():
    for p in v["seeds"]:
        if not p:
            continue
        vt = p["vs_twin"] or {}
        L_.append(f"| {k} | {p['seed']} | {p['mis_P']:.3f} → {p['mis_L']:.3f} ({p['floor']:.3f}) | {p['r']:.2f} | "
                  f"{p['psnr_P']:.2f} → {p['psnr_L']:.2f} | {p['lpips_P']:.3f} → {p['lpips_L']:.3f} | "
                  f"{p['de_P']:.1f} → {p['de_L']:.1f} | {p['travel_P']:.1f} → {p['travel_L']:.1f} | "
                  f"{(vt.get('red_outer_shift') or 0):+.2f} | {vt.get('d_outer_med', 0):.2f} / {vt.get('d_outer_p90', 0):.2f} | "
                  f"{p['mis_Lrule'] if p['mis_Lrule'] is not None else float('nan'):.3f} |")
L_ += ["", f"C1 colour {'PASS' if C1 else 'FAIL'}: " + ", ".join(f"{k} {'pass' if v['passed'] else 'fail'}"
                                                              for k, v in c1.items()),
       f"C2 geometry {'PASS' if C2 else 'FAIL'}: {miss} of {cells} cells worse than the tag's worst V by more than its spread",
       f"C3 at rest {'PASS' if C3 else 'FAIL'}; C4 pieces {'PASS' if C4 else 'FAIL'}; gate {'PASS' if res['PASS'] else 'FAIL'}"]
L_ += ["", "| run | windows | minutes | kin end | ind IoU | yard sil | yard shade | display | thin | g_share |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for n, r in runs.items():
    e = r["end"]
    L_.append(f"| {n} | {r['windows']} | {r['minutes']:.1f} | {r['kin_end']:.2e} | {e.get('ind_sil', 0):.4f} | "
              f"{e.get('yard_sil', 0):.2e} | {e.get('yard_pbr', 0):.2e} | {e.get('display', 0):.4f} | {e.get('thin', 0):.3f} | "
              f"{(r['g_share'] or 0):.2f} |")
(out / "table.md").write_text("\n".join(L_) + "\n")
print("\n".join(L_))

# ---- figures (one question per panel; twin blue, full orange: the default palette's slots 1 and 2) ----
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
TW, FU, INK, INK2, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
labels = {"bunny_half": "bunny\nhalf (control)", "bunny_band": "bunny\nband (short)", "bunny_ears": "bunny\nears (medium)",
          "bunny_turn": "bunny\nturn (long)", "dragon_band": "dragon\nband (short)"}
keys = list(c1)
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for i, k in enumerate(keys):
    ps = [p for p in c1[k]["seeds"] if p]
    if not ps:
        continue
    for j, (arm, c) in enumerate((("P", TW), ("L", FU))):
        vals = [p[f"mis_{arm}"] for p in ps]
        ax.bar(i + (j - .5) * .36, np.mean(vals), .32, color=c, edgecolor=SURF, linewidth=2,
               label=("dF_c + layer u (twin)" if arm == "P" else FULL_LABEL) if i == 0 else None)
        ax.scatter([i + (j - .5) * .36] * len(vals), vals, s=18, color=INK, zorder=3)
    ax.hlines(np.mean([p["floor"] for p in ps]), i - .4, i + .4, colors=INK2, linestyles="--", linewidth=1,
              label="a perfect sample's own share" if i == 0 else None)
ax.set_xticks(range(len(keys)), [labels[k] for k in keys])
ax.set_ylabel("share of the surface in the wrong colour")
ax.set_title("How much of the surface ends in the wrong colour? (dots: seeds 97, 99)", color=INK, loc="left")
ax.grid(axis="y", color="#e6e5e1", linewidth=.8)
ax.set_axisbelow(True)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(out / "fig_miscoloured.png", dpi=160)
plt.close(fig)

fig, axs = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.2), sharey=False)
for ax, k in zip(np.atleast_1d(axs), keys):
    for p in c1[k]["seeds"]:
        if not p:
            continue
        ls = "-" if p["seed"] == 97 else "--"
        for arm, c in (("P", TW), ("L", FU)):
            tl = p[f"tl_{arm}"]
            if tl:
                t, m = zip(*tl)
                ax.plot(t, m, ls, color=c, linewidth=2)
    ax.set_title(labels[k].replace("\n", " "), color=INK, fontsize=10, loc="left")
    ax.set_xlabel("time (s)")
    ax.grid(color="#e6e5e1", linewidth=.8)
np.atleast_1d(axs)[0].set_ylabel("wrong-colour share")
from matplotlib.lines import Line2D  # noqa: E402
fig.legend([Line2D([], [], color=TW, lw=2), Line2D([], [], color=FU, lw=2), Line2D([], [], color=INK2, lw=2),
            Line2D([], [], color=INK2, lw=2, ls="--")],
           ["dF_c + layer u (twin)", FULL_LABEL, "seed 97", "seed 99"],
           frameon=False, loc="upper center", ncol=4)
fig.suptitle("When does the colour get placed?", x=0.01, ha="left", y=0.86, color=INK)
fig.tight_layout(rect=(0, 0, 1, .82))
fig.savefig(out / "fig_timeline.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 3.6))
for i, k in enumerate(keys):
    for p in c1[k]["seeds"]:
        if not p:
            continue
        ax.plot([i - .15, i + .15], [p["psnr_P"], p["psnr_L"]], color=INK2, linewidth=1, zorder=1)
        ax.scatter([i - .15], [p["psnr_P"]], s=40, color=TW, zorder=2, label="dF_c + layer u (twin)" if i == 0 and p["seed"] == 97 else None)
        ax.scatter([i + .15], [p["psnr_L"]], s=40, color=FU, zorder=2,
                   label=FULL_LABEL if i == 0 and p["seed"] == 97 else None)
ax.set_xticks(range(len(keys)), [labels[k] for k in keys])
ax.set_ylabel("PSNR (dB), higher is closer")
ax.set_title("Do pictures from new viewpoints look like the painted target? (46 held-out views)", color=INK, loc="left")
ax.grid(axis="y", color="#e6e5e1", linewidth=.8)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(out / "fig_heldout_psnr.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 3.6))
for i, k in enumerate(keys):
    for p in c1[k]["seeds"]:
        vt = (p or {}).get("vs_twin") or {}
        if vt.get("red_outer_shift") is None:
            continue
        ax.scatter([i], [vt["red_outer_shift"]], s=40, color=FU, zorder=2)
ax.axhline(0, color=INK2, linewidth=1)
ax.set_xticks(range(len(keys)), [labels[k] for k in keys])
ax.set_ylabel("pitches toward where red is wanted")
ax.set_title("How far did the render move the red surface material, beyond the twin? (dots: seeds)", color=INK, loc="left")
ax.grid(axis="y", color="#e6e5e1", linewidth=.8)
fig.tight_layout()
fig.savefig(out / "fig_red_shift.png", dpi=160)
plt.close(fig)
print("wrote", out)
