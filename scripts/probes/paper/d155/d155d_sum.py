"""d155d_sum.py GATE_DIR OUT_DIR -- D155d's 40k gate as pre-registered (2026-10-21 12:18 CDT; the tag's range as D155's gate: its re-run seeds and B3's): items (1), (3), (4), (5) and
the render lead at seed 97; (2) by the letter (seed 97) and with dragon and homer read by the median of the ramp arm's
three seeds; (6a) the ramp's largest mean J <= 1.03; (6b) the end's mean J within the tag's range or nearer 1 than its
nearest seed, J's p99 not above the tag's largest; each on >= 4 of 5. Tables: the volume and the elastic energy (A3D
against the tag's and A3G's re-runs with --record_elastic), the end per measure with the seeds, timing, motion, the
render's influence. Figure fig_volume_d.png: mean J and the volumetric elastic energy over time, bunny / dragon / homer."""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GD, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
R = {r["name"]: r for r in (json.loads(line) for line in open(f"{GD}/gate_all.jsonl"))}
MESH = ("bunny", "dragon", "teapot", "armadilo", "homer")
SEEDS = (97, 99, 101)
END = (("end.ind_sil", "high"), ("end.yard_sil", "low"), ("end.yard_pbr", "low"), ("end.display", "low"),
       ("end.thin", "low"))
md = open(f"{OUT}/gate_tables.md", "w")


def p(s=""):
    print(s)
    md.write(s + "\n")


def g(r, path):
    x = r
    for k in path.split("."):
        x = None if x is None else x.get(k)
    return x


B3 = {r["name"]: r for r in (json.loads(line) for line in open(f"{os.path.dirname(GD)}/gate_b3_tag.jsonl"))}


def tag(m, path, arm="V"):
    """the tag's range as D155's gate defined it (pre-registered 2026-10-17, kept by D155b-d): its three re-run seeds and
    B3's three where B3 has the measure (d155_gate_sum.tag)"""
    return [x for src in (R, B3) for x in (g(src.get(f"{m}40k_{arm}{s}"), path) for s in SEEDS) if x is not None]


def yes(b):
    return "–" if b is None else ("yes" if b else "NO")


def f_(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def rng(v, nd=3):
    return f"{min(v):.{nd}f}–{max(v):.{nd}f}" if v else "–"


def end_ok(m, path, better, x):
    tv = tag(m, path)
    if not tv or x is None:
        return None
    return x >= min(tv) if better == "high" else x <= max(tv)


crit = {}
for m in MESH:
    a = R.get(f"{m}40k_A3DL97")
    if a is None:
        continue
    c = {}
    c["1"] = bool(tag(m, "move_after_1s")) and a["move_after_1s"] >= max(tag(m, "move_after_1s")) + 0.15
    c["2 letter"] = {k: end_ok(m, k, b, g(a, k)) for k, b in END}
    three = [R.get(f"{m}40k_A3DL{s}") for s in SEEDS]
    if m in ("dragon", "homer") and all(three) and all(g(r, k) is not None for r in three for k, _ in END):
        c["2 seeds"] = {k: end_ok(m, k, b, float(np.median([g(r, k) for r in three]))) for k, b in END}
        c["2 seeds vals"] = {k: [g(r, k) for r in three] for k, _ in END}
    else:
        c["2 seeds"] = c["2 letter"]
    pp = R.get(f"{m}40k_A3DP97")
    c["lead"] = (None if pp is None or g(a, "end.yard_sil") is None or g(pp, "end.yard_sil") is None
                 else g(a, "end.yard_sil") < g(pp, "end.yard_sil"))
    c["3"] = bool(tag(m, "kin_end")) and a["kin_end"] <= 1.25 * max(tag(m, "kin_end"))
    c["4"] = bool(tag(m, "kin_peak")) and a["kin_peak"] <= 0.5 * min(tag(m, "kin_peak"))
    ta, tb = tag(m, "apart_w3_12.max"), tag(m, "visible_w3_10.bump_max")
    c["5 apart"] = None if not ta or g(a, "apart_w3_12.max") is None else g(a, "apart_w3_12.max") <= max(ta)
    c["5 bumps"] = (None if not tb or g(a, "visible_w3_10.bump_max") is None
                    else g(a, "visible_w3_10.bump_max") <= max(tb))
    jr = g(a, "J_ramp_max")
    c["6a"] = None if jr is None else jr <= 1.03
    tm, tq = tag(m, "J_end.mean"), tag(m, "J_end.p99")
    jm, jq = g(a, "J_end.mean"), g(a, "J_end.p99")
    c["6b mean"] = (None if not tm or jm is None
                    else (min(tm) <= jm <= max(tm) or abs(jm - 1) <= min(abs(x - 1) for x in tm)))
    c["6b p99"] = None if not tq or jq is None else jq <= max(tq)
    crit[m] = c


def n_ok(f):
    return sum(1 for c in crit.values() if f(c))


cnt = {k: n_ok(lambda c, k=k: c[k]) for k in ("1", "lead", "3", "4", "5 apart", "5 bumps", "6a", "6b mean", "6b p99")}
letter = {k: n_ok(lambda c, k=k: c["2 letter"][k]) for k, _ in END}
seeds = {k: n_ok(lambda c, k=k: c["2 seeds"][k]) for k, _ in END}
item2 = all(v >= 4 for v in letter.values()) or all(v >= 4 for v in seeds.values())
passed = len(crit) == 5 and all(v >= 4 for v in cnt.values()) and item2
p("## D155d criteria (A3D render arm against the tag's three seeds; pre-registered 2026-10-21 12:18 CDT)")
p("| mesh | 1 | 2 IoU | 2 yard sil | 2 yard shading | 2 display | 2 thin | 2 with the 3-seed median: IoU / sil / shading / "
  "display / thin | lead | 3 | 4 | 5 apart | 5 bumps | 6a | 6b mean | 6b p99 |")
p("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for m, c in crit.items():
    L, S3 = c["2 letter"], c["2 seeds"]
    p(f"| {m} | {yes(c['1'])} | " + " | ".join(yes(L[k]) for k, _ in END) + " | "
      + " / ".join(yes(S3[k]) for k, _ in END) + f" | {yes(c['lead'])} | {yes(c['3'])} | {yes(c['4'])} | "
      f"{yes(c['5 apart'])} | {yes(c['5 bumps'])} | {yes(c['6a'])} | {yes(c['6b mean'])} | {yes(c['6b p99'])} |")
p()
p("counts of 5: " + json.dumps(cnt) + "; item 2 by the letter " + json.dumps(letter) + "; with the 3-seed median "
  + json.dumps(seeds))
p(f"verdict: {'PASS' if passed else 'FAIL'} (each item on >= 4 of 5; item 2 on either reading)")
p()
p("## Item 2 with the seeds (dragon, homer: A3D L97 / L99 / L101 and their median; the tag's three seeds)")
p("| mesh | measure | tag | A3D 97 / 99 / 101 | median |")
p("|---|---|---|---|---|")
for m in ("dragon", "homer"):
    c = crit.get(m, {})
    for k, _ in END:
        v = c.get("2 seeds vals", {}).get(k)
        med = f_(float(np.median(v)), 4) if v else "–"
        p(f"| {m} | {k[4:]} | {rng(tag(m, k), 4)} | " + (" / ".join(f_(x, 4) for x in v) if v else "–") + f" | {med} |")
p()
p("## The volume (render arm; the tag re-run with the record VE97 / D155b re-run A3GE97 / D155d A3DL97)")
p("| mesh | ramp's largest mean J | mean J at 3.3 s | det Fp p99 at 3.3 s | end mean J: tag's 3 seeds / A3D | end J p99: tag's largest / A3D |")
p("|---|---|---|---|---|---|")
for m in MESH:
    ve, ge, d = (R.get(f"{m}40k_{x}") for x in ("VE97", "A3GE97", "A3DL97"))
    tq = tag(m, "J_end.p99")
    p(f"| {m} | {f_(g(ve, 'J_ramp_max'), 4)} / {f_(g(ge, 'J_ramp_max'), 4)} / {f_(g(d, 'J_ramp_max'), 4)} | "
      f"{f_(g(ve, 'J_at33.mean'), 4)} / {f_(g(ge, 'J_at33.mean'), 4)} / {f_(g(d, 'J_at33.mean'), 4)} | "
      f"{f_(g(ve, 'J_at33.Jp_p99'))} / {f_(g(ge, 'J_at33.Jp_p99'))} / {f_(g(d, 'J_at33.Jp_p99'))} | "
      f"{rng(tag(m, 'J_end.mean'), 4)} / {f_(g(d, 'J_end.mean'), 4)} | {f_(max(tq) if tq else None)} / {f_(g(d, 'J_end.p99'))} |")
p()
p("## The elastic energy the body carries (sum V0 psi(F Fp^-1); tag VE97 / A3GE97 / A3DL97)")
p("| mesh | largest on [0, 3.3 s] (A3D's time s) | its volumetric part, largest | at 3.3 s | at the end | J_e p01..p99 on the ramp, A3D | J_e p99 at the end | kinetic peak over 3-4.5 s |")
p("|---|---|---|---|---|---|---|---|")
for m in MESH:
    ve, ge, d = (R.get(f"{m}40k_{x}") for x in ("VE97", "A3GE97", "A3DL97"))

    def e(r, k):
        return g(r, f"elastic.{k}")
    p(f"| {m} | {f_(e(ve, 'ramp_max'), 2)} / {f_(e(ge, 'ramp_max'), 2)} / {f_(e(d, 'ramp_max'), 2)} ({f_(e(d, 'ramp_max_t'), 2)}) | "
      f"{f_(e(ve, 'ramp_vol_max'), 2)} / {f_(e(ge, 'ramp_vol_max'), 2)} / {f_(e(d, 'ramp_vol_max'), 2)} | "
      f"{f_(e(ve, 'at33'), 2)} / {f_(e(ge, 'at33'), 2)} / {f_(e(d, 'at33'), 2)} | "
      f"{f_(e(ve, 'end'), 2)} / {f_(e(ge, 'end'), 2)} / {f_(e(d, 'end'), 2)} | "
      f"{f_(e(d, 'Je_p01_ramp_min'))}..{f_(e(d, 'Je_p99_ramp_max'))} | "
      f"{f_(e(ve, 'Je_p99_end'))} / {f_(e(ge, 'Je_p99_end'))} / {f_(e(d, 'Je_p99_end'))} | "
      f"{f_(g(ve, 'kin_release'), 4)} / {f_(g(ge, 'kin_release'), 4)} / {f_(g(d, 'kin_release'), 4)} |")
p()
p("## The end against the independent sample (render arm L, twin P: the tag's three seeds / A3G / A3C / A3D, seed 97)")
p("| mesh, arm | IoU ind. | yardstick sil. 1e-3 | yardstick shading 1e-3 | display | thin |")
p("|---|---|---|---|---|---|")
for m in MESH:
    for arm in ("L", "P"):
        cells = []
        for k, _ in END:
            sc = 1e3 if "yard" in k else 1.0
            tv = [x * sc for x in tag(m, k, "V" if arm == "L" else "W")]
            vals = [g(R.get(f"{m}40k_{x}{arm}97"), k) for x in ("A3G", "A3C", "A3D")]
            cells.append(f"{rng(tv, 4)} / " + " / ".join(f_(None if v is None else v * sc, 4) for v in vals))
        p(f"| {m} {arm} | " + " | ".join(cells) + " |")
p()
p("## Timing, motion, floating pieces, the render's influence, cost (render arm: tag V97 / A3G / A3D)")
p("| mesh | move after 1 s | physics / render P90 s (A3D) | peak kinetic | kinetic end 1e-5 | apart w3-12 | bumps w3-10 | ramp raw cos | render's merit share on the ramp | windows | minutes |")
p("|---|---|---|---|---|---|---|---|---|---|---|")
for m in MESH:
    rs = [R.get(f"{m}40k_V97"), R.get(f"{m}40k_A3GL97"), R.get(f"{m}40k_A3DL97")]

    def j(k, nd=3, s=1.0):
        return " / ".join(f_(None if g(r, k) is None else g(r, k) * s, nd) for r in rs)
    d = rs[2]
    p(f"| {m} | {j('move_after_1s')} | {f_(g(d, 'lin.P90_s'), 2)} / {f_(g(d, 'lin.R90_s'), 2)} | {j('kin_peak')} | "
      f"{j('kin_end', 2, 1e5)} | {j('apart_w3_12.max', 0)} | {j('visible_w3_10.bump_max', 0)} | {j('ramp.raw_cos')} | "
      f"{j('ramp.merit_share')} | {j('windows', 0)} | {j('minutes', 2)} |")
p()
p("## Twins (lambda = 0) against their render arms, A3D: the ramp's largest mean J / end mean J / end J p99")
p("| mesh | render arm L | twin P |")
p("|---|---|---|")
for m in MESH:
    L, P_ = R.get(f"{m}40k_A3DL97"), R.get(f"{m}40k_A3DP97")
    p(f"| {m} | {f_(g(L, 'J_ramp_max'), 4)} / {f_(g(L, 'J_end.mean'), 4)} / {f_(g(L, 'J_end.p99'))} | "
      f"{f_(g(P_, 'J_ramp_max'), 4)} / {f_(g(P_, 'J_end.mean'), 4)} / {f_(g(P_, 'J_end.p99'))} |")
md.close()

INK2, MUTED, GRID, SURF = "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9, "savefig.dpi": 160, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "lines.linewidth": 2.0, "legend.frameon": False, "legend.fontsize": 7.5})
fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.2))
for jx, m in enumerate(("bunny", "dragon", "homer")):
    for name, col, lab in ((f"{m}40k_VE97", MUTED, "the tag (no moving target)"),
                           (f"{m}40k_A3GE97", BLUE, "D155b: the volume made rest volume on the way"),
                           (f"{m}40k_A3DL97", ORANGE, "D155d: the volume held elastically on the way")):
        r = R.get(name)
        if r is None:
            continue
        t, J = r["curves"]["t"], r["curves"]["J"]
        ax[0, jx].plot([a for a, b in zip(t, J) if b is not None], [b for b in J if b is not None], color=col, label=lab)
        E = r["curves"].get("E_vol") or []
        ax[1, jx].plot([a for a, b in zip(t, E) if b is not None], [b for b in E if b is not None], color=col, label=lab)
    ax[0, jx].axhline(1.0, color=MUTED, lw=1.0)
    ax[0, jx].set_ylabel("the body's volume over its source volume")
    ax[0, jx].set_title(f"{m} 40k: does the body give the volume back?", fontsize=9)
    ax[1, jx].set_ylabel("elastic energy held in the volume")
    ax[1, jx].set_title(f"{m} 40k: what the body holds elastically instead", fontsize=9)
    for i in (0, 1):
        ax[i, jx].axvline(3.26, color=INK2, lw=0.9, ls=":")
        ax[i, jx].set_xlim(0, 8)
        ax[i, jx].set_xlabel("simulated time (s); dotted: the target stops moving")
        ax[i, jx].legend(loc="upper right")
fig.suptitle("D155d: the targets on the way no longer set the body's rest volume", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_volume_d.png")
print("written", OUT)
