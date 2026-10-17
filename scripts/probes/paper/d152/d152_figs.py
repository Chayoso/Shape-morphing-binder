"""d152_figs.py OUT_DIR -- D152's figures from output/gpu/d152 (h345v2, tables, iou_calib.json); CPU only.
fig1_displacement_persistence.png  H3: the render-caused displacement per window and what becomes of it
fig2_render_loss_floor.png         H2: the run's own silhouette term per window, arm and twin, against its floor
fig3_error_forms.png               H1: the share of the twin's excess over the floor that the arm removes, per measure;
                                   the IoU's response to a uniform normal shift of the surface
fig4_reach_pcgrad.png              H4 / H5: what a control step realises of the render's demand; what PCGrad removes"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = "/data/relcfd/chayo/physmorph_v2/output/gpu/d152"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "savefig.dpi": 160,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "legend.frameon": False, "legend.fontsize": 7.5})


def runmed(x, y, w=7):
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    m = np.array([np.nanmedian(y[max(0, i - w // 2): i + w // 2 + 1]) for i in range(len(y))])
    return x, m


H = {m: json.load(open(f"{D}/h345v2/{m}_h345.json")) for m in ("bunny", "dragon")}

# ---------------- fig1 ----------------
fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
for i, m in enumerate(("bunny", "dragon")):
    rows = [r for r in H[m]["rows"] if r.get("t")]
    t = np.array([r["t"] for r in rows])
    ax = axs[i, 0]
    for key, col, lab in (("dmed_rim3", ORANGE, "silhouette rim"), ("dmed_layer", BLUE, "outer layer"),
                          ("dmed_thin", AQUA, "thin parts"), ("dmed_bulk", MUTED, "bulk (> 2 pitches deep)")):
        y = np.array([r[key] for r in rows])
        ax.plot(t, y, color=col, lw=1.4, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("median |x(render) - x(lambda = 0)|, pitches")
    ax.set_title(f"{m} 300k: what the render moves in one window\n(the same window from the same start, render on vs off)",
                 fontsize=9)
    ax.legend(loc="upper right")
    ax = axs[i, 1]
    tl = [r["t"] for r in rows if r.get("undo_phys_next") is not None]
    ul = [r["undo_phys_next"] for r in rows if r.get("undo_phys_next") is not None]
    t5 = [r["t"] for r in rows if r.get("c_5") is not None]
    c5 = [r["c_5"] for r in rows if r.get("c_5") is not None]
    ax.scatter(tl, ul, s=9, color=ORANGE, alpha=0.45, linewidths=0)
    ax.plot(*runmed(tl, ul), color=ORANGE, label="taken back by the next window\nwithout the render (share)")
    ax.scatter(t5, c5, s=9, color=BLUE, alpha=0.45, linewidths=0)
    ax.plot(*runmed(t5, c5), color=BLUE, label="the arm's own drift along it\nover the next 5 windows (share)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.set_ylim(-2, 2)
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("share of the window's render displacement")
    ax.set_title("what becomes of it (outer layer; running median of 7)", fontsize=9)
    ax.legend(loc="lower right")
    ax = axs[i, 2]
    late = [r for r in rows if r["t"] >= 3.0 and r.get("sil_gain") is not None and r.get("sil_drift_phys") is not None]
    tt = np.array([r["t"] for r in late])
    g = np.cumsum([r["sil_gain"] for r in late])
    d = np.cumsum([r["sil_drift_phys"] for r in late])
    net = np.array([late[0]["d_sil_r"] - r["d_sil_r"] for r in late])
    ax.plot(tt, g, color=BLUE, label="summed in-window gains of the render")
    ax.plot(tt, d, color=ORANGE, label="summed give-back of a window without it")
    ax.plot(tt, net, color=INK, lw=1.2, ls="--", label="the arm's actual net decrease")
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("silhouette term (the run's own, 188 px)")
    ax.set_title(f"the silhouette term from 3 s on: gains vs give-back\n(give-back / gain = {d[-1] / g[-1]:.2f})", fontsize=9)
    ax.legend(loc="upper left")
fig.savefig(os.path.join(out, "fig1_displacement_persistence.png"))
plt.close(fig)

# ---------------- fig2 ----------------
series = json.load(open(f"{D}/tables/window_series.json"))
h2 = json.load(open(f"{D}/tables/h2_table.json")) if os.path.exists(f"{D}/tables/h2_table.json") else {}
MESHES = ["bunny", "dragon", "teapot", "bob", "armadilo", "homer"]
D114 = {"bunny": 7.781673921272159e-4, "dragon": 6.567617529071867e-4}
fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True, sharey=False)
for ax, m in zip(axs.ravel(), MESHES):
    for name, s in series.items():
        if not name.startswith(m + "_"):
            continue
        arm = name.split("_")[1][0]
        ax.plot(s["t"], s["d_sil"], color=BLUE if arm == "L" else ORANGE, lw=1.0, alpha=0.8)
    fl = [v["floor_sil"] for k, v in h2.items() if k.startswith(m + "_")]
    ow = [v.get("own_sil") for k, v in h2.items() if k.startswith(m + "_") and v.get("own_sil")]
    if not fl and m in D114:                     # D114's measurement (seed 97; the same target pictures as the tag's)
        fl = [D114[m]]
    if fl:
        ax.axhspan(min(fl), max(fl) * 1.0001, color=INK2, alpha=0.18, lw=0)
        ax.axhline(np.median(fl), color=INK, lw=1.2, ls="--")
        ax.text(0.98, np.median(fl), "floor",
                transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7, color=INK)
    if ow:
        ax.axhline(np.median(ow), color=MUTED, lw=1.0, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(5e-5, 0.3)
    ax.set_title(m, fontsize=9)
    ax.set_xlabel("simulated time (s)")
    ax.set_ylabel("silhouette term (own pictures)")
axs[0, 0].plot([], [], color=BLUE, label="render arm (all seeds)")
axs[0, 0].plot([], [], color=ORANGE, label="lambda = 0 twin (all seeds)")
axs[0, 0].plot([], [], color=INK, ls="--", label="floor: an independent sample (median; band = seeds)")
axs[0, 0].plot([], [], color=MUTED, ls=":", label="the run's own target sample")
axs[0, 0].legend(loc="upper right")
fig.suptitle("The run's own silhouette term at every committed window against what an independent sample of the target "
             "scores (300k, the tag)", fontsize=10)
fig.savefig(os.path.join(out, "fig2_render_loss_floor.png"))
plt.close(fig)

# ---------------- fig3 ----------------
h1 = json.load(open(f"{D}/tables/h1_table.json"))
cal = json.load(open(f"{D}/iou_calib.json"))
MEAS = [("sil_iou_ind", "silhouette IoU\n(particle splat, 128 px)"), ("yard_par_sil", "yardstick silhouette,\nparticles (96 px)"),
        ("yard_ext_sil", "yardstick silhouette,\nexterior (96 px)"), ("yard_ext_pbr", "yardstick shading,\nexterior (96 px)"),
        ("disp_front", "display front 1 - IoU"), ("disp_back", "display back 1 - IoU"),
        ("disp_front_diff", "display front\nimage difference")]
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
ax = axs[0]
cols = {"bunny": BLUE, "dragon": ORANGE, "teapot": AQUA, "bob": VIOLET, "armadilo": INK2, "homer": MUTED}
mk = {"bunny": "o", "dragon": "s", "teapot": "^", "bob": "D", "armadilo": "v", "homer": "P"}
for j, (key, lab) in enumerate(MEAS):
    for k, m in enumerate(MESHES):
        v = h1.get(m, {}).get(key)
        if not v or not v.get("gap_share"):
            continue
        y = j + (k - 2.5) * 0.11
        med_, lo, hi = v["gap_share"]
        lo, hi, med_ = max(lo, -0.5), min(hi, 2.0), min(max(med_, -0.5), 2.0)
        ax.plot([lo, hi], [y, y], color=cols[m], lw=1.0, alpha=0.7)
        ax.plot(med_, y, mk[m], color=cols[m], ms=5, label=m if j == 0 else None)
ax.axvline(0, color=INK2, lw=0.8)
ax.axvline(1, color=INK, lw=1.0, ls="--")
ax.text(1.02, -0.45, "arm at the floor", fontsize=7, color=INK)
ax.set_yticks(range(len(MEAS)), [l for _, l in MEAS])
ax.invert_yaxis()
ax.set_xlim(-0.5, 2.0)
ax.set_xlabel("share of the twin's excess over the floor that the render arm removes\n(median over seeds, bar = seed range; "
              "0 = no gain, 1 = at the floor)")
ax.legend(loc="lower right", ncol=2)
ax.set_title("H1: the same runs, read in error form against each measure's floor (300k, the tag)", fontsize=9)
ax = axs[1]
for spec, r in cal.items():
    m = r["mesh"]
    xs = sorted(float(k) for k in r["shift"])
    ys = [r["shift"][str(x) if str(x) in r["shift"] else repr(x)] for x in xs]
    ax.plot(xs, ys, "-o", ms=3, color=BLUE if m == "bunny" else ORANGE, lw=1.0, alpha=0.8,
            label=f"{m}" if r["seed"] == 97 else None)
    dLP = r["res128"]["err_L"] - r["res128"]["err_P"]
    ax.axhline(dLP, color=BLUE if m == "bunny" else ORANGE, lw=0.8, ls=":", alpha=0.8)
ax.axhline(0, color=INK2, lw=0.8)
ax.axvline(0, color=INK2, lw=0.8)
ax.set_xlabel("the render arm's outer layer moved along its outward normal (pitches)")
ax.set_ylabel("change of 1 - IoU (the gallery's silhouette IoU)")
ax.set_title("how much a uniform shift moves the IoU\n(dotted: the render's whole effect, arm - twin, per seed)", fontsize=9)
ax.legend(loc="upper right")
fig.savefig(os.path.join(out, "fig3_error_forms.png"))
plt.close(fig)

# ---------------- fig4 ----------------
h5 = json.load(open(f"{D}/tables/h5_table.json"))
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
for m, ls in (("bunny", "-"), ("dragon", "--")):
    rows = [r for r in H[m]["rows"] if r.get("t")]
    t = [r["t"] for r in rows]
    ax = axs[0]
    ax.plot(*runmed(t, [r["reach_r_u"] for r in rows], 5), color=AQUA, ls=ls, label=f"render demand via u, {m}")
    ax.plot(*runmed(t, [r["reach_r_dfc"] for r in rows], 5), color=BLUE, ls=ls, label=f"render demand via dFc, {m}")
    ax.plot(*runmed(t, [r["reach_p_dfc"] for r in rows], 5), color=ORANGE, ls=ls, label=f"physics demand via dFc, {m}")
    ax = axs[1]
    ax.plot(*runmed(t, [r["low_r"] for r in rows], 5), color=BLUE, ls=ls, label=f"render, {m}")
    ax.plot(*runmed(t, [r["low_p"] for r in rows], 5), color=ORANGE, ls=ls, label=f"physics, {m}")
axs[0].set_ylim(0, 1)
axs[0].set_xscale("log")
axs[0].set_xlabel("simulated time (s)")
axs[0].set_ylabel("cos(wanted move, realised move)")
axs[0].set_title("H4: how much of a channel's position demand one\ncontrol step realises (running median of 5)", fontsize=9)
axs[0].legend(loc="center right", fontsize=6.5, bbox_to_anchor=(1.0, 0.62))
axs[1].set_ylim(0, 1)
axs[1].set_xscale("log")
axs[1].set_xlabel("simulated time (s)")
axs[1].set_ylabel("share of |demand|^2 at the MPM cell scale")
axs[1].set_title("H4: the demand's cell-scale part\n(B-spline P2G then G2P on the MPM grid)", fontsize=9)
axs[1].legend(loc="upper right")
ax = axs[2]
for m in MESHES:
    s = series.get(f"{m}_L97")
    if not s:
        continue
    c = np.array([v if v is not None else np.nan for v in s["g_raw_cos"]])
    rem = np.where(c < 0, -c, 0.0)
    ax.plot(*runmed(s["t"], rem, 9), color=cols[m], lw=1.2, label=m)
ax.set_xlabel("simulated time (s)")
ax.set_ylabel("share of |g_render| removed by PCGrad")
ax.set_ylim(0, 0.2)
ax.set_title("H5: PCGrad's cut of the render gradient\n(|cos| when cos < 0, seed 97, running median of 9)", fontsize=9)
ax.legend(loc="upper left", ncol=2)
fig.savefig(os.path.join(out, "fig4_reach_pcgrad.png"))
plt.close(fig)
print("written", out)
