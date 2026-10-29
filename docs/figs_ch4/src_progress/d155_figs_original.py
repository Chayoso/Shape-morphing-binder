"""d155_figs.py OUT_DIR -- D155's figures from output/gpu/d155 (records.json, frames_summary.json, b7_*.json); CPU.
fig1_progress.png        do the physics core and the render term fall together? (linear and log progress, seconds)
fig2_motion_phases.png   where the visible motion happens, what is still to come, and who causes it per window
fig3_tail.png            what the tail after 3 s still changes (each measure over its end value)"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155"
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                      "#008300", "#4a3aa7")
SEQ = ["#0d366b", "#1d5aa6", "#2a78d6", "#79aef0", "#c4dcfa"]     # one hue, dark (early) to light (late)
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "savefig.dpi": 160,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "legend.frameon": False, "legend.fontsize": 7.5})
R = json.load(open(f"{D}/records.json"))
FR = json.load(open(f"{D}/frames_summary.json"))
MESH = ("bunny", "dragon", "teapot", "bob", "armadilo", "homer")


def runs(mesh, arm):
    return sorted((r for r in R if r["mesh"] == mesh and r["arm"] == arm), key=lambda r: r["seed"] != 97)


def runmed(y, w=5):
    y = np.asarray(y, float)
    h = w // 2
    return np.array([np.nanmedian(y[max(0, i - h):i + h + 1]) for i in range(len(y))])


# ---------------- fig 1: progress ----------------
fig, ax = plt.subplots(2, 2, figsize=(13.5, 7.6))
for i, mesh in enumerate(("bunny", "dragon")):
    for j, scale in enumerate(("lin", "log")):
        a = ax[i, j]
        for r in runs(mesh, "L"):
            c = r["curves"]
            t = np.array(c["t"])
            bold = r["seed"] == 97
            kw = dict(lw=2.0 if bold else 0.8, alpha=1.0 if bold else 0.45)
            a.plot(t, c["c" + scale]["P"], color=BLUE, label="physics core (transport + end drift + released motion)"
                   if bold else None, **kw)
            a.plot(t, c["c" + scale]["R"], color=ORANGE, label="render term (silhouette + shading)" if bold else None,
                   **kw)
            if scale == "lin" and "kmove" in c:
                st = np.array(c["kmove"]["state"]) / 240.0
                mv = np.cumsum(c["kmove"]["move"])
                a.plot(st, mv / mv[-1], color=AQUA, ls="--", label="particle movement done (share of the run's total)"
                       if bold else None, **kw)
        r97 = runs(mesh, "L")[0]
        a.axvline(r97["T"] / 3, color=MUTED, ls=":", lw=1.0)
        a.text(r97["T"] / 3, 0.04, " a third of the run", color=INK2, fontsize=7.5)
        a.set_ylim(-0.03, 1.05)
        a.set_xlim(0, r97["T"])
        a.set_xlabel("simulated time (s)")
        if scale == "lin":
            a.set_ylabel("share of the drop already done")
            a.set_title(f"{mesh} 300k render arm: linear progress (value)\n"
                        f"physics 90 % at {r97['lin']['P90_s']:.2f} s, render 90 % at {r97['lin']['R90_s']:.2f} s "
                        f"(seed 97; thin lines: other seeds)", fontsize=9)
        else:
            a.set_ylabel("share of the decades already fallen")
            a.set_title(f"{mesh} 300k render arm: log progress (decades)\n"
                        f"physics 90 % at {r97['log']['P90_s']:.1f} s, render 90 % at {r97['log']['R90_s']:.1f} s",
                        fontsize=9)
        a.legend(loc="lower right")
fig.suptitle("D155 (tag freeze-2026-10-15, B1): both terms fall in the first 1-2 s, together; the log scale shows the "
             "long sub-pitch tail of both", fontsize=10)
fig.tight_layout()
fig.savefig(f"{out}/fig1_progress.png")
plt.close(fig)

# ---------------- fig 2: motion by phase ----------------
fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.4))
a = ax[0, 0]
BINS = ("0-1", "1-2", "2-3", "3-6", "6-end")
LAB = ("0-1 s", "1-2 s", "2-3 s", "3-6 s", "6 s to the end")
for k, mesh in enumerate(MESH):
    rr = runs(mesh, "L")
    left = 0.0
    for b, (key, lab) in enumerate(zip(BINS, LAB)):
        v = float(np.median([r["move_bins_s"][key] for r in rr]))
        a.barh(k, v, left=left, color=SEQ[b], edgecolor=SURF, linewidth=1.5, height=0.62, label=lab if k == 0 else None)
        if v > 0.06:
            a.text(left + v / 2, k, f"{100 * v:.0f}%", ha="center", va="center", fontsize=7.5,
                   color="white" if b < 3 else INK)
        left += v
    T = float(np.median([r["T"] for r in rr]))
    a.text(1.01, k, f"{T:.0f} s run", va="center", fontsize=7.5, color=INK2)
a.set_yticks(range(len(MESH)))
a.set_yticklabels(MESH)
a.invert_yaxis()
a.set_xlim(0, 1.13)
a.set_xticks([0, .25, .5, .75, 1])
a.set_xlabel("share of the run's summed particle movement (render arm, median of seeds)")
a.set_title("where the movement happens, by simulated time", fontsize=9)
a.legend(loc="lower center", bbox_to_anchor=(0.45, -0.32), ncol=5)
a.grid(False)

a = ax[1, 0]
cols = dict(zip(MESH, (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET)))
for mesh in MESH:
    e = FR.get(f"{mesh}300k_L97")
    if not e:
        continue
    t = np.array(e["t"])
    y = np.maximum(np.array(e["lag_n"]), 1e-4)
    a.semilogy(t[t > 0.25], y[t > 0.25], color=cols[mesh], label=f"{mesh} (1 pitch = {e['px_per_pitch']:.0f} px at 4K)")
a.axhline(0.1, color=MUTED, ls=":", lw=1.0)
a.text(0.3, 0.11, "0.1 pitch (about 1 px at 4K)", fontsize=7.5, color=INK2)
a.set_xlim(0, 12)
a.set_ylim(1e-2, 30)
a.set_xlabel("simulated time (s)")
a.set_ylabel("surface displacement still to come\n(normal part, rms over the outer layer, pitches)")
a.set_title("how far the visible surface still has to move (seed 97 render arm)", fontsize=9)
a.legend(loc="upper right")

for i, mesh in enumerate(("bunny", "dragon")):
    a = ax[i, 1]
    b = json.load(open(f"{D}/b7_{mesh}.json"))
    rows = [w for w in b["rows"] if w["chained"] and w["t"] is not None]
    t = np.array([w["t"] for w in rows])
    a.semilogy(t, [w["layer"]["D"] for w in rows], color=INK, label="actual move of the window")
    a.semilogy(t, [w["layer"]["D0"] for w in rows], color=BLUE, lw=1.0,
               label="the same window without the render (physics alone)")
    a.semilogy(t, [w["layer"]["d"] for w in rows], color=ORANGE, label="render-caused part (with minus without)")
    a.set_xlabel("simulated time (s)")
    a.set_ylabel("outer-layer move per window, rms (pitches)")
    ph = json.load(open(f"{D}/b7_phases.json"))[mesh]["phases"]
    txt = "render's share of the actual move\n(projection): " + ", ".join(
        f"{k} {100 * ph[k]['proj']:.0f}%" for k in ("0-1 s", "1-2 s", "2-3 s", "3-6 s", "6 s-end") if k in ph)
    a.text(0.98, 0.62, txt, transform=a.transAxes, ha="right", fontsize=7.5, color=INK2)
    a.set_title(f"{mesh} 300k (B7, seed 97): who moves the surface in each window", fontsize=9)
    a.legend(loc="upper right")
fig.suptitle("D155: the visible deformation is front-loaded; late windows move the surface by hundredths of a pitch, "
             "mostly by the physics' own refit", fontsize=10)
fig.tight_layout()
fig.savefig(f"{out}/fig2_motion_phases.png")
plt.close(fig)

# ---------------- fig 3: the tail ----------------
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
for i, mesh in enumerate(("bunny", "dragon")):
    a = ax[i]
    r = runs(mesh, "L")[0]
    c = r["curves"]
    t = np.array(c["t"])
    m = t >= 2.0
    for key, lab, col in (("TE", "transport (geometry energy)", BLUE), ("R", "the run's own render term", ORANGE),
                          ("kin", "kinetic energy at the window end", YELLOW)):
        y = np.array(c[key])
        a.semilogy(t[m], y[m] / y[-1], color=col, label=lab)
    if "yard" in c:
        ys = np.array(c["yard"]["state"]) / 240.0
        Y = np.array(c["yard"]["Y"])
        mm = ys >= 2.0
        a.semilogy(ys[mm], Y[mm] / Y[-1], color=AQUA, label="yardstick: exterior silhouette + shading vs an "
                                                            "independent sample")
    if "disp" in c:
        es = np.array(c["disp"]["state"]) / 240.0
        Dd = np.array(c["disp"]["D"])
        mm = es >= 2.0
        a.semilogy(es[mm], Dd[mm] / Dd[-1], color=MAGENTA, label="display front 1 - IoU vs the independent sample")
    a.axhline(1.0, color=MUTED, lw=0.8)
    a.set_xlim(2.0, r["T"])
    a.set_xlabel("simulated time (s)")
    a.set_ylabel("value / its value at the delivered end")
    a.set_title(f"{mesh} 300k render arm, seed 97: what the tail after 2 s still lowers", fontsize=9)
    a.legend(loc="upper right")
fig.suptitle("D155: in the tail every measure keeps falling slowly, the kinetic energy and the transport most; the surface "
             "still has 0.2-0.3 pitch to go at 3 s on the bunny, 0.8-1.2 on the dragon", fontsize=10)
fig.tight_layout()
fig.savefig(f"{out}/fig3_tail.png")
plt.close(fig)
print("figures written to", out)
