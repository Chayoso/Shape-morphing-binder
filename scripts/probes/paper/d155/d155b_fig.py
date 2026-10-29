"""d155b_fig.py OUT_PNG -- D155b's root-cause figure: (top) the volume the in-between target asks for, (the target's median
8th-neighbour spacing over the final target's)^3, per built pack (the run log's "[target] shading target: spacing" lines,
the first being the final pack), against the window's time; (bottom) the body's volume over its rest volume (the records'
Jx_mean), the tag (seed 97) against A3 and A3G. Bunny, dragon, homer at 40k, render arm."""
import json
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "savefig.dpi": 160, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.8, "legend.frameon": False,
                     "legend.fontsize": 7.5})
meshes = ("bunny", "dragon", "homer")
fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.0))
for j, m in enumerate(meshes):
    log = open(f"{O}/{m}40k_A3GL97.log").read()
    sp = [float(x) for x in re.findall(r"shading target: spacing ([0-9.]+)", log)]
    a = next(iter(json.load(open(f"{O}/{m}40k_A3GL97.json"))["arms"].values()))
    built = [h for h in a["history"] if (h.get("morph_build_s") or 0) > 0 and (h.get("morph_s") or 1) < 1]
    t = [h["frame_end"] / 240 if h.get("frame_end") else None for h in built]
    vol = [(s / sp[0]) ** 3 for s in sp[1:1 + len(built)]]
    pairs = [(x, v) for x, v in zip(t, vol) if x is not None]
    ax[0, j].plot([p[0] for p in pairs], [p[1] for p in pairs], color=ORANGE, marker="o", ms=3)
    ax[0, j].axhline(1.0, color=MUTED, lw=0.9)
    ax[0, j].set_ylabel("volume the in-between target asks for\n(over the body's)")
    ax[0, j].set_title(f"{m} 40k: the target on the way is roomier than the body", fontsize=9)
    for name, col, lab in ((f"{m}40k_V97", MUTED, "the tag"), (f"{m}40k_A3L97", ORANGE, "A3 (moving target)"),
                           (f"{m}40k_A3GL97", BLUE, "A3G (moving target, u on the final)")):
        a = next(iter(json.load(open(f"{O}/{name}.json"))["arms"].values()))
        acc = [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit") and h.get("Jx_mean")]
        ax[1, j].plot([h["frame_end"] / 240 for h in acc], [h["Jx_mean"] for h in acc], color=col, label=lab)
    ax[1, j].axhline(1.0, color=MUTED, lw=0.9)
    ax[1, j].set_ylabel("the body's volume over its rest volume")
    ax[1, j].set_title(f"{m} 40k: the body swells to follow it and keeps part of it", fontsize=9)
    for i in (0, 1):
        ax[i, j].set_xlim(0, 10)
        ax[i, j].set_xlabel("simulated time (s)")
    ax[1, j].legend(loc="upper right")
fig.suptitle("D155b: why the end lags the tag: the displacement interpolation's in-between shapes hold more volume than "
             "the volume-conserving body, which dilates and assimilates it", fontsize=10)
fig.tight_layout()
fig.savefig(sys.argv[1])
print("written", sys.argv[1])
