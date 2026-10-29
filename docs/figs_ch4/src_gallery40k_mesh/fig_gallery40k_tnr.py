"""fig_gallery40k_tnr.py G40_TABLE_JSON OUT_PNG -- FIGS_V2 (2026-10-22): the dissertation's Figure (ch4:fig_g40) drawn
directly instead of cut from D154's five-panel plot (d154_g40_fig.py): the same three panels (exact binary 1 - IoU at
1024 px, outline shape error, normal angle at the nearest mesh point), the same marks, colours, limits and words, no
title and no bottom label (the cut version had neither), every text in Times New Roman (the user, 2026-10-22).
Per mesh (19 rows): the 40k render arm's error over its twin's (dot = median over seeds 97 / 99 / 101, bar = range; grey
tick = a perfect particle sample, own / W; dashed = the twin)."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/fonts")
import tnr  # noqa: E402

T = json.load(open(sys.argv[1]))
BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
tnr.mpl()
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c8c7c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#ecebe7", "grid.linewidth": 0.6})
panels = [("iou1024", "1 - IoU, exact silhouettes\n(1024 px)"), ("outline_sd", "outline shape error\n(offset spread)"),
          ("normal_cl", "normal angle at the\nnearest mesh point")]
meshes = list(T["outline_sd"]["meshes"])
fig, axes = plt.subplots(1, len(panels), figsize=(8.06, 5.9), sharey=True, gridspec_kw=dict(width_ratios=[1, 1, 1]))
for ax, (k, title) in zip(axes, panels):
    for i, mesh in enumerate(meshes):
        d = T[k]["meshes"].get(mesh)
        if not d:
            continue
        y = len(meshes) - 1 - i
        r = d["ratio"]
        ax.plot([r[1], r[2]], [y, y], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(r[0], y, "o", color=BLUE, ms=5.5, mec="white", mew=1.0, zorder=3)
        ownr = [p["own"] / p["W"] for p in d["per_seed"] if p["W"]]
        ax.plot(np.median(ownr), y, "|", color=GREY, ms=9, mew=1.8, zorder=2)
    ax.axvline(1.0, color=ORANGE, lw=1.5, ls="--")
    p = T[k]["pairs"]
    ax.set_title(f"{title}\narm lower in {p['v_lower']} / {p['n']}", color=INK, fontsize=10)
    ax.set_xlim(0.7, 1.35)
axes[0].set_yticks(range(len(meshes)))
axes[0].set_yticklabels(meshes[::-1])
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=160)
print("written", sys.argv[2])
