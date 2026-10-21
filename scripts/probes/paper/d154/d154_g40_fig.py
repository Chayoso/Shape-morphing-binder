"""d154_g40_fig.py OUT_DIR -- D154 follow-ups (3) / (3b)'s figure from g40_table.json: one panel per measure, one row per
mesh (19), the 40k render arm's error relative to its twin (V / W; dot = median over seeds 97 / 99 / 101, bar = range;
grey tick = a perfect particle sample of the mesh, own / W; dashed = the twin). One question per panel: does the
render arm's error sit left of its twin's on this measure? Colours: the reference palette's slot 1 (arm) and slot 2
(twin), grey for the perfect sample."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

out = Path(sys.argv[1])
T = json.load(open(out / "g40_table.json"))
BLUE, ORANGE, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c8c7c2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#ecebe7", "grid.linewidth": 0.6})
panels = [("iou1024", "1 - IoU, exact silhouettes\n(1024 px)"), ("yard_mesh_sil", "yardstick silhouette\n(vs the exact mesh)"),
          ("yard_mesh_pbr", "yardstick shading\n(vs the exact mesh)"), ("outline_sd", "outline shape error\n(offset spread)"),
          ("normal_cl", "normal angle at the\nnearest mesh point")]
meshes = list(T["outline_sd"]["meshes"])
fig, axes = plt.subplots(1, len(panels), figsize=(13, 6.6), sharey=True)
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
    ax.set_title(f"{title}\narm lower in {p['v_lower']} / {p['n']}", color=INK, fontsize=9)
    ax.set_xlim(0.7, 1.35)
axes[0].set_yticks(range(len(meshes)))
axes[0].set_yticklabels(meshes[::-1])
fig.supxlabel("error of the 40k render arm relative to its twin (twin = 1, dashed; dot = median over seeds 97 / 99 / 101, "
              "bar = range; grey tick = a perfect particle sample)", color=INK2)
fig.suptitle("40k gallery (19 meshes) against the exact mesh: the render arm's outline shape is better on every mesh, "
             "its silhouette area is not (slightly worse)",
             color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(out / "fig_g40_vs_mesh.png", dpi=160)
print("written", out / "fig_g40_vs_mesh.png")
