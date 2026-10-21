"""d154_g40_fig.py OUT_DIR -- D154 follow-up (3)'s figure from g40_table.json: per mesh and measure the 40k render arm's
error relative to its twin (V / W; dot = median over seeds, bar = seed range; grey tick = a perfect particle sample of
the mesh, own / W), the reference palette's slot 1 and the twin's dashed line in slot 2."""
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
rows = [("iou1024", "1 - IoU, exact\nsilhouettes, 1024 px"), ("yard_mesh_sil", "yardstick silhouette\n(vs the exact mesh)"),
        ("yard_mesh_pbr", "yardstick shading\n(vs the exact mesh)"), ("outline_sd", "outline shape error\n(offset spread)"),
        ("normal_cl", "normal angle at the\nnearest mesh point")]
meshes = ["armadilo", "bunny", "dragon", "homer", "teapot"]
fig, axes = plt.subplots(1, 5, figsize=(12, 3.9), sharey=True, sharex=True)
for ax, mesh in zip(axes, meshes):
    for i, (k, _) in enumerate(rows):
        d = T[k]["meshes"].get(mesh)
        if not d:
            continue
        y = len(rows) - 1 - i
        r = d["ratio"]
        ax.plot([r[1], r[2]], [y, y], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot(r[0], y, "o", color=BLUE, ms=6, mec="white", mew=1.2, zorder=3)
        ownr = [p["own"] / p["W"] for p in d["per_seed"] if p["W"]]
        ax.plot(np.median(ownr), y, "|", color=GREY, ms=12, mew=2, zorder=2)
    ax.axvline(1.0, color=ORANGE, lw=1.5, ls="--")
    ax.set_title(mesh, color=INK)
    ax.set_xlim(0.75, 1.3)
axes[0].set_yticks(range(len(rows)))
axes[0].set_yticklabels([lab for _, lab in rows][::-1])
fig.supxlabel("error of the 40k render arm relative to its twin (twin = 1, dashed; dot = median over seeds 97 / 99 / 101, "
              "bar = range; grey tick = a perfect particle sample)", color=INK2)
fig.suptitle("40k gallery against the exact mesh: the silhouette area does not move, the outline's shape does",
             color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(out / "fig_g40_vs_mesh.png", dpi=160)
print("written", out / "fig_g40_vs_mesh.png")
