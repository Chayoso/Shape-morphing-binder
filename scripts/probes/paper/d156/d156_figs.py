"""d156_figs.py TABLE_JSON OUT_DIR -- D156's figure: how far each end state got from the hull toward the true mesh.

Two panels, one per measure (the image-visible surface; the volume of the hull's excess), one group per mesh, the
mean over seeds 97 / 99 with each seed as a dot: P (render off, the hull as its only target), L (render on, the true
mesh's pictures), and the tag's render arm V with the mesh itself as the target (the scale: what a 3D target gives).
0 = at the hull, 1 = at the true mesh."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

T = json.load(open(sys.argv[1]))["table"]
out = Path(sys.argv[2])
MESHES = ["bob", "A", "C", "dragon", "cheburashka", "bunny", "armadilo", "cow", "homer"]
SERIES = [("P", "hull", "dF_c + layer u (λ = 0), visual hull as target", "#2a78d6"),
          ("L", "hull", "full (+ render loss on the true pictures), visual hull as target", "#eb6834"),
          ("V", "mesh", "full, the mesh itself as target (tag)", "#1baf7a")]
PANELS = (("surf_deep", "concave surface (>= 0.5 pitch below the hull)"), ("vol", "volume of the hull's excess"),
          ("sil_held", "silhouettes in 46 held-out views"), ("img_shaded_mse", "studio images, held-out views"))
fig, axes = plt.subplots(2, 2, figsize=(13, 8.6), sharey=True)
axes = axes.ravel()
for ax, (key, title) in zip(axes, PANELS):
    w = 0.26
    for k, (lab, frame, name, col) in enumerate(SERIES):
        means, pts = [], []
        for i, m in enumerate(MESHES):
            v = [T[f"{m}_{s}_{frame}"][lab][key] for s in (97, 99)
                 if f"{m}_{s}_{frame}" in T and lab in T[f"{m}_{s}_{frame}"] and T[f"{m}_{s}_{frame}"][lab][key] is not None]
            means.append(np.mean(v) if v else np.nan)
            pts += [(i + (k - 1) * w, x) for x in v]
        x = np.arange(len(MESHES)) + (k - 1) * w
        ax.bar(x, means, w - 0.03, color=col, label=name, zorder=2)
        if pts:
            ax.scatter(*zip(*pts), s=10, color="#0b0b0b", zorder=3)
    ax.axhline(0, color="#52514e", lw=0.8, zorder=1)
    ax.axhline(1, color="#52514e", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(np.arange(len(MESHES)), MESHES, rotation=30, ha="right")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color="#e4e3df", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for a in (axes[0], axes[2]):
    a.set_ylabel("share of the way (0 = hull, 1 = true mesh)")
fig.suptitle("D156, 40k: how far each end got from the visual hull toward the true mesh", fontsize=12, y=0.995)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 0.965))
fig.text(0.5, 0.005, "40k, bars = mean of seeds 97 / 99, dots = each seed; meshes ordered by the hull's excess depth",
         ha="center", fontsize=8, color="#52514e")
fig.tight_layout(rect=(0, 0.03, 1, 0.89))
fig.savefig(out / "fig_d156_shares40.png", dpi=130)
print(out / "fig_d156_shares40.png")
