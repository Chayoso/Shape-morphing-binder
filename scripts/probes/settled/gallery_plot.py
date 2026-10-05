"""gallery_plot.py IND_LOG OUT_PNG — D92: the 40k gallery against an independent sample of each mesh: per mesh the
silhouette error 1 - IoU of the physics-only twin (SP), the render arm with one target sample (SG) and with eight
(SK), beside the floor (the target sample itself against the independent one); and chamfer."""
import json, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

by = defaultdict(dict)
for l in open(sys.argv[1]):
    if l.strip():
        r = json.loads(l)
        mesh, arm = r["tag"].rsplit("_", 1)
        by[mesh][arm] = r
meshes = [m for m in by if all(a in by[m] for a in ("SK", "SG", "SP"))]
fig, ax = plt.subplots(2, 1, figsize=(16, 9))
x = np.arange(len(meshes))
for k, (arm, label, c) in enumerate((("SP", "physics-only twin", "0.25"), ("SG", "render, one target sample", "tab:blue"),
                                     ("SK", "render, eight target samples (D91)", "tab:red"))):
    ax[0].bar(x + (k - 1) * 0.27, [1 - by[m][arm]["ind"]["sil"] for m in meshes], 0.27, color=c, label=label)
    ax[1].bar(x + (k - 1) * 0.27, [by[m][arm]["ind"]["chamfer"] for m in meshes], 0.27, color=c, label=label)
ax[0].plot(x, [1 - by[m]["SK"]["floor"]["sil"] for m in meshes], "k_", ms=18, mew=2, label="floor (target sample vs independent)")
ax[0].set_ylabel("1 - silhouette IoU (independent sample)")
ax[1].set_ylabel("chamfer to the independent sample")
for a in ax:
    a.set_xticks(x)
    a.set_xticklabels(meshes, rotation=30, fontsize=9)
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8)
fig.suptitle("40k gallery (19 meshes), every arm with the minimum spacing, D81's weight, D89's one resolution: "
             "read against an independent sample of each mesh", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(sys.argv[2], dpi=90)
print("wrote", sys.argv[2])
