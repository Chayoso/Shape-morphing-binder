"""d155c_fig.py OUT_PNG -- D155c's figure. Top: the volume each target on the way asks, as the transport's cells see it
(the occupied cells of the gate grid, cells holding at least half the median count of the final target's, over the
final target's), per committed ramp window of D155b's run (targets as built) and of D155c's (rescaled by the window's
recorded k), rebuilt from the dumped D_Y. Bottom: the body's volume over its rest volume (records' Jx_mean), the tag
(seed 97), D155b (A3G) and D155c (A3C). Bunny, dragon, homer at 40k, render arm."""
import json
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

O = "/data/relcfd/chayo/physmorph_v2/output/gpu/d155/gate"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "savefig.dpi": 160, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2.0, "legend.frameon": False,
                     "legend.fontsize": 7.5})


def hist(name):
    a = next(iter(json.load(open(f"{O}/{name}.json"))["arms"].values()))
    return [h for h in a["history"] if h.get("frame_end") and not h.get("null_commit") and h["frame_end"] <= a["deliver_n"]]


meshes = ("bunny", "dragon", "homer")
fig, ax = plt.subplots(2, 3, figsize=(13.5, 7.2))
for j, m in enumerate(meshes):
    z = np.load(f"{O}/zdump_{m}40k.npz")
    Y, D, gdx, t_e = z["Y"].astype(np.float64), z["D"].astype(np.float64), float(z["gdx"]), float(z["t_m"])
    lo = np.minimum(Y.min(0), (Y + D).min(0)) - 4 * gdx

    def cells(p):
        return np.unique(np.floor((p - lo) / gdx).astype(np.int64), axis=0, return_counts=True)[1]
    cy = cells(Y)
    full = np.median(cy)
    occ_y = np.sum(cy >= 0.5 * full)
    for arm, col, lab in (("A3GL97", BLUE, "D155b's targets (as interpolated)"),
                          ("A3CL97", ORANGE, "D155c's targets (rescaled by the spacing)")):
        pts = []
        for h in hist(f"{m}40k_{arm}"):
            s = h.get("morph_s", 1.0)
            if s >= 1.0:
                continue
            Z = Y + (1 - s) * D
            if h.get("morph_k") is not None:
                c = Z.mean(0)
                Z = (Z - c) * h["morph_k"] + c
            pts.append((h["frame_end"] / 240, np.sum(cells(Z) >= 0.5 * full) / occ_y))
        ax[0, j].plot([p[0] for p in pts], [p[1] for p in pts], color=col, marker="o", ms=3.5, label=lab)
    ax[0, j].axhline(1.0, color=MUTED, lw=1.0)
    ax[0, j].text(0.05, 1.002, "the body's own volume", color=INK2, fontsize=7.5, va="bottom")
    ax[0, j].set_ylabel("volume the target on the way asks\n(over the body's, as the transport's cells see it)")
    ax[0, j].set_title(f"{m} 40k: what the targets on the way ask", fontsize=9)
    for name, col, lab in ((f"{m}40k_V97", MUTED, "the tag (no moving target)"),
                           (f"{m}40k_A3GL97", BLUE, "D155b"), (f"{m}40k_A3CL97", ORANGE, "D155c")):
        acc = [h for h in hist(name) if h.get("Jx_mean")]
        ax[1, j].plot([h["frame_end"] / 240 for h in acc], [h["Jx_mean"] for h in acc], color=col, label=lab)
    ax[1, j].axhline(1.0, color=MUTED, lw=1.0)
    ax[1, j].set_ylabel("the body's volume over its rest volume")
    ax[1, j].set_title(f"{m} 40k: the body follows up, and does not come back down", fontsize=9)
    for i in (0, 1):
        ax[i, j].axvline(t_e, color=INK2, lw=0.9, ls=":")
        ax[i, j].set_xlim(0, 8)
        ax[i, j].set_xlabel("simulated time (s)")
    ax[1, j].text(t_e + 0.08, ax[1, j].get_ylim()[1], "the target stops moving", color=INK2, fontsize=7.5, va="top")
    ax[0, j].legend(loc="lower right")
    ax[1, j].legend(loc="upper right")
fig.suptitle("D155c: the rescale made the first targets larger than the body; neither run's body gives back the volume "
             "it took on the way (the assimilation makes it rest volume)", fontsize=10)
fig.tight_layout()
fig.savefig(sys.argv[1])
print("written", sys.argv[1])
