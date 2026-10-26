"""Chapter 4 evidence figures for the render terms (run on the GPU server's Python with matplotlib, never locally).

    python plot_render_evidence.py DATA_DIR OUT_DIR

Reads (all copied unchanged from the result folders, see README_ko.md):
  h1_table.json   end states of the 300k runs against an independent sample (render arm L, lambda = 0 twin P, floor)
  table1.json     the same end states against the exact target mesh (L, P, a perfect particle sample "own")
  iou_calib.json  the binary silhouette IoU's response to a uniform normal shift of the arm's outer layer
  switch.json     the render term switched off / on at 6 s (bunny, dragon x seeds 97, 99)
Writes fig_render_effect_end.png, fig_iou_sensitivity.png, fig_render_switch.png into OUT_DIR.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# reference palette (validated: blue / orange / aqua, light mode); text in ink, never in series colours
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#b9b8b3", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 10, "axes.titlecolor": INK, "figure.facecolor": "white", "axes.facecolor": SURF,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 220, "legend.frameon": False,
})
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/fonts")
import tnr  # noqa: E402

tnr.mpl()                                     # every text in Times New Roman (the user, 2026-10-22)
MESHES = ["bunny", "dragon", "teapot", "bob", "armadilo", "homer"]
MESH_LABEL = {"armadilo": "armadillo"}


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def seed_ratios(entry, ref_key):
    """median / min / max over seeds of L / P, and the median of ref / P (the reference tick)."""
    r = [s["L"] / s["P"] for s in entry["per_seed"] if s["P"] > 0]
    t = [s[ref_key] / s["P"] for s in entry["per_seed"] if s["P"] > 0 and s.get(ref_key) is not None]
    return float(np.median(r)), float(np.min(r)), float(np.max(r)), (float(np.median(t)) if t else None)


# ---------------------------------------------------------------- figure 1: the end, arm against twin, six meshes
h1, t1 = load("h1_table.json"), load("table1.json")
ROWS = [  # (source, key, label, reference key)
    ("h1", "sil_iou_ind", "particle silhouette, 1 - IoU\n(binary splat, 128 px)", "floor"),
    ("h1", "chamfer_ind", "chamfer distance", "floor"),
    ("h1", "yard_ext_sil", "exterior silhouette residual\n(96 px)", "floor"),
    ("h1", "yard_ext_pbr", "exterior shading residual\n(96 px)", "floor"),
    ("h1", "disp_front", "displayed surface, 1 - IoU\n(front view, 4K)", "floor"),
    ("t1", "iou1024", "binary silhouette, 1 - IoU\n(exact, 1024 px)", "own"),
    ("t1", "outline_sd", "outline shape error\n(offset spread)", "own"),
    ("t1", "normal_cl", "normal angle at the\nnearest mesh point", "own"),
    ("t1", "shading", "headlight shading error\n(exact, 1024 px)", "own"),
]
fig, axes = plt.subplots(1, len(MESHES), figsize=(13.2, 5.4), sharey=True)
ys = np.arange(len(ROWS))[::-1].astype(float)
ys[5:] -= 0.6                                            # a gap between the two references
for ax, mesh in zip(axes, MESHES):
    ax.axvline(1.0, color=ORANGE, lw=2, ls=(0, (4, 3)), zorder=1)
    for y, (src, key, _, ref) in zip(ys, ROWS):
        entry = (h1 if src == "h1" else t1)[mesh][key]
        med, lo, hi, tick = seed_ratios(entry, ref)
        ax.plot([lo, hi], [y, y], color=BLUE, lw=2, solid_capstyle="round", zorder=3)
        ax.plot([med], [y], "o", ms=6.5, color=BLUE, mec=SURF, mew=1.5, zorder=4)
        if tick is not None:
            ax.plot([tick], [y], marker="|", ms=13, mew=2.2, color="#8c8b86", zorder=2)
    ax.set_title(MESH_LABEL.get(mesh, mesh))
    ax.set_xlim(0.3, 1.45)
    ax.set_xticks([0.5, 1.0])
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(ys)
axes[0].set_yticklabels([r[2] for r in ROWS], fontsize=8.5)
for ax in axes:
    ax.axhline((ys[4] + ys[5]) / 2, color="#b9b8b3", lw=0.8)
fig.text(0.004, (0.5 + 0.86) / 2 + 0.06, "against an independent\nsample of the target", fontsize=8.5, color=INK2,
         rotation=90, va="center")
fig.text(0.004, 0.27, "against the exact\ntarget mesh", fontsize=8.5, color=INK2, rotation=90, va="center")
fig.supxlabel("error of the full method relative to its $\\lambda = 0$ twin (twin = 1, dashed)", fontsize=9.5,
              color=INK2)
from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([], [], color=BLUE, marker="o", ms=6, mec=SURF, lw=2, label="full method: median over seeds, bar = seed range"),
           Line2D([], [], color="#8c8b86", marker="|", ms=12, mew=2, lw=0, label="a perfect particle sample of the mesh"),
           Line2D([], [], color=ORANGE, lw=2, ls=(0, (4, 3)), label="$\\lambda = 0$ twin")]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.56, 1.0), fontsize=9)
fig.tight_layout(rect=(0.03, 0.02, 1, 0.94))
fig.savefig(os.path.join(OUT, "fig_render_effect_end.png"))
plt.close(fig)

# ---------------------------------------------------------------- figure 2: the binary IoU against a uniform shift
cal = load("iou_calib.json")
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for key in sorted(cal):
    c = cal[key]
    mesh = c["mesh"]
    col = BLUE if mesh == "bunny" else ORANGE
    sh = sorted((float(k), v) for k, v in c["shift"].items())
    xs = [d for d, _ in sh]
    vs = [v for _, v in sh]
    xs.insert(3, 0.0)
    vs.insert(3, 0.0)
    ax.plot(xs, vs, "-o", color=col, lw=1.6, ms=4.5, mec=SURF, mew=1.0, alpha=0.95,
            label=f"{mesh}, seed {c['seed']}")
    lead = c["res128"]["err_L"] - c["res128"]["err_P"]
    ax.axhline(lead, color=col, lw=1.0, ls=(0, (1, 2)))
ax.axhline(0, color="#8c8b86", lw=1)
ax.axvline(0, color="#8c8b86", lw=1)
ax.set_xlabel("outer layer moved along its outward normal (sampling pitches)")
ax.set_ylabel("change of 1 - IoU\n(binary silhouette, 128 px)")
h, lab = ax.get_legend_handles_labels()
seen, H, L = set(), [], []
for hh, ll in zip(h, lab):
    m = ll.split(",")[0]
    if m not in seen:
        seen.add(m)
        H.append(hh)
        L.append(m + " (three seeds)")
H.append(Line2D([], [], color=INK2, lw=1, ls=(0, (1, 2))))
L.append("whole difference, full method - twin")
ax.legend(H, L, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_iou_sensitivity.png"))
plt.close(fig)

# ---------------------------------------------------------------- figure 3: the render switched at 6 s
sw = load("switch.json")["series"]
cols = [("bunny", 97), ("bunny", 99), ("dragon", 97), ("dragon", 99)]
fig, axes = plt.subplots(2, 4, figsize=(13.2, 5.6), sharey=True)
for r, (arm, title) in enumerate((("Loff6", "render off from 6 s"), ("Pon6", "render on from 6 s"))):
    for c, (mesh, seed) in enumerate(cols):
        ax = axes[r, c]
        rows = sw[f"{mesh}_{seed}_{arm}"]["yard_sil"]["rows"]
        t = np.array([q["t"] for q in rows])
        L = np.array([q["L"] / q["P"] if q.get("L") and q.get("P") else np.nan for q in rows])
        X = np.array([q["X"] / q["P"] if q.get("X") is not None and q.get("P") else np.nan for q in rows])
        ax.axhline(1.0, color=ORANGE, lw=2, ls=(0, (4, 3)))
        ax.axvline(6.0, color="#8c8b86", lw=1)
        ax.plot(t, L, color=BLUE, lw=2)
        ok = np.isfinite(X)
        ax.plot(t[ok], X[ok], "-o", color=AQUA, lw=1.6, ms=3.5, mec=SURF, mew=0.6)
        ax.set_title(f"{mesh}, seed {seed}: {title}", fontsize=9.5)
        ax.set_xlim(3, 16.5)
        if r == 1:
            ax.set_xlabel("simulated time (s)")
        if c == 0:
            ax.set_ylabel("exterior silhouette residual\nrelative to the twin")
handles = [Line2D([], [], color=BLUE, lw=2, label="full method (render on throughout)"),
           Line2D([], [], color=AQUA, lw=1.6, marker="o", ms=4, mec=SURF, label="render switched at 6 s"),
           Line2D([], [], color=ORANGE, lw=2, ls=(0, (4, 3)), label="$\\lambda = 0$ twin (= 1)")]
fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(os.path.join(OUT, "fig_render_switch.png"))
plt.close(fig)
print("wrote", OUT)
