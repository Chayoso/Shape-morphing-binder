"""fig_render_effect_v2.py OUT_PNG -- FIGS_V2 (2026-10-22): the dissertation's new Figure 4.3, what the render loss changes
at the end of a 300k morph, read against the exact target mesh only (CPU, no recomputation of any metric).

(a) outline: Boundary IoU error [Cheng et al. 2021] of the end body's silhouette, views not used in training (46), 1024 px,
    band 2 % of the diagonal, against the exact target mesh grown by the offset a perfect particle sample has (D157's
    decision metric sil/zb/cor/1024/held/biou2pct_err);
(b) 3D surface: 1 - F-score at half a particle spacing [Knapitsch et al. 2017] of the visible exterior surface against
    the same reference (D157's d3/zb/cor/f0.5);
    per mesh-seed pair (tag freeze-2026-10-15, 300k, B1: bunny / dragon x 5 seeds, the other four x 3) the twin
    "dF_c + layer u" (grey) and the full method "dF_c + layer u + render loss (full)" (orange), and a perfect particle
    sample of the target (black tick: the sampling floor, D157's own);
(c) where: the end silhouettes of seed 97 on two crops (bunny's ears, three-quarter view; dragon's tail tip, side view),
    both views held out; red = the body covers what the target does not, blue = the target covers what the body misses
    (D157's demo arrays output/gpu/d157/demo/*_end.npz, its crops and its counts inside the crop).
Inputs: output/gpu/d157/read300_*.jsonl and output/gpu/d157/demo/{bunny,dragon}_end.npz."""
import glob
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.ndimage import binary_erosion, distance_transform_cdt  # noqa: E402

G = "/data/relcfd/chayo/physmorph_v2/output/gpu/d157"
OUT = sys.argv[1]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
TWIN, FULL = "#9a9993", "#eb6834"
RED, BLUE, AGREE = "#e34948", "#2a78d6", "#d9d8d4"
sys.path.insert(0, "/data/relcfd/chayo/physmorph_v2/tmp/fonts")
import tnr  # noqa: E402

tnr.mpl()                                     # every text in Times New Roman (the user, 2026-10-22)
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "white"})
MESHES = [("bunny", "bunny"), ("dragon", "dragon"), ("armadilo", "armadillo"), ("bob", "bob"), ("homer", "homer"),
          ("teapot", "teapot")]

rows = {}
for f in sorted(glob.glob(f"{G}/read300_*.jsonl")):
    for ln in open(f):
        r = json.loads(ln)
        rows[(r["mesh"], r["seed"], r["label"])] = r
biou = lambda r: r["sil"]["zb"]["cor"]["1024"]["held"]["biou2pct_err"]   # noqa: E731
f05 = lambda r: 1.0 - r["d3"]["zb"]["cor"]["f0.5"]                       # noqa: E731

pairs = []                                    # (mesh label, seed) in display order
for m, lab in MESHES:
    for s in sorted({k[1] for k in rows if k[0] == m}):
        if all((m, s, x) in rows for x in ("P", "L", "own")):
            pairs.append((m, lab, s))


def dumbbell(ax, fn, title, xlabel, xmax, show_names):
    y, ticks, labels, seps = 0.0, [], [], []
    down = 0
    rel = []
    prev = None
    for m, lab, s in pairs:
        if prev is not None and m != prev:
            y += 0.9
            seps.append(y - 0.95)
        prev = m
        t, a, o = fn(rows[(m, s, "P")]), fn(rows[(m, s, "L")]), fn(rows[(m, s, "own")])
        down += a < t
        rel.append((t - a) / t)
        ax.plot([t, a], [y, y], color="#c9c8c3", lw=1.6, zorder=1, solid_capstyle="round")
        ax.annotate("", xy=(a + 0.004 * xmax, y), xytext=(t, y),
                    arrowprops=dict(arrowstyle="-|>", color="#b5b4ae", lw=0, mutation_scale=7), zorder=1)
        ax.plot([o, o], [y - 0.34, y + 0.34], color=INK, lw=1.1, zorder=2)
        ax.scatter([t], [y], s=26, color=TWIN, zorder=3, edgecolor="white", linewidth=0.6)
        ax.scatter([a], [y], s=26, color=FULL, zorder=4, edgecolor="white", linewidth=0.6)
        ticks.append(y)
        labels.append(f"{lab}  {s}" if show_names else f"{s}")
        y += 1.0
    for sy in seps:
        ax.axhline(sy, color=GRID, lw=0.8, zorder=0)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(0, xmax)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=8.5)
    med = 100 * float(np.median(rel))
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=16)
    ax.text(0.0, 1.005, f"lower in {down} of {len(pairs)} pairs; median {med:.0f} % lower",
            transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
    return down, med


def unpack(z, k, n):
    return np.unpackbits(z[k])[:n * 1024 * 1024].reshape(n, 1024, 1024).astype(bool)


def biou_px(S, T, k=29):
    def band(m):
        cd = distance_transform_cdt(np.pad(m, 1), metric="chessboard")[1:-1, 1:-1]
        return m & (cd <= k)
    bs, bt = band(S), band(T)
    return int((bs & bt).sum()) / max(int((bs | bt).sum()), 1)


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def overlay(S, T):
    S, T = S.T[::-1], T.T[::-1]
    img = np.ones(S.shape + (3,))
    img[S & T] = hexrgb(AGREE)
    img[S & ~T] = hexrgb(RED)
    img[~S & T] = hexrgb(BLUE)
    img[T & ~binary_erosion(T)] = hexrgb(INK)
    return img


def crop_box(Ls, Ps, Ts, vi, f0, f1, g0, g1):
    u = Ls[vi].T[::-1] | Ps[vi].T[::-1] | Ts[vi].T[::-1]
    r, c = np.where(u)
    b0, b1, d0, d1 = r.min(), r.max(), c.min(), c.max()
    r0, r1 = int(b0 + f0 * (b1 - b0)), int(b0 + f1 * (b1 - b0))
    c0, c1 = int(d0 + g0 * (d1 - d0)), int(d0 + g1 * (d1 - d0))
    side = max(r1 - r0, c1 - c0)
    rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
    r0, c0 = max(rc - side // 2, 0), max(cc - side // 2, 0)
    return r0, min(r0 + side, 1024), c0, min(c0 + side, 1024)


CROPS = [("bunny", 1, (0.0, 0.40, 0.0, 0.55), "bunny, ears (held-out view)"),
         ("dragon", 0, (0.15, 0.48, 0.78, 1.0), "dragon, tail tip (held-out view)")]

fig = plt.figure(figsize=(7.4, 8.3))
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.55, 1.0], hspace=0.30, wspace=0.16,
              left=0.13, right=0.975, top=0.875, bottom=0.06)
axa = fig.add_subplot(gs[0, 0])
axb = fig.add_subplot(gs[0, 1])
da, ma = dumbbell(axa, biou, "(a) Is the end outline closer to the exact target?",
                  "outline error: 1 − boundary IoU, views not used in training", 0.10, True)
db, mb = dumbbell(axb, f05, "(b) Is the end surface closer in 3D?",
                  "surface error: 1 − F-score at half a particle spacing", 0.22, False)
handles = [Line2D([], [], marker="o", ls="", color=TWIN, markersize=6, label="dF_c + layer u (the twin: no render loss)"),
           Line2D([], [], marker="o", ls="", color=FULL, markersize=6, label="dF_c + layer u + render loss (full)"),
           Line2D([], [], marker="|", ls="", color=INK, markersize=9, markeredgewidth=1.3,
                  label="a perfect particle sample of the target (the floor)")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.977),
           columnspacing=1.4, handletextpad=0.3)
fig.text(0.5, 0.995, "300k morphs at their end, against the exact target mesh; one row = one mesh and seed",
         fontsize=8.5, color=INK, va="top", ha="center")

sub = GridSpec(1, 4, figure=fig, left=0.03, right=0.985, top=gs[1, 0].get_position(fig).y1 - 0.025,
               bottom=0.06, wspace=0.04)
fig.text(0.03, gs[1, 0].get_position(fig).y1 + 0.012, "(c) Where do they differ? The rims of the thin parts, seed 97",
         fontsize=9.5, color=INK, va="bottom")
k = 0
counts = []
for mesh, vi, fr, name in CROPS:
    z = np.load(f"{G}/demo/{mesh}_end.npz")
    n = len(list(z["views"]))
    Ls, Ps, Ts = unpack(z, "L_mask", n), unpack(z, "P_mask", n), unpack(z, "cor_mask", n)
    r0, r1, c0, c1 = crop_box(Ls, Ps, Ts, vi, *fr)
    win = np.zeros((1024, 1024), bool)
    win[r0:r1, c0:c1] = True
    win = win[::-1].T
    for lab, S in (("dF_c + layer u", Ps[vi]), ("full", Ls[vi])):
        ax = fig.add_subplot(sub[0, k])
        ax.imshow(overlay(S, Ts[vi])[r0:r1, c0:c1], interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#d6d5d0")
        mis = int(((S ^ Ts[vi]) & win).sum())
        counts.append((mesh, lab, mis, biou_px(S & win, Ts[vi] & win)))
        ax.set_title("+ render loss (full)" if lab == "full" else lab, fontsize=8.5, color=FULL if lab == "full" else INK2, pad=3)
        ax.text(0.5, -0.04, f"{mis:,} mismatched pixels", transform=ax.transAxes, ha="center", va="top", fontsize=8,
                color=INK)
        if lab != "full":
            ax.text(0.0, 1.13, name, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")
        k += 1
fig.legend(handles=[Patch(fc=AGREE, label="body and target agree"), Patch(fc=RED, label="body covers, target does not"),
                    Patch(fc=BLUE, label="target covers, body misses"),
                    Patch(fc="white", ec=INK, label="target outline")],
           loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.005), handlelength=1.2,
           columnspacing=1.2)
fig.savefig(OUT, dpi=300)
print("wrote", OUT, "pairs", len(pairs), "a", da, round(ma, 1), "b", db, round(mb, 1), "crops", counts)
