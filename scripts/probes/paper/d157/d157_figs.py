"""d157_figs.py TABLE_JSON OUT_DIR -- D157's three figures (matplotlib, static PNG).

fig1_metrics.png: which standard metrics see the render: the decision set's median r (the share of the twin's error
    the arm removes, %) with the pairs' interquartile range, 300k and 40k side by side; filled = sees it by the
    pre-registered rule, hollow = does not.
fig2_permesh.png: per mesh the r of three lead metrics at both N (dots: mesh-seeds; bar: the mesh's median).
fig3_offset.png: (4a) the arm - twin signed offset from the mesh by the distance to the nearest training outline, and
    (4b) the move the silhouette term asks of the twin under (1, 1) and (2, 1) weights against the observed arm - twin
    offset."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

T = json.load(open(sys.argv[1]))
out = sys.argv[2]
BLUE, ORANGE, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
DEC = [("sil/zb/cor/1024/held/biou2pct_err", "Boundary IoU (2 % diag.)"),
       ("sil/zb/cor/1024/held/bfdavis_err", "Boundary F (DAVIS)"),
       ("sil/zb/cor/1024/held/chamfer_a", "2D contour Chamfer"),
       ("sil/zb/cor/1024/held/hd95_a", "2D contour HD95"),
       ("d3/zb/cor/f0.5", "3D F-score @ 0.5 pitch"),
       ("d3/zb/cor/f1", "3D F-score @ 1 pitch"),
       ("d3/zb/cor/nc", "3D normal consistency"),
       ("d3/zb/cor/chamfer_l1_a", "3D Chamfer-L1"),
       ("img/cor/shaded/held/mse", "shaded PSNR (as MSE)"),
       ("img/cor/shaded/held/ssim", "shaded SSIM"),
       ("img/cor/shaded/held/lpips", "shaded LPIPS"),
       ("img/cor/normal/held/mse", "normal-map PSNR (as MSE)"),
       ("img/cor/normal/held/ssim", "normal-map SSIM"),
       ("img/cor/normal/held/lpips", "normal-map LPIPS")]
REF = [("run/sil_iou", "mask IoU, the gallery's (128 px)"),
       ("sil/zb/raw/1024/held/iou_err", "mask IoU, exact mesh, 1024 px")]
M = T["metrics"]


def fig1():
    rows = REF + DEC
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.8), sharey=True)
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[len(REF):] -= 0.6                                      # a gap between the reference rows and the decision set
    for ax, N in zip(axes, ("300k", "40k")):
        for i, (y, (k, lab)) in enumerate(zip(ys, rows)):
            x = M.get(f"{N}:{k}")
            if x is None:
                continue
            col = GREY if i < len(REF) else BLUE
            m, lo, hi = 100 * x["r_med"], 100 * x["r_q25"], 100 * x["r_q75"]
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(m, y, "o", ms=8, mfc=col if x["sees"] else "white", mec=col, mew=2)
            ax.text(max(hi, m) + 1.5, y, f"{m:+.0f} %", va="center", fontsize=8, color=INK2)
        ax.axvline(0, color=INK2, lw=1)
        for v in (-5, 5):
            ax.axvline(v, color=GREY, lw=0.8, ls=(0, (3, 3)))
        ax.set_xlabel("share of the twin's error the render arm removes (%)\nmedian over mesh-seed pairs, bar = IQR")
        npairs = max((M[f"{N}:{k}"]["pairs"] for k, _ in DEC if f"{N}:{k}" in M), default=0)
        nmesh = max((M[f"{N}:{k}"]["meshes"] for k, _ in DEC if f"{N}:{k}" in M), default=0)
        ax.set_title(f"{N}: {nmesh} meshes, {npairs} mesh-seed pairs", color=INK, fontsize=10, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for yb in (ys[len(REF)] + 0.8, ys[len(REF) + 4] + 0.5, ys[len(REF) + 8] + 0.5):
            ax.axhline(yb, color=GRID, lw=1)
    axes[0].set_yticks(ys, [lab for _, lab in rows])
    for t in axes[0].get_yticklabels()[:len(REF)]:
        t.set_color(GREY)
    h1 = plt.Line2D([], [], marker="o", ls="", ms=8, mfc=BLUE, mec=BLUE, mew=2, label="sees the render (pre-registered rule)")
    h2 = plt.Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=BLUE, mew=2, label="does not")
    fig.legend(handles=[h1, h2], loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.01, 0.955))
    fig.suptitle("Standard metrics against the exact mesh (grown by the exterior's own offset), held-out views",
                 x=0.02, ha="left", y=0.995, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(f"{out}/fig1_metrics.png", dpi=160)


def fig2(keys):
    fig, axes = plt.subplots(len(keys), 2, figsize=(10, 2.6 * len(keys)), squeeze=False)
    for i, (k, lab) in enumerate(keys):
        for j, N in enumerate(("300k", "40k")):
            ax = axes[i, j]
            x = M.get(f"{N}:{k}")
            if x is None:
                ax.axis("off")
                continue
            pm = x["per_mesh_r"]
            order = sorted(pm, key=lambda m: pm[m])
            rows = [r for r in T.get("pairs", {}).get(f"{N}:{k}", [])]
            vals = [100 * pm[m] for m in order]
            ax.bar(range(len(order)), [max(v, -60) for v in vals], color=BLUE, width=0.6)
            for xi, v in enumerate(vals):
                if v < -60:                                    # clipped: a near-zero twin error makes r explode
                    ax.text(xi, -58, f"{v:.0f} %", rotation=90, va="bottom", ha="center", fontsize=7, color="white")
            ax.axhline(0, color=INK2, lw=1)
            ax.set_xticks(range(len(order)), order, rotation=60, ha="right", fontsize=7)
            ax.set_ylabel("arm removes (%)")
            ax.set_title(f"{lab}, {N}", fontsize=9, loc="left", color=INK)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
        lims = [axes[i, j].get_ylim() for j in range(2) if axes[i, j].axison]
        if lims:                                               # one scale per row: 300k and 40k comparable
            for j in range(2):
                axes[i, j].set_ylim(max(-62, min(l[0] for l in lims)), max(l[1] for l in lims))
    fig.tight_layout()
    fig.savefig(f"{out}/fig2_permesh.png", dpi=160)


def fig3():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.5, 1]))
    bins = [("bin0", "0–0.1"), ("bin1", "0.1–0.25"), ("bin2", "0.25–0.5"), ("bin3", "0.5–1"), ("bin4", "1–2"),
            ("bin5", "> 2"), ("headon", "head-on"), ("crevice", "crevice")]
    for N, col, dx in (("300k", BLUE, -0.15), ("40k", ORANGE, 0.15)):
        R = T["region"].get(N, {})
        xs, ms, lo, hi = [], [], [], []
        for i, (b, _) in enumerate(bins):
            if b in R:
                xs.append(i + dx); ms.append(R[b]["d_med"]); lo.append(R[b]["d_q25"]); hi.append(R[b]["d_q75"])
        a1.vlines(xs, lo, hi, color=col, lw=2)
        a1.plot(xs, ms, "o", color=col, ms=7, label=f"{N}")
    a1.axhline(0, color=INK2, lw=1)
    a1.set_xticks(range(len(bins)), [b[1] for b in bins], fontsize=8)
    a1.set_xlabel("distance of the nearest mesh point to a training-view outline (run's render pixels)")
    a1.set_ylabel("arm − twin offset from the mesh\n(pitches, + = arm farther out)")
    a1.legend(frameon=False, title="median, bar = IQR", fontsize=8, title_fontsize=8, loc="upper left")
    a1.grid(axis="y", color=GRID, lw=0.6)
    a1.set_axisbelow(True)
    a1.set_title("(4a) where the arm's exterior moves relative to the twin's", fontsize=9, loc="left")
    B = [b for b in T["bias"] if b["res"] == b["run_res"]]
    labels, vals = [], []
    for N in ("300k", "40k"):
        bb = [b for b in B if b["n"] == N]
        if not bb:
            continue
        for q, lab in (("P_argmin11" if N == "300k" else "W_argmin11", "twin, weights 1:1"),
                       ("P_argmin21" if N == "300k" else "W_argmin21", "twin, weights 2:1 (the run)"),
                       ("obs_arm_minus_twin", "observed: arm − twin")):
            v = [b[q] for b in bb if b.get(q) is not None]
            labels.append(f"{N}: {lab}")
            vals.append(v)
    for i, v in enumerate(vals):
        a2.plot(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, "o", color=GREY, ms=3)
        a2.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=BLUE if i < 3 else ORANGE, lw=2.5)
    a2.axhline(0, color=INK2, lw=1)
    a2.set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=7)
    a2.set_ylabel("uniform move along the normals (pitches)")
    a2.set_title("(4b) the silhouette term's optimum for the twin", fontsize=9, loc="left")
    a2.grid(axis="y", color=GRID, lw=0.6)
    a2.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{out}/fig3_offset.png", dpi=160)


fig1()
fig2([("sil/zb/cor/1024/held/biou2pct_err", "Boundary IoU (2 % diag.), held-out, 1024 px"),
      ("d3/zb/cor/f0.5", "3D F-score @ 0.5 pitch"), ("img/cor/normal/held/mse", "normal-map PSNR (as MSE), held-out")])
fig3()
