"""demo_render.py DIR [CROPS_JSON] -- D157 demo: the PNGs from demo_compute.py's arrays (no recomputation).

Silhouette overlays (render off | render on, held-out views, 1024 px): red = the body covers, the mesh does not; blue =
the mesh covers, the body does not; light grey = both; the referee's outline a thin dark line; per panel the mismatched
pixels and the boundary IoU (Cheng et al. 2021, band 2 % of the image diagonal). Heatmaps: the visible display surface
coloured by its signed distance to the exact mesh minus the perfect sample's offset c (pitches; one fixed scale)."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.ndimage import binary_erosion, distance_transform_cdt  # noqa: E402

D = sys.argv[1]
CROPS = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
RED, BLUE, AGREE, INK, INK2, BG = "#e34948", "#2a78d6", "#d9d8d4", "#0b0b0b", "#52514e", "#ffffff"
plt.rcParams.update({"font.size": 10, "figure.facecolor": "white"})
LAB = {"P": "render OFF (physics only)", "L": "render ON"}
VIEWNAME = {"bunny": {16: "side", 13: "three-quarter", 2: "from above (held-out, far)"},
            "dragon": {16: "side", 14: "raised side", 38: "from below (held-out, far)"}}
SCALE = 0.5                                                    # pitches, the heatmap's fixed range
CMAP = LinearSegmentedColormap.from_list("d", ["#184f95", BLUE, "#9ec5f4", "#f0efec", "#f3a7a6", RED, "#a12a2a"])


def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def unpack(z, k, n):
    return np.unpackbits(z[k])[:n * 1024 * 1024].reshape(n, 1024, 1024).astype(bool)


def biou(S, T, k=29):
    def band(m):
        cd = distance_transform_cdt(np.pad(m, 1), metric="chessboard")[1:-1, 1:-1]
        return m & (cd <= k)
    bs, bt = band(S), band(T)
    return int((bs & bt).sum()) / max(int((bs | bt).sum()), 1)


def overlay(S, T):
    """Display image (up at the top) of state mask S against referee mask T."""
    S, T = S.T[::-1], T.T[::-1]
    img = np.ones(S.shape + (3,)) * hexrgb(BG)
    img[S & T] = hexrgb(AGREE)
    img[S & ~T] = hexrgb(RED)
    img[~S & T] = hexrgb(BLUE)
    edge = T & ~binary_erosion(T)
    img[edge] = hexrgb(INK)
    return img


def bbox(*ms, pad=24):
    u = np.zeros_like(ms[0])
    for m in ms:
        u |= m.T[::-1]
    r, c = np.where(u)
    return max(r.min() - pad, 0), min(r.max() + pad, 1023), max(c.min() - pad, 0), min(c.max() + pad, 1023)


def label(ax, S, T):
    mis = int((S ^ T).sum())
    ax.text(0.02, 0.02, f"mismatched pixels {mis:,}\nboundary IoU {biou(S, T):.3f}", transform=ax.transAxes,
            fontsize=9, color=INK, va="bottom", bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))


def legend(fig, y=0.005):
    from matplotlib.patches import Patch
    hs = [Patch(fc=AGREE, label="body and target agree"), Patch(fc=RED, label="body covers, target does not"),
          Patch(fc=BLUE, label="target covers, body misses"), Patch(fc="white", ec=INK, label="target outline")]
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, y), fontsize=9)


def sil_figure(mesh, z, ref, out):
    views = list(z["views"])
    n = len(views)
    Ls, Ps, Ts = unpack(z, "L_mask", n), unpack(z, "P_mask", n), unpack(z, f"{ref}_mask", n)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4.6 * n))
    for i, v in enumerate(views):
        r0, r1, c0, c1 = bbox(Ls[i], Ps[i], Ts[i])
        for j, lab in enumerate(("P", "L")):
            S = (Ps if lab == "P" else Ls)[i]
            ax = axes[i, j]
            ax.imshow(overlay(S, Ts[i])[r0:r1, c0:c1], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            label(ax, S, Ts[i])
            if i == 0:
                ax.set_title(LAB[lab], fontsize=12, color=INK)
        axes[i, 0].set_ylabel(VIEWNAME[mesh].get(int(v), f"view {v}"), fontsize=10, color=INK2)
    what = ("the exact target mesh grown by the 0.8-spacing offset that even a perfect particle sample has"
            if ref == "cor" else "the exact target mesh as it is (raw: every body sits ~0.8 spacing outside it)")
    fig.suptitle(f"{mesh} 300k, end state, views not used in training (1024 px)\nreference: {what}", fontsize=11,
                 color=INK, y=0.995)
    legend(fig)
    fig.tight_layout(rect=(0, 0.025, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def zoom_figure(mesh, z, out):
    """CROPS[mesh] = [[view_index_in_list, r0, r1, c0, c1, name], ...] in display pixels of the 1024-px image."""
    crops = CROPS.get(mesh, [])
    if not crops:
        return
    views = list(z["views"])
    n = len(views)
    Ls, Ps, Ts = unpack(z, "L_mask", n), unpack(z, "P_mask", n), unpack(z, "cor_mask", n)
    fig, axes = plt.subplots(len(crops), 2, figsize=(9, 4.3 * len(crops)), squeeze=False)
    for i, (vi, f0, f1, g0, g1, name) in enumerate(crops):
        b0, b1, d0, d1 = bbox(Ls[vi], Ps[vi], Ts[vi], pad=0)      # the crop: fractions of the view's bounding box,
        r0, r1 = int(b0 + f0 * (b1 - b0)), int(b0 + f1 * (b1 - b0))  # made square about its centre
        c0, c1 = int(d0 + g0 * (d1 - d0)), int(d0 + g1 * (d1 - d0))
        side = max(r1 - r0, c1 - c0)
        rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
        r0, c0 = max(rc - side // 2, 0), max(cc - side // 2, 0)
        r1, c1 = min(r0 + side, 1024), min(c0 + side, 1024)
        for j, lab in enumerate(("P", "L")):
            S = (Ps if lab == "P" else Ls)[vi]
            ax = axes[i, j]
            win = np.zeros_like(S)
            sub = np.zeros((1024, 1024), bool)
            sub[r0:r1, c0:c1] = True
            win = sub[::-1].T                                  # the window in mask coordinates
            ax.imshow(overlay(S, Ts[vi])[r0:r1, c0:c1], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            label(ax, S & win, Ts[vi] & win)
            if i == 0:
                ax.set_title(LAB[lab], fontsize=12, color=INK)
        axes[i, 0].set_ylabel(name, fontsize=10, color=INK2)
    fig.suptitle(f"{mesh} 300k, zoom on the thin parts (offset-corrected reference, 1024 px; counts inside the crop)",
                 fontsize=11, color=INK, y=0.995)
    legend(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


def heat_figure(mesh, z, out, rows=(0, 1)):
    c_a = float(z["c"]) / float(z["pitch"])
    fig, axes = plt.subplots(len(rows), 2, figsize=(10, 4.8 * len(rows)), squeeze=False)
    views = list(z["views"])
    for i, vi in enumerate(rows):
        boxes = []
        for lab in ("P", "L"):
            d = z[f"{lab}_dist"][vi].astype(np.float32)[:, ::-1]
            r, cc = np.where(np.isfinite(d))
            boxes.append((r.min(), r.max(), cc.min(), cc.max()))
        r0 = max(min(b[0] for b in boxes) - 16, 0); r1 = min(max(b[1] for b in boxes) + 16, 1023)
        c0 = max(min(b[2] for b in boxes) - 16, 0); c1 = min(max(b[3] for b in boxes) + 16, 1023)
        for j, lab in enumerate(("P", "L")):
            d = z[f"{lab}_dist"][vi].astype(np.float32)[:, ::-1] - c_a     # mirrored to the overlays' orientation
            sh = z[f"{lab}_shade"][vi].astype(np.float32)[:, ::-1]
            rgb = CMAP(np.clip((d + SCALE) / (2 * SCALE), 0, 1))[..., :3] * (0.5 + 0.5 * sh[..., None])
            rgb[~np.isfinite(d)] = 1.0
            ax = axes[i, j]
            ax.imshow(rgb[r0:r1, c0:c1], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            fin = np.isfinite(d)
            ax.text(0.02, 0.02, f"typical distance {np.median(np.abs(d[fin])):.2f} spacing\n"
                                f"on average {np.mean(d[fin]):+.2f} (+ = sticking out)\n"
                                f"share farther than 0.25: {100 * np.mean(np.abs(d[fin]) > 0.25):.0f} %",
                    transform=ax.transAxes, fontsize=9, color=INK, va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
            if i == 0:
                ax.set_title(LAB[lab], fontsize=12, color=INK)
        axes[i, 0].set_ylabel(VIEWNAME[mesh].get(int(views[vi]), ""), fontsize=10, color=INK2)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(-SCALE, SCALE))
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.03, pad=0.03, aspect=40)
    cb.set_label("how far the visible body surface lies from where a perfect particle sample's surface would be "
                 "(particle spacings)\nblue = sunk inside, grey = on target, red = sticking out; same scale for both "
                 "panels and both shapes", fontsize=9)
    fig.suptitle(f"{mesh} 300k, end state: distance of the displayed surface to the exact target", fontsize=11,
                 color=INK, y=0.995)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def time_figure(mesh, zt, ze, out):
    T = np.unpackbits(ze["cor_mask"])[:1024 * 1024].reshape(1024, 1024).astype(bool)     # first view
    cols = [(120, "0.5 s"), (240, "1 s"), (480, "2 s"), (960, "4 s"), ("end", "end")]
    fig, axes = plt.subplots(2, len(cols), figsize=(3.4 * len(cols), 7.2))
    n = len(list(ze["views"]))
    Ms = {}
    for lab in ("P", "L"):
        for rw, _ in cols:
            if rw == "end":
                Ms[(lab, rw)] = unpack(ze, f"{lab}_mask", n)[0]
            else:
                Ms[(lab, rw)] = np.unpackbits(zt[f"{lab}_t{rw}"])[:1024 * 1024].reshape(1024, 1024).astype(bool)
    r0, r1, c0, c1 = bbox(T, *Ms.values())
    for i, lab in enumerate(("P", "L")):
        for j, (rw, name) in enumerate(cols):
            S = Ms[(lab, rw)]
            ax = axes[i, j]
            ax.imshow(overlay(S, T)[r0:r1, c0:c1], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            label(ax, S, T)
            if i == 0:
                ax.set_title(name, fontsize=12, color=INK)
        axes[i, 0].set_ylabel(LAB[lab], fontsize=11, color=INK)
    fig.suptitle(f"{mesh} 300k, side view (held-out), offset-corrected reference: when the difference appears",
                 fontsize=11, color=INK, y=0.995)
    legend(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)


for mesh in ("bunny", "dragon"):
    try:
        z = np.load(f"{D}/{mesh}_end.npz")
    except FileNotFoundError:
        continue
    sil_figure(mesh, z, "cor", f"{D}/{mesh}_1_silhouette.png")
    sil_figure(mesh, z, "raw", f"{D}/{mesh}_1b_silhouette_raw.png")
    zoom_figure(mesh, z, f"{D}/{mesh}_2_zoom.png")
    heat_figure(mesh, z, f"{D}/{mesh}_3_distance_heatmap.png")
    try:
        zt = np.load(f"{D}/{mesh}_time.npz")
        time_figure(mesh, zt, z, f"{D}/{mesh}_4_time_strip.png")
    except FileNotFoundError:
        pass
    print("rendered", mesh, flush=True)
