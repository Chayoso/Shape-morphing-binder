"""d158_figs.py OUT_DIR FIG_DIR [mesh:az:el:cx0,cx1,cy0,cy1 ...] -- D158's figures (CPU, matplotlib).

map_MESH.png   one dump (the last kept frame): the camera's view of the discs (orthographic, the nearest disc facing the
               camera per pixel), the whole body with the zoom's box, and four zoomed maps on one colour scale: D115's
               relief (its layer height minus the smooth part above 10.8 pitches), what one value per outer particle can
               hold of it, what is left, and what the loop's own path (through the exterior field) can hold.
shares.png     how much of the relief one value per outer particle can hold, over the run (energy share), both meshes.
pixel.png      does the picture register the relief: the relief's render gain at D115's pixel and at the loop's, against
               the dense pictures and against the run's own loss targets, the last kept frame.
The crop is given in fractions of the whole view's box (x0, x1 from the left, y0, y1 from the top)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap             # noqa: E402

out, figd = Path(sys.argv[1]), Path(sys.argv[2])
figd.mkdir(parents=True, exist_ok=True)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
S1, S2 = "#2a78d6", "#eb6834"                                      # categorical slots 1, 2
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a8f", "#c0302f"])
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
specs = {}
for s in sys.argv[3:]:
    m, az, el, box = s.split(":")
    specs[m] = (float(az), float(el), [float(v) for v in box.split(",")])
summary = json.load(open(out / "summary.json"))


def view(p, n, az, el):
    az, el = np.radians(az), np.radians(el)
    d = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    r = np.cross([0., 1., 0.], d)
    r /= np.linalg.norm(r)
    u = np.cross(d, r)
    return p @ r, p @ u, p @ d, n @ d


def zbuffer(X, Y, Z, F, vals, lo, hi, px):
    """The value of the nearest camera-facing disc per pixel, each disc splatted on 3 x 3 pixels."""
    W, H = int(np.ceil((hi[0] - lo[0]) / px)), int(np.ceil((hi[1] - lo[1]) / px))
    ok = (F > 0.05) & (X >= lo[0]) & (X < hi[0]) & (Y >= lo[1]) & (Y < hi[1])
    ix, iy, z = ((X[ok] - lo[0]) / px).astype(int), ((hi[1] - Y[ok]) / px).astype(int), Z[ok]
    best = np.full((H, W), -np.inf)
    img = [np.full((H, W), np.nan) for _ in vals]
    vv = [v[ok] for v in vals]
    order = np.argsort(z)                                          # far to near: the nearest written last
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            jx, jy = np.clip(ix + dx, 0, W - 1), np.clip(iy + dy, 0, H - 1)
            key = jy * W + jx
            o = order[np.argsort(key[order], kind="stable")]
            last = np.r_[key[o][1:] != key[o][:-1], True]       # the last (nearest) per pixel within this offset
            sel = o[last]
            k = key[sel]
            upd = z[sel] > best.reshape(-1)[k]
            best.reshape(-1)[k[upd]] = z[sel][upd]
            for im, v in zip(img, vv):
                im.reshape(-1)[k[upd]] = v[sel][upd]
    return img


for mesh, (az, el, box) in specs.items():
    d = out / f"{mesh}300k_L97"
    dumps = sorted(d.glob("dump_*.npz"))
    z = np.load(dumps[-1])
    p, n, a = z["p"], z["n"], summary[mesh]["head"]["pitch_a"]
    rel, relW, relF = z["h_rel"] / a, z["rel_uW"] / a, z["rel_uF"] / a
    X, Y, Z, F = view(p, n, az, el)
    lo, hi = np.array([X.min(), Y.min()]), np.array([X.max(), Y.max()])
    ext = hi - lo
    clo = lo + np.array([box[0] * ext[0], (1 - box[3]) * ext[1]])
    chi = lo + np.array([box[1] * ext[0], (1 - box[2]) * ext[1]])
    full = zbuffer(X, Y, Z, F, [rel], lo, hi, 0.5 * a)[0]
    maps = zbuffer(X, Y, Z, F, [rel, relW, rel - relW, relF], clo, chi, 0.3 * a)
    inside = (X > clo[0]) & (X < chi[0]) & (Y > clo[1]) & (Y < chi[1]) & (F > 0.05) & (z["arrived"] > 0.5)
    sc = float(np.ceil(np.nanpercentile(np.abs(rel[inside]), 98) * 20) / 20) if inside.any() else 0.2
    fig, ax = plt.subplots(1, 5, figsize=(17, 4.2), gridspec_kw=dict(width_ratios=[0.8, 1, 1, 1, 1]))
    t_end = summary[mesh]["rows"][-1]["t"]
    titles = ["The whole body (the box: the zoom)", "D115's relief (layer height,\nsmooth part removed)",
              "What one value per outer particle\ncan hold of it", "What is left\n(finer than the particles hold)",
              "What the loop's own path to u\n(through the exterior field) can hold"]
    ax[0].imshow(full, cmap=DIV, vmin=-sc, vmax=sc, interpolation="nearest")
    H0, W0 = full.shape
    bx = [(clo[0] - lo[0]) / (0.5 * a), (chi[0] - lo[0]) / (0.5 * a)]
    by = [(hi[1] - chi[1]) / (0.5 * a), (hi[1] - clo[1]) / (0.5 * a)]
    ax[0].plot([bx[0], bx[1], bx[1], bx[0], bx[0]], [by[0], by[0], by[1], by[1], by[0]], color=INK, lw=1)
    for i, im in enumerate(maps):
        hnd = ax[i + 1].imshow(im, cmap=DIV, vmin=-sc, vmax=sc, interpolation="nearest")
    for i in range(5):
        ax[i].set_title(titles[i], fontsize=10, color=INK)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_facecolor("#f9f9f7")
    cb = fig.colorbar(hnd, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label("height along the normal (pitches)", color=INK2)
    r = summary[mesh]["rows"][-1]
    fig.suptitle(f"{mesh} 300k, render arm (dF_c + layer u + render loss (full)), seed 97, last kept frame (t = {t_end:.1f} s): "
                 f"one value per outer particle holds {100 * r['all']['rel_uW']:.0f} % of the relief, the loop's path "
                 f"{100 * r['all']['rel_uF']:.0f} %\n(left out of every panel: the layer's smooth part, mostly a uniform inward "
                 f"shift, mean {r['h_mean']:+.2f} pitch, which re-draws the finer field's offset)", fontsize=11, color=INK, y=1.07)
    fig.savefig(figd / f"map_{mesh}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

meshes = [m for m in ("bunny", "dragon") if m in summary]
fig, ax = plt.subplots(2, len(meshes), figsize=(5.4 * len(meshes), 7.2), sharey="row", squeeze=False)
for j, m in enumerate(meshes):
    rows = [r for r in summary[m]["rows"] if r["arrived"] >= 0.05]   # the relief exists where the body has arrived
    t = [r["t"] for r in rows]
    for i, (getter, ylab, ttl) in enumerate((
            (lambda r, k: r["all"][f"rel_u{k}"], "share of the relief's energy (%)",
             f"{m} 300k: how much of D115's relief\nthe layer control u can hold"),
            (lambda r, k: r["set"]["dense_ac"][f"sh{k}"], "share of the relief's render gain (%)",
             f"{m} 300k: how much of the relief's effect on D115's\nown render term the held part keeps"))):
        a_ = ax[i, j]
        for key, col, lab in (("W", S1, "one value per outer particle"), ("F", S2, "through the exterior field (the loop's path)")):
            y = [100 * getter(r, key) for r in rows]
            a_.plot(t, y, color=col, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=lab)
            a_.annotate(f"{y[-1]:.0f} %", (t[-1], y[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK2)
        a_.set_xscale("log")
        a_.set_xticks(t)
        a_.set_xticklabels([f"{v:g}" for v in t])
        a_.set_ylim(0, 100)
        a_.grid(axis="y", color=GRID, lw=0.8)
        a_.set_title(ttl, fontsize=10)
        a_.set_xlabel("time in the morph (s)")
        if j == 0:
            a_.set_ylabel(ylab)
            if i == 0:
                a_.legend(frameon=False, loc="lower left", fontsize=9)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(figd / "shares.png", dpi=130, bbox_inches="tight")
plt.close(fig)

def raw_end(m):
    p = out / f"{m}300k_L97" / "raw.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(s) for s in open(p)]
    return max(rows, key=lambda r: r["kept"])["raw"]


tot = lambda t: t[0] + t[1]                                       # noqa: E731
fig, ax = plt.subplots(2, len(meshes), figsize=(6.6 * len(meshes), 6.4), squeeze=False)
for j, m in enumerate(meshes):
    st = summary[m]["rows"][-1]["set"]
    rw = raw_end(m)
    # row 0: the pixel. The relief on top of its smooth part, against the dense pictures, at either pixel.
    lab0 = ["D115's pixel (0.58 pitch)", "the loop's pixel (188 px)"]
    v0 = [-100 * st[s]["rel_gain"] for s in ("dense_ac", "dense_loop")]
    # row 1: the loop's own targets, drawn as the loop draws (the field's own normals; d158_raw.py): the relief added
    # to the run's own surface; the dense pictures' value with the same normals beside it
    lab1, v1 = [], []
    if rw is not None:
        for s, lab in (("dense_ac", "dense pictures, D115's pixel"), ("dense_loop", "dense pictures, the loop's pixel")):
            v1.append(100 * (tot(rw[s]["h"]) - tot(rw[s]["low"])) / tot(rw[s]["low"]))
            lab1.append(lab)
        for s, lab in (("loop_ac", "the loop's own targets, D115's pixel"), ("loop_loop", "the loop's own targets, the loop's pixel")):
            v1.append(100 * (tot(rw[s]["rel"]) - tot(rw[s]["zero"])) / tot(rw[s]["zero"]))
            lab1.append(lab)
    for i, (labs, vals, ttl) in enumerate((
            (lab0, v0, f"{m} 300k: does the loop's coarser pixel see D115's relief?\n(dense pictures; the relief on its smooth part)"),
            (lab1, v1, f"{m} 300k: do the loop's own target pictures ask for it?\n(drawn as the loop draws: the field's own normals)"))):
        a_ = ax[i, j]
        if not vals:
            a_.set_axis_off()
            continue
        yy = np.arange(len(vals))[::-1]
        cols = [S1 if "dense" in l or i == 0 else S2 for l in labs]
        a_.barh(yy, vals, height=0.6, color=cols, edgecolor=SURF, linewidth=2)
        for y_, v in zip(yy, vals):
            a_.annotate(f"{v:+.1f} %", (max(v, 0.0), y_), xytext=(4, 0), textcoords="offset points",
                        ha="left", va="center", color=INK2, fontsize=9)    # right of the bar's end or of zero
        lo_, hi_ = min(min(vals), 0.0), max(max(vals), 0.0)
        a_.set_xlim(lo_ - 0.05 * (hi_ - lo_), hi_ + 0.25 * (hi_ - lo_) + 1.0)
        a_.axvline(0, color="#c3c2b7", lw=1)
        a_.set_yticks(yy)
        a_.set_yticklabels(labs)
        a_.set_title(ttl, fontsize=10)
        a_.set_xlabel("change of silhouette + shading (%; below 0: the relief helps)")
        a_.grid(axis="x", color=GRID, lw=0.8)
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(figd / "pixel.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("wrote", figd)
