"""render_effect_map.py RENDER_NPZ TWIN_NPZ REF_NPZ OUT_PNG TITLE — what the render gradient leaves on the body: the
render arm and its physics-only twin (run12.sh's TAG_frames12.npz) against an independent sample of the target
(REF_NPZ's `tgt`, D90). Each outer-layer particle's signed offset from the target's surface (the nearest point of the
reference sample's layer, along that point's outward normal; > 0 outside), in display pitches.
Top two rows: the last kept frame of each arm from three cameras (the display camera and two more around the
vertical axis; each pixel shows its front-most particle), one colour scale. Bottom: on every kept frame, the outer
layer's mean |offset|, its 90th percentile and the share farther than one pitch, both arms."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize  # noqa: E402
from physmorph.pipeline.window.layer import layer_by_asymmetry, layer_spacing  # noqa: E402

out, title = sys.argv[4], sys.argv[5]
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RENDER, PHYS = "#2a78d6", "#eb6834"
div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f29b98", "#b8302f"])
div.set_bad(alpha=0.)
under = ListedColormap([GRID])
under.set_bad(alpha=0.)
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.titlecolor": INK})
W = 420
ref = torch.as_tensor(np.load(sys.argv[3])["tgt"], device="cuda").float()
pitch = .708 * float(torch.cdist(ref[::50], ref).topk(9, largest=False).values[:, 8].median())   # momentum_probe's
r_lay, r_nrm = layer_by_asymmetry(ref, layer_spacing(ref))
r_pts, r_nrm = ref[r_lay], r_nrm[r_lay]
tree = gpu.KNN(r_pts)


def offsets(x):
    """(layer mask, signed offset of every particle in pitches, spacing)."""
    x = torch.as_tensor(np.asarray(x, np.float32), device="cuda")
    sp = layer_spacing(x)
    lay = layer_by_asymmetry(x, sp)[0]
    _, i = tree.query(x, 1)
    i = i.reshape(-1)
    return lay.cpu().numpy(), (((x - r_pts[i]) * r_nrm[i]).sum(1) / pitch).cpu().numpy(), sp


arms = []
for path, name, colour in ((sys.argv[1], "render arm", RENDER), (sys.argv[2], "physics-only twin", PHYS)):
    z = np.load(path, allow_pickle=True)
    raws, fr = np.asarray(z["raws"]), z["frames"]
    series = []
    for k in range(len(fr)):
        lay, off, sp = offsets(fr[k])
        o = np.abs(off[lay])
        series.append((float(o.mean()), float(np.percentile(o, 90)), float((o > 1.).mean())))
    arms.append(dict(name=name, colour=colour, raws=raws, series=np.array(series), x=np.asarray(fr[-1], np.float32),
                     lay=lay, off=off, sp=sp))
    s = np.array(series)
    print(f"{name}: last frame mean |offset| {s[-1, 0]:.4f} pitches, p90 {s[-1, 1]:.4f}, beyond one pitch "
          f"{100 * s[-1, 2]:.3f} %; mean over the last third of the kept frames {s[-len(s) // 3:, 0].mean():.4f}", flush=True)

lim = float(np.percentile(np.abs(np.concatenate([arm["off"][arm["lay"]] for arm in arms])), 99))
cams = []
for azim in (35., 125., 215.):
    az, el = np.radians(azim), np.radians(18.)
    view = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    right = np.cross([0., 1., 0.], view); right /= np.linalg.norm(right)
    cams.append((azim, view, right, np.cross(view, right)))


def picture(x, val, sp, cam):
    _, view, right, up = cam
    u, v, d = x @ right, x @ up, x @ view
    span = 1.08 * max(np.ptp(u), np.ptp(v))
    s = (W - 1) / span
    px = np.round((u - u.mean()) * s + W / 2).astype(int)
    py = np.round((v.mean() - v) * s + W / 2).astype(int)
    r = max(1, int(round(.5 * sp * s)))
    o = np.arange(-r, r + 1)
    dx, dy = (a.ravel() for a in np.meshgrid(o, o))
    PX, PY = (px[:, None] + dx).ravel(), (py[:, None] + dy).ravel()
    I = np.repeat(np.arange(len(x)), len(dx))
    ok = (PX >= 0) & (PX < W) & (PY >= 0) & (PY < W)
    pix, I, D = (PY * W + PX)[ok], I[ok], np.repeat(d, len(dx))[ok]
    order = np.argsort(-D, kind="stable")                       # nearest to the camera first
    pix, I = pix[order], I[order]
    _, first = np.unique(pix, return_index=True)
    cover, img = np.full(W * W, np.nan), np.full(W * W, np.nan)
    cover[pix[first]] = 0.
    img[pix[first]] = val[I[first]]
    return cover.reshape(W, W), img.reshape(W, W)


fig = plt.figure(figsize=(16, 16))
grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, .7))
for col, cam in enumerate(cams):
    for row, arm in enumerate(arms):
        ax = fig.add_subplot(grid[row, col])
        cover, img = picture(arm["x"], np.where(arm["lay"], arm["off"], np.nan), arm["sp"], cam)
        ax.imshow(cover, cmap=under, interpolation="nearest")
        im = ax.imshow(img, cmap=div, norm=Normalize(-lim, lim), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.set_title(f"{arm['name']}, last frame, camera {cam[0]:.0f} deg", fontsize=9)
        if col == 2:
            fig.colorbar(im, ax=ax, fraction=.04, pad=.01, label="offset from the target surface, pitches (red outside)")
for j, label in enumerate(("mean |offset| of the outer layer, pitches", "90th percentile of |offset|, pitches",
                           "share of the outer layer farther than one pitch")):
    ax = fig.add_subplot(grid[2, j])
    for arm in arms:
        ax.plot(arm["raws"], arm["series"][:, j], color=arm["colour"], lw=2, label=arm["name"])
    ax.set_title(label, fontsize=10); ax.set_xlabel("simulated frame"); ax.grid(color=GRID, lw=.6)
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
fig.suptitle(title + ": where the body sits against the target, with and without the render gradient", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .97))
fig.savefig(out, dpi=80)
print("wrote", out)
