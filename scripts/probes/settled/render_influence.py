"""render_influence.py DUMP_DIR FRAMES12_NPZ OUT_DIR TITLE — how the render gradient acts on a run, from its per-window
gradient dumps (`pipeline_run.py --grad_dump DIR`) and the run's target sample (run12.sh's TAG_frames12.npz).

At the first gradient of each window: the weighted render term's position gradient on every particle (lambda times
silhouette plus shading), the physics objective's (everything but the cleanup), and what each does when it alone
drives the controls (write_grad_dump's linear-response rollouts, each channel's gradient scaled to the window's
accepted change: the stress control dFc and the layer control u). "Push" = the descent direction -g along the
particle's outward normal (> 0 moves the particle out). "Offset" = the particle's signed distance to the target's
surface (> 0 outside; the nearest point of the target sample's layer and its outward normal), in display pitches.
"Toward" = sum(push * -offset) / sum(|push| |offset|) over the outer layer: +1 when every push points to the target
surface, 0 when the pushes are blind to it.

OUT_DIR/maps_XXXX.png, one per window (the body from the display camera; each pixel shows its front-most particle):
  offset from the target | |physics gradient| | |render gradient| (one log scale for both)
  physics push           | render push        | log10(render-only step move / physics-only step move)
OUT_DIR/summary.png, per window: the render's share of the step; the two gradients' cosine; toward, render and
physics; the outer layer's move under each channel alone; the gradients' share on the outer layer and in their top
5 % particles. OUT_DIR/rows.txt: the numbers behind summary.png."""
import glob, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
import matplotlib                                              # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Normalize, SymLogNorm  # noqa: E402
from physmorph.pipeline.window.layer import layer_by_asymmetry, layer_spacing  # noqa: E402

files = sorted(glob.glob(os.path.join(sys.argv[1], "win_*.npz")))
tgt = torch.as_tensor(np.load(sys.argv[2])["tgt"], device="cuda").float()
out, title = sys.argv[3], sys.argv[4]
os.makedirs(out, exist_ok=True)
SURF, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RENDER, PHYS = "#2a78d6", "#eb6834"                           # categorical slots 1 and 2: the render term, the physics
seq = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
div = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f29b98", "#b8302f"])
for cm in (seq, div):
    cm.set_bad(alpha=0.)
under = ListedColormap([GRID])
under.set_bad(alpha=0.)
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": GRID, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.titlecolor": INK})
az, el = np.radians(35.), np.radians(18.)                     # the display camera (StudioRaster(..., 35., 18.))
view = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
right = np.cross([0., 1., 0.], view); right /= np.linalg.norm(right)
up = np.cross(view, right)
W = 420                                                         # pixels across a panel

pitch = .708 * float(torch.cdist(tgt[::50], tgt).topk(9, largest=False).values[:, 8].median())   # momentum_probe's
t_lay, t_nrm = layer_by_asymmetry(tgt, layer_spacing(tgt))
t_pts, t_nrm = tgt[t_lay], t_nrm[t_lay]
t_tree = gpu.KNN(t_pts)


def top_share(n2, q=.05):
    s = np.sort(n2)[::-1]
    return float(s[:max(1, int(q * len(s)))].sum() / max(s.sum(), 1e-30))


def toward(push, off):
    return float((push * -off).sum() / max((np.abs(push) * np.abs(off)).sum(), 1e-30))


def window(z):
    x = torch.as_tensor(z["xT0"], device="cuda").float()
    sp = layer_spacing(x)
    lay, nrm = layer_by_asymmetry(x, sp)
    _, i = t_tree.query(x, 1)
    i = i.reshape(-1)
    off = (((x - t_pts[i]) * t_nrm[i]).sum(1) / pitch).cpu().numpy()
    lay, nrm = lay.cpu().numpy(), nrm.cpu().numpy()
    gr = float(z["lam_r"]) * (z["gx_sil"] + z["gx_pbr"])     # the weighted render term (w_pbr = 1, the default)
    gp = z["gx_phys"]
    push_r, push_p = -(gr * nrm).sum(1), -(gp * nrm).sum(1)
    mv = {}
    for k, base in (("rend", "xT_base"), ("phys", "xT_base"), ("rend_u", "xT_base_u0"), ("phys_u", "xT_base_u0")):
        key = "xT_" + k
        mv[k] = (np.linalg.norm(z[key] - z[base], axis=1) / pitch if key in z.files and base in z.files
                 else np.full(len(gr), np.nan))                   # a slimmed dump (slim_dumps.py) has no rollouts
    nr2, np2 = (gr ** 2).sum(1), (gp ** 2).sum(1)
    row = dict(lam=float(z["lam_r"]), g_share=float(z["g_share"]),
               cos=float((gr * gp).sum() / (np.sqrt(nr2.sum() * np2.sum()) + 1e-30)),
               cos_layer=float((gr[lay] * gp[lay]).sum() / (np.sqrt(nr2[lay].sum() * np2[lay].sum()) + 1e-30)),
               toward_r=toward(push_r[lay], off[lay]), toward_p=toward(push_p[lay], off[lay]),
               layer_r=float(nr2[lay].sum() / max(nr2.sum(), 1e-30)), layer_p=float(np2[lay].sum() / max(np2.sum(), 1e-30)),
               top_r=top_share(nr2), top_p=top_share(np2), off_abs=float(np.abs(off[lay]).mean()),
               **{"move_" + k: float(v[lay].mean()) for k, v in mv.items()})
    return row, dict(x=z["xT0"], sp=sp, lay=lay, off=off, nr=np.sqrt(nr2), np=np.sqrt(np2), push_r=push_r, push_p=push_p,
                     ratio=np.log10((mv["rend"] + mv["rend_u"] + 1e-12) / (mv["phys"] + mv["phys_u"] + 1e-12)))


def front(x, sp):
    """(pixel index (M,), particle index (M,)) of each covered pixel's front-most particle; a particle covers a square
    of about one spacing."""
    u, v, d = x @ right, x @ up, x @ view
    s = (W - 1) / (xlim[1] - xlim[0])
    px, py = np.round((u - xlim[0]) * s).astype(int), np.round((ylim[1] - v) * s).astype(int)
    r = max(1, int(round(.5 * sp * s)))
    o = np.arange(-r, r + 1)
    dx, dy = (a.ravel() for a in np.meshgrid(o, o))
    PX, PY = (px[:, None] + dx).ravel(), (py[:, None] + dy).ravel()
    I = np.repeat(np.arange(len(x)), len(dx))
    ok = (PX >= 0) & (PX < W) & (PY >= 0) & (PY < H)
    pix, I, D = (PY * W + PX)[ok], I[ok], np.repeat(d, len(dx))[ok]
    order = np.argsort(-D, kind="stable")                       # nearest to the camera first
    pix, I = pix[order], I[order]
    _, first = np.unique(pix, return_index=True)
    return pix[first], I[first]


def image(pix, idx, val):
    img = np.full(H * W, np.nan)
    img[pix] = val[idx]
    return img.reshape(H, W)


def panel(a, pix, idx, s, val, cmap, norm, name, layer_only=False):
    a.imshow(image(pix, idx, np.zeros(len(val))), cmap=under, interpolation="nearest")
    v = np.where(s["lay"], val, np.nan) if layer_only else val
    im = a.imshow(image(pix, idx, v), cmap=cmap, norm=norm, interpolation="nearest")
    a.set_title(name, fontsize=9)
    plt.gcf().colorbar(im, ax=a, fraction=.04, pad=.01)


def symlog(v, mask):
    lim = float(np.percentile(np.abs(v[mask]), 99.5)) + 1e-30
    return SymLogNorm(lim / 300., vmin=-lim, vmax=lim, base=10)


first0 = np.load(files[0])["xT0"]
bound = np.concatenate([first0, tgt.cpu().numpy()])
bu, bv = bound @ right, bound @ up
pad = .04 * max(np.ptp(bu), np.ptp(bv))
xlim, ylim = (bu.min() - pad, bu.max() + pad), (bv.min() - pad, bv.max() + pad)
H = int(round(W * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])))
rows = []
for f in files:
    k = int(os.path.basename(f)[4:8])
    row, s = window(np.load(f))
    rows.append((k, row))
    print(f"window {k:3d}: " + " ".join(f"{a} {b:.4g}" for a, b in row.items()), flush=True)
    pix, idx = front(s["x"], s["sp"])
    fig, ax = plt.subplots(2, 3, figsize=(16, 2 * 5.3 * H / W + .8))
    hi = float(np.percentile(np.concatenate([s["nr"], s["np"]]), 99.5))
    lo = hi * 1e-4
    lim = float(np.percentile(np.abs(s["off"][s["lay"]]), 99))
    panel(ax[0, 0], pix, idx, s, s["off"], div, Normalize(-lim, lim), "offset from the target surface, pitches (red outside)",
          layer_only=True)
    panel(ax[0, 1], pix, idx, s, np.clip(s["np"], lo, hi), seq, LogNorm(lo, hi), "|physics gradient|")
    panel(ax[0, 2], pix, idx, s, np.clip(s["nr"], lo, hi), seq, LogNorm(lo, hi),
          "|render gradient| = |lambda (silhouette + shading)|")
    panel(ax[1, 0], pix, idx, s, s["push_p"], div, symlog(s["push_p"], s["lay"]), "physics push along the normal (red out)",
          layer_only=True)
    panel(ax[1, 1], pix, idx, s, s["push_r"], div, symlog(s["push_r"], s["lay"]), "render push along the normal (red out)",
          layer_only=True)
    panel(ax[1, 2], pix, idx, s, np.clip(s["ratio"], -1.5, 1.5), div, Normalize(-1.5, 1.5),
          "log10(render-only step move / physics-only step move)")
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
        for sp_ in a.spines.values():
            sp_.set_visible(False)
    fig.suptitle(f"{title}, window {k}: lambda {row['lam']:.3g}, render share of the step {row['g_share']:.2f}, "
                 f"toward the target: render {row['toward_r']:+.2f}, physics {row['toward_p']:+.2f}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(os.path.join(out, f"maps_{k:04d}.png"), dpi=80)
    plt.close(fig)

w = np.array([r[0] for r in rows])
R = [r[1] for r in rows]
with open(os.path.join(out, "rows.txt"), "w") as fh:
    keys = list(R[0])
    fh.write("window " + " ".join(keys) + "\n")
    for k, r in zip(w, R):
        fh.write(f"{k} " + " ".join(f"{r[a]:.5g}" for a in keys) + "\n")
fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))


def lines(a, pairs, ttl, log=False, zero=False):
    for key, lab, c, ls in pairs:
        a.plot(w, [r[key] for r in R], color=c, lw=2, ls=ls, label=lab)
    if zero:
        a.axhline(0, color=MUTED, lw=1)
    if log:
        a.set_yscale("log")
    a.set_title(ttl, fontsize=10)
    if len(pairs) > 1:
        a.legend(fontsize=8, frameon=False)


lines(ax[0, 0], (("g_share", "render", RENDER, "-"),), "render's share of the step (first gradient of the window)")
ax[0, 0].set_ylim(0, 1)
lines(ax[0, 1], (("cos", "all particles", RENDER, "-"), ("cos_layer", "outer layer", RENDER, "--")),
      "cosine of the render and physics position gradients", zero=True)
lines(ax[0, 2], (("toward_r", "render", RENDER, "-"), ("toward_p", "physics", PHYS, "-")),
      "push toward the target surface (+1: every push points to it)", zero=True)
ax[0, 2].set_ylim(-1, 1)
lines(ax[1, 0], (("move_rend", "render, stress control dFc", RENDER, "-"), ("move_phys", "physics, stress control dFc", PHYS, "-"),
                 ("move_rend_u", "render, layer control u", RENDER, "--"), ("move_phys_u", "physics, layer control u", PHYS, "--")),
      "outer layer's mean move, pitches, when one channel alone drives the step", log=True)
lines(ax[1, 1], (("layer_r", "render", RENDER, "-"), ("layer_p", "physics", PHYS, "-")),
      "share of the gradient (squared norm) on the outer layer")
lines(ax[1, 2], (("top_r", "render", RENDER, "-"), ("top_p", "physics", PHYS, "-")),
      "share of the gradient (squared norm) in its top 5 % particles")
for a in ax.flat:
    a.grid(color=GRID, lw=.6); a.set_xlabel("window")
fig.suptitle(title + ": the render gradient per window", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(os.path.join(out, "summary.png"), dpi=90)
print("wrote", out)
