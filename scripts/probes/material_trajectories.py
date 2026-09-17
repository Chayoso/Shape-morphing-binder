"""Material study: do the material constants change the morph TRAJECTORY, not only the end?

    material_trajectories.py <out_dir> <target> <base_prefix> <prefix2> [<prefix3> ...] [--png plot.png]

Loads <prefix>_<target>_render_full_dt_iso_nn.npz archives (same seed -> the same particle
indices), and reports per run: windows, wall time, end chamfer / silIoU, path length
(mean over particles of the summed per-step displacement, wu), peak speed, and the
TRAJECTORY DIVERGENCE from the base run: mean |x_run(k) - x_base(k)| over particles at
matched archived frames k, in world units and in particle spacings, at 10 / 25 / 50 / 100 %
of the shorter run, plus the frame at which the divergence first exceeds one spacing. Also
the chamfer-to-target curve at matched frames. Writes a PNG of both curves when --png is given."""
import sys
import numpy as np
from scipy.spatial import cKDTree

args = [a for a in sys.argv[1:] if not a.startswith("--")]
png = None
if "--png" in sys.argv:
    png = sys.argv[sys.argv.index("--png") + 1]
    if png in args:
        args.remove(png)
out, tgt_name, base = args[0], args[1], args[2]
prefixes = [base] + args[3:]
ARM = "render_full_dt_iso_nn"


def load(p):
    z = np.load(f"{out}/{p}_{tgt_name}_{ARM}.npz", allow_pickle=True)
    fr = np.asarray(z["frames"], np.float32)
    dn = int(z["deliver_n"]) if "deliver_n" in z.files else len(fr)
    return fr[:dn], np.asarray(z["tgt"], np.float32)


runs = {p: load(p) for p in prefixes}
tgt = runs[base][1]
tree = cKDTree(tgt)
spacing = float(np.median(tree.query(tgt, k=2, workers=-1)[0][:, 1]))
fb = runs[base][0]
print(f"target {tgt_name}: particle spacing {spacing:.4f} wu; base run {base}: {len(fb)} archived frames")


def chamfer(x, step=4):
    d1 = tree.query(x[::step], workers=-1)[0].mean()
    d2 = cKDTree(x[::step]).query(tgt[::step], workers=-1)[0].mean()
    return 0.5 * (d1 + d2)


curves = {}
for p, (fr, _) in runs.items():
    n = len(fr)
    steps = np.linalg.norm(np.diff(fr[:: max(1, n // 400)], axis=0), axis=2)      # (n', N)
    path = float(steps.sum(0).mean())
    vmax = float(steps.max())
    m = min(n, len(fb))
    div = np.array([np.linalg.norm(fr[k] - fb[k], axis=1).mean() for k in range(0, m, max(1, m // 200))])
    ks = np.arange(0, m, max(1, m // 200))
    first = next((int(ks[i]) for i, d in enumerate(div) if d > spacing), -1)
    cham = np.array([chamfer(fr[k]) for k in range(0, n, max(1, n // 60))])
    curves[p] = (ks, div, np.arange(0, n, max(1, n // 60)), cham)
    q = [div[int(len(div) * f) - 1] for f in (0.1, 0.25, 0.5, 1.0)]
    print(f"{p:14s} frames {n:5d} path {path:6.2f} wu  peak step {vmax:.3f} wu  end chamfer {cham[-1]:.4f}  "
          f"divergence from {base}: 10% {q[0]:.3f} 25% {q[1]:.3f} 50% {q[2]:.3f} end {q[3]:.3f} wu "
          f"(= {q[3] / spacing:.1f} spacings); first > 1 spacing at frame {first}")
if png:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for p, (ks, div, kc, cham) in curves.items():
        ax[0].plot(ks, div / spacing, label=p)
        ax[1].plot(kc, cham, label=p)
    ax[0].set_xlabel("archived frame"); ax[0].set_ylabel(f"mean |x - x_{base}| (spacings)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("archived frame"); ax[1].set_ylabel("chamfer to target (wu)"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle(f"{tgt_name}: trajectory divergence and progress by material")
    fig.tight_layout(); fig.savefig(png, dpi=110)
    print("saved", png)
