"""ridge_closed_loop.py ARCHIVE_NPZ RUN_JSON WAVELENGTH_WU_DESIGN [DX=0.30] — D15: how much of a ridged target's relief a
morph carries, at the end of the controlled half and at the end of every committed window.

The target is make_ridge_slab.py's box (ridges y = A sin(2 pi x / wavelength) on the top face). The relief is the
sinusoid of that wavelength in the heights of the top layer (the pipeline's layer rule, facing up, away from the
box's edges), fitted by least squares together with a quadratic surface (the body's large-scale shape while it
morphs). The run's target sample gives the reference: its own fitted amplitude and phase. For the body, per window:
the share of the target's relief it carries in phase with the target, its amplitude whatever the phase, and the rms
height error against the target's fitted surface, in particle spacings."""
import json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from physmorph import gpu                                      # noqa: E402  (before torch: CuPy binds its CUDA 12 compiler first)
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402
from physmorph.pipeline.window.layer import layer_by_asymmetry, layer_spacing  # noqa: E402

dev = torch.device("cuda")
z = np.load(sys.argv[1], allow_pickle=True)
arm = json.load(open(sys.argv[2]))["arms"]["render_full_dt_iso_nn"]
lam_design = float(sys.argv[3])
DX = float(sys.argv[4]) if len(sys.argv) > 4 else 0.30
L_DESIGN = 4.454
frames = z["frames"]
n_del = int(z["deliver_n"])
tgt = torch.as_tensor(np.asarray(z["tgt"], np.float32), device=dev)
N = len(tgt)
a = float(((tgt[:, 0].max() - tgt[:, 0].min()) * (tgt[:, 1].max() - tgt[:, 1].min()) * (tgt[:, 2].max() - tgt[:, 2].min()) / N) ** (1 / 3))
half = 0.5 * float(tgt[:, 0].max() - tgt[:, 0].min() + a)
scale = 2 * half / L_DESIGN


def top(x):
    mask, nrm = layer_by_asymmetry(x, layer_spacing(x))
    m = mask & (nrm[:, 1] > .7) & (x[:, 1] > 0) & (x[:, 0].abs() < half - 1.5 * DX) & (x[:, 2].abs() < half - 1.5 * DX)
    return x[m]


def fit(p, lam):
    """Least squares: height = quadratic surface + alpha sin + beta cos. Returns (alpha, beta), coefficients, residual rms."""
    kx = 2 * math.pi * p[:, 0] / lam
    B = torch.stack([torch.ones_like(kx), p[:, 0], p[:, 2], p[:, 0] ** 2, p[:, 2] ** 2, p[:, 0] * p[:, 2], torch.sin(kx), torch.cos(kx)], 1).double()
    sol = torch.linalg.lstsq(B, p[:, 1].double()[:, None]).solution[:, 0]
    res = float((B @ sol - p[:, 1].double()).pow(2).mean().sqrt())
    return sol[6:8], sol, res


tt = top(tgt)
lams = [lam_design * scale * (1 + e) for e in np.linspace(-.04, .04, 41)]
lam = max(lams, key=lambda l: float(fit(tt, l)[0].norm()))
ab_t, sol_t, res_t = fit(tt, lam)
A_t = float(ab_t.norm())
print(f"N {N}; spacing {a:.4f} wu; target: scale {scale:.4f}, wavelength {lam:.4f} wu = {lam / a:.1f} spacings = {lam / DX:.2f} cells; relief in the target sample {A_t:.4f} wu = {A_t / a:.2f} spacings "
      f"({A_t / (lam / 8):.2f} of the mesh's), top layer {len(tt)} points, residual {res_t / a:.2f} spacings")


def state(raw):
    p = top(torch.as_tensor(np.asarray(frames[raw], np.float32), device=dev))
    if len(p) < 50:
        return None
    ab, sol, res = fit(p, lam)
    kx = 2 * math.pi * p[:, 0] / lam
    Bt = torch.stack([torch.ones_like(kx), p[:, 0], p[:, 2], p[:, 0] ** 2, p[:, 2] ** 2, p[:, 0] * p[:, 2], torch.sin(kx), torch.cos(kx)], 1).double()
    err = float((p[:, 1].double() - Bt @ sol_t).pow(2).mean().sqrt())
    return float((ab * ab_t).sum()) / A_t ** 2, float(ab.norm()) / A_t, err / a, len(p)


com = [r for r in arm["history"] if r.get("frame_end") and not r.get("null_commit") and r["frame_end"] <= n_del]
print(f"committed windows {len(com)}, delivered frames {n_del}, silhouette IoU {arm['metrics']['sil_iou']:.4f}, minutes {arm['seconds'] / 60:.1f}, "
      f"lambda of window 0 {com[0].get('lambda'):.3g}, median g_share {float(np.median([r['g_share'] for r in com if r.get('g_share') is not None])):.2f}")
print("window | end of the controlled half: in-phase share, amplitude share, height rms (spacings) | end of the window: the same | top-layer points")
rows = []
for i, r in enumerate(com):
    fe = r["frame_end"]
    d, e = state(fe - 21), state(fe - 1)
    if d is None or e is None:
        continue
    rows.append((i, d, e))
    if i < 8 or i % 5 == 0 or i >= len(com) - 3:
        print(f"{i:4d} | {d[0]:+.2f} {d[1]:.2f} {d[2]:5.2f} | {e[0]:+.2f} {e[1]:.2f} {e[2]:5.2f} | {e[3]}")
tail = rows[-10:]
if tail:
    m = lambda k, j: float(np.mean([t[k][j] for t in tail]))
    print(f"last {len(tail)} windows: in-phase share {m(1, 0):+.2f} at the end of the controlled half, {m(2, 0):+.2f} at the window's end; amplitude share {m(1, 1):.2f}, {m(2, 1):.2f}; "
          f"height rms {m(1, 2):.2f}, {m(2, 2):.2f} spacings")
    best = max(rows, key=lambda t: t[2][0])
    print(f"largest in-phase share at a window's end: {best[2][0]:+.2f} at window {best[0]}")
