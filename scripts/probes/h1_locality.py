"""Locality probe on a delivered state (REVISION 3 amendment, docs/root_analysis.md).

At the delivered state of a solid run, compare the D_vol (L2, local) gradient with the
H^-1 (non-local) gradient of the same residual:
  * where each gradient's norm lives (ear-region particles vs body),
  * for BODY particles, the mean component of the descent direction toward the ear
    region (positive = mass is told to move into the ears),
  * the cosine between the two per-particle gradient fields.
The claim under test: D_vol gives body particles no ear-directed signal (the wrong fixed
point), H^-1 does.

usage: python scripts/probes/h1_locality.py RUN.npz [--loss_res 64] [--ear_frac 0.10]
"""
import argparse

import numpy as np
import torch

from physmorph.losses.volumetric import d_h1, d_vol, rasterize_mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--loss_res", type=int, default=64)
    ap.add_argument("--domain", type=float, default=32.0, help="MPM domain edge (wu)")
    ap.add_argument("--ear_frac", type=float, default=0.10)
    a = ap.parse_args()
    z = np.load(a.npz)
    dn = int(z["deliver_n"]) if "deliver_n" in z else len(z["frames"])
    x = torch.tensor(z["frames"][dn - 1], dtype=torch.float32)
    t = torch.tensor(z["tgt"], dtype=torch.float32)
    N, M = len(x), len(t)
    m = torch.ones(N); mt = torch.ones(M) * (N / M)
    dims = (a.loss_res,) * 3
    dx = a.domain / a.loss_res
    gmin = torch.full((3,), -a.domain / 2)
    tg = rasterize_mass(t, mt, gmin, dx, dims)
    # ear region: top ear_frac of the TARGET along its longest axis (same census as runner)
    ext = (t.max(0).values - t.min(0).values); ax = int(ext.argmax())
    thr = torch.quantile(t[:, ax], 1 - a.ear_frac)
    ear_t = t[t[:, ax] >= thr]; ear_c = ear_t.mean(0)
    ear_p = x[:, ax] >= thr
    print(f"N={N} M={M} loss_res={a.loss_res} dx={dx:.3f}  ear axis={ax} thr={float(thr):.3f} "
          f"ear particles {int(ear_p.sum())} ({ear_p.float().mean():.3f} of N; target {a.ear_frac})")
    xg = x.clone().requires_grad_(True)
    (gv,) = torch.autograd.grad(d_vol(xg, m, tg, gmin, dx, dims), xg)
    (gh,) = torch.autograd.grad(d_h1(xg, m, tg, gmin, dx, dims), xg)
    body = ~ear_p
    to_ear = ear_c[None, :] - x; to_ear = to_ear / to_ear.norm(dim=1, keepdim=True).clamp_min(1e-9)
    for name, g in (("D_vol (L2)", gv), ("H^-1", gh)):
        n = g.norm(dim=1)
        tot = float(n.sum())
        comp = (-g * to_ear).sum(1)                   # ear-directed descent component
        rel = comp / n.clamp_min(1e-12)
        print(f"{name:11s} |g| share: ears {float(n[ear_p].sum()) / tot:.3f} body {float(n[body].sum()) / tot:.3f} | "
              f"body ear-directed component: mean rel {float(rel[body].mean()):+.3f} "
              f"(frac of body particles with rel>0.5: {float((rel[body] > 0.5).float().mean()):.3f}) | "
              f"ears: mean rel {float(rel[ear_p].mean()):+.3f}")
    cos = torch.nn.functional.cosine_similarity(gv, gh, dim=1)
    print(f"cos(D_vol, H^-1) per particle: median body {float(cos[body].median()):+.3f} ears {float(cos[ear_p].median()):+.3f}")
    # net flux check: mean descent of the body along the ear direction, in units of
    # the body's mean gradient magnitude
    for name, g in (("D_vol (L2)", gv), ("H^-1", gh)):
        net = float((-g[body] * to_ear[body]).sum(1).mean()) / float(g[body].norm(dim=1).mean())
        print(f"{name:11s} body net ear-directed / mean|g|: {net:+.4f}")


if __name__ == "__main__":
    main()
