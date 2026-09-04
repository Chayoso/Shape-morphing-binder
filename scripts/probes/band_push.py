"""Band-outward push probe (docs/experiments.md 2026-09-04): at a delivered state whose
residual is an equatorial surplus band with deficits at both poles, measure for the band's
particles the descent component OUTWARD from the band centre along the long axis, for the
local D_vol (L2) gradient and the non-local H^-1 gradient. The claim: L2 pushes only at the
band edges (zero inside a uniform surplus), H^-1 pushes every band particle toward a pole.

usage: python scripts/probes/band_push.py RUN.npz [--axis 1] [--band 0.5] [--core 0.25]
"""
import argparse

import numpy as np
import torch

from physmorph.losses.volumetric import d_h1, d_vol, rasterize_mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--axis", type=int, default=1)
    ap.add_argument("--band", type=float, default=0.5)
    ap.add_argument("--core", type=float, default=0.25)
    ap.add_argument("--centre", type=float, default=0.0)
    ap.add_argument("--loss_res", type=int, default=64)
    ap.add_argument("--domain", type=float, default=32.0)
    a = ap.parse_args()
    z = np.load(a.npz)
    dn = int(z["deliver_n"]) if "deliver_n" in z else len(z["frames"])
    x = torch.tensor(z["frames"][dn - 1], dtype=torch.float32)
    t = torch.tensor(z["tgt"], dtype=torch.float32)
    N, M = len(x), len(t)
    m = torch.ones(N); mt = torch.ones(M) * (N / M)
    dims = (a.loss_res,) * 3; dx = a.domain / a.loss_res
    gmin = torch.full((3,), -a.domain / 2)
    tg = rasterize_mass(t, mt, gmin, dx, dims)
    xg = x.clone().requires_grad_(True)
    (gv,) = torch.autograd.grad(d_vol(xg, m, tg, gmin, dx, dims), xg)
    (gh,) = torch.autograd.grad(d_h1(xg, m, tg, gmin, dx, dims), xg)
    y = x[:, a.axis] - a.centre
    band = y.abs() < a.band; core = y.abs() < a.core
    out = torch.sign(y)
    print(f"N={N} band particles {int(band.sum())} core {int(core.sum())}")
    for name, g in (("D_vol L2", gv), ("H^-1", gh)):
        d = -g[:, a.axis] * out; n = g.norm(dim=1); rel = d / n.clamp_min(1e-12)
        print(f"{name:9s} outward/mean|g|: band {float(d[band].mean() / n[band].mean()):+.3f} "
              f"core {float(d[core].mean() / n[core].mean()):+.3f} | outward-dominant (rel>0.5): "
              f"band {float((rel[band] > 0.5).float().mean()):.3f} core {float((rel[core] > 0.5).float().mean()):.3f} | "
              f"|g| share in band {float(n[band].sum() / n.sum()):.3f}")
    # profile of the outward component along the axis (both gradients), 0.5 wu bins
    edges = np.arange(-3.5, 3.6, 0.5)
    print("axis bin | L2 outward rel | H^-1 outward rel | n")
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (y >= lo) & (y < hi)
        if int(s.sum()) < 50:
            continue
        r2 = float(((-gv[:, a.axis] * out) / gv.norm(dim=1).clamp_min(1e-12))[s].mean())
        rh = float(((-gh[:, a.axis] * out) / gh.norm(dim=1).clamp_min(1e-12))[s].mean())
        print(f"[{lo:+.1f},{hi:+.1f}) {r2:+.3f} {rh:+.3f} {int(s.sum())}")


if __name__ == "__main__":
    main()
