"""Band-outward push probe (docs/experiments.md 2026-09-04): at a delivered state whose
residual is an equatorial surplus band with deficits at both poles, measure for the band's
particles the descent component OUTWARD from the band centre along the long axis, for the
local D_vol (log-mass L2) gradient and the non-local H^-1 gradient. Also (REFUTE F6) the
LATERAL component — radial in the plane transverse to the long axis, i.e. expansion through
the silhouette — and, for particles near/outside the target surface, the component along
the outward target-normal proxy (particle minus nearest target point).

usage: python scripts/probes/band_push.py RUN.npz [--axis 1] [--band 0.5] [--core 0.25]
"""
import argparse

import numpy as np
import torch
from scipy.spatial import cKDTree

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
    # lateral radial direction in the transverse plane (about the long axis)
    tr = [i for i in range(3) if i != a.axis]
    rad = torch.zeros_like(x); rad[:, tr] = x[:, tr] - x[:, tr].mean(0)
    rad = rad / rad.norm(dim=1, keepdim=True).clamp_min(1e-9)
    # outward target-normal proxy: particle minus nearest target point (meaningful off-surface)
    tt = cKDTree(t.numpy()); dist, idx = tt.query(x.numpy())
    sp = float(np.median(tt.query(t.numpy(), k=2)[0][:, 1]))
    nrm = x - t[torch.as_tensor(idx)]; nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-9)
    near = torch.as_tensor(dist > 0.5 * sp)          # off the target support (outside or in gaps)
    print(f"N={N} band particles {int(band.sum())} core {int(core.sum())} | particles >0.5sp off target: {int(near.sum())}")
    for name, g in (("D_vol L2", gv), ("H^-1", gh)):
        n = g.norm(dim=1).clamp_min(1e-12)
        d = -g[:, a.axis] * out; rel = d / n
        lat = (-g * rad).sum(1) / n
        nout = (-g * nrm).sum(1) / n
        print(f"{name:9s} poleward/|g|: band {float(d[band].mean() / n[band].mean()):+.3f} core {float(d[core].mean() / n[core].mean()):+.3f} "
              f"(rel>0.5: band {float((rel[band] > 0.5).float().mean()):.3f}) | LATERAL radial/|g|: band {float(lat[band].mean()):+.3f} "
              f"all {float(lat.mean()):+.3f} (rel>0.5: {float((lat[band] > 0.5).float().mean()):.3f}) | outward-normal on off-support particles: {float(nout[near].mean()):+.3f} "
              f"| |g| share band {float(n[band].sum() / n.sum()):.3f}")
    edges = np.arange(-3.5, 3.6, 0.5)
    print("axis bin | L2 pole  H^-1 pole | L2 lateral  H^-1 lateral | n")
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (y >= lo) & (y < hi)
        if int(s.sum()) < 50:
            continue
        vals = []
        for g in (gv, gh):
            n = g.norm(dim=1).clamp_min(1e-12)
            vals.append(float(((-g[:, a.axis] * out) / n)[s].mean()))
        for g in (gv, gh):
            n = g.norm(dim=1).clamp_min(1e-12)
            vals.append(float(((-g * rad).sum(1) / n)[s].mean()))
        print(f"[{lo:+.1f},{hi:+.1f}) {vals[0]:+.3f} {vals[1]:+.3f} | {vals[2]:+.3f} {vals[3]:+.3f} | {int(s.sum())}")


if __name__ == "__main__":
    main()
