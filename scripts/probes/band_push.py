"""Band-outward push probe (docs/experiments.md 2026-09-04): at a delivered state whose
residual is an equatorial surplus band with deficits at both poles, measure for the band's
particles the descent component OUTWARD from the band centre along the long axis, for the
local D_vol (log-mass L2) gradient and the non-local H^-1 gradient. Also (REFUTE Codex F6)
the LATERAL component — radial in the plane transverse to the long axis — and (REFUTE Opus
F6) the outward-normal component on the ON-SUPPORT SURFACE SHELL: particles whose loss cell
is within one cell of an empty target cell, with the outward normal taken from the blurred
target occupancy (the stratum where a surplus band's self-field can push mass through the
silhouette). Off-support particles are reported separately.

usage: python scripts/probes/band_push.py RUN.npz [--axis 1] [--band 0.5] [--core 0.25]
"""
import argparse

import numpy as np
import torch
from scipy.ndimage import binary_dilation, gaussian_filter
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
    ap.add_argument("--no_self_correct", action="store_true")
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
    (gh,) = torch.autograd.grad(d_h1(xg, m, tg, gmin, dx, dims, self_correct=not a.no_self_correct), xg)
    y = x[:, a.axis] - a.centre
    band = y.abs() < a.band; core = y.abs() < a.core
    out = torch.sign(y)
    tr = [i for i in range(3) if i != a.axis]
    rad = torch.zeros_like(x); rad[:, tr] = x[:, tr] - x[:, tr].mean(0)
    rad = rad / rad.norm(dim=1, keepdim=True).clamp_min(1e-9)
    # --- surface shell of the TARGET support on the loss grid ---
    occ = (tg.reshape(dims).numpy() > 0)
    empty = ~occ
    surf = occ & binary_dilation(empty)                       # occupied cells touching empty
    shell_cells = binary_dilation(surf)                        # within one cell of the surface
    blur = gaussian_filter(occ.astype(np.float64), 1.5)
    gx, gy, gz = np.gradient(blur)                             # points INTO the body
    cid = np.clip(np.floor((x.numpy() + a.domain / 2) / dx).astype(int), 0, a.loss_res - 1)
    on_support = occ[cid[:, 0], cid[:, 1], cid[:, 2]]
    in_shell = shell_cells[cid[:, 0], cid[:, 1], cid[:, 2]] & on_support
    nrm = -np.stack([gx, gy, gz], -1)[cid[:, 0], cid[:, 1], cid[:, 2]]   # outward
    nrm = torch.as_tensor(nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12), dtype=torch.float32)
    in_shell = torch.as_tensor(in_shell); off = ~torch.as_tensor(on_support)
    print(f"N={N} band particles {int(band.sum())} core {int(core.sum())} | on-support shell {int(in_shell.sum())} "
          f"(band {int((in_shell & band).sum())}) | off-support {int(off.sum())} | self_correct={not a.no_self_correct}")
    for name, g in (("D_vol L2", gv), ("H^-1", gh)):
        n = g.norm(dim=1).clamp_min(1e-12)
        d = -g[:, a.axis] * out; rel = d / n
        lat = (-g * rad).sum(1) / n
        nout = (-g * nrm).sum(1) / n
        sh, shb = in_shell, in_shell & band
        print(f"{name:9s} poleward/|g|: band {float(d[band].mean() / n[band].mean()):+.3f} core {float(d[core].mean() / n[core].mean()):+.3f} "
              f"(rel>0.5: {float((rel[band] > 0.5).float().mean()):.3f}) | lateral radial/|g|: band {float(lat[band].mean()):+.3f} "
              f"| OUTWARD-NORMAL on shell: all {float(nout[sh].mean()):+.3f} (frac>0.5: {float((nout[sh] > 0.5).float().mean()):.3f}) "
              f"band-shell {float(nout[shb].mean()):+.3f} (frac>0.5: {float((nout[shb] > 0.5).float().mean()):.3f}) | off-support {float(nout[off].mean()) if int(off.sum()) else float('nan'):+.3f} "
              f"| |g| share band {float(n[band].sum() / n.sum()):.3f}")
    edges = np.arange(-3.5, 3.6, 0.5)
    print("axis bin | L2 pole  H^-1 pole | L2 lateral  H^-1 lateral | L2 shell-out  H^-1 shell-out | n, n_shell")
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (y >= lo) & (y < hi)
        if int(s.sum()) < 50:
            continue
        vals = []
        for g in (gv, gh):
            n = g.norm(dim=1).clamp_min(1e-12); vals.append(float(((-g[:, a.axis] * out) / n)[s].mean()))
        for g in (gv, gh):
            n = g.norm(dim=1).clamp_min(1e-12); vals.append(float(((-g * rad).sum(1) / n)[s].mean()))
        ss = s & in_shell
        for g in (gv, gh):
            n = g.norm(dim=1).clamp_min(1e-12)
            vals.append(float(((-g * nrm).sum(1) / n)[ss].mean()) if int(ss.sum()) else float("nan"))
        print(f"[{lo:+.1f},{hi:+.1f}) {vals[0]:+.3f} {vals[1]:+.3f} | {vals[2]:+.3f} {vals[3]:+.3f} | {vals[4]:+.3f} {vals[5]:+.3f} | {int(s.sum())}, {int(ss.sum())}")


if __name__ == "__main__":
    main()
