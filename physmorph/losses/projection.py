"""Support-preserving projection of a per-particle displacement (docs/method.md 10.22).

The paced target advects the cloud along the transport plan by at most one cell per window
(optimizer.py, the ot_pace/ot_shape block). For a volume-preserving but anisotropic map the
straight-ray (displacement) interpolant is not volume preserving in transit: with the map's
Jacobian J, det(I + t(J - I)) is not 1 for 0 < t < 1, so the paced density falls below bulk
where the map stretches (a fan converging into a thin feature) and rises above it where the
rays converge. On the 300k bunny the cloud realises that literally: a bulge at the ear base
(1.5-1.7x the target mass) feeding a sub-cell filament at 50-70 % of the target thickness,
which breaks into pieces at the native spacing (docs/experiments.md 2026-09-24, ear_slab).

The projection removes the divergent part of the paced step on the occupied cells: a Chorin
projection on a MAC grid with the pressure zero outside the body (the density constraint of
Maury, Roudneff-Chupin and Santambrogio 2010: the admissible velocity is the projection of the
desired one; the Hele-Shaw growth of Perthame, Quiros and Vazquez 2014: pressure lives on the
saturated set and vanishes on the free surface). The advected cloud then keeps bulk density and
a thin feature is extruded from the body instead of being assembled from a sparse stream.

Discretisation on the loss grid's node lattice: face velocities by the mass-weighted CIC deposit
of the step at the lattice shifted by half a cell along each axis; a node belongs to the body
where its CIC mass is at least half the bulk cell mass (its position inside the material to within
the CIC ramp); -Lap p = -div u on the body with p = 0 outside (conjugate gradients); the
correction -grad p on the faces is gathered back at the particles (FLIP-style: the particle keeps
its own step, minus the grid's divergent part). No constant is tuned: the half-bulk occupancy is
the CIC value of a node on the surface, the tolerance is a solver residual.
"""
from __future__ import annotations

import torch

from .volumetric import gather_cic, rasterize_mass


def _cg(apply, b, iters: int, tol: float):
    p = torch.zeros_like(b)
    r = b.clone()
    z = r.clone()
    rr = float((r * r).sum())
    b_norm = float(b.norm()) + 1e-30
    n_it = 0
    if rr ** 0.5 <= tol * b_norm:
        return p, 0, rr ** 0.5 / b_norm
    for n_it in range(1, iters + 1):
        Az = apply(z)
        a = rr / max(float((z * Az).sum()), 1e-30)
        p = p + a * z
        r = r - a * Az
        rr_new = float((r * r).sum())
        if rr_new ** 0.5 <= tol * b_norm:
            rr = rr_new
            break
        z = r + (rr_new / rr) * z
        rr = rr_new
    return p, n_it, rr ** 0.5 / b_norm


def bulk_mode(mass: torch.Tensor, nb: int = 64) -> float:
    """The bulk node mass of a CIC mass grid: the mode of the node-mass histogram over the positive
    nodes, referenced to their median (range 0 .. 3 median; the lowest quarter of the median
    excluded as the CIC halo). Meant for a well-sampled uniform body — the TARGET grid — whose
    interior nodes form the peak. On a cloud in transit it is not reliable: while the straight-ray
    target spreads the crown into a sparse fan, most positive nodes are partial, the median falls
    (l300 windows 3–6: 178 -> 20) and the mode with it; a high-quantile reference fails the other
    way on a compressed pocket. The optimizer therefore reads the target's value once and passes it.
    """
    pos = mass[mass > 0]
    if pos.numel() == 0:
        return 0.0
    med = float(torch.median(pos))
    top = 3.0 * med
    hist = torch.histc(pos, bins=nb, min=0.0, max=top)
    lo = int(nb * 0.25 / 3.0)
    k = int(torch.argmax(hist[lo:])) + lo
    return (k + 0.5) * top / nb


def project_step(x: torch.Tensor, d: torch.Tensor, m, grid_min: torch.Tensor, dx: float, dims,
                 body_frac: float = 0.5, iters: int = 400, tol: float = 1e-4, bulk=None):
    """Return (d_projected, stats) for particle positions x (N,3) and per-particle steps d (N,3).

    stats: body (nodes in the body), bulk (the bulk node mass), cg_iters, cg_res, div0/div1 (rms
    of the discrete divergence of the step over the body, before/after: a relative volume change
    per window), div0_p95, corr_med/corr_p95 (the correction |grad p| gathered at the particles,
    in cells), pic_change (median |gathered step - own step| / |own step|: how far the grid's
    view of the step sits from the particle's own).
    """
    nx, ny, nz = [int(v) for v in dims]
    dev, dtype = x.device, x.dtype
    x = x.detach()
    d = d.detach().to(dtype)
    m = torch.as_tensor(m, device=dev, dtype=dtype)
    if m.dim() == 0:
        m = m.expand(x.shape[0])
    mass = rasterize_mass(x, m, grid_min, dx, dims)
    pos = mass[mass > 0]
    # the bulk node mass = the MODE of the node-mass histogram above a quarter of its 99th
    # percentile: the interior nodes sit at the bulk value (± the sampling noise), the surface
    # ramp and the CIC halo spread thinly below it, and a compressed pocket (a pile-up at a
    # feature's root, up to several times bulk) is a minority — a median above half the maximum
    # followed that pocket and shrank the body to a few hundred nodes (n300, first launch)
    # the bulk node mass: the target's value when given (the reference the cloud converges to),
    # else the cloud's own mode (a ball at rest, the tests)
    bulk = float(bulk) if (bulk is not None and float(bulk) > 0) else bulk_mode(mass)
    peak, dense_frac, peak_pos = 0.0, 0.0, (float("nan"),) * 3
    if pos.numel() and bulk > 0:
        peak = float(pos.max()) / bulk
        dense_frac = float(mass[mass >= 2.0 * bulk].sum() / mass.sum())
        i_pk = int(torch.argmax(mass))
        ijk = (i_pk // (ny * nz), (i_pk // nz) % ny, i_pk % nz)
        peak_pos = tuple(float(grid_min[a]) + dx * ijk[a] for a in range(3))
    body = (mass >= body_frac * bulk).view(nx, ny, nz) if bulk > 0 else torch.zeros(nx, ny, nz, dtype=torch.bool, device=dev)
    # MAC face velocities: the mass-weighted step deposited at the lattice shifted by half a cell
    # along each axis (face index i sits between nodes i and i+1)
    uf = []
    xs_all = []
    for ax in range(3):
        off = torch.zeros(3, device=dev, dtype=dtype)
        off[ax] = 0.5 * dx
        xs = x - off
        xs_all.append(xs)
        mf = rasterize_mass(xs, m, grid_min, dx, dims)
        pf = rasterize_mass(xs, m * d[:, ax], grid_min, dx, dims)
        uf.append(torch.where(mf > 0, pf / mf.clamp_min(1e-30), torch.zeros_like(pf)).view(nx, ny, nz))

    def div(u):
        out = u[0].clone()
        out[1:] -= u[0][:-1]
        out += u[1]
        out[:, 1:] -= u[1][:, :-1]
        out += u[2]
        out[:, :, 1:] -= u[2][:, :, :-1]
        return out / dx

    div0 = div(uf)
    dx2 = dx * dx
    zero = torch.zeros(nx, ny, nz, device=dev, dtype=dtype)

    def apply(p):
        p = torch.where(body, p, zero)
        q = torch.zeros_like(p)
        q[1:] += p[:-1]
        q[:-1] += p[1:]
        q[:, 1:] += p[:, :-1]
        q[:, :-1] += p[:, 1:]
        q[:, :, 1:] += p[:, :, :-1]
        q[:, :, :-1] += p[:, :, 1:]
        return torch.where(body, (6.0 * p - q) / dx2, zero)

    b = torch.where(body, -div0, zero)
    p, n_it, res = _cg(apply, b, iters, tol)
    p = torch.where(body, p, zero)
    # the correction on the faces: -(p[c+e] - p[c]) / dx, p = 0 outside the body (the Dirichlet
    # free surface); faces between two outside nodes carry no correction
    corr = [torch.zeros_like(u) for u in uf]
    corr[0][:-1] = -(p[1:] - p[:-1]) / dx
    corr[1][:, :-1] = -(p[:, 1:] - p[:, :-1]) / dx
    corr[2][:, :, :-1] = -(p[:, :, 1:] - p[:, :, :-1]) / dx
    div1 = div([u + c for u, c in zip(uf, corr)])
    out = d.clone()
    g_corr = torch.zeros_like(d)
    g_pic = torch.zeros_like(d)
    for ax in range(3):
        g_corr[:, ax] = gather_cic(corr[ax].reshape(-1), xs_all[ax], grid_min, dx, dims)
        g_pic[:, ax] = gather_cic(uf[ax].reshape(-1), xs_all[ax], grid_min, dx, dims)
    out = out + g_corr
    dn = d.norm(dim=1).clamp_min(1e-12)
    cn = g_corr.norm(dim=1) / dx
    nb = int(body.sum())
    stats = {
        "body": nb, "bulk": bulk, "peak": peak, "dense_frac": dense_frac, "peak_pos": peak_pos,
        "cg_iters": int(n_it), "cg_res": float(res),
        "div0": float(div0[body].pow(2).mean().sqrt()) if nb else 0.0,
        "div1": float(div1[body].pow(2).mean().sqrt()) if nb else 0.0,
        "div0_p95": float(div0[body].abs().quantile(0.95)) if nb else 0.0,
        "corr_med": float(cn.median()), "corr_p95": float(cn.quantile(0.95)),
        "pic_change": float(((g_pic - d).norm(dim=1) / dn).median()),
    }
    return out, stats
