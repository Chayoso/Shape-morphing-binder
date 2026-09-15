"""Dimensionless D_vol (loss_units="density"): same minimiser as eq (13), value
approximately invariant to the loss-grid resolution, gradient O(1)-commensurable."""
import numpy as np
import torch

from physmorph.losses.volumetric import (d_vol, d_vol_density, density_units,
                                         target_mass_grid)


def _ball(n, r, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return (v * r * rng.uniform(0, 1, (n, 1)) ** (1 / 3)).astype(np.float32)


def _setup(res, n=300000):
    # SAME samples for source and target (source = anisotropically scaled target), so
    # the residual is the deterministic shape mismatch and not Poisson sampling noise,
    # which is resolution-dependent by nature (fewer particles per finer cell)
    t = torch.tensor(_ball(n, 1.0, 1))
    x = t * torch.tensor([1.2, 0.9, 1.0])
    m = torch.ones(n)
    gmin = torch.tensor([-2.0, -2.0, -2.0])
    dx = 4.0 / res
    tg = target_mass_grid(t, m, gmin, dx, (res,) * 3)
    return x, m, tg, gmin, dx, (res,) * 3


def test_density_form_is_resolution_invariant_where_legacy_is_not():
    vals_leg, vals_den = [], []
    for res in (24, 48):
        x, m, tg, gmin, dx, dims = _setup(res)
        m_ref, n_sup = density_units(tg)
        vals_leg.append(float(d_vol(x, m, tg, gmin, dx, dims)))
        vals_den.append(float(d_vol_density(x, m, tg, gmin, dx, dims, m_ref, n_sup)))
    ratio_leg = vals_leg[1] / vals_leg[0]
    ratio_den = vals_den[1] / vals_den[0]
    # legacy grows with the CELL COUNT (~8x per halving of dx on a 3-D grid, minus the
    # log saturation); the density form only carries the Riemann-sum convergence of an
    # under-resolved boundary band (the mismatch shell is ~1 cell at res 24)
    assert ratio_leg > 2.5, (vals_leg, vals_den)
    assert ratio_den < 1.6 and ratio_den < 0.5 * ratio_leg, (vals_leg, vals_den)


def test_same_minimiser_zero_at_target():
    x, m, tg, gmin, dx, dims = _setup(16)
    t = torch.tensor(_ball(300000, 1.0, 1))
    m_ref, n_sup = density_units(tg)
    assert float(d_vol_density(t, m, tg, gmin, dx, dims, m_ref, n_sup)) < 1e-9
    assert float(d_vol(t, m, tg, gmin, dx, dims)) < 1e-6


def test_density_gradient_is_a_descent_direction_with_small_units():
    x, m, tg, gmin, dx, dims = _setup(24)
    m_ref, n_sup = density_units(tg)
    xg = x.clone().requires_grad_(True)
    L0 = d_vol_density(xg, m, tg, gmin, dx, dims, m_ref, n_sup)
    g_den = torch.autograd.grad(L0, xg)[0]
    xg2 = x.clone().requires_grad_(True)
    g_leg = torch.autograd.grad(d_vol(xg2, m, tg, gmin, dx, dims), xg2)[0]
    assert torch.isfinite(g_den).all()
    assert g_den.norm() < g_leg.norm()           # units shrink, not grow
    # a small step along -g decreases the loss (it is a real gradient of the scalar)
    step = 0.05 * dx / g_den.norm() * (len(x) ** 0.5)
    with torch.no_grad():
        L1 = d_vol_density(x - step * g_den, m, tg, gmin, dx, dims, m_ref, n_sup)
    assert float(L1) < float(L0)
