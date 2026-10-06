"""The comparison baseline's loss (D112): Xu et al.'s EndLayerMassLoss as the C++ oracle computes it
(legacy/DiffMPMLib3D/CompGraph.cpp, losses/volumetric.d_vol_xu): the cubic B-spline P2G keeps the mass, the value is
the oracle's, and the gradient is the value's own inside the target and five times it where the target is empty."""
import torch

from physmorph.losses.volumetric import d_vol_xu, rasterize_mass_cubic

GMIN, DX, DIMS = torch.tensor([-2.0, -2.0, -2.0]), 0.25, (16, 16, 16)


def test_the_cubic_p2g_keeps_the_mass():
    x = torch.rand(500, 3) * 1.5 - 0.75                       # well inside the grid: every stencil whole
    m = torch.rand(500)
    assert torch.allclose(rasterize_mass_cubic(x, m, GMIN, DX, DIMS).sum(), m.sum(), rtol=1e-5)


def test_the_value_and_the_out_of_target_gradient():
    tgt = torch.rand(400, 3) * 0.6 - 0.3                      # a small target blob around the origin
    t_grid = rasterize_mass_cubic(tgt, torch.full((400,), 0.05), GMIN, DX, DIMS)
    far = torch.tensor([[1.2, 1.2, 1.2]])                     # a particle whose whole stencil is empty in the target
    x = torch.cat((tgt[:200] * 0.8, far)).requires_grad_(True)
    m = torch.full((201,), 0.05)
    loss = d_vol_xu(x, m, t_grid, GMIN, DX, DIMS)
    cur = rasterize_mass_cubic(x.detach(), m, GMIN, DX, DIMS)
    log_diff = torch.log(cur + 1 + 1e-4) - torch.log(t_grid + 1 + 1e-4)
    want = 0.5 * log_diff.pow(2).sum() + ((1e-3 - cur).pow(2) * (cur < 1e-3)).sum()
    assert torch.allclose(loss.detach(), want, rtol=1e-6)     # the oracle's value
    g, = torch.autograd.grad(loss, x)
    # the value's own gradient (what plain autograd of the oracle's value gives)
    xv = x.detach().clone().requires_grad_(True)
    cv = rasterize_mass_cubic(xv, m, GMIN, DX, DIMS)
    lv = 0.5 * (torch.log(cv + 1 + 1e-4) - torch.log(t_grid + 1 + 1e-4)).pow(2).sum() \
        + ((1e-3 - cv).pow(2) * (cv < 1e-3)).sum()
    gv, = torch.autograd.grad(lv, xv)
    assert torch.allclose(g[-1], 5.0 * gv[-1], rtol=1e-5, atol=1e-12)   # outside the target: five times
    inside = (t_grid > 1e-12).reshape(DIMS)
    assert bool(inside.any())
