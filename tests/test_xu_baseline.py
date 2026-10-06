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


def test_the_published_form_is_its_own_gradient():
    """xu_form "paper" (Xu et al., arXiv 2409.15746): 1/2 sum (ln(m+1) - ln(m*+1))^2, no min-mass penalty, and the
    gradient the value's own everywhere, so a line search on the value reads the step it was given."""
    from physmorph.pipeline.config import PipelineConfig
    kw = PipelineConfig(xu_form="paper").xu_kw()
    tgt = torch.rand(400, 3) * 0.6 - 0.3
    t_grid = rasterize_mass_cubic(tgt, torch.full((400,), 0.05), GMIN, DX, DIMS)
    x = torch.cat((tgt[:200] * 0.8, torch.tensor([[1.2, 1.2, 1.2]]))).requires_grad_(True)
    m = torch.full((201,), 0.05)
    loss = d_vol_xu(x, m, t_grid, GMIN, DX, DIMS, **kw)
    g, = torch.autograd.grad(loss, x)
    xv = x.detach().clone().requires_grad_(True)
    lv = 0.5 * (torch.log(rasterize_mass_cubic(xv, m, GMIN, DX, DIMS) + 1) - torch.log(t_grid + 1)).pow(2).sum()
    gv, = torch.autograd.grad(lv, xv)
    assert torch.allclose(loss.detach(), lv.detach(), rtol=1e-6)
    assert torch.allclose(g, gv, rtol=1e-5, atol=1e-12)


def test_the_journal_form_is_its_own_gradient():
    """xu_form "tvcg" (Xu et al., IEEE TVCG 2025, Eqs. 7-9): L_mass = 1/2 sum (ln(m+1+eps) - ln(m*+1+eps))^2 with
    eps = 1e-4, plus L_penalty = sum 1{m < m_min} w (m_min - m)^2 with w = 10 (m_min 1e-3, the C++ copy's); no
    out-of-target factor, so the gradient is the value's own, the penalty's included (nodes at the body's fringe)."""
    from physmorph.pipeline.config import PipelineConfig
    kw = PipelineConfig(xu_form="tvcg").xu_kw()
    assert kw == dict(out_of_target=1.0, penalty_weight=10.0, eps=1e-4, min_mass=1e-3)
    tgt = torch.rand(400, 3) * 0.6 - 0.3
    t_grid = rasterize_mass_cubic(tgt, torch.full((400,), 0.05), GMIN, DX, DIMS)
    x = torch.cat((tgt[:200] * 0.8, torch.tensor([[1.2, 1.2, 1.2]]))).requires_grad_(True)
    m = torch.full((201,), 0.05)
    loss = d_vol_xu(x, m, t_grid, GMIN, DX, DIMS, **kw)
    g, = torch.autograd.grad(loss, x)
    xv = x.detach().clone().requires_grad_(True)
    cv = rasterize_mass_cubic(xv, m, GMIN, DX, DIMS)
    lv = (0.5 * (torch.log(cv + 1 + 1e-4) - torch.log(t_grid + 1 + 1e-4)).pow(2).sum()
          + (10.0 * (1e-3 - cv).pow(2) * (cv < 1e-3)).sum())
    gv, = torch.autograd.grad(lv, xv)
    fringe = ((cv > 0) & (cv < 1e-3)).sum()
    assert int(fringe) > 0                                    # the penalty acts on some node with mass
    assert torch.allclose(loss.detach(), lv.detach(), rtol=1e-6)
    assert torch.allclose(g, gv, rtol=1e-5, atol=1e-12)
