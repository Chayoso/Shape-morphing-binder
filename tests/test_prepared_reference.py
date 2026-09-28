"""Prepared observation ownership and operator parity on a CPU-only cloud."""
from types import SimpleNamespace

import pytest
import torch

from physmorph.pipeline.prepared_reference import PreparedReference
from physmorph.pipeline.render_loss import target_silhouettes, shade_targets, d_render, d_pbr
from physmorph.losses.volumetric import rasterize_mass, d_vol_density


def scene():
    torch.manual_seed(43)
    x = .3*torch.randn(24, 3, dtype=torch.float64)
    tgt = SimpleNamespace(m=torch.ones(24, dtype=x.dtype), lgmin=x.new_tensor([-2.]*3),
        ldx=.5, ldims=(9,)*3, m_ref=2., n_support=35, views=[(0., .2), (1., -.2)],
        extent=2., pgmin=x.new_tensor([-2.]*3), pdx=.5, pdims=(9,)*3, pblur=.5,
        gauss=None, surface_gs=None)
    cfg = SimpleNamespace(loss_units='density', phys_loss='ot_pace', render_res=12,
        dvol_form='log', sil_k=1.5, w_hole=2., w_spray=1., w_pbr=.7,
        pbr_denoised=True, pbr_ambient=.25)
    # PBR density normals use float32 unit masses internally.
    x = x.float(); tgt.m = tgt.m.float(); tgt.lgmin = tgt.lgmin.float(); tgt.pgmin = tgt.pgmin.float()
    grid = rasterize_mass(x, tgt.m, tgt.lgmin, tgt.ldx, tgt.ldims)
    alpha = target_silhouettes(x, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k)
    shade = shade_targets(x, tgt.views, cfg.render_res, tgt.extent,
                          tgt.pgmin, tgt.pdx, tgt.pdims, cfg.sil_k, cfg.pbr_ambient,
                          blur_cells=tgt.pblur)
    return x, cfg, tgt, grid, alpha, shade


def test_owned_reference_matches_live_density_and_pbr():
    x, cfg, tgt, grid, alpha, shade = scene()
    ref = PreparedReference.capture(cfg, tgt, grid, alpha, shade, True, 'paced')
    y = (x+.035).requires_grad_()
    observed = ref.terms(y)
    expected = d_vol_density(y, tgt.m, grid, tgt.lgmin, tgt.ldx, tgt.ldims, tgt.m_ref, tgt.n_support)
    silhouette = d_render(y, alpha, tgt.views, cfg.render_res, tgt.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray)
    pbr = d_pbr(y, shade, tgt.views, cfg.render_res, tgt.extent, tgt.pgmin, tgt.pdx,
                tgt.pdims, cfg.sil_k, cfg.pbr_ambient, tgt.pblur)
    torch.testing.assert_close(observed['volume'], expected, rtol=0, atol=0)
    torch.testing.assert_close(observed['render'], silhouette+cfg.w_pbr*pbr, rtol=0, atol=0)
    baseline = {k:v.detach().clone() for k,v in observed.items()}
    grid.zero_(); alpha[0].zero_(); shade[0].zero_(); tgt.m.zero_(); tgt.lgmin.add_(20)
    tgt.views.clear()
    for k,v in ref.terms(y).items():
        torch.testing.assert_close(v, baseline[k], rtol=0, atol=0)
    assert bool(torch.isfinite(torch.autograd.grad(observed['render'], y)[0]).all())


def test_reference_rejects_unsupported_objective():
    x, cfg, tgt, grid, alpha, shade = scene()
    cfg.phys_loss = 'ot_leash'
    with pytest.raises(ValueError, match='ot_pace'):
        PreparedReference.capture(cfg, tgt, grid, alpha, shade, True, 'paced')
