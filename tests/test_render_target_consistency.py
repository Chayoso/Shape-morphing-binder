import numpy as np
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.runner import build_target
from physmorph.pipeline.render_loss import d_pbr


def test_matched_shading_has_zero_residual_and_force_without_surface_reconstruction(monkeypatch):
    from physmorph.render import surface_recon

    def unavailable_surface(*args, **kwargs):
        raise AssertionError("matched shading must not require unused surface reconstruction")

    monkeypatch.setattr(surface_recon, "target_surface_normals", unavailable_surface)
    pts = np.random.default_rng(4).uniform(-1., 1., (300, 3)).astype(np.float32)
    prm = MPMParams(dx=.25, nx=32, ny=32, nz=32, grid_min=(-4., -4., -4.))
    cfg = PipelineConfig(device='cpu', render_views=2, render_elevs=(0.,), render_res=16,
                         loss_res=16, lambda_auto=.5, w_pbr=1., pbr_denoised=True,
                         pbr_target_mode='matched')
    pack = build_target(pts, prm, cfg)
    x = torch.tensor(pts, requires_grad=True)
    loss = d_pbr(x, pack.shade, pack.views, cfg.render_res, pack.extent,
                 pack.pgmin, pack.pdx, pack.pdims, cfg.sil_k, cfg.pbr_ambient, pack.pblur)
    g, = torch.autograd.grad(loss, x)
    assert float(loss) < 1e-10
    assert float(g.norm()) < 1e-6
