"""P302 continuous primitive adjoints and frozen observation semantics (CPU only)."""
from types import SimpleNamespace

import pytest
import torch

from physmorph.render.studio import DensityNormals
from physmorph.render.surface_gaussians import SurfaceGaussians, tangent_covariance, pack_covariance
from physmorph.pipeline.surface_render_loss import (coverage_error, image_edges, edge_roi,
                                                    SurfaceWindowLoss, ViewTarget)
from physmorph.pipeline.endpoint_contract import validate_endpoint_config
from physmorph.pipeline.config import PipelineConfig


def test_repeated_tangent_eigenvalue_has_finite_correct_adjoint():
    normals = torch.tensor([[0., 0., 1.], [0., 1., 0.]], dtype=torch.double, requires_grad=True)
    sigma = torch.tensor([.2, .3], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda n, s: pack_covariance(tangent_covariance(n, s)),
                                    (normals, sigma))
    expected = torch.tensor([[.04/16, .04, .04], [.09/16, .09, .09]], dtype=torch.double)
    torch.testing.assert_close(torch.linalg.eigvalsh(tangent_covariance(normals, sigma)), expected)


def test_density_normal_continuous_path_matches_directional_difference():
    gen = torch.Generator().manual_seed(41)
    x = (.3*torch.randn(40, 3, generator=gen, dtype=torch.double)).requires_grad_()
    model = DensityNormals(torch.zeros(3, dtype=torch.double), 1., .2)
    direction = torch.randn(x.shape, generator=gen, dtype=x.dtype)
    def objective(p):
        n, magnitude = model(p)
        return (n[:, 1]*magnitude).mean()
    gradient, = torch.autograd.grad(objective(x), x)
    step = 1e-6
    fd = (objective(x.detach()+step*direction)-objective(x.detach()-step*direction))/(2*step)
    torch.testing.assert_close((gradient*direction).sum(), fd, atol=1e-7, rtol=1e-4)


def test_duplicate_positions_and_zero_normals_have_positive_covariance(monkeypatch):
    import physmorph.render.surface_gaussians as module
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(40, 3, generator=gen, dtype=torch.double)*.25
    x[:2] = 0.
    indices = torch.cdist(x, x).argsort(1)[:, :33]
    # Deliberately detached distance output: production Warp's distances are not differentiable.
    monkeypatch.setattr(module, 'knn_self_torch', lambda p, k: (torch.zeros(40, k), indices[:, :k]))
    model = SurfaceGaussians(torch.zeros(3, dtype=x.dtype), 1., .02, .1, cuda_only=False)
    model.normals = lambda p: (p*0, p[:, 0]*0)
    before = model.metadata()
    x.requires_grad_()
    primitive = model(x)
    assert bool((torch.linalg.eigvalsh(primitive.covariance) > 0).all())
    loss = primitive.covariance.sum()+primitive.sigma.sum()
    gradient, = torch.autograd.grad(loss, x)
    assert bool(torch.isfinite(gradient).all()) and float(gradient.norm()) > 0
    model(x.detach()*1.1)
    assert model.metadata() == before  # candidate does not change reference calibration


def test_deficit_excess_edges_and_target_only_roi():
    target = torch.zeros(32, 48); target[8:24, 19:36] = 1
    assert coverage_error(torch.zeros(1), torch.ones(1)).item() == 2
    assert coverage_error(torch.ones(1), torch.zeros(1)).item() == 1
    assert image_edges(torch.ones(10, 10)).abs().max() == 0
    roi = edge_roi(target, 8)
    assert roi == edge_roi(target.clone(), 8)
    y, x, size = roi
    assert float(image_edges(target).square().sum(0)[y:y+size, x:x+size].sum()) > 0


def test_view_loss_is_mean_and_targets_remain_frozen():
    class Raster:
        def coverage(self, x, primitive):
            return x[0, 0].expand(16, 16)
    x = torch.ones(33, 3, requires_grad=True)
    target = ViewTarget(torch.zeros(16, 16), torch.zeros(8, 8), torch.zeros(2, 8, 8), (4, 4, 8))
    def make(count):
        shared = SimpleNamespace(geometry=lambda p: None, coarse=[Raster()]*count, detail=[Raster()]*count,
                                 deficit_weight=2., excess_weight=1.)
        return SurfaceWindowLoss(shared, [target]*count, 'paced')
    a, b = make(1), make(4)
    torch.testing.assert_close(a(x)[0], b(x)[0])
    torch.testing.assert_close(torch.autograd.grad(a(x)[0], x)[0], torch.autograd.grad(b(x)[0], x)[0])
    a(x*.5)
    assert a.targets[0].roi == (4, 4, 8) and not target.coarse.any()


def test_surface_contract_is_opt_in_cuda_shared_endpoint():
    cfg = PipelineConfig(surface_gs_loss=True, compute_backend='cuda', device='cuda:0',
                         commit_pic=True, commit_pic_objective=True)
    validate_endpoint_config(cfg)
    cfg.compute_backend = 'numpy'
    with pytest.raises(ValueError, match='CUDA'):
        validate_endpoint_config(cfg)


def test_default_cameras_cover_opposite_azimuths():
    import math
    from physmorph.pipeline.render_loss import make_views
    from physmorph.pipeline.surface_render_loss import select_surface_views
    views = select_surface_views(make_views(6), 4)
    assert len(views) == 4
    assert any(math.cos(a-views[0][0]) < -.8 for a, e in views[1:])
    assert max(a for a, e in views)-min(a for a, e in views) > math.pi
