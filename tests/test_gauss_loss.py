import types

import pytest
import torch

from physmorph.pipeline import gauss_loss


def test_gaussian_covariance_matches_sigma_ff_transpose_and_is_differentiable():
    F = torch.tensor([[[2.0, 0.5, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.25]]], requires_grad=True)
    cov, cov6 = gauss_loss.gaussian_covariance(F, 0.4, jitter=0.0)
    expected = 0.16 * (F @ F.transpose(1, 2))
    assert torch.allclose(cov, expected)
    assert torch.allclose(cov6, expected[:, (0, 0, 0, 1, 1, 2),
                                          (0, 1, 2, 1, 2, 2)])
    cov6.sum().backward()
    assert F.grad is not None and torch.isfinite(F.grad).all()
    assert float(F.grad.abs().sum()) > 0


@pytest.mark.parametrize("has_norm,cov_key", [
    (True, "cov3Ds_precomp"),
    (False, "cov3D_precomp"),
])
def test_render_uses_backend_specific_precomputed_covariance(monkeypatch, has_norm, cov_key):
    calls = []

    class Rasterizer:
        def __init__(self, raster_settings):
            self.settings = raster_settings

        def __call__(self, **kw):
            calls.append(kw)
            # A fake differentiable image depending on both centers and covariance.
            value = kw["means3D"].sum() + kw[cov_key].sum()
            return value.expand(3, 2, 2), None

    mod = types.SimpleNamespace(GaussianRasterizer=Rasterizer)
    monkeypatch.setattr(gauss_loss, "_gs", lambda: (mod, has_norm))
    bundle = gauss_loss.GaussViews.__new__(gauss_loss.GaussViews)
    bundle.sigma0 = 0.3
    bundle.dev = "cpu"

    x = torch.randn(4, 3, requires_grad=True)
    F = torch.eye(3).repeat(4, 1, 1).requires_grad_()
    image = bundle._render(x, object(), F)
    image.sum().backward()

    assert cov_key in calls[0]
    assert "scales" not in calls[0] and "rotations" not in calls[0]
    assert ("norm3Ds_precomp" in calls[0]) is has_norm
    assert x.grad is not None and float(x.grad.abs().sum()) > 0
    assert F.grad is not None and float(F.grad.abs().sum()) > 0


def test_f_none_keeps_isotropic_scale_rotation_fallback(monkeypatch):
    calls = []

    class Rasterizer:
        def __init__(self, raster_settings):
            pass

        def __call__(self, **kw):
            calls.append(kw)
            return kw["means3D"].sum().expand(3, 1, 1)

    mod = types.SimpleNamespace(GaussianRasterizer=Rasterizer)
    monkeypatch.setattr(gauss_loss, "_gs", lambda: (mod, False))
    bundle = gauss_loss.GaussViews.__new__(gauss_loss.GaussViews)
    bundle.sigma0 = 0.25
    bundle.dev = "cpu"
    bundle._render(torch.zeros(2, 3), object())

    assert torch.allclose(calls[0]["scales"], torch.full((2, 3), 0.25))
    assert torch.allclose(calls[0]["rotations"][:, 0], torch.ones(2))
    assert "cov3D_precomp" not in calls[0] and "cov3Ds_precomp" not in calls[0]


def test_surface_mask_removes_interior_gaussians_and_their_gradients(monkeypatch):
    calls = []

    class Rasterizer:
        def __init__(self, raster_settings):
            pass

        def __call__(self, **kw):
            calls.append(kw)
            value = kw["means3D"].sum() + kw["cov3D_precomp"].sum()
            return value.expand(3, 1, 1)

    mod = types.SimpleNamespace(GaussianRasterizer=Rasterizer)
    monkeypatch.setattr(gauss_loss, "_gs", lambda: (mod, False))
    bundle = gauss_loss.GaussViews.__new__(gauss_loss.GaussViews)
    bundle.sigma0 = 0.25
    bundle.dev = "cpu"
    x = torch.randn(4, 3, requires_grad=True)
    F = torch.eye(3).repeat(4, 1, 1).requires_grad_()
    mask = torch.tensor([True, False, True, False])
    bundle._render(x, object(), F, mask).sum().backward()

    assert calls[0]["means3D"].shape[0] == 2
    assert float(x.grad[~mask].abs().sum()) == 0.0
    assert float(F.grad[~mask].abs().sum()) == 0.0
    assert float(x.grad[mask].abs().sum()) > 0.0
    assert float(F.grad[mask].abs().sum()) > 0.0


def test_gaussian_shape_diagnostics_exposes_anisotropy_and_radius():
    F = torch.diag_embed(torch.tensor([[10.0, 1.0, 0.1], [2.0, 1.0, 0.5]]))
    d = gauss_loss.gaussian_shape_diagnostics(F, sigma0=0.2,
                                              reference_spacing=0.1,
                                              radius_sigma=3.0)
    assert d["gauss_condition_max"] == pytest.approx(100.0, rel=1e-5)
    assert d["gauss_radius_max"] == pytest.approx(6.0)
    assert d["gauss_radius_over_spacing_max"] == pytest.approx(60.0)
