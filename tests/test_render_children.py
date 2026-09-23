import types

import numpy as np
import torch

from physmorph.pipeline import gauss_loss
from physmorph.render.children import (expand_children_torch,
                                       tangent_child_offsets)


def test_tangent_offsets_are_centered_and_lie_in_planar_tangent():
    a = np.linspace(-0.2, 0.2, 4, dtype=np.float32)
    x = np.stack(np.meshgrid(a, a, [0.0], indexing="ij"), -1).reshape(-1, 3)
    offsets = tangent_child_offsets(x, np.ones(len(x), bool), sigma0=0.04,
                                    count=4, offset_scale=0.35, k=8)
    np.testing.assert_allclose(offsets.mean(1), 0.0, atol=1e-8)
    np.testing.assert_allclose(offsets[..., 2], 0.0, atol=1e-7)
    assert offsets.shape == (len(x), 4, 3)


def test_child_centers_follow_x_plus_F_delta_and_aggregate_gradients():
    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]], requires_grad=True)
    F = torch.stack([torch.diag(torch.tensor([2.0, 3.0, 4.0])),
                     torch.tensor([[1.0, 0.2, 0.0], [0.0, 0.8, 0.1],
                                   [0.0, 0.0, 1.2]])]).requires_grad_()
    offsets = torch.tensor([[[0.1, 0.2, 0.0], [-0.1, -0.2, 0.0]],
                            [[0.0, 0.2, 0.1], [0.0, -0.2, -0.1]]])
    xc, Fc = expand_children_torch(x, F, offsets)
    expected = torch.stack([x[i] + F[i] @ offsets[i, c]
                            for i in range(2) for c in range(2)])
    assert torch.allclose(xc, expected)
    assert Fc.shape == (4, 3, 3)
    weights = torch.arange(1, 13, dtype=x.dtype).reshape(4, 3)
    loss = (xc * weights).sum()
    loss.backward()
    parent_weights = weights.reshape(2, 2, 3)
    expected_x_grad = parent_weights.sum(1)
    expected_F_grad = torch.einsum("nci,ncj->nij", parent_weights, offsets)
    assert torch.allclose(x.grad, expected_x_grad)
    assert torch.allclose(F.grad, expected_F_grad)


def test_mask_is_applied_before_child_expansion_and_gradients_return_to_parents(monkeypatch):
    calls = []

    class Rasterizer:
        def __init__(self, raster_settings):
            pass

        def __call__(self, **kw):
            calls.append(kw)
            value = kw["means3D"].square().sum() + kw["cov3D_precomp"].sum()
            return value.expand(3, 1, 1)

    monkeypatch.setattr(gauss_loss, "_gs", lambda: (
        types.SimpleNamespace(GaussianRasterizer=Rasterizer), False))
    bundle = gauss_loss.GaussViews.__new__(gauss_loss.GaussViews)
    bundle.sigma0 = 0.04
    bundle.child_sigma_scale = 0.55
    x = torch.randn(4, 3, requires_grad=True)
    F = torch.eye(3).repeat(4, 1, 1).requires_grad_()
    offsets = torch.zeros(4, 4, 3)
    offsets[:, :, :2] = torch.tensor([[0.01, 0.01], [0.01, -0.01],
                                      [-0.01, 0.01], [-0.01, -0.01]])
    mask = torch.tensor([True, False, True, False])
    bundle._render(x, object(), F, mask, offsets).sum().backward()
    assert calls[0]["means3D"].shape == (int(mask.sum()) * 4, 3)
    assert float(x.grad[~mask].abs().sum()) == 0.0
    assert float(F.grad[~mask].abs().sum()) == 0.0
    assert float(x.grad[mask].abs().sum()) > 0.0
    assert float(F.grad[mask].abs().sum()) > 0.0


def test_target_bake_uses_the_same_masked_child_representation(monkeypatch):
    calls = []

    class Rasterizer:
        def __init__(self, raster_settings):
            pass

        def __call__(self, **kw):
            calls.append(kw)
            return kw["means3D"].sum().expand(3, 1, 1)

    monkeypatch.setattr(gauss_loss, "_gs", lambda: (
        types.SimpleNamespace(GaussianRasterizer=Rasterizer), False))
    bundle = gauss_loss.GaussViews.__new__(gauss_loss.GaussViews)
    bundle.sigma0 = 0.04
    bundle.child_count = 4
    bundle.child_sigma_scale = 0.55
    bundle.child_offset_scale = 0.35
    bundle.child_k = 3
    bundle.dev = "cpu"
    bundle.cams = [object()]
    bundle.targets = None
    bundle.target_offsets = None
    target = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                           [0.0, 0.1, 0.0], [0.1, 0.1, 0.0],
                           [0.2, 0.1, 0.0]])
    mask = torch.tensor([True, True, True, True, False])
    bundle.bake_targets(target, mask=mask)

    assert bundle.target_offsets.shape == (5, 4, 3)
    assert calls[0]["means3D"].shape == (int(mask.sum()) * 4, 3)
    assert torch.allclose(calls[0]["scales"], torch.full((16, 3), 0.04 * 0.55))
    assert bundle.targets[0].requires_grad is False
