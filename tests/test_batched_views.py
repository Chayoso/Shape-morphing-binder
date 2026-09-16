"""Batched multi-view rasterisation (2026-09-16 speed) equals the per-view functions."""
import numpy as np
import torch

from physmorph.losses.silhouette import soft_silhouette, soft_silhouette_multi
from physmorph.pipeline.render_loss import (d_pbr, d_render, field_normals, make_views,
                                            shaded_view, shaded_views_multi, target_silhouettes)


def test_multi_view_silhouette_and_shading_match_per_view():
    torch.manual_seed(0)
    x = torch.rand(3000, 3) * 2 - 1
    views = make_views(4, (0.0, 0.5))
    res, ext = 24, 1.4
    multi = soft_silhouette_multi(x, views, res, ext, 1.5)
    for v, (th, ph) in enumerate(views):
        single = soft_silhouette(x, th, res, ext, 1.5, ph)
        assert torch.allclose(multi[v], single, atol=1e-5)
    gmin = torch.tensor([-1.5, -1.5, -1.5]); dims = (12, 12, 12); dx = 3.0 / 12
    n_p, sw = field_normals(x, gmin, dx, dims)
    sm, am = shaded_views_multi(x, views, res, ext, n_p, sw)
    for v, (th, ph) in enumerate(views):
        s1, a1 = shaded_view(x, th, ph, res, ext, n_p, sw)
        assert torch.allclose(sm[v], s1, atol=1e-5) and torch.allclose(am[v], a1, atol=1e-5)


def test_batched_losses_are_differentiable_and_match_reference():
    torch.manual_seed(1)
    x = (torch.rand(2000, 3) * 2 - 1).requires_grad_(True)
    t = torch.rand(2000, 3) * 2 - 1
    views = make_views(3, (0.0,))
    res, ext = 20, 1.3
    sils = target_silhouettes(t, [v[0] for v in views], res, ext)
    L = d_render(x, sils, views, res, ext)
    ref = sum(((torch.clamp(a_t - soft_silhouette(x, th, res, ext, 1.5, ph), min=0).pow(2) * 2.0
                + torch.clamp(soft_silhouette(x, th, res, ext, 1.5, ph) - a_t, min=0).pow(2)).mean()
               for a_t, (th, ph) in zip(sils, views))) / len(views)
    assert torch.allclose(L, ref, atol=1e-6)
    (g,) = torch.autograd.grad(L, x)
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0
