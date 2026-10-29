"""D178, the masked rule (render_split "masked": the dFc block's held weight on the render residual inside the content
mask, target.content_masks), CPU:
(1) d_exterior_masked's full terms are d_exterior's bit for bit (grey and colour) in value, and its gradient to rounding
    (the silhouette and shading branches are summed in another order);
(2) a mask of ones gives the full terms, a mask of zeros (or None) gives zero, and the masked and unmasked parts add up
    to the full term (the same per-pixel penalties and normalisation);
(3) the configuration refuses the rule off the exterior's sample pictures.
The pipeline-level check (the rule is Hu's bit for bit where the pictures are the 3D target's own) is the V3 gate."""
import pytest
import torch

from physmorph.losses.silhouette import set_kernel
from physmorph.pipeline import PipelineConfig
from physmorph.pipeline.render_loss import d_exterior, d_exterior_masked, exterior_targets, make_views
from physmorph.render.exterior import Lattice, Tracked, ZhuBridson


def _ball(radius=6., seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    r = torch.arange(-int(radius) - 1, int(radius) + 2, dtype=dtype)
    x = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    x = x + .2 * (2 * torch.rand(x.shape, generator=gen, dtype=dtype) - 1)
    return x[x.norm(dim=1) < radius]


def _setup(colour, dtype=torch.float64):
    set_kernel("cic")
    x = _ball(dtype=dtype)
    tr = Tracked(ZhuBridson(x, 1.), Lattice(torch.zeros(3, dtype=x.dtype), 30.), .8, 1.)
    views, res, extent = make_views(4, (0., .5)), 32, 10.
    p, n, _ = tr.read(x)
    alb = None
    if colour:
        red = (x[:, 0] > 0)[:, None]
        alb = torch.where(red, torch.tensor([.9, .25, .2], dtype=dtype), torch.ones(3, dtype=dtype))
        ref = alb.mean(0)
        _, _, _, c = tr.read(x, alb, ref)
        sils, shade = exterior_targets(p, n, views, res, extent, albedo=c)
    else:
        sils, shade = exterior_targets(p, n, views, res, extent)
    return x, tr, alb, views, res, extent, sils, shade


def _read(tr, x, alb):
    if alb is None:
        p, n, _ = tr.read(x)
        return p, n, None
    p, n, _, c = tr.read(x, alb, alb.mean(0))
    return p, n, c


@pytest.mark.parametrize("colour", [False, True])
def test_the_full_terms_are_d_exterior_bit_for_bit(colour):
    x, tr, alb, views, res, extent, sils, shade = _setup(colour)
    shift = torch.tensor([.6, -.3, .2], dtype=x.dtype)
    xa = (x + shift).requires_grad_(True)
    xb = (x + shift).requires_grad_(True)
    pa, na, ca = _read(tr, xa, alb)
    sil_a, shd_a = d_exterior(pa, na, sils, shade, views, res, extent, albedo=ca)
    pb, nb, cb = _read(tr, xb, alb)
    m_s = torch.ones(len(views), res, res, dtype=x.dtype)
    m_c = torch.ones(3, len(views), res, res, dtype=x.dtype) if colour else m_s
    sil_b, shd_b, sil_m, shd_m = d_exterior_masked(pb, nb, sils, shade, m_s, m_c, views, res, extent, albedo=cb)
    assert float(shd_a) > 0 and torch.equal(sil_a, sil_b) and torch.equal(shd_a, shd_b)
    ga, = torch.autograd.grad(sil_a + shd_a, xa)
    gb, = torch.autograd.grad(sil_b + shd_b, xb, retain_graph=True)
    assert torch.allclose(ga, gb, rtol=1e-12, atol=1e-18)    # the two graphs sum their branches in another order
    assert torch.allclose(sil_m, sil_b, rtol=1e-12, atol=0) and torch.allclose(shd_m, shd_b, rtol=1e-12, atol=0)


@pytest.mark.parametrize("colour", [False, True])
def test_the_masked_and_unmasked_parts_add_up(colour):
    x, tr, alb, views, res, extent, sils, shade = _setup(colour)
    xs = x + torch.tensor([.6, -.3, .2], dtype=x.dtype)
    p, n, c = _read(tr, xs, alb)
    gen = torch.Generator().manual_seed(3)
    m_s = (torch.rand(len(views), res, res, generator=gen) > .5).to(x.dtype)
    m_c = ((torch.rand(3, len(views), res, res, generator=gen) > .5).to(x.dtype) if colour
           else (torch.rand(len(views), res, res, generator=gen) > .5).to(x.dtype))
    sil, shd, sil_1, shd_1 = d_exterior_masked(p, n, sils, shade, m_s, m_c, views, res, extent, albedo=c)
    _, _, sil_0, shd_0 = d_exterior_masked(p, n, sils, shade, 1 - m_s, 1 - m_c, views, res, extent, albedo=c)
    assert torch.allclose(sil_1 + sil_0, sil, rtol=1e-12, atol=0) and torch.allclose(shd_1 + shd_0, shd, rtol=1e-12, atol=0)
    _, _, z_s, z_c = d_exterior_masked(p, n, sils, shade, None, None, views, res, extent, albedo=c)
    assert float(z_s) == 0.0 and float(z_c) == 0.0


def test_the_rule_needs_the_exterior_sample_pictures():
    with pytest.raises(ValueError):
        PipelineConfig(render_split="masked", render_exterior=False)
    with pytest.raises(ValueError):
        PipelineConfig(render_split="masked", render_exterior=True, render_target="mesh_shell")
    PipelineConfig(render_split="masked", render_exterior=True)
