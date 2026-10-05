"""The exterior (render/exterior.py) and the render terms read on it (D62), CPU: (1) the discs of a ball of particles
lie on the field's zero set, a sphere, with radial normals, and the field read without its gradient has the same
values; (2) tracked discs are where they were found, and follow a
translation of the particles along their normals; (3) a loss on the tracked discs reaches the particles and its
gradient matches central finite differences; (4) the render terms vanish on the target's own discs and not on a
shifted body."""
import torch

from physmorph.losses.silhouette import set_kernel
from physmorph.pipeline.render_loss import d_exterior, exterior_targets, make_views
from physmorph.render.exterior import Lattice, Tracked, ZhuBridson


def _ball(radius=6., seed=0, dtype=torch.float64):
    """A ball of particles on a jittered grid of unit pitch."""
    gen = torch.Generator().manual_seed(seed)
    r = torch.arange(-int(radius) - 1, int(radius) + 2, dtype=dtype)
    x = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    x = x + .2 * (2 * torch.rand(x.shape, generator=gen, dtype=dtype) - 1)
    return x[x.norm(dim=1) < radius]


def _tracked():
    x = _ball()
    return x, Tracked(ZhuBridson(x, 1.), Lattice(torch.zeros(3, dtype=x.dtype), 30.), .8, 1.)


def test_discs_lie_on_the_zero_set_with_radial_normals():
    x = _ball()
    field = ZhuBridson(x, 1.)
    pts, g, cells, _ = Lattice(torch.zeros(3, dtype=x.dtype), 30.).discs(field, .4)
    assert len(pts) > 500 and cells >= len(pts)
    assert float(field(pts)[0].abs().max()) < .02
    r = pts.norm(dim=1)
    assert 5. < float(r.mean()) < 7. and float(r.std()) < .35
    assert float((torch.nn.functional.normalize(g, dim=1) * torch.nn.functional.normalize(pts, dim=1)).sum(1).mean()) > .95


def test_the_field_without_its_gradient_has_the_same_values():
    x = _ball()
    field = ZhuBridson(x, 1.)
    q = Lattice(torch.zeros(3, dtype=x.dtype), 30.).at(torch.randint(0, 40, (2000, 3)), .4)
    with torch.no_grad():                                       # as the window's objective calls it
        f, g, s = field(q)
        f2, g2, s2 = field(q, grad=False)
    assert g2 is None and g.shape == q.shape
    assert torch.equal(f, f2) and torch.equal(s, s2)


def test_tracked_discs_stay_where_found_and_follow_a_translation():
    x, tr = _tracked()
    p, n, move = tr.read(x)
    assert float(move.abs().max()) < .03 and float((p - tr.p0).abs().max()) < .03
    assert float((n * tr.n0).sum(1).min()) > .99
    d = torch.tensor([.3, 0., 0.], dtype=x.dtype)
    move = tr.read(x + d)[2]
    assert float((move - tr.n0 @ d).abs().mean()) < .15 * .3            # the surface's displacement along n0, to first order


def test_a_loss_on_the_discs_reaches_the_particles():
    x, tr = _tracked()
    loss = lambda y: (lambda p, n, _: (p[:, 0] * n[:, 1]).sum() + p.square().sum())(*tr.read(y))  # noqa: E731
    xg = x.clone().requires_grad_(True)
    g, = torch.autograd.grad(loss(xg), xg)
    assert bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0.
    v = torch.randn(x.shape, generator=torch.Generator().manual_seed(1), dtype=x.dtype)
    eps = 1e-5
    fd = float(loss(x + eps * v) - loss(x - eps * v)) / (2 * eps)
    assert abs(fd - float((g * v).sum())) < 1e-4 * max(abs(fd), 1.)


def test_the_render_terms_vanish_on_the_target_and_not_on_a_shifted_body():
    set_kernel("cic")
    x, tr = _tracked()
    views, res, extent = make_views(4, (0., .5)), 32, 10.
    p, n, _ = tr.read(x)
    sils, shade = exterior_targets(p, n, views, res, extent)
    sil, pbr = d_exterior(p, n, sils, shade, views, res, extent)
    assert float(sil) < 1e-12 and float(pbr) < 1e-12
    p, n, _ = tr.read(x + torch.tensor([.8, 0., 0.], dtype=x.dtype))
    sil, pbr = d_exterior(p, n, sils, shade, views, res, extent)
    assert float(sil) > 1e-5 and float(pbr) > 1e-6
