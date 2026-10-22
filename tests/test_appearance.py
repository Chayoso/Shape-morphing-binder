"""D160, the render routes material by its colour (pipeline/appearance.py, render_loss.shaded_discs with albedo,
render/exterior.disc_albedo), CPU:
(1) the reduction: one channel of ones against a white target is the shading term without colour, bit for bit, in
    value and in gradient (float32 and float64), and the discs read the same points and normals;
(2) three equal channels give the grey term to rounding;
(3) a disc's albedo is its particles' under its centre's kernel weights, and a loss on it reaches the particles with
    the gradient of central finite differences;
(4) with the body's geometry exactly the target's and only the paint shifted, the silhouette and the grey shading
    vanish and the colour term does not; a small step down its gradient lowers it, and the material at the colour
    front moves toward where its colour is wanted;
(5) the null space: every physics term (the transport with its surface proximity, the end drift, the released motion,
    the spray cleanup, the near band) is a function of the set of particles, the same to rounding under any
    permutation of them, and it never reads an albedo; the colour term is not invariant under a permutation of the
    albedos;
(6) the scenes: the volumes painted are the quantiles they name, and the target paint is one function of position
    for every draw.
The pipeline-level checks (CUDA, deterministic) are in test_appearance_pipeline.py."""
import numpy as np
import pytest
import torch

from physmorph.losses.silhouette import set_kernel
from physmorph.pipeline.render_loss import d_exterior, exterior_targets, make_views, shaded_discs
from physmorph.render.exterior import Lattice, Tracked, ZhuBridson, disc_albedo, found_disc_albedo


def _ball(radius=6., seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    r = torch.arange(-int(radius) - 1, int(radius) + 2, dtype=dtype)
    x = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    x = x + .2 * (2 * torch.rand(x.shape, generator=gen, dtype=dtype) - 1)
    return x[x.norm(dim=1) < radius]


def _tracked(dtype=torch.float64):
    x = _ball(dtype=dtype)
    return x, Tracked(ZhuBridson(x, 1.), Lattice(torch.zeros(3, dtype=x.dtype), 30.), .8, 1.)


RED = torch.tensor([.9, .25, .2])


def _paint(x, cut=0.):
    """Two albedos, red where x_0 > cut."""
    red = (x[:, 0] > cut)[:, None]
    return torch.where(red, RED.to(x.dtype), torch.ones(3, dtype=x.dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_one_channel_of_ones_is_the_shading_term_without_colour_bit_for_bit(dtype):
    set_kernel("cic")
    x, tr = _tracked(dtype)
    views, res, extent = make_views(4, (0., .5)), 32, 10.
    ones, ref = torch.ones(len(x), 1, dtype=dtype), torch.ones(1, dtype=dtype)
    p, n, move = tr.read(x)
    p1, n1, move1, c = tr.read(x, ones, ref)
    assert torch.equal(p, p1) and torch.equal(n, n1) and torch.equal(move, move1)
    assert torch.equal(c, torch.ones_like(c))
    sils, shade = exterior_targets(p, n, views, res, extent)
    sils1, shade1 = exterior_targets(p, n, views, res, extent, albedo=torch.ones(len(p), 1, dtype=dtype))
    assert all(torch.equal(a, b) for a, b in zip(sils, sils1))
    assert all(torch.equal(a, b[0]) for a, b in zip(shade, shade1))
    s, al = shaded_discs(p, n, views, res, extent)
    s1, al1 = shaded_discs(p, n, views, res, extent, albedo=torch.ones(len(p), 1, dtype=dtype))
    assert torch.equal(s, s1[0]) and torch.equal(al, al1)
    shift = torch.tensor([.6, -.3, .2], dtype=dtype)
    xa = (x + shift).requires_grad_(True)
    xb = (x + shift).requires_grad_(True)
    pa, na, _ = tr.read(xa)
    sil_a, pbr_a = d_exterior(pa, na, sils, shade, views, res, extent)
    pb, nb, _, cb = tr.read(xb, ones, ref)
    sil_b, pbr_b = d_exterior(pb, nb, sils1, shade1, views, res, extent, albedo=cb)
    assert float(pbr_a) > 0 and torch.equal(sil_a, sil_b) and torch.equal(pbr_a, pbr_b)
    ga, = torch.autograd.grad(sil_a + pbr_a, xa)
    gb, = torch.autograd.grad(sil_b + pbr_b, xb)
    assert float(ga.abs().max()) > 0 and torch.equal(ga, gb)


def test_three_equal_channels_are_the_grey_term_to_rounding():
    set_kernel("cic")
    x, tr = _tracked()
    views, res, extent = make_views(4, (0., .5)), 32, 10.
    p, n, _ = tr.read(x)
    sils, shade = exterior_targets(p, n, views, res, extent)
    sils3, shade3 = exterior_targets(p, n, views, res, extent, albedo=torch.ones(len(p), 3, dtype=x.dtype))
    xa = (x + torch.tensor([.6, -.3, .2], dtype=x.dtype)).requires_grad_(True)
    pa, na, _ = tr.read(xa)
    _, pbr = d_exterior(pa, na, sils, shade, views, res, extent)
    pb, nb, _, c = tr.read(xa, torch.ones(len(x), 3, dtype=x.dtype), torch.ones(3, dtype=x.dtype))
    _, pbr3 = d_exterior(pb, nb, sils3, shade3, views, res, extent, albedo=c)
    assert float(pbr) > 0
    torch.testing.assert_close(pbr3, pbr, rtol=1e-12, atol=0.)
    g, = torch.autograd.grad(pbr, xa, retain_graph=True)
    g3, = torch.autograd.grad(pbr3, xa)
    torch.testing.assert_close(g3, g, rtol=1e-9, atol=1e-14)


def test_the_disc_albedo_is_its_particles_under_the_centre_weights_and_reaches_them():
    x, tr = _tracked()
    a = _paint(x)
    ref = a.mean(0)
    c = tr.read(x, a, ref)[3]
    valid = tr.idx >= 0
    q = x[tr.idx.clamp_min(0)]
    w = (1. - (tr.p0[:, None] - q).square().sum(-1) / tr.radius ** 2).clamp_min(0.) ** 3 * valid
    direct = (w[..., None] * a[tr.idx.clamp_min(0)]).sum(1) / w.sum(1)[:, None]
    torch.testing.assert_close(c, direct, rtol=1e-12, atol=1e-12)
    assert float(c[:, 1].min()) < .3 and float(c[:, 1].max()) == pytest.approx(1.)      # both colours on the surface
    found = found_disc_albedo(ZhuBridson(x, 1.), tr.p0, a, ref, 1.)
    torch.testing.assert_close(found, c, rtol=1e-12, atol=1e-12)                         # the target's discs alike
    wt = torch.linspace(.5, 1.5, 3, dtype=x.dtype)
    loss = lambda y: ((tr.read(y, a, ref)[3] * wt).sum(1) ** 2).sum()  # noqa: E731
    xg = x.clone().requires_grad_(True)
    g, = torch.autograd.grad(loss(xg), xg)
    assert float(g.abs().sum()) > 0
    v = torch.randn(x.shape, generator=torch.Generator().manual_seed(2), dtype=x.dtype)
    eps = 1e-5
    fd = float(loss(x + eps * v) - loss(x - eps * v)) / (2 * eps)
    assert abs(fd - float((g * v).sum())) < 1e-5 * max(abs(fd), 1.)


def test_a_one_colour_body_carries_exactly_no_colour_gradient():
    x, tr = _tracked()
    a = torch.full((len(x), 3), .7, dtype=x.dtype)
    xg = x.clone().requires_grad_(True)
    c = tr.read(xg, a, a.mean(0))[3]
    assert torch.equal(c, torch.full_like(c, .7))
    g, = torch.autograd.grad((c * torch.arange(1., 4., dtype=x.dtype)).sum(), xg)
    assert float(g.abs().max()) == 0.


def test_only_the_paint_differs_and_the_colour_term_moves_the_front_toward_its_colour():
    """The body is the target's geometry, its paint red for x_0 > 0; the target's red for x_0 > -1.5 (one and a half
    pitches further). The silhouette and the grey shading are zero; the colour term is not; one small step down its
    gradient lowers it, and the descent moves the material at the front (|x_0| < 2.5, outer two pitches) toward
    -x_0, where red is wanted."""
    set_kernel("cic")
    x, tr = _tracked()
    views, res, extent = make_views(6, (0., .5, -.5)), 48, 8.
    a_body, a_tgt = _paint(x, 0.), _paint(x, -1.5)
    ref = a_body.mean(0)
    p, n, _, _ = tr.read(x, a_body, ref)
    c_t = found_disc_albedo(ZhuBridson(x, 1.), tr.p0, a_tgt, ref, 1.)
    sils, shade = exterior_targets(p, n, views, res, extent, albedo=c_t)
    _, grey = exterior_targets(p, n, views, res, extent)
    xg = x.clone().requires_grad_(True)
    pg, ng, _, cg = tr.read(xg, a_body, ref)
    sil, col = d_exterior(pg, ng, sils, shade, views, res, extent, albedo=cg)
    assert float(sil) < 1e-12 and float(d_exterior(pg, ng, sils, grey, views, res, extent)[1]) < 1e-12
    assert float(col) > 1e-4
    g, = torch.autograd.grad(col, xg)
    step = 1e-2 / float(g.abs().max())
    p2, n2, _, c2 = tr.read(x - step * g, a_body, ref)
    assert float(d_exterior(p2, n2, sils, shade, views, res, extent, albedo=c2)[1]) < float(col)
    front = (x[:, 0].abs() < 2.5) & (x.norm(dim=1) > 4.)
    assert int(front.sum()) > 30 and float(-g[front, 0].sum()) < 0.


def _phys_terms(x, v, V, perm=None):
    """The physics terms of the window objective on a small state (the frozen definitions, losses/ and window/):
    the transport divergence with its surface proximity, the end drift, the released motion, the spray cleanup and the
    near band read at the current state."""
    from physmorph import gpu
    from physmorph.losses.grid_ot import GridSinkhornLoss
    from physmorph.losses.support import SurfaceProximity
    from physmorph.losses.volumetric import d_nn_band_current, d_w1, rasterize_mass, target_dt_grid
    from physmorph.pipeline.window.objective import end_drift, released_motion
    rng = np.random.default_rng(5)
    tgt = torch.tensor(rng.normal(0, .5, (len(x), 3)) * np.array([1.2, .8, 1.]) + 2.).clamp(.6, 3.4)
    m = torch.ones(len(x), dtype=x.dtype)
    origin, dims, dx = torch.zeros(3, dtype=x.dtype), (8, 8, 8), .5
    grid = rasterize_mass(tgt, m, origin, dx, dims)
    ot = GridSinkhornLoss(grid, origin, dx, dims, eps=1., iters=4000, tol=1e-9,
                          support=SurfaceProximity(tgt.numpy()))
    dt3 = target_dt_grid(rasterize_mass(tgt, m, origin, .25, (16,) * 3), .25, (16,) * 3)
    if perm is not None:
        x, v, V = x[perm], v[perm], V[:, perm]
    return dict(transport=ot.state_energy(x, m), drift=end_drift(v, .08), released=released_motion(V, 3, .08),
                spray=d_w1(x, m, dt3, origin, .25, (16,) * 3),
                near=d_nn_band_current(x, m, tgt, torch.ones_like(m), .05, gpu.KNN(tgt), far=.5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the device KD-tree (gpu.KNN) needs CUDA")
def test_the_physics_terms_are_functions_of_the_set_and_the_colour_term_is_not():
    rng = np.random.default_rng(3)
    x = torch.tensor(rng.normal(0, .55, (300, 3)) + 2.).clamp(.6, 3.4)
    v = torch.tensor(rng.normal(0, .1, (300, 3)))
    V = torch.tensor(rng.normal(0, .1, (6, 300, 3)))
    perm = torch.randperm(300, generator=torch.Generator().manual_seed(4))
    base, swapped = _phys_terms(x, v, V), _phys_terms(x, v, V, perm)
    for k in base:
        assert bool(torch.isfinite(base[k])) and float(base[k].abs()) > 0, k
        torch.testing.assert_close(swapped[k], base[k], rtol=1e-9, atol=1e-12, msg=k)
    # the colour term reads which particle carries which albedo: a permutation of the albedos changes it
    set_kernel("cic")
    xb, tr = _tracked()
    views, res, extent = make_views(4, (0., .5)), 32, 10.
    a = _paint(xb)
    pa = torch.randperm(len(xb), generator=torch.Generator().manual_seed(6))
    p, n, _, c = tr.read(xb, a, a.mean(0))
    sils, shade = exterior_targets(p, n, views, res, extent, albedo=found_disc_albedo(ZhuBridson(xb, 1.), tr.p0, a,
                                                                                         a.mean(0), 1.))
    same = d_exterior(p, n, sils, shade, views, res, extent, albedo=c)[1]
    c2 = tr.read(xb, a[pa], a.mean(0))[3]
    other = d_exterior(p, n, sils, shade, views, res, extent, albedo=c2)[1]
    assert float(same) < 1e-12 and float(other) > 1e-4


def test_the_physics_never_reads_an_albedo():
    """No physics module takes an albedo: the objective's physics terms are built without one (window/objective.py's
    phys_core and cleanup read xT, vT, V, the target pack and the gates; the albedos live on the render's Exterior)."""
    import inspect
    from physmorph.pipeline.window import objective
    src = inspect.getsource(objective.Objective)
    phys = src[src.index("def phys_core"):src.index("def near_band_far")]
    assert "albedo" not in phys and "albedo" not in inspect.getsource(objective.Objective.losses).replace(
        "render_terms", "")
    import physmorph.losses.grid_ot as go
    import physmorph.losses.support as su
    import physmorph.losses.volumetric as vo
    assert all("albedo" not in inspect.getsource(m) for m in (go, su, vo))


def test_the_scenes_paint_the_quantiles_they_name_and_one_target_field():
    from physmorph.pipeline.appearance import RED, WHITE, make_appearance, target_red
    rng = np.random.default_rng(7)
    src = rng.normal(0, 1, (4000, 3)).astype(np.float32)
    tgt = (rng.normal(0, 1, (4000, 3)) * np.array([1., 1.5, .8])).astype(np.float32)
    draw = (rng.normal(0, 1, (4000, 3)) * np.array([1., 1.5, .8])).astype(np.float32)
    for scene, s_share, t_share in (("half", .5, .5), ("band", .5, .35)):
        app = make_appearance(scene, src, tgt, [draw])
        red_s = (app.src == np.asarray(RED, np.float32)).all(1)
        red_t = (app.tgt == np.asarray(RED, np.float32)).all(1)
        assert ((app.src == np.asarray(WHITE, np.float32)).all(1) | red_s).all()
        assert abs(float(red_s.mean()) - s_share) < 2e-3 and abs(float(red_t.mean()) - t_share) < 2e-3
        assert np.array_equal(app.draws[0] == np.asarray(RED, np.float32), np.repeat(
            target_red(app.params, draw)[:, None], 3, 1))
        assert len(app.target_albedos()) == 2
    white = make_appearance("white", src, tgt, [draw])
    assert white.src.shape == (4000, 1) and (white.src == 1).all() and (white.draws[0] == 1).all()
    with pytest.raises(ValueError, match="unknown appearance"):
        make_appearance("plaid", src, tgt)


def test_the_bunny_scenes_paint_the_ears_and_the_head_side():
    """ears: the target red above the bunny's ears' base, the source's top 5 %; turn: the target red on the head's side
    of its median plane (half the target sample), the source's top half."""
    from physmorph.pipeline.appearance import BUNNY_EAR_BASE, BUNNY_HEAD, make_appearance, source_red, target_red
    rng = np.random.default_rng(8)
    src = rng.normal(0, 1, (5000, 3)).astype(np.float32)
    tgt = (rng.normal(0, 1, (5000, 3)) * np.array([1., 1.6, 1.])).astype(np.float32)
    ears = make_appearance("ears", src, tgt)
    assert abs(float(source_red(ears.params, src).mean()) - .05) < 2e-3
    assert np.array_equal(target_red(ears.params, tgt), tgt[:, 1] >= BUNNY_EAR_BASE)
    turn = make_appearance("turn", src, tgt)
    assert abs(float(source_red(turn.params, src).mean()) - .5) < 2e-3
    assert abs(float(target_red(turn.params, tgt).mean()) - .5) < 2e-3
    h = np.asarray(BUNNY_HEAD) / np.linalg.norm(BUNNY_HEAD)
    far = tgt[np.argmax(tgt @ h)][None]
    assert bool(target_red(turn.params, far)[0]) and not bool(target_red(turn.params, -far)[0])


def test_d160d_colour_term_is_the_shading_term_of_the_colour_pictures():
    """D160d's colour term alone (render_loss.d_colour) is d_exterior's shading term with albedos, bit for bit, and with
    the points and normals held its gradient reaches the particles through the discs' albedos alone."""
    from physmorph.pipeline.render_loss import d_colour
    set_kernel("cic")
    x, tr = _tracked()
    views, res, extent = make_views(4, (0., .5)), 32, 10.
    a_body, a_tgt = _paint(x, 0.), _paint(x, -1.5)
    ref = a_body.mean(0)
    p, n, _, _ = tr.read(x, a_body, ref)
    c_t = found_disc_albedo(ZhuBridson(x, 1.), tr.p0, a_tgt, ref, 1.)
    sils, shade = exterior_targets(p, n, views, res, extent, albedo=c_t)
    xg = x.clone().requires_grad_(True)
    pg, ng, _, cg = tr.read(xg, a_body, ref)
    _, col = d_exterior(pg, ng, sils, shade, views, res, extent, albedo=cg)
    assert torch.equal(d_colour(pg, ng, shade, views, res, extent, 1.5, .25, cg), col)
    held = d_colour(pg.detach(), ng.detach(), shade, views, res, extent, 1.5, .25, cg)
    g_held, = torch.autograd.grad(held, xg, retain_graph=True)
    xr = x.clone().requires_grad_(True)
    c_only = tr.read(xr, a_body, ref)[3]
    g_route, = torch.autograd.grad(d_colour(p.detach(), n.detach(), shade, views, res, extent, 1.5, .25, c_only), xr)
    torch.testing.assert_close(g_held, g_route, rtol=1e-10, atol=1e-14)
    assert float(g_held.abs().max()) > 0
