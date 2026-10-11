"""D144: Zhu and Bridson's field with Solenthaler et al.'s correction (render/exterior.ZhuBridsonSolenthaler,
`--exterior_field zbs`).

(1) The largest real eigenvalue in closed form against torch.linalg.eigvals; the factor f against its formula.
(2) Identical to Zhu and Bridson's on a convex blob: where the centre moves no faster than the query point the field
    and the discs are Zhu and Bridson's.
(3) Two parallel slabs 5 pitches apart (the mouth's gap: EVmax about 18 at its middle): Zhu and Bridson's draws a
    blob in the gap, the corrected field draws nothing there; slabs 1.5 pitches apart (their offset surfaces overlap)
    stay bridged, as Zhu and Bridson's. Between, a blob shrinks with f (EVmax 1.7 at 2 pitches, 2.3 at 2.5, 3 at 3,
    4 at 3.5: gone from 3.5 pitches on, D144's record).
(4) The gradient (Zhu and Bridson's less offset f' grad EVmax) against central differences of the field.
(5) The Warp kernel (CUDA) against the tensor form: values, gradients, EVmax, the same discs.
(6) Tracked discs read where they were found, and their read's gradient (float64 gradcheck).
(7) The switch: zb stays the default; zbs reaches the target's pictures and the render terms of a small run.
"""
import numpy as np
import pytest
import torch

from physmorph.render.exterior import (Lattice, TrackedSolenthaler, ZhuBridson, ZhuBridsonSolenthaler, ev_max,
                                       make_field, sol_factor, tracked)


def _ball(radius=5., seed=0, jitter=.1, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    r = torch.arange(-int(radius) - 1, int(radius) + 2, dtype=dtype)
    x = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    x = x + jitter * (2 * torch.rand(x.shape, generator=gen, dtype=dtype) - 1)
    return x[x.norm(dim=1) < radius]


def _sheets(gap=2.5, n=13, layers=2, seed=1, dtype=torch.float64):
    """Two slabs of `layers` particle layers, one pitch apart, facing each other across `gap` pitches."""
    gen = torch.Generator().manual_seed(seed)
    r = torch.arange(n, dtype=dtype) - (n - 1) / 2
    g = torch.stack(torch.meshgrid(r, r, indexing="ij"), -1).reshape(-1, 2)
    zs = [-(k + gap / 2) for k in range(layers)] + [k + gap / 2 for k in range(layers)]
    x = torch.cat([torch.cat([g, torch.full((len(g), 1), z, dtype=dtype)], 1) for z in zs])
    return x + .05 * (2 * torch.rand(x.shape, generator=gen, dtype=dtype) - 1)


def test_the_largest_eigenvalue_and_the_factor():
    gen = torch.Generator().manual_seed(2)
    J = torch.randn(20000, 3, 3, generator=gen, dtype=torch.float64)
    J = torch.cat([J, torch.eye(3, dtype=torch.float64).expand(4, 3, 3), torch.diag_embed(torch.tensor(
        [[1., 1., .3], [2., 1., 1.], [.5, .5, .5]], dtype=torch.float64))])
    torch.testing.assert_close(ev_max(J), torch.linalg.eigvals(J).real.max(-1).values, rtol=1e-6, atol=1e-6)
    ev = torch.tensor([.5, 1., 1.5, 2.25, 3., 3.5, 9.], dtype=torch.float64)
    f, df = sol_factor(ev, 1., 3.5)
    g = (3.5 - ev) / 2.5
    want = torch.where(ev <= 1., torch.ones_like(ev), (g ** 3 - 3 * g ** 2 + 3 * g).clamp(0., 1.))
    torch.testing.assert_close(f, want)
    assert float(f[0]) == 1. and float(f[-1]) == 0. and float(f[-2]) == 0.
    eps = 1e-6
    fd = (sol_factor(ev + eps, 1., 3.5)[0] - sol_factor(ev - eps, 1., 3.5)[0]) / (2 * eps)
    torch.testing.assert_close(df[2:5], fd[2:5], rtol=1e-5, atol=1e-8)


def test_identical_to_zhu_bridson_on_a_convex_blob():
    x = _ball()
    zb, zs = ZhuBridson(x, 1.), ZhuBridsonSolenthaler(x, 1.)
    lat = Lattice(torch.zeros(3, dtype=x.dtype), 30.)
    p0, _, _, _ = lat.discs(zb, .4, refine=False)
    ev = zs.ev_grad(p0)[0]
    f = sol_factor(ev)[0]
    assert float((f == 1.).double().mean()) > .5 and float(f.min()) > .99      # convex: the centre lags the point
    with torch.no_grad():
        dphi = (zs(p0, grad=False)[0] - zb(p0, grad=False)[0]).abs()
    assert float(dphi.max()) < .01                                           # a hundredth of a pitch at most
    p1, _, _, _ = lat.discs(zs, .4, refine=False)
    assert abs(len(p1) - len(p0)) <= 2
    d = torch.cdist(p1, p0).min(1).values
    assert float(d.max()) < .02


def test_no_blob_between_two_slabs_at_the_mouths_gap():
    x = _sheets(gap=5.)
    zb, zs = ZhuBridson(x, 1.), ZhuBridsonSolenthaler(x, 1.)
    g = torch.stack(torch.meshgrid(torch.linspace(-3, 3, 13, dtype=x.dtype), torch.linspace(-3, 3, 13, dtype=x.dtype),
                                   indexing="ij"), -1).reshape(-1, 2)
    mid = torch.cat([g, torch.zeros(len(g), 1, dtype=x.dtype)], 1)
    with torch.no_grad():
        assert bool((zb(mid, grad=False)[0] < 0).all())                      # the centroid falls in the gap: a blob
        assert bool((zs(mid, grad=False)[0] > 0).all())                      # corrected: nothing drawn there
    p1, _, _, _ = Lattice(torch.zeros(3, dtype=x.dtype), 30.).discs(zs, .3, refine=False)
    inner = p1[(p1[:, :2].abs() < 3).all(1)]
    assert len(inner) > 200 and not bool((inner[:, 2].abs() < 1.).any())
    # the slabs' outer faces are Zhu and Bridson's
    outer = torch.cat([g, torch.full((len(g), 1), 3.5 + .8, dtype=x.dtype)], 1)
    with torch.no_grad():
        assert float((zs(outer, grad=False)[0] - zb(outer, grad=False)[0]).abs().max()) < .01


def test_the_gradient_is_the_fields_by_central_differences():
    x = _sheets(gap=3.)
    zs = ZhuBridsonSolenthaler(x, 1.)
    gen = torch.Generator().manual_seed(5)
    q = (torch.rand(400, 3, generator=gen, dtype=x.dtype) - .5) * torch.tensor([6., 6., 5.], dtype=x.dtype)
    f, g, _ = zs(q)
    ev = zs.ev_grad(q)[0]
    vary = (ev > zs.t_low) & (ev < zs.t_high) & torch.isfinite(f)
    assert int(vary.sum()) > 20                                              # points where f varies are tested
    eps = 1e-5
    fd = torch.stack([(zs(q + eps * e, grad=False)[0] - zs(q - eps * e, grad=False)[0]) / (2 * eps)
                      for e in torch.eye(3, dtype=q.dtype)], 1)
    ok = torch.isfinite(f)
    assert float((fd[ok] - g[ok]).abs().max()) < 1e-3 * max(1., float(g[ok].abs().max()))


def test_device_field_is_the_tensor_field_to_rounding():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    x = torch.cat([_ball(5., dtype=torch.float32), _sheets(dtype=torch.float32) + torch.tensor([0., 0., 9.])]).cuda()
    ten, dev = ZhuBridsonSolenthaler(x, 1.), ZhuBridsonSolenthaler(x, 1., device_field=True)
    lat = Lattice(torch.zeros(3, device="cuda"), 30.)
    q = lat.at(torch.randint(55, 105, (20000, 3), device="cuda"), .4)
    ft, gt, st = ten(q)
    fd, gd, sd = dev(q)
    fin = torch.isfinite(ft)
    assert torch.equal(fin, torch.isfinite(fd)) and int(fin.sum()) > 1000
    assert float((ft[fin] - fd[fin]).abs().max()) < 1e-3
    assert float((gt[fin] - gd[fin]).abs().max()) < 2e-2
    et, ed = ten.ev_grad(q)[0], dev.ev_grad(q)[0]
    assert float((et[fin] - ed[fin]).abs().max()) < 1e-2      # float32 near a double root: the root's sqrt sensitivity
    pt, _, ct, _ = lat.discs(ten, .4, refine=False)
    pd, _, cd, _ = lat.discs(dev, .4, refine=False)
    assert abs(ct - cd) <= 3 and abs(len(pt) - len(pd)) <= 3 and len(pt) > 500
    d = torch.cdist(pd, pt, compute_mode="donot_use_mm_for_euclid_dist").min(1).values
    assert float(d.max()) < 2e-3


def test_tracked_discs_read_where_found_and_their_gradient_checks():
    x = torch.cat([_ball(3.2), _sheets(gap=2.5, n=7, layers=1) + torch.tensor([0., 0., 7.], dtype=torch.float64)])
    lat = Lattice(torch.zeros(3, dtype=x.dtype), 30.)
    tr = tracked(ZhuBridsonSolenthaler(x, 1.), lat, .5, 1.)
    assert isinstance(tr, TrackedSolenthaler) and len(tr.p0) > 100
    p, n, move = tr.read(x)
    assert float(move.abs().max()) < .05
    assert float((n - tr.n0).abs().max()) < 1e-3
    gen = torch.Generator().manual_seed(7)
    wp_, wn = torch.randn(p.shape, dtype=x.dtype, generator=gen), torch.randn(p.shape, dtype=x.dtype, generator=gen)

    def loss(xx):
        pp, nn, _ = tr.read(xx)
        return (wp_ * pp).sum() + (wn * nn).sum()

    assert torch.autograd.gradcheck(loss, (x.clone().requires_grad_(),), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_the_switch():
    from physmorph.pipeline.config import PipelineConfig
    assert PipelineConfig().exterior_field == "zb"
    assert PipelineConfig(exterior_field="zbs").exterior_field == "zbs"
    x = _ball()
    assert type(make_field("zbs", x, 1.)) is ZhuBridsonSolenthaler and type(make_field("zb", x, 1.)) is ZhuBridson


def test_a_small_run_reads_the_corrected_field():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import physmorph.pipeline.window.objective as O
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig, run_pipeline
    seen = []
    orig = O.tracked

    def spy(field, *a):
        seen.append(type(field).__name__)
        return orig(field, *a)

    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=4, iters=2, animations=2, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=3, render_exterior=True, exterior_field="zbs")
    O.tracked = spy
    try:
        run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    finally:
        O.tracked = orig
    assert seen and set(seen) == {"ZhuBridsonSolenthaler"}


def test_touching_slabs_stay_bridged():
    """Slabs 1.5 pitches apart: their surfaces, 0.8 pitches off each, overlap, so the material is drawn as one; the
    centre does not outrun the point (EVmax about 1) and the corrected field keeps Zhu and Bridson's bridge."""
    x = _sheets(gap=1.5)
    zb, zs = ZhuBridson(x, 1.), ZhuBridsonSolenthaler(x, 1.)
    g = torch.stack(torch.meshgrid(torch.linspace(-3, 3, 13, dtype=x.dtype), torch.linspace(-3, 3, 13, dtype=x.dtype),
                                   indexing="ij"), -1).reshape(-1, 2)
    mid = torch.cat([g, torch.zeros(len(g), 1, dtype=x.dtype)], 1)
    with torch.no_grad():
        a, b = zb(mid, grad=False)[0], zs(mid, grad=False)[0]
    assert bool((a < 0).all()) and bool((b < 0).all()) and float((a - b).abs().max()) < .05
