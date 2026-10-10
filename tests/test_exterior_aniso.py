"""D140: the anisotropic exterior field (render/exterior.Anisotropic, `--exterior_field aniso`).

(1) An isolated particle's surface sits at 0.8 pitches, as Zhu and Bridson's does, with the field a distance there.
(2) A flat sheet of particles one pitch apart is one continuous thin surface: inside between its particles, of a steady
    thickness (no beads), its discs one connected set.
(3) Two parallel sheets with an empty gap: nothing drawn in the gap, where Zhu and Bridson's centroid falls between
    the sheets and draws a blob (the mouth's beads, D140's inventory).
(4) The tensor form's gradient against central differences; the Warp kernel (CUDA) against the tensor form: values,
    gradients and densities to float rounding, the same discs.
(5) The tracked discs (TrackedAniso): at the state they were found at they read where they were found; a loss on
    them reaches the particles, through the centres and the covariances, as float64 gradcheck finds.
(6) The search grows its nodes where the zero set lies beyond the ring of two cells; without the growth the far
    surface is missed.
(7) The switch: zb, the default, is Zhu and Bridson's field; aniso reaches the target's pictures and the render terms
    of a small run.
"""
import numpy as np
import pytest
import torch

from physmorph.render.exterior import (Anisotropic, Lattice, TrackedAniso, ZhuBridson, connected_sets, make_field,
                                       tracked)


def _sheet(n=15, z=0., seed=0, dtype=torch.float64, zjit=True):
    """A square sheet of particles one pitch apart in the plane at height z, jittered by a tenth of a pitch (zjit: also
    off the plane)."""
    gen = torch.Generator().manual_seed(seed)
    r = torch.arange(n, dtype=dtype) - (n - 1) / 2
    g = torch.stack(torch.meshgrid(r, r, indexing="ij"), -1).reshape(-1, 2)
    x = torch.cat([g, torch.full((len(g), 1), z, dtype=dtype)], 1)
    j = .1 * (2 * torch.rand(x.shape, generator=gen, dtype=dtype) - 1)
    return x + (j if zjit else j * torch.tensor([1., 1., 0.], dtype=dtype))


def _ball(radius=4., seed=0, dtype=torch.float64):
    gen = torch.Generator().manual_seed(seed)
    r = torch.arange(-int(radius) - 1, int(radius) + 2, dtype=dtype)
    x = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    x = x + .2 * (2 * torch.rand(x.shape, generator=gen, dtype=dtype) - 1)
    return x[x.norm(dim=1) < radius]


def _height(field, xy, top=3., n=3001):
    """The outermost zero crossing of the field above the points xy (a vertical scan, then the linear root)."""
    z = torch.linspace(0., top, n, dtype=torch.float64)
    q = torch.cat([xy[:, None, :].expand(-1, n, -1), z[None, :, None].expand(len(xy), -1, -1)], -1).reshape(-1, 3)
    f = field(q, grad=False)[0].reshape(len(xy), n)
    inside = f < 0
    last = n - 1 - torch.flip(inside, [1]).float().argmax(1)
    f0, f1 = f[torch.arange(len(xy)), last], f[torch.arange(len(xy)), (last + 1).clamp(max=n - 1)]
    return z[last] + (z[1] - z[0]) * f0 / (f0 - f1)


def test_isolated_particle_surface_at_0_8_pitches():
    x = torch.tensor([[0., 0., 0.], [40., 0., 0.]], dtype=torch.float64)        # two particles far apart: both alone
    field = Anisotropic(x, 1.)
    dirs = torch.nn.functional.normalize(torch.randn(50, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(1)), dim=1)
    f, g, _ = field(.8 * dirs)
    assert float(f.abs().max()) < 1e-9                           # the surface at 0.8 pitches in every direction
    assert float((g - dirs).abs().max()) < 1e-9                  # the field a distance there: unit outward gradient
    assert float(ZhuBridson(x, 1.)(.8 * dirs)[0].abs().max()) < 1e-9    # as Zhu and Bridson's
    pts, nrm, _, _ = Lattice(torch.zeros(3, dtype=x.dtype), 60.).discs(field, .2)
    near = pts[pts.norm(dim=1) < 5.]
    assert len(near) > 20 and float((near.norm(dim=1) - .8).abs().max()) < .02


def test_flat_sheet_is_one_continuous_thin_surface():
    x = _sheet(zjit=False)                                       # in-plane jitter: the thickness is the field's
    field = Anisotropic(x, 1.)
    g = torch.stack(torch.meshgrid(torch.arange(-5, 5, dtype=torch.float64) + .5,
                                   torch.arange(-5, 5, dtype=torch.float64) + .5, indexing="ij"), -1).reshape(-1, 2)
    mid = torch.cat([g, torch.zeros(len(g), 1, dtype=g.dtype)], 1)
    assert bool((field(mid, grad=False)[0] < 0).all())          # inside between the particles: no hole
    over = x[(x[:, :2].abs() < 5).all(1), :2]
    h_between, h_over = _height(field, g), _height(field, over)
    h = torch.cat([h_between, h_over])
    assert .5 < float(h.min()) and float(h.max()) < 1.0          # a thin sheet, half thickness 0.5-1 pitch
    assert float(h.min()) > .8 * float(h.max())                  # of steady thickness: no beads
    pts, _, _, _ = Lattice(torch.zeros(3, dtype=x.dtype), 30.).discs(field, .3)
    n_sets, _ = connected_sets(pts, .3)
    assert len(pts) > 1000 and n_sets == 1


def test_two_sheets_with_an_empty_gap_draw_nothing_in_the_gap():
    x = torch.cat([_sheet(z=0., seed=1), _sheet(z=2.5, seed=2)])
    g = torch.stack(torch.meshgrid(torch.linspace(-5, 5, 41, dtype=torch.float64),
                                   torch.linspace(-5, 5, 41, dtype=torch.float64), indexing="ij"), -1).reshape(-1, 2)
    mid = torch.cat([g, torch.full((len(g), 1), 1.25, dtype=g.dtype)], 1)
    aniso, zb = Anisotropic(x, 1.), ZhuBridson(x, 1.)
    assert bool((aniso(mid, grad=False)[0] > 0).all())          # nothing in the gap
    assert bool((zb(mid, grad=False)[0] < 0).any())              # where the centroid's field draws a blob
    pts, _, _, _ = Lattice(torch.zeros(3, dtype=x.dtype), 30.).discs(aniso, .3)
    inner = pts[(pts[:, :2].abs() < 5).all(1)]
    assert len(inner) > 500 and not bool(((inner[:, 2] - 1.25).abs() < .25).any())


def test_the_gradient_is_the_fields_by_central_differences():
    x = _ball()
    field = Anisotropic(x, 1.)
    gen = torch.Generator().manual_seed(5)
    q = torch.nn.functional.normalize(torch.randn(300, 3, dtype=torch.float64, generator=gen), dim=1)
    q = q * (3. + 2. * torch.rand(300, 1, dtype=torch.float64, generator=gen))
    f, g, _ = field(q)
    ok = torch.isfinite(f)
    assert int(ok.sum()) > 100
    eps = 1e-6
    fd = torch.stack([(field(q + eps * e, grad=False)[0] - field(q - eps * e, grad=False)[0]) / (2 * eps)
                      for e in torch.eye(3, dtype=q.dtype)], 1)
    assert float((fd[ok] - g[ok]).abs().max()) < 1e-6 * max(1., float(g[ok].abs().max()))


def test_device_field_is_the_tensor_field_to_rounding():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    x = torch.cat([_ball(6., dtype=torch.float32), _sheet(z=8., dtype=torch.float32)]).cuda()
    dev = Anisotropic(x, 1.)
    ten = Anisotropic(x, 1.)
    ten.device_field = False
    assert dev.device_field
    lat = Lattice(torch.zeros(3, device="cuda"), 30.)
    q = lat.at(torch.randint(55, 105, (40000, 3), device="cuda"), .4)
    fd, gd, sd = dev(q)
    ft, gt, st = ten(q)
    fin = torch.isfinite(ft)
    assert torch.equal(fin, torch.isfinite(fd)) and int(fin.sum()) > 2000
    assert float((ft[fin] - fd[fin]).abs().max()) < 1e-4
    assert float((gt[fin] - gd[fin]).abs().max()) < 1e-3
    torch.testing.assert_close(sd, st, rtol=1e-4, atol=1e-6)
    pt, _, ct, _ = lat.discs(ten, .4, refine=False)
    pd, _, cd, _ = lat.discs(dev, .4, refine=False)
    assert abs(ct - cd) <= 2 and abs(len(pt) - len(pd)) <= 2 and len(pt) > 500
    d = torch.cdist(pd, pt, compute_mode="donot_use_mm_for_euclid_dist").min(1).values
    assert float(d.max()) < 1e-3


def test_tracked_discs_read_where_found_and_their_gradient_checks():
    x = _ball(3.2)
    lat = Lattice(torch.zeros(3, dtype=x.dtype), 30.)
    tr = tracked(Anisotropic(x, 1.), lat, .5, 1.)
    assert isinstance(tr, TrackedAniso) and len(tr.p0) > 100
    p, n, move = tr.read(x)
    assert float(move.abs().max()) < .021                        # the projection's tolerance, 0.02 pitches
    assert float((n - tr.n0).abs().max()) < 1e-6
    shift = torch.tensor([.05, 0., 0.], dtype=x.dtype)
    p, _, move = tr.read(x + shift)                               # a small translation: the discs follow it
    assert float((move - (tr.n0 @ shift)).abs().max()) < .02
    gen = torch.Generator().manual_seed(7)
    wp_, wn = torch.randn(p.shape, dtype=x.dtype, generator=gen), torch.randn(p.shape, dtype=x.dtype, generator=gen)

    def loss(xx):
        pp, nn, _ = tr.read(xx)
        return (wp_ * pp).sum() + (wn * nn).sum()

    assert torch.autograd.gradcheck(loss, (x.clone().requires_grad_(),), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_the_search_grows_past_the_ring_of_two_cells():
    """A sheet's surface lies 0.76 pitches off its plane: with a lattice of 0.15 pitches the ring of two cells (0.3
    to 0.45 pitches) falls short, and without the growth no disc is found on its faces."""
    x = _sheet()
    field = Anisotropic(x, 1.)
    lat = Lattice(torch.zeros(3, dtype=x.dtype), 30.)
    pts, _, _, _ = lat.discs(field, .15, refine=False)
    face = pts[(pts[:, :2].abs() < 5).all(1)]
    assert len(face) > 1000 and float(face[:, 2].abs().min()) > .6
    field.grows = False
    pts, _, _, _ = lat.discs(field, .15, refine=False)
    assert int((pts[:, :2].abs() < 5).all(1).sum()) < .1 * len(face)


def test_the_switch():
    from physmorph.pipeline.config import PipelineConfig
    assert PipelineConfig().exterior_field == "zb"
    x = _ball()
    assert type(make_field("zb", x, 1.)) is ZhuBridson and type(make_field("aniso", x, 1.)) is Anisotropic
    with pytest.raises(ValueError):
        PipelineConfig(exterior_field="aniso", exterior_radius=2.5)
    with pytest.raises(ValueError):
        make_field("other", x, 1.)


def test_a_small_run_reads_the_anisotropic_field():
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
                         render_res=24, dt_res=32, patience=3, render_exterior=True, exterior_field="aniso")
    O.tracked = spy
    try:
        run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    finally:
        O.tracked = orig
    assert seen and set(seen) == {"Anisotropic"}


def test_the_largest_eigenvalue_in_closed_form():
    from physmorph.render.exterior import largest_eigenvalue
    gen = torch.Generator().manual_seed(9)
    A = torch.randn(2000, 3, 3, dtype=torch.float64, generator=gen)
    C = A @ A.transpose(1, 2) + .01 * torch.eye(3, dtype=torch.float64)
    C = torch.cat([C, (1. / 12.) * torch.eye(3, dtype=torch.float64).expand(5, 3, 3),          # isotropic: one triple root
                   torch.diag_embed(torch.tensor([[1., 1., 3.], [3., 1., 1.], [2., 2., 0.5]], dtype=torch.float64))])
    torch.testing.assert_close(largest_eigenvalue(C), torch.linalg.eigvalsh(C)[:, -1], rtol=1e-6, atol=1e-12)
