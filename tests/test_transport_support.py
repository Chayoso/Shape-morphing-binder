"""Contracts for the opt-in, transport-bounded particle-support penalty."""
import math
import numpy as np
import pytest
import torch

from physmorph.pipeline import PipelineConfig


def cloud():
    return np.random.default_rng(13).normal(0, .2, (48, 3))


def test_sampling_berth_uses_target_spacing_and_is_rigid_scale_invariant():
    from physmorph.prepare import sampling_berth
    # Cube corners + centre: median NN=sqrt(3)/2, median eighth NN=sqrt(3).
    target = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.)
                       for z in (0., 1.)] + [[.5, .5, .5]])
    assert sampling_berth(target, 1000.) == pytest.approx(2.)
    transformed = 3 * target[:, [1, 2, 0]] + 2
    assert sampling_berth(transformed, 1000.) == pytest.approx(2.)


@pytest.mark.parametrize('target', [np.zeros((9, 3)), np.zeros((8, 3)),
                                    np.full((9, 3), np.nan)])
def test_sampling_berth_rejects_undefined_spacing(target):
    from physmorph.prepare import sampling_berth
    with pytest.raises(ValueError, match='sampling berth'):
        sampling_berth(target, 1000.)


def test_sampling_berth_must_stay_inside_the_far_band():
    from physmorph.prepare import sampling_berth
    with pytest.raises(ValueError, match='nn_far_k'):
        sampling_berth(cloud(), .5)


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_support_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match='support_weight'):
        PipelineConfig(support_weight=weight)


def test_support_matches_brute_force_neighbors_and_has_true_gradient():
    from physmorph.losses.support import TransportSupport
    target = cloud()
    support = TransportSupport(target, weight=8.)
    x = torch.tensor(target * 1.7, requires_grad=True)
    td = np.linalg.norm(target[:, None] - target[None], axis=2)
    np.fill_diagonal(td, np.inf)
    td = np.sort(td, axis=1)[:, :32]
    radius = np.median(td[:, 7])
    h = .5 * radius
    floor = .5 * np.median(np.exp(-td ** 2 / (2 * h * h)).sum(1))
    xd = np.linalg.norm(x.detach().numpy()[:, None] - x.detach().numpy()[None], axis=2)
    np.fill_diagonal(xd, np.inf)
    rho = np.exp(-np.sort(xd, axis=1)[:, :32] ** 2 / (2 * h * h)).sum(1)
    expected = radius ** 2 * np.maximum(np.log(floor) - np.log(rho), 0) ** 2
    assert float(support.penalty(x)) == pytest.approx(expected.mean(), rel=1e-10)
    direction = torch.tensor(np.random.default_rng(19).normal(size=x.shape))
    direction /= direction.norm()
    energy = lambda q: support((q - .1).square().mean(), q)
    g = torch.autograd.grad(energy(x), x)[0]
    eps = 1e-5
    fd = (energy(x.detach() + eps * direction) - energy(x.detach() - eps * direction)) / (2 * eps)
    assert float((g * direction).sum()) == pytest.approx(float(fd), rel=3e-4, abs=1e-8)


def test_bounded_support_vanishes_without_transport_error():
    from physmorph.losses.support import TransportSupport
    support = TransportSupport(cloud(), weight=8.)
    x = torch.tensor(cloud() * 3, requires_grad=True)
    assert float(support.penalty(x)) > 0
    zero = support(torch.tensor(0., dtype=x.dtype), x)
    assert float(zero) == 0.
    assert torch.count_nonzero(torch.autograd.grad(zero, x)[0]) == 0
    for value in [1e-6, .1, 100.]:
        result = support(torch.tensor(value, dtype=x.dtype), x)
        assert value < float(result) <= 2 * value


def test_zero_weight_and_inadmissible_transport_do_not_query_bad_positions():
    from physmorph.losses.support import TransportSupport
    bad_x = torch.full((48, 3), float('nan'))
    energy = torch.tensor(2.)
    assert TransportSupport(cloud(), weight=0.)(energy, bad_x) is energy
    support = TransportSupport(cloud(), weight=8.)
    for value in [float('inf'), float('nan'), -.001]:
        energy = torch.tensor(value)
        assert support(energy, bad_x) is energy


@pytest.mark.parametrize('weight', [1e39, 1e300])
def test_large_finite_weights_preserve_bounded_cost_and_finite_gradients(weight):
    from physmorph.losses.support import TransportSupport
    support = TransportSupport(cloud(), weight=weight)
    for value in [0., .1, 1e30]:
        x = torch.tensor(cloud() * 3, dtype=torch.float32, requires_grad=True)
        energy = torch.tensor(value, dtype=x.dtype, requires_grad=True)
        result = support(energy, x)
        gx, ge = torch.autograd.grad(result, (x, energy))
        assert result.dtype == energy.dtype
        assert torch.isfinite(result) and energy <= result <= 2 * energy
        assert torch.isfinite(gx).all() and torch.isfinite(ge)
        if value == 0:
            assert torch.count_nonzero(gx) == 0


def test_support_preserves_rigid_motion_and_length_squared_units():
    from physmorph.losses.support import TransportSupport
    target = cloud()
    x = torch.tensor(target * 1.5)
    R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    a = TransportSupport(target, weight=8.)(torch.tensor(.2), x)
    transformed = TransportSupport(3 * target @ R.T + 2, weight=8.)
    b = transformed(torch.tensor(1.8), 3 * x @ torch.tensor(R.T) + 2)
    assert float(b) == pytest.approx(9 * float(a), rel=1e-6)


def test_shared_transport_energy_preserves_disabled_path_and_trial_values():
    from physmorph.losses.grid_ot import GridSinkhornLoss
    from physmorph.losses.support import TransportSupport
    from physmorph.losses.volumetric import rasterize_mass
    target = torch.tensor(cloud() + 1.5)
    mass = torch.ones(len(target), dtype=target.dtype)
    origin, dims = torch.zeros(3, dtype=target.dtype), (4, 4, 4)
    grid = rasterize_mass(target, mass, origin, 1., dims)
    kw = dict(eps=1., iters=2000, tol=1e-9)
    plain = GridSinkhornLoss(grid, origin, 1., dims, **kw)
    disabled = GridSinkhornLoss(grid, origin, 1., dims, support=None, **kw)
    support = TransportSupport(target.numpy(), weight=8.)
    enabled = GridSinkhornLoss(grid, origin, 1., dims, support=support, **kw)
    x = (target * 1.25).requires_grad_(True)
    v = torch.zeros_like(x)
    base = plain.state_energy(x, mass, v, .1)
    assert torch.equal(disabled.state_energy(x, mass, v, .1), base)
    got = enabled.state_energy(x, mass, v, .1)
    expected = support(base, x)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(torch.autograd.grad(got, x, retain_graph=True)[0],
                               torch.autograd.grad(expected, x)[0])
    with torch.no_grad():
        torch.testing.assert_close(enabled.state_energy(x, mass, v, .1), got.detach())


def test_target_build_attaches_support():
    from physmorph.pipeline.target import build_target
    from physmorph.mpm.state import MPMParams
    target = cloud().astype(np.float32)
    cfg = PipelineConfig(loss_res=8, render_res=16, dt_res=16)
    prm = MPMParams(dx=.25, nx=16, ny=16, nz=16, grid_min=(-2., -2., -2.))
    pack = build_target(target, prm, cfg)
    assert pack.support is not None and pack.support.weight == cfg.support_weight
    assert build_target(target, prm, PipelineConfig(loss_res=8, render_res=16, dt_res=16,
                                                    support_weight=0.)).support is None

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gpu_neighbors_match_exact_distances_including_duplicates():
    from physmorph.render.knn_gpu import knn_self_torch
    from scipy.spatial import cKDTree
    points = np.random.default_rng(29).normal(size=(4096, 3)).astype(np.float32)
    points[:40] = points[0]
    d, j = knn_self_torch(torch.tensor(points, device='cuda'), 33)
    expected = cKDTree(points).query(points, k=33)[0]
    np.testing.assert_allclose(d.cpu(), expected, atol=2e-6, rtol=2e-6)
    actual = np.linalg.norm(points[j.cpu().numpy()] - points[:, None], axis=2)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gpu_neighbor_radius_expansion_reuses_completed_rows(monkeypatch):
    import physmorph.render.knn_gpu as knn
    from scipy.spatial import cKDTree
    points = np.random.default_rng(31).uniform(-1., 1., (8192, 3)).astype(np.float32)
    points[-1] = [3., 0., 0.]  # This isolated row needs a wider search than the bulk.
    N, k = len(points), 33
    allocations = []
    zeros = knn.wp.zeros

    def counted(shape, *args, **kwargs):
        if shape in ((N, k), N):
            allocations.append(shape)
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(knn.wp, 'zeros', counted)
    d, j = knn.knn_self_torch(torch.tensor(points, device='cuda'), k)
    expected = cKDTree(points).query(points, k=k)[0]
    np.testing.assert_allclose(d.cpu(), expected, atol=2e-6, rtol=2e-6)
    actual = np.linalg.norm(points[j.cpu().numpy()] - points[:, None], axis=2)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    # Three result buffers per query, not three new buffers per doubled radius.
    assert allocations == [(N, k), (N, k), N]


def sheet_on_block(z=.7):
    """A thick block of lattice points with a one-layer sheet above it at height z: the sheet is
    sampled at the block's spacing but has fewer neighbours, so its kernel density is lower."""
    g = np.arange(6) * .1
    block = np.array([[x, y, z] for x in g for y in g for z in g])
    s = np.arange(10) * .1
    sheet = np.array([[x, y, z] for x in s for y in s])
    return np.concatenate([block, sheet])


def test_target_referenced_floor_never_penalises_the_target():
    from physmorph.losses.support import TransportSupport
    target = sheet_on_block()
    x = torch.tensor(target)
    # the global floor (half the median density) asks the sheet to be denser than it is
    assert float(TransportSupport(target, weight=8.).penalty(x)) > 0
    ref = TransportSupport(target, weight=8., target_ref=True)
    assert float(ref.penalty(x)) == 0.
    assert ref.floor(x).shape == (len(target),)


def test_target_referenced_floor_is_the_target_density_at_the_particle_and_continuous():
    """A'': at every target point the floor is half the point's own leave-one-out density (the per-point floor);
    between the block's top face and the sheet the nearest target point switches, where the per-point floors differ
    by more than 10 %, and the floor is continuous there; three spacings off the target it is below zero (no penalty, no
    gradient: the W1 cleanup's job); a body that is the target stretched by 1.5 pays, with a true gradient that
    includes the floor's own dependence on the position."""
    from physmorph.losses.support import TransportSupport
    target = sheet_on_block()
    ref = TransportSupport(target, weight=8., target_ref=True, form="ratio")
    y = torch.tensor(target)
    torch.testing.assert_close(ref.floor(y), ref.log_floor_pt.to(device=y.device, dtype=y.dtype), atol=1e-9, rtol=1e-9)
    below, above = torch.tensor([[.25, .25, .6 - 1e-4]]), torch.tensor([[.25, .25, .6 + 1e-4]])
    i_lo, i_hi = ref.tree.query(below, 1)[1][0, 0], ref.tree.query(above, 1)[1][0, 0]
    assert abs(float(ref.log_floor_pt[i_lo] - ref.log_floor_pt[i_hi])) > .1     # the old floor stepped here
    assert abs(float(ref.floor(below) - ref.floor(above))) < 1e-3                # the new one does not
    far = torch.tensor([[.25, .25, -.3]], requires_grad=True)                       # three spacings off the block
    assert float(ref.penalty_per_point(torch.cat([far, y[1:]]))[0]) == 0.
    g_far = torch.autograd.grad(ref.penalty(torch.cat([far, y[1:]])), far)[0]
    assert float(g_far.abs().max()) == 0.
    far32 = far.detach().float().requires_grad_(True)                              # float32, as the pipeline runs:
    x32 = torch.cat([far32, y[1:].float()])                                         # the zero-penalty region must
    g32 = torch.autograd.grad(ref.penalty(x32), far32)[0]                           # keep a finite gradient
    assert float(ref.penalty_per_point(x32)[0]) == 0. and bool(torch.isfinite(g32).all()) and float(g32.abs().max()) == 0.
    jitter = torch.tensor(np.random.default_rng(47).uniform(-.005, .005, (216, 3)))     # no lattice ties: a tie at
    x = (torch.tensor(target[:216]) - .25) * 1.5 + .25 + jitter                   # the k-th neighbour would swap
    x = x.requires_grad_(True)
    per = ref.penalty_per_point(x)
    assert float(per.max()) > .1 * ref.radius ** 2 and float(per.min()) >= 0.
    energy = lambda q: ref((q - .1).square().mean(), q)
    g = torch.autograd.grad(energy(x), x)[0]
    direction = torch.tensor(np.random.default_rng(23).normal(size=x.shape))
    direction /= direction.norm()
    eps = 1e-6
    fd = (energy(x.detach() + eps * direction) - energy(x.detach() - eps * direction)) / (2 * eps)
    assert float((g * direction).sum()) == pytest.approx(float(fd), rel=1e-3, abs=1e-9)


def test_loss_grid_follows_the_particle_count_above_the_reference():
    from physmorph.prepare import prepare
    kw = dict(seed=3, cell_diag=26., young=1.4e5, poisson=.2, far_k=1000., log=lambda s: None)
    base = prepare("assets/isosphere.obj", "assets/bunny.obj", 4000, **kw)
    same = prepare("assets/isosphere.obj", "assets/bunny.obj", 4000, loss_ref_n=4000, **kw)
    fine = prepare("assets/isosphere.obj", "assets/bunny.obj", 4000, loss_ref_n=500, **kw)
    assert base.loss_res == base.prm.nx == same.loss_res            # at or below the reference: the MPM cell
    assert fine.prm.nx == base.prm.nx                                # the MPM grid never changes
    assert fine.loss_res == int(np.ceil(base.prm.nx * 2.0))          # (4000 / 500)^(1/3) = 2


def test_transport_gate_stays_on_the_mpm_cell_grid():
    from physmorph.pipeline.target import build_target
    from physmorph.mpm.state import MPMParams
    target = cloud().astype(np.float32)
    prm = MPMParams(dx=.25, nx=16, ny=16, nz=16, grid_min=(-2., -2., -2.))
    same = build_target(target, prm, PipelineConfig(loss_res=16, render_res=16, dt_res=16))
    assert same.gate[0] is same.grid and same.gate[1] == same.ldx
    fine = build_target(target, prm, PipelineConfig(loss_res=32, render_res=16, dt_res=16, loss_follows_n=True))
    assert fine.ldims == (32,) * 3 and fine.gate[2] == (16,) * 3
    assert fine.gate[1] == pytest.approx(.25) and float(fine.gate[0].sum()) == pytest.approx(float(fine.grid.sum()))


def test_ratio_form_is_bounded_and_agrees_with_the_log_form_near_the_floor():
    from physmorph.losses.support import deficit_penalty
    t = torch.tensor([-1., 0., .01, .1, 5., 50.], dtype=torch.float64)      # log f - log s
    lg, rt = deficit_penalty(t, "log"), deficit_penalty(t, "ratio")
    assert float(lg[0]) == float(rt[0]) == float(rt[1]) == 0.               # above the floor: free
    assert float(rt[2] / lg[2]) == pytest.approx(1., abs=.011)             # s -> f: the same penalty
    assert float(rt.max()) <= 1. and float(lg[-1]) == 2500.                  # bounded against unbounded
    with pytest.raises(ValueError, match="support form"):
        from physmorph.losses.support import TransportSupport
        TransportSupport(cloud(), weight=8., form="cap")
    with pytest.raises(ValueError, match="support_form"):
        PipelineConfig(support_form="cap")


def test_ratio_form_bounds_an_isolated_particle_and_leaves_it_no_gradient():
    from physmorph.losses.support import TransportSupport
    target = cloud()
    x = torch.tensor(target.copy())
    x[0] = torch.tensor([4., 4., 4.], dtype=x.dtype)                        # far off the body
    x.requires_grad_(True)
    ratio = TransportSupport(target, weight=8., form="ratio")
    per = ratio.penalty_per_point(x)
    assert float(per[0]) == pytest.approx(ratio.radius ** 2) and float(per.max()) <= ratio.radius ** 2
    g = torch.autograd.grad(per[0], x)[0]
    assert float(g.abs().max()) < 1e-12                                      # left to the W1 cleanup
    log = TransportSupport(target, weight=8., form="log")
    assert float(log.penalty_per_point(x.detach())[0]) > 1e3 * log.radius ** 2
    assert torch.equal(log.penalty(x.detach()), TransportSupport(target, weight=8.).penalty(x.detach()))


def test_ratio_form_has_true_gradient_and_spares_the_target_with_its_own_floor():
    from physmorph.losses.support import TransportSupport
    target = cloud()
    support = TransportSupport(target, weight=8., form="ratio")
    x = torch.tensor(target * 1.7, requires_grad=True)
    assert float(support.penalty(x)) > 0
    energy = lambda q: support((q - .1).square().mean(), q)
    g = torch.autograd.grad(energy(x), x)[0]
    direction = torch.tensor(np.random.default_rng(37).normal(size=x.shape))
    direction /= direction.norm()
    eps = 1e-5
    fd = (energy(x.detach() + eps * direction) - energy(x.detach() - eps * direction)) / (2 * eps)
    assert float((g * direction).sum()) == pytest.approx(float(fd), rel=3e-4, abs=1e-8)
    sheet = sheet_on_block()
    ref = TransportSupport(sheet, weight=8., target_ref=True, form="ratio")
    assert float(ref.penalty(torch.tensor(sheet))) == 0.


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_thin_set_finds_the_sheet_and_measures_its_coverage():
    from physmorph.thin import thin_metrics, thin_set
    target = sheet_on_block(z=.9)                  # four spacings above: a separate feature (at two the
    ts = thin_set(target, cell=.15, ref_n=len(target))   # sampling's covering radius bridges the gap)
    pts = ts.points.cpu().numpy()
    on_sheet = np.isclose(pts[:, 2], .9, atol=1e-6)
    assert on_sheet.sum() == 100                                        # the one-layer sheet is thin
    assert thin_metrics(target, ts)["thin_uncovered"] == 0.
    m = thin_metrics(target[:216], ts)                                  # the block alone
    assert m["thin_uncovered"] >= on_sheet.mean() - 1e-9 and m["thin_gap_median_sp"] > 1.5



def test_surface_proximity_charges_the_missing_sheet_and_spares_the_target():
    """The proximity reads the target's outer points; its threshold is about 1.5 spacings (sqrt(sp^2 + 2 h^2 ln 2));
    the target itself pays nothing; with the sheet missing the sheet's points pay about radius^2 each and the block's
    outer points nothing; a body particle within one spacing of a sheet point clears it and one at two spacings does
    not; descent moves the nearest block particles up toward the sheet; the gradient is true (finite differences)."""
    from physmorph.losses.support import SurfaceProximity
    target = sheet_on_block(z=.9)
    prox = SurfaceProximity(target)
    assert 0 < len(prox.y) < len(target) and prox.weight is None
    thr = math.sqrt(prox.spacing ** 2 + 2 * prox.h ** 2 * math.log(2))
    assert 1.3 * prox.spacing < thr < 1.8 * prox.spacing
    on_sheet = torch.isclose(prox.y[:, 2].cpu(), torch.tensor(.9, dtype=prox.y.dtype))
    assert int(on_sheet.sum()) >= 60
    assert float(prox.penalty(torch.tensor(target))) == 0.
    jitter = np.random.default_rng(11).uniform(-.005, .005, (216, 3))
    block = torch.tensor(target[:216] + jitter, requires_grad=True)
    per = prox.penalty_per_point(block)
    assert float(per[on_sheet].min()) > .9 * prox.radius ** 2 and float(per[~on_sheet].max()) == 0.
    one = torch.tensor(np.concatenate([target[:216] + jitter, [[.45, .45, .9 - 1.0 * prox.spacing]]]))
    two = torch.tensor(np.concatenate([target[:216] + jitter, [[.45, .45, .9 - 2.0 * prox.spacing]]]))
    j = int(torch.nonzero(on_sheet & torch.isclose(prox.y[:, 0].cpu(), torch.tensor(.4, dtype=prox.y.dtype))
                          & torch.isclose(prox.y[:, 1].cpu(), torch.tensor(.4, dtype=prox.y.dtype)))[0])
    assert float(prox.penalty_per_point(one)[j]) == 0. and float(prox.penalty_per_point(two)[j]) > .1 * prox.radius ** 2
    g = torch.autograd.grad(prox.penalty(block), block)[0]
    top = block.detach()[:, 2] > .45
    assert float(g[top, 2].sum()) < 0. and float(g[~top].abs().sum()) == 0.   # only the nearest particles move, up
    energy = lambda q: prox((q - .1).square().mean(), q)
    gE = torch.autograd.grad(energy(block), block)[0]
    direction = torch.tensor(np.random.default_rng(41).normal(size=block.shape))
    direction /= direction.norm()
    eps = 1e-6
    fd = (energy(block.detach() + eps * direction) - energy(block.detach() - eps * direction)) / (2 * eps)
    assert float((gE * direction).sum()) == pytest.approx(float(fd), rel=1e-3, abs=1e-9)
