"""Contracts for the opt-in, transport-bounded particle-support penalty."""
import numpy as np
import pytest
import torch

from physmorph.pipeline import PipelineConfig


def cloud():
    return np.random.default_rng(13).normal(0, .2, (48, 3))


def _berth_args(**overrides):
    from argparse import Namespace
    options = dict(nn_sampling_berth=True, nn_berth_k=1., nn_far_k=1000., w_nn=.2,
                   solver_mode='settled_transport', support_weight=8.,
                   arms='render_full_dt_iso_nn')
    options.update(overrides)
    return Namespace(**options)


def test_sampling_berth_disabled_does_not_inspect_target_or_change_manual_value():
    from scripts import pipeline_run
    assert pipeline_run._resolve_nn_berth(
        _berth_args(nn_sampling_berth=False, nn_berth_k=1.25, solver_mode='legacy'), None) == 1.25


def test_sampling_berth_uses_target_spacing_and_is_rigid_scale_invariant():
    from scripts import pipeline_run
    # Cube corners + centre: median NN=sqrt(3)/2, median eighth NN=sqrt(3).
    target = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.)
                       for z in (0., 1.)] + [[.5, .5, .5]])
    assert pipeline_run._resolve_nn_berth(_berth_args(), target) == pytest.approx(2.)
    transformed = 3 * target[:, [1, 2, 0]] + 2
    assert pipeline_run._resolve_nn_berth(_berth_args(), transformed) == pytest.approx(2.)


@pytest.mark.parametrize('override', [dict(solver_mode='legacy'), dict(support_weight=0.),
    dict(support_weight=float('nan')), dict(nn_berth_k=1.5), dict(nn_far_k=.5),
    dict(w_nn=0.), dict(w_nn=-1.), dict(w_nn=float('nan')),
    dict(arms='phys')])
def test_sampling_berth_rejects_incompatible_options(override):
    from scripts import pipeline_run
    with pytest.raises(ValueError, match='nn_sampling_berth'):
        pipeline_run._resolve_nn_berth(_berth_args(**override), cloud())


@pytest.mark.parametrize('target', [np.zeros((9, 3)), np.zeros((8, 3)),
                                    np.full((9, 3), np.nan)])
def test_sampling_berth_rejects_undefined_spacing(target):
    from scripts import pipeline_run
    with pytest.raises(ValueError, match='nn_sampling_berth'):
        pipeline_run._resolve_nn_berth(_berth_args(), target)


def test_support_is_opt_in_and_requires_settled_transport():
    assert getattr(PipelineConfig(), 'support_weight', None) == 0.
    with pytest.raises(ValueError, match='settled_transport'):
        PipelineConfig(support_weight=8.)
    assert PipelineConfig(solver_mode='settled_transport', support_weight=8.).support_weight == 8.


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_support_rejects_invalid_weights(weight):
    with pytest.raises(ValueError, match='support_weight'):
        PipelineConfig(solver_mode='settled_transport', support_weight=weight)


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
    from physmorph.losses.ot import GridSinkhornLoss
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


def test_target_build_attaches_support_and_rejects_nonuniform_particle_masses():
    from physmorph.pipeline.runner import build_target
    from physmorph.mpm.state import MPMParams
    target = cloud().astype(np.float32)
    cfg = PipelineConfig(device='cpu', solver_mode='settled_transport', support_weight=8., loss_res=8)
    prm = MPMParams(dx=1., nx=16, ny=16, nz=16)
    pack = build_target(target, prm, cfg)
    assert pack.support is not None
    with pytest.raises(ValueError, match='uniform'):
        build_target(target, prm, cfg, w_src=np.linspace(.5, 1.5, len(target)))
    with pytest.raises(ValueError, match='uniform'):
        build_target(target, prm, cfg, w_tgt=np.linspace(.5, 1.5, len(target)))


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
