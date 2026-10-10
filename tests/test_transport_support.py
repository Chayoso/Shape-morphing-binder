"""Contracts of the geometry energy's fine term (the surface proximity), the sampling berth, the device neighbours, the
loss and gate grids and the thin set."""
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
    assert sampling_berth(target) == pytest.approx(2.)
    transformed = 3 * target[:, [1, 2, 0]] + 2
    assert sampling_berth(transformed) == pytest.approx(2.)


@pytest.mark.parametrize('target', [np.zeros((9, 3)), np.zeros((8, 3)),
                                    np.full((9, 3), np.nan)])
def test_sampling_berth_rejects_undefined_spacing(target):
    from physmorph.prepare import sampling_berth
    with pytest.raises(ValueError, match='sampling berth'):
        sampling_berth(target)


def test_shared_transport_energy_preserves_disabled_path_and_trial_values():
    from physmorph.losses.grid_ot import GridSinkhornLoss
    from physmorph.losses.support import SurfaceProximity
    from physmorph.losses.volumetric import rasterize_mass
    target = torch.tensor(cloud() + 1.5)
    mass = torch.ones(len(target), dtype=target.dtype)
    origin, dims = torch.zeros(3, dtype=target.dtype), (4, 4, 4)
    grid = rasterize_mass(target, mass, origin, 1., dims)
    kw = dict(eps=1., iters=2000, tol=1e-9)
    plain = GridSinkhornLoss(grid, origin, 1., dims, **kw)
    disabled = GridSinkhornLoss(grid, origin, 1., dims, support=None, **kw)
    support = SurfaceProximity(target.numpy())
    enabled = GridSinkhornLoss(grid, origin, 1., dims, support=support, **kw)
    x = (target * 1.25).requires_grad_(True)
    v = torch.zeros_like(x)
    base = plain.state_energy(x, mass)
    assert torch.equal(disabled.state_energy(x, mass), base)
    got = enabled.state_energy(x, mass)
    expected = support(base, x)
    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(torch.autograd.grad(got, x, retain_graph=True)[0],
                               torch.autograd.grad(expected, x)[0])
    with torch.no_grad():
        torch.testing.assert_close(enabled.state_energy(x, mass), got.detach())


def test_target_build_attaches_support():
    from physmorph.pipeline.target import build_target
    from physmorph.mpm.state import MPMParams
    target = cloud().astype(np.float32)
    cfg = PipelineConfig(loss_res=8, render_res=16, dt_res=16)
    prm = MPMParams(dx=.25, nx=16, ny=16, nz=16, grid_min=(-2., -2., -2.))
    pack = build_target(target, prm, cfg)
    from physmorph.losses.support import SurfaceProximity
    assert isinstance(pack.support, SurfaceProximity)


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


def test_loss_grid_follows_the_particle_count_above_the_reference():
    from physmorph.prepare import prepare
    kw = dict(seed=3, cell_diag=26., young=1.4e5, poisson=.2, log=lambda s: None)
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
    assert 0 < len(prox.y) < len(target)
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
