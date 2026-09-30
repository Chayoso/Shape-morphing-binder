"""Fixed-target grid Sinkhorn divergence (losses/grid_ot.py): consistent mass, gradient and
equilibrium; rejected unsolved trials; the separable log-domain transform against the dense
cost; the CUDA-graph solve against the plain one; the transport state energy (drift in time
units, no hidden escaped mass); the label-free transport map."""
import numpy as np
import pytest
import torch

def test_grid_sinkhorn_loss_has_consistent_mass_gradient_and_equilibrium():
    from physmorph.losses.grid_ot import GridSinkhornLoss
    target = torch.tensor([.2, .3, .5], dtype=torch.float64)
    loss = GridSinkhornLoss(target, torch.zeros(3, dtype=torch.float64), 1., (3, 1, 1),
                            eps=.3, iters=2000, tol=1e-9)
    matched = target.clone().requires_grad_(True)
    value = loss(matched)
    grad = torch.autograd.grad(value, matched)[0]
    assert abs(float(value)) < 1e-10
    assert float(grad.abs().max()) < 1e-8
    current = torch.tensor([.5, .3, .2], dtype=torch.float64, requires_grad=True)
    value = loss(current)
    grad = torch.autograd.grad(value, current)[0]
    direction = torch.tensor([-.2, .1, .1], dtype=torch.float64)
    h = 1e-4
    fd = (loss(current.detach() + h * direction) - loss(current.detach() - h * direction)) / (2 * h)
    assert float(value) > 0
    assert float(grad @ direction) < 0
    assert float(grad @ direction) == pytest.approx(float(fd), rel=2e-4, abs=1e-6)
    assert float(loss(current.detach() * 7)) == pytest.approx(float(value), abs=1e-8)


def test_grid_sinkhorn_rejects_missing_physical_mass_and_unconverged_duals():
    from physmorph.losses.grid_ot import GridSinkhornLoss
    target = torch.tensor([.2, .3, .5], dtype=torch.float64)
    with pytest.raises(ValueError, match='converge'):
        GridSinkhornLoss(target, torch.zeros(3), 1., (3, 1, 1), eps=4., iters=1, tol=1e-9)
    loss = GridSinkhornLoss(target, torch.zeros(3), 1., (3, 1, 1),
                            eps=.3, iters=2000, tol=1e-8, mass_total=1.)
    lost = (.5 * target).requires_grad_(True)
    value = loss(lost)
    assert bool(torch.isinf(value))
    assert bool(torch.isfinite(torch.autograd.grad(value, lost)[0]).all())


def test_grid_sinkhorn_rejects_unsolved_trials_without_inventing_gradients():
    from physmorph.losses.grid_ot import GridSinkhornLoss
    target = torch.tensor([.2, .3, .5], dtype=torch.float64)
    loss = GridSinkhornLoss(target, torch.zeros(3), 1., (3, 1, 1),
                            eps=.3, iters=2000, tol=1e-9)
    loss.iters = 1
    current = target.flip(0)
    assert bool(torch.isinf(loss(current)))
    with pytest.raises(ValueError, match='converge'):
        loss(current.requires_grad_(True))


def test_grid_sinkhorn_keeps_gradients_into_empty_cic_nodes():
    from physmorph.losses.grid_ot import GridSinkhornLoss
    from physmorph.losses.volumetric import rasterize_mass
    origin, dims = torch.zeros(3, dtype=torch.float64), (3, 3, 3)
    y = torch.tensor([[1., 1., 1.]], dtype=torch.float64)
    mass = torch.ones(1, dtype=torch.float64)
    target = rasterize_mass(y, mass, origin, 1., dims)
    loss = GridSinkhornLoss(target, origin, 1., dims, eps=.3, iters=2000, tol=1e-9)
    x = torch.tensor([[0., 1., 1.]], dtype=torch.float64, requires_grad=True)
    def value(z):
        return loss(rasterize_mass(z, mass, origin, 1., dims))
    before = value(x)
    grad = torch.autograd.grad(before, x)[0]
    step = torch.tensor([[1e-5, 0., 0.]], dtype=torch.float64)
    fd = float((value(x.detach() + step) - before.detach()) / step[0, 0])
    assert float(grad[0, 0]) == pytest.approx(fd, rel=2e-3)
    assert float(grad[0, 0]) < -1.


def test_grid_sinkhorn_separable_transform_matches_dense_cost():
    from physmorph.losses.grid_ot import GridSinkhornLoss, _grid_measure
    gen = torch.Generator().manual_seed(8)
    target = torch.rand(24, generator=gen, dtype=torch.float64)
    target[::3] = 0.
    origin, dims, dx = torch.zeros(3, dtype=torch.float64), (3, 4, 2), .5
    loss = GridSinkhornLoss(target, origin, dx, dims, eps=.3, iters=2000, tol=1e-8)
    dual = torch.rand(24, generator=gen, dtype=torch.float64)
    nodes, _ = _grid_measure(torch.ones_like(target), origin, dx, dims)
    cost = torch.cdist(nodes, nodes).square()
    log_weights = (target / target.sum()).log()
    dense = -.3 * torch.logsumexp((dual[None, :] - cost) / .3 + log_weights[None, :], 1)
    assert torch.allclose(loss.transform(dual, log_weights, .3), dense, atol=1e-10)


def test_grid_sinkhorn_self_solve_reuses_identical_dual_transforms(monkeypatch):
    from physmorph.losses.grid_ot import GridSinkhornLoss
    mass = torch.tensor([.2, .3, .5], dtype=torch.float64)
    loss = GridSinkhornLoss(mass, torch.zeros(3), 1., (3, 1, 1),
                            eps=.3, iters=2000, tol=1e-9)
    transform, calls = loss.transform, []
    def counted(*args):
        calls.append(1)
        return transform(*args)
    monkeypatch.setattr(loss, 'transform', counted)
    reference = loss.solve(mass, mass.clone())
    count = len(calls)
    calls.clear()
    result = loss.solve(mass, mass)
    assert all(torch.equal(a, b) for a, b in zip(reference, result))
    assert len(calls) * 2 == count


@pytest.mark.parametrize('requires_grad', [False, True])
def test_grid_sinkhorn_value_avoids_float32_self_energy_cancellation(requires_grad):
    from physmorph.losses.grid_ot import GridSinkhornLoss
    gen = torch.Generator().manual_seed(19)
    target = torch.rand(210, generator=gen) + .1
    target /= target.sum()
    direction = torch.randn(210, generator=gen)
    direction -= direction.mean()
    direction *= target.min() / direction.abs().max()
    loss = GridSinkhornLoss(target, torch.zeros(3), .25, (7, 6, 5),
                            eps=.15, iters=2000, tol=1e-3)
    current = (target + .1 * direction).requires_grad_(requires_grad)
    a = current.detach() / current.detach().sum()
    f, g = loss.solve(a, loss.b)
    fs, _ = loss.solve(a, a)
    ft, _ = loss.solve(loss.b, loss.b)
    reference = ((a.double() * (f.double() - fs.double())).sum()
                 + (loss.b.double() * (g.double() - ft.double())).sum())
    assert float(loss(current)) == pytest.approx(float(reference), rel=1e-5, abs=1e-12)


def test_end_drift_charges_the_released_end_and_the_released_motion_is_a_record():
    """The stability term is the residual drift at the released end: (T dt)^2 x the mean |v_T|^2; zero at rest;
    invariant under v -> 2v with the horizon halved; the driven phase is not costed. The released-motion record
    sees an oscillating release that ends at rest, which the drift does not (R11: charging it fought the transport)."""
    from physmorph.pipeline.window.objective import end_drift, released_motion
    T, dt = 4, .5
    vT = torch.zeros(3, 3); vT[:, 0] = .2
    value = end_drift(vT, T * dt)
    assert float(value) == pytest.approx((T * dt * .2) ** 2)
    assert float(end_drift(2 * vT, T * dt / 2)) == pytest.approx(float(value))
    assert float(end_drift(torch.zeros(3, 3), T * dt)) == 0.
    leaf = vT.clone().requires_grad_(True)
    assert float(torch.autograd.grad(end_drift(leaf, T * dt), leaf)[0][:, 0].min()) > 0.
    osc = torch.zeros(2 * T, 3, 3); osc[T:, :, 0] = torch.tensor([.2, -.2, .2, 0.])[:, None]
    assert float(end_drift(osc[-1], T * dt)) == 0. and float(released_motion(osc, T, T * dt)) > 0.


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('iters', [1600, 1601])
def test_grid_sinkhorn_cuda_blocks_preserve_values_gradients_and_history_independence(iters, device):
    from physmorph.losses.grid_ot import GridSinkhornLoss
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA graph execution')
    gen = torch.Generator().manual_seed(19)
    target = (torch.rand(120, generator=gen) + .1).to(device)
    target /= target.sum()
    kw = dict(eps=.15, iters=iters, tol=1e-3)
    cold = GridSinkhornLoss(target, torch.zeros(3, device=device), .25, (6, 5, 4), **kw)
    graph = GridSinkhornLoss(target, torch.zeros(3, device=device), .25, (6, 5, 4),
                             cuda_blocks=True, **kw)
    def evaluate(loss, current):
        leaf = current.detach().clone().requires_grad_(True)
        value = loss(leaf)
        return value.detach(), torch.autograd.grad(value, leaf)[0]
    shifted = target.roll(13)
    first = evaluate(graph, shifted)
    for current in (shifted, target.flip(0), target, shifted):
        actual, expected = evaluate(graph, current), evaluate(cold, current)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=1e-7, rtol=1e-3)
    for before, after in zip(first, evaluate(graph, shifted)):
        torch.testing.assert_close(before, after, atol=0., rtol=0.)
    _, at_target = evaluate(graph, target)
    assert float(at_target.abs().max()) < 1e-7
    # Cached graphs must not silently bypass a changed convergence budget.
    graph.iters = 4
    assert bool(torch.isinf(graph(shifted)))
    with pytest.raises(ValueError, match='converge'):
        evaluate(graph, shifted)


@pytest.mark.parametrize('temperature', [.003, 30.])
def test_grid_transform_cuda_uses_volume_sized_workspace(temperature):
    from physmorph.losses.grid_ot import GridSinkhornLoss
    if not torch.cuda.is_available():
        pytest.skip('CUDA transport workspace')
    dims = (18, 17, 16)
    gen = torch.Generator().manual_seed(41)
    target = torch.ones(np.prod(dims), device='cuda')
    loss = GridSinkhornLoss(target, torch.zeros(3, device='cuda'), .25, dims,
                            eps=.3, iters=1600, tol=1e-3)
    dual = torch.rand(target.numel(), generator=gen).cuda()
    weights = torch.rand(target.numel(), generator=gen).cuda()
    weights[:dims[1] * dims[2]] = 0.  # Empty grid lines must stay -inf, not NaN.
    log_weights = (weights / weights.sum()).log()
    with torch.no_grad():
        expected = loss.transform(dual, log_weights, temperature)
        loss.cuda_blocks = True
        loss.transform(dual, log_weights, temperature)  # Warm kernel compilation.
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = loss.transform(dual, log_weights, temperature)
        torch.cuda.synchronize()
        workspace = torch.cuda.max_memory_allocated() - allocated
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    # A separable transform needs grid-sized buffers, not a grid x axis tensor.
    assert workspace <= 8 * dual.numel() * dual.element_size()
    for bad_weights in (torch.full_like(log_weights, -float('inf')),
                        log_weights.clone().index_fill(0, torch.tensor([1], device='cuda'), float('nan'))):
        with torch.no_grad():
            loss.cuda_blocks = False
            expected = loss.transform(dual, bad_weights, temperature)
            loss.cuda_blocks = True
            actual = loss.transform(dual, bad_weights, temperature)
        torch.testing.assert_close(actual, expected, equal_nan=True)
    # Direct differentiable calls still use autograd, not a detached fast path.
    leaf = dual.detach().requires_grad_(True)
    actual = loss.transform(leaf, log_weights, temperature)
    ga, = torch.autograd.grad(actual.sum(), leaf)
    loss.cuda_blocks = False
    expected = loss.transform(leaf, log_weights, temperature)
    ge, = torch.autograd.grad(expected.sum(), leaf)
    torch.testing.assert_close(ga, ge, atol=0., rtol=0.)


def test_transport_state_energy_cannot_hide_escaped_mass():
    from physmorph.losses.grid_ot import GridSinkhornLoss
    from physmorph.losses.volumetric import rasterize_mass
    origin, dims = torch.zeros(3, dtype=torch.float64), (4, 4, 4)
    mass = torch.ones(1, dtype=torch.float64)
    target = rasterize_mass(torch.tensor([[1., 1., 1.]], dtype=torch.float64),
                            mass, origin, 1., dims)
    loss = GridSinkhornLoss(target, origin, 1., dims, eps=1., mass_total=1.)
    x = torch.tensor([[-1., 1., 1.]], dtype=torch.float64, requires_grad=True)
    value = loss.state_energy(x, mass)
    assert bool(torch.isinf(value))
    assert bool(torch.isfinite(torch.autograd.grad(value, x)[0]).all())


def test_grid_transport_must_actually_solve_at_requested_blur():
    from physmorph.losses.grid_ot import grid_transport_displacement
    from physmorph.losses.volumetric import rasterize_mass
    x = torch.tensor([[0., 0., 0.], [1., 0., 0.]])
    mass, origin, dims = torch.ones(2), torch.zeros(3), (3, 3, 3)
    target = rasterize_mass(x, mass, origin, 1., dims)
    with pytest.raises(ValueError, match='requested blur'):
        grid_transport_displacement(x, mass, target, origin, 1., dims,
                                    eps=.5, iters=4, tol=.01)


def test_grid_transport_is_stationary_under_particle_relabeling():
    from physmorph.losses.grid_ot import grid_transport_displacement
    from physmorph.losses.volumetric import rasterize_mass
    gen = torch.Generator().manual_seed(14)
    y = torch.rand(150, 3, generator=gen) * 1.5 + .7
    mass = torch.rand(150, generator=gen) + .2
    origin, dx, dims = torch.zeros(3), .5, (7, 7, 7)
    target = rasterize_mass(y, mass, origin, dx, dims)
    idx = torch.randperm(len(y), generator=gen)
    d = grid_transport_displacement(y[idx], mass[idx], target, origin, dx, dims,
                                    eps=.1, iters=400, tol=1e-4)
    assert float(d.abs().max()) < 2e-5


def test_grid_transport_moves_mass_toward_target_and_ignores_mass_units():
    from physmorph.losses.grid_ot import grid_transport_displacement
    from physmorph.losses.volumetric import rasterize_mass
    gen = torch.Generator().manual_seed(14)
    y = torch.rand(150, 3, generator=gen) * 1.5 + .7
    mass = torch.rand(150, generator=gen) + .2
    origin, dx, dims = torch.zeros(3), .5, (8, 8, 8)
    target = rasterize_mass(y, mass, origin, dx, dims)
    x = y + torch.tensor([.1, 0., 0.])
    d = grid_transport_displacement(x, mass, target, origin, dx, dims,
                                    eps=dx ** 2, iters=400, tol=1e-4)
    scaled = grid_transport_displacement(x, mass * 7., target * 7., origin, dx, dims,
                                         eps=dx ** 2, iters=400, tol=1e-4)
    assert float((d - torch.tensor([-.1, 0., 0.])).norm(dim=1).mean()) < .025
    assert torch.allclose(scaled, d, atol=2e-5)

