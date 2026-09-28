"""Owned post-layer position sequences and their adjoints, Warp CPU only."""
from dataclasses import replace

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.function import (
    PersistentAdjoint, RolloutSpec, _WarpMPMExt, warp_mpm_ext,
    warp_mpm_ext_with_previous, warp_mpm_ext_with_positions,
)
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory


def case(T=3, *, body=True):
    rng = np.random.default_rng(297)
    x = rng.uniform(-.6, .6, (24, 3)).astype(np.float32)
    n = len(x)
    ids = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    prm = MPMParams(dx=.5, dt=.015, grid_min=(-3., -3., -3.), nx=12, ny=12, nz=12,
                    drag=.02, smoothing=.955, eta_sym=1, eta_mode=1)
    nbr = np.stack([(np.arange(n)+i) % n for i in (1, 2, 3)], 1).astype(np.int32)
    layer = (np.ones(n, np.float32), np.tile([0., 1., 0.], (n, 1)).astype(np.float32),
             nbr, np.full((n, 3), 1/3, np.float32), .4/T, None, 0.,
             np.linspace(.2, 1., n, dtype=np.float32))
    pin = (np.arange(n) % 6 == 0).astype(np.float32)
    s = RolloutSpec(x, 1., 80., 40., prm, T, device='cpu',
                    vol0=np.full(n, .02, np.float32), layer=layer, pin=pin,
                    body_ctrl=body, body_modes=2 if body else 1,
                    v0=rng.normal(0., .02, x.shape).astype(np.float32),
                    C0=rng.normal(0., .01, ids.shape).astype(np.float32),
                    F0=ids+rng.normal(0., .002, ids.shape).astype(np.float32),
                    Fp=ids+rng.normal(0., .001, ids.shape).astype(np.float32),
                    eta=np.linspace(.1, .3, n, dtype=np.float32), Fg0=ids.copy())
    dc = .025*torch.randn(T, n, 3, 3, generator=torch.Generator().manual_seed(19))
    u = torch.linspace(-.003, .004, n)
    b = .001*torch.randn(2*n, 3, generator=torch.Generator().manual_seed(20)) if body else None
    return s, dc, u, b


def bridge(kind, spec):
    if kind == 'ordinary':
        return lambda dc, **kw: warp_mpm_ext_with_positions(dc, spec, **kw), None
    adj = PersistentAdjoint(spec, position_sequence=True)
    return adj.apply_with_positions, adj


def fd_check(loss, leaf, eps, rtol=.025, atol=3e-5):
    grad, = torch.autograd.grad(loss(leaf), leaf)
    assert torch.isfinite(grad).all() and grad.abs().max() > 20*atol
    # A resolved strong coordinate avoids false passing at near-zero derivatives.
    index = int(grad.abs().argmax())
    plus, minus = leaf.detach().clone(), leaf.detach().clone()
    plus.reshape(-1)[index] += eps
    minus.reshape(-1)[index] -= eps
    with torch.no_grad():
        actual = (loss(plus)-loss(minus))/(2*eps)
    assert float(grad.reshape(-1)[index]) == pytest.approx(float(actual), rel=rtol, abs=atol)
    return grad


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
def test_sequence_matches_actual_post_layer_states_and_preserves_old_outputs(kind):
    s, dc, u, b = case()
    run, adj = bridge(kind, s)
    out = run(dc, u_t=u, body_t=b)
    old = warp_mpm_ext(dc, s, u_t=u, body_t=b)
    previous = warp_mpm_ext_with_previous(dc, s, u_t=u, body_t=b)
    assert len(out) == 6 and out[-1].shape == (s.T, len(s.x0), 3)
    for actual, expected in zip(out[:5], old):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(out[-1][-2], previous[-1], rtol=0, atol=0)
    seq = [wp.from_torch(dc[t], dtype=wp.mat33) for t in range(s.T)]
    tr = Trajectory(s.x0, s.m, s.lam, s.mu, s.prm, s.T,
                    dFc=seq, F0=s.F0, Fp=s.Fp, v0=s.v0, C0=s.C0, eta=s.eta,
                    pin=s.pin, vol0=s.vol0, layer=s.layer,
                    layer_u=wp.from_torch(u, dtype=wp.float32),
                    body_control=wp.from_torch(b, dtype=wp.vec3),
                    track_geom=True, Fg0=s.Fg0, device='cpu', requires_grad=False)
    tr.rollout()
    expected = torch.stack([wp.to_torch(tr.x[t]) for t in range(1, s.T+1)])
    pre_layer = torch.stack([wp.to_torch(tr.xu[t]) for t in range(1, s.T+1)])
    torch.testing.assert_close(out[-1], expected, rtol=0, atol=0)
    assert torch.count_nonzero(expected-pre_layer) > 0
    snapshots = tuple(value.clone() for value in out)
    run(dc*1.4, u_t=u*.8, body_t=b*.9)
    if adj is not None:
        for value in adj.traj.x:
            value.zero_()
    for actual, expected in zip(out, snapshots):
        assert torch.equal(actual, expected)
    out[-1].zero_()
    assert torch.equal(out[0], snapshots[0])  # X and xT are independently owned.


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
@pytest.mark.parametrize('channel', ['dfc', 'body', 'u'])
def test_position_sequence_finite_differences_through_mixed_controls(kind, channel):
    s, dc, u, b = case()
    run, _ = bridge(kind, s)
    weights = torch.randn(s.T, len(s.x0), 3, generator=torch.Generator().manual_seed(25))
    def loss(value):
        d, surface, force = (value if channel == 'dfc' else dc,
                             value if channel == 'u' else u,
                             value if channel == 'body' else b)
        out = run(d, u_t=surface, body_t=force)
        delta = out[-1]-torch.from_numpy(s.x0)[None]
        return (delta.double()*weights).sum() + .01*out[4].double().square().sum()
    leaf = {'dfc': dc, 'body': b, 'u': u}[channel].clone().requires_grad_()
    fd_check(loss, leaf, 2e-3 if channel == 'dfc' else 2e-4)


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
@pytest.mark.parametrize('T', [1, 3])
def test_simultaneous_terminal_position_and_velocity_seeds_merge(kind, T):
    s, dc, u, b = case(T=T, body=T > 1)
    run, _ = bridge(kind, s)
    d = dc.clone().requires_grad_()
    out = run(d, u_t=u, body_t=b)
    w = torch.randn(4, len(s.x0), 3, generator=torch.Generator().manual_seed(26))
    loss = (out[0]*w[0] + out[-1][-1]*w[1] + out[2]*w[2] + out[4][-1]*w[3]).sum()
    g, = torch.autograd.grad(loss, d)
    ref = dc.clone().requires_grad_()
    legacy = warp_mpm_ext(ref, s, u_t=u, body_t=b)
    gold, = torch.autograd.grad((legacy[0]*(w[0]+w[1]) + legacy[2]*(w[2]+w[3])).sum(), ref)
    torch.testing.assert_close(g, gold, rtol=0, atol=0)
    assert out[-1].shape[0] == T


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
def test_early_position_causality_pinned_outputs_and_no_x0_gradient(kind):
    s, dc, u, b = case()
    x0 = torch.from_numpy(s.x0.copy()).requires_grad_()
    s = replace(s, x0=x0)
    run, _ = bridge(kind, s)
    d, surface, force = (value.clone().requires_grad_() for value in (dc, u, b))
    out = run(d, u_t=surface, body_t=force)
    pinned = torch.from_numpy(s.pin > .5)
    assert torch.equal(out[-1][:, pinned], x0.detach()[pinned].expand(s.T, -1, -1))
    g, gx0 = torch.autograd.grad(out[-1][0].square().sum(), (d, x0), allow_unused=True, retain_graph=True)
    assert gx0 is None
    assert g[0].abs().max() > 1e-7 and torch.count_nonzero(g[1:]) == 0
    grads = torch.autograd.grad(out[-1][:, pinned].sum(), (d, surface, force))
    assert all(torch.count_nonzero(value) == 0 for value in grads)


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
def test_repeated_seeds_and_missing_X_seed_do_not_leak(kind):
    s, dc, u, b = case()
    run, _ = bridge(kind, s)
    d = dc.clone().requires_grad_()
    out = run(d, u_t=u, body_t=b)
    first, = torch.autograd.grad(out[-1][0].square().sum(), d, retain_graph=True)
    velocity, = torch.autograd.grad(out[4].square().sum(), d, retain_graph=True)
    terminal, = torch.autograd.grad(out[0].square().sum(), d, retain_graph=True)
    again, = torch.autograd.grad(out[-1][0].square().sum(), d)
    assert torch.equal(first, again)
    ref = dc.clone().requires_grad_()
    old = warp_mpm_ext(ref, s, u_t=u, body_t=b)
    expected_v, = torch.autograd.grad(old[4].square().sum(), ref, retain_graph=True)
    expected_x, = torch.autograd.grad(old[0].square().sum(), ref)
    torch.testing.assert_close(velocity, expected_v, rtol=0, atol=0)
    torch.testing.assert_close(terminal, expected_x, rtol=0, atol=0)


def test_prepared_legacy_apply_clears_full_position_seed_buffers():
    s, dc, u, b = case()
    adj = PersistentAdjoint(s, position_sequence=True)
    d = dc.clone().requires_grad_()
    out = adj.apply_with_positions(d, u_t=u, body_t=b)
    torch.autograd.grad(out[-1].square().sum(), d)
    assert torch.count_nonzero(adj.sX) > 0
    old = adj.apply(d, u_t=u, body_t=b)
    assert len(old) == 5
    got, = torch.autograd.grad(old[0].square().sum(), d)
    assert torch.count_nonzero(adj.sX) == 0
    ref = dc.clone().requires_grad_()
    expected, = torch.autograd.grad(warp_mpm_ext(ref, s, u_t=u, body_t=b)[0].square().sum(), ref)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize('advance', ['positions', 'legacy', 'direct', 'failed_copy'])
def test_stale_persistent_sequence_context_rejected(advance):
    s, dc, u, b = case()
    adj = PersistentAdjoint(s, position_sequence=True)
    d = dc.clone().requires_grad_()
    first = adj.apply_with_positions(d, u_t=u, body_t=b)
    owned = first[-1].clone()
    if advance == 'positions':
        adj.apply_with_positions(d*2, u_t=u, body_t=b)
    elif advance == 'legacy':
        adj.apply(d*2, u_t=u, body_t=b)
    elif advance == 'direct':
        adj.forward()
    else:
        with pytest.raises(RuntimeError):
            adj.apply_with_positions(d*2, u_t=torch.zeros(len(s.x0)+1), body_t=b)
    assert torch.equal(first[-1], owned)
    with pytest.raises(RuntimeError, match='stale persistent forward'):
        torch.autograd.grad(first[-1].square().sum(), d)


def test_ordinary_aliasing_input_mutation_is_rejected_before_backward():
    s, dc, u, b = case()
    d = dc.clone().requires_grad_()
    out = warp_mpm_ext_with_positions(d, s, u_t=u, body_t=b)
    with torch.no_grad():
        d.add_(.01)
    with pytest.raises(RuntimeError, match='modified by an inplace operation'):
        torch.autograd.grad(out[-1].square().sum(), d)


def test_material_only_sequence_gradient_and_mutually_exclusive_modes():
    s, dc, u, b = case(T=2)
    s = replace(s, prm=replace(s.prm, dt=.04))
    dc = dc * 2
    lam = torch.full((len(s.x0),), 80., requires_grad=True)
    weights = torch.randn(s.T, len(s.x0), 3, generator=torch.Generator().manual_seed(27))
    def loss(value):
        X = warp_mpm_ext_with_positions(dc, s, lam_t=value, u_t=u, body_t=b)[-1]
        return ((X-torch.from_numpy(s.x0)).double()*weights).sum()
    fd_check(loss, lam, .5, rtol=.06, atol=2e-8)
    with pytest.raises(ValueError, match='mutually exclusive'):
        PersistentAdjoint(s, previous_position=True, position_sequence=True)
    with pytest.raises(ValueError, match='mutually exclusive'):
        _WarpMPMExt.apply(dc, None, None, u, b, s, True, True)
    with pytest.raises(ValueError, match='position_sequence=True'):
        PersistentAdjoint(s).apply_with_positions(dc, u_t=u, body_t=b)
    with pytest.raises(ValueError, match='previous_position=True'):
        PersistentAdjoint(s, position_sequence=True).apply_with_previous(dc, u_t=u, body_t=b)
    with pytest.raises(ValueError, match='position_sequence=True'):
        PersistentAdjoint(s, previous_position=True).apply_with_positions(dc, u_t=u, body_t=b)
    empty = replace(s, T=0)
    with pytest.raises(ValueError, match='T>=1'):
        warp_mpm_ext_with_positions(dc[:0], empty, u_t=u, body_t=b)
    with pytest.raises(ValueError, match='T>=1'):
        PersistentAdjoint(empty, position_sequence=True)
