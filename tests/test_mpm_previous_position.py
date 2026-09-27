"""Opt-in previous-position bridge contracts, Warp CPU only."""
from dataclasses import replace

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.function import (
    PersistentAdjoint, RolloutSpec, warp_mpm_ext, warp_mpm_ext_with_previous,
)
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory


def case(T=3, *, layer=False, body=0, pin=False, elastic=True):
    x = np.random.default_rng(292).uniform(-.65, .65, (24, 3)).astype(np.float32)
    n = len(x)
    prm = MPMParams(dx=.5, dt=.01, grid_min=(-3., -3., -3.), nx=12, ny=12, nz=12,
                    drag=0., smoothing=.955)
    layer_args = None
    if layer:
        normals = np.tile([0., 1., 0.], (n, 1)).astype(np.float32)
        layer_args = (np.ones(n, np.float32), normals, np.arange(n, dtype=np.int32)[:, None],
                      np.ones((n, 1), np.float32), 0.)
    pins = (np.arange(n) % 4 == 0).astype(np.float32) if pin else None
    s = RolloutSpec(x, 1., 80. if elastic else 0., 40. if elastic else 0., prm, T,
                    device='cpu', vol0=np.full(n, .02, np.float32), layer=layer_args,
                    pin=pins, body_ctrl=bool(body), body_modes=body or 1)
    dc = .01 * torch.randn(T, n, 3, 3, generator=torch.Generator().manual_seed(71))
    return s, dc


def bridge(kind, s):
    if kind == 'ordinary':
        return lambda d, **kw: warp_mpm_ext_with_previous(d, s, **kw)
    adj = PersistentAdjoint(s, previous_position=True)
    return lambda d, **kw: adj.apply_with_previous(d, **kw)


def direction_fd(loss, leaf, eps=2e-3, *, rtol=.035, atol=2e-6):
    value = loss(leaf)
    grad, = torch.autograd.grad(value, leaf)
    assert torch.isfinite(grad).all() and grad.abs().max() > 5 * atol
    # Strongest coordinate avoids a fortuitously near-zero directional derivative.
    index = int(grad.abs().argmax())
    plus, minus = leaf.detach().clone(), leaf.detach().clone()
    plus.reshape(-1)[index] += eps
    minus.reshape(-1)[index] -= eps
    with torch.no_grad():
        fd = (loss(plus) - loss(minus)) / (2 * eps)
    assert float(grad.reshape(-1)[index]) == pytest.approx(float(fd), rel=rtol, abs=atol)
    return grad


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
def test_owned_previous_matches_actual_trajectory_and_legacy_outputs(kind):
    s, dc = case()
    run = bridge(kind, s)
    old = warp_mpm_ext(dc, s)
    new = run(dc)
    assert len(old) == 5 and len(new) == 6
    for a, b in zip(old, new[:5]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    seq = [wp.from_torch(dc[t].contiguous(), dtype=wp.mat33) for t in range(s.T)]
    tr = Trajectory(s.x0, s.m, s.lam, s.mu, s.prm, s.T, dFc=seq,
                    device='cpu', requires_grad=False, vol0=s.vol0, track_geom=True)
    tr.rollout()
    torch.testing.assert_close(new[-1], wp.to_torch(tr.x[s.T-1]), rtol=0, atol=0)
    saved = tuple(t.detach().clone() for t in new)
    run(dc * 1.4)
    for a, b in zip(new, saved):
        assert torch.equal(a, b)  # outputs do not alias reusable trajectory arrays.


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
@pytest.mark.parametrize('loss_kind', ['previous', 'raw_step'])
def test_sequence_control_previous_and_step_finite_difference(kind, loss_kind):
    s, dc = case(T=3)
    s = replace(s, prm=replace(s.prm, dt=.03))
    dc = dc * 5
    run = bridge(kind, s)
    weights = torch.randn(len(s.x0), 3, generator=torch.Generator().manual_seed(3))
    x0 = torch.from_numpy(s.x0)
    def loss(d):
        out = run(d)
        if loss_kind == 'previous':
            return ((out[-1]-x0) * weights).sum()
        return ((out[0]-out[-1]) / s.prm.dt).square().sum()
    grad = direction_fd(loss, dc.requires_grad_())
    if loss_kind == 'previous':
        assert torch.count_nonzero(grad[-1]) == 0  # last-step control cannot change x[T-1].
        assert grad[:-1].abs().max() > 1e-6


def test_previous_material_only_leaf_finite_difference():
    s, dc = case(T=2)
    s = replace(s, prm=replace(s.prm, dt=.04))
    dc = dc * 5
    weights = torch.randn(len(s.x0), 3, generator=torch.Generator().manual_seed(4))
    def loss(lam):
        out = warp_mpm_ext_with_previous(dc, s, lam_t=lam)
        return ((out[-1]-torch.from_numpy(s.x0)) * weights).sum()
    leaf = torch.full((len(s.x0),), 80., requires_grad=True)
    direction_fd(loss, leaf, eps=.5, rtol=.06, atol=2e-7)


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
@pytest.mark.parametrize('modes', [1, 2])
def test_body_previous_finite_difference_and_pins(kind, modes):
    s, dc = case(T=2, body=modes, pin=True)
    run = bridge(kind, s)
    body = (.001 * torch.randn(modes * len(s.x0), 3,
                              generator=torch.Generator().manual_seed(6))).requires_grad_()
    weights = torch.randn(len(s.x0), 3, generator=torch.Generator().manual_seed(7))
    def loss(b):
        out = run(dc, body_t=b)
        mask = torch.from_numpy(s.pin > .5)
        assert torch.equal(out[-1][mask], torch.from_numpy(s.x0)[mask])
        assert torch.equal(out[0][mask], torch.from_numpy(s.x0)[mask])
        assert torch.count_nonzero(out[2][mask]) == 0
        return ((out[-1]-torch.from_numpy(s.x0))*weights).sum()
    direction_fd(loss, body, eps=2e-4, rtol=.015, atol=3e-4)


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
def test_layer_previous_finite_difference_and_pins(kind):
    s, dc = case(T=2, layer=True, pin=True)
    run = bridge(kind, s)
    u = torch.linspace(-.001, .001, len(s.x0)).requires_grad_()
    weights = torch.linspace(.5, 1.5, len(s.x0))
    def loss(value):
        out = run(dc, u_t=value)
        mask = torch.from_numpy(s.pin > .5)
        assert torch.equal(out[-1][mask], torch.from_numpy(s.x0)[mask])
        return ((out[-1]-torch.from_numpy(s.x0))[:, 1]*weights).sum()
    grad = direction_fd(loss, u, eps=2e-4, rtol=.015)
    assert torch.count_nonzero(grad[torch.from_numpy(s.pin > .5)]) == 0


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
def test_constant_geometric_velocity_exposes_detached_previous_error(kind):
    s, dc = case(T=2, layer=True, elastic=False)
    s = replace(s, v0=np.tile([0., .1, 0.], (len(s.x0), 1)).astype(np.float32))
    dc.zero_()
    out = bridge(kind, s)(dc, u_t=torch.full((len(s.x0),), .002, requires_grad=True))
    # Uniform u adds .001 each step, so raw geometric velocity is .2, stored v=.1.
    v_geom = (out[0]-out[-1])/s.prm.dt
    torch.testing.assert_close(v_geom[:, 1], torch.full((len(s.x0),), .2), rtol=0, atol=8e-6)
    torch.testing.assert_close(out[2][:, 1], torch.full((len(s.x0),), .1), rtol=0, atol=1e-6)
    # A single shared amplitude gives an exact derivative 1/(T*dt), not 1/dt.
    amplitude = torch.tensor(.002, requires_grad=True)
    out = bridge(kind, s)(dc, u_t=amplitude.expand(len(s.x0)).contiguous())
    good, = torch.autograd.grad(((out[0]-out[-1])[:, 1]/s.prm.dt).mean(), amplitude,
                                retain_graph=True)
    bad, = torch.autograd.grad(((out[0]-out[-1].detach())[:, 1]/s.prm.dt).mean(), amplitude)
    assert float(good) == pytest.approx(1/(s.T*s.prm.dt), rel=2e-5)
    assert float(bad) == pytest.approx(s.T*float(good), rel=2e-5)


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
def test_t1_previous_is_owned_constant_and_terminal_gradient_matches_legacy(kind):
    s, dc = case(T=1)
    d = dc.requires_grad_()
    out = bridge(kind, s)(d)
    assert not out[-1].requires_grad
    assert torch.equal(out[-1], torch.from_numpy(s.x0))
    assert out[-1].data_ptr() != torch.from_numpy(s.x0).data_ptr()
    g, = torch.autograd.grad((out[0]-out[-1]).square().sum(), d)
    ref = dc.detach().clone().requires_grad_()
    legacy = warp_mpm_ext(ref, s)
    gold, = torch.autograd.grad((legacy[0]-torch.from_numpy(s.x0)).square().sum(), ref)
    torch.testing.assert_close(g, gold, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['ordinary', 'persistent'])
def test_repeated_backward_and_missing_previous_seed(kind):
    s, dc = case(T=2)
    run = bridge(kind, s)
    d = dc.requires_grad_()
    out = run(d)
    w = torch.randn(len(s.x0), 3, generator=torch.Generator().manual_seed(5))
    gprev, = torch.autograd.grad((out[-1]*w).sum(), d, retain_graph=True)
    gx, = torch.autograd.grad((out[0]*w).sum(), d, retain_graph=True)
    gv, = torch.autograd.grad(out[4].square().sum(), d, retain_graph=True)
    gprev_again, = torch.autograd.grad((out[-1]*w).sum(), d)
    assert torch.equal(gprev, gprev_again)
    ref = dc.detach().clone().requires_grad_()
    plain = warp_mpm_ext(ref, s)
    refx, = torch.autograd.grad((plain[0]*w).sum(), ref, retain_graph=True)
    refv, = torch.autograd.grad(plain[4].square().sum(), ref)
    torch.testing.assert_close(gx, refx, rtol=0, atol=0)
    torch.testing.assert_close(gv, refv, rtol=0, atol=0)


def test_prepared_legacy_apply_keeps_five_outputs_and_clears_extra_seed():
    s, dc = case(T=2)
    adj = PersistentAdjoint(s, previous_position=True)
    d = dc.requires_grad_()
    previous = adj.apply_with_previous(d)[-1]
    torch.autograd.grad(previous.square().sum(), d)
    assert torch.count_nonzero(adj.sx_previous) > 0
    out = adj.apply(d)
    assert len(out) == 5
    g, = torch.autograd.grad(out[0].square().sum(), d)
    assert torch.count_nonzero(adj.sx_previous) == 0
    ref = dc.detach().clone().requires_grad_()
    gold, = torch.autograd.grad(warp_mpm_ext(ref, s)[0].square().sum(), ref)
    torch.testing.assert_close(g, gold, rtol=0, atol=0)


@pytest.mark.parametrize('advance', ['new_api', 'legacy_api', 'direct_forward'])
def test_stale_persistent_context_rejected_but_outputs_stay_owned(advance):
    s, dc = case(T=2)
    adj = PersistentAdjoint(s, previous_position=True)
    d = dc.requires_grad_()
    first = adj.apply_with_previous(d)
    saved = tuple(t.detach().clone() for t in first)
    if advance == 'new_api':
        adj.apply_with_previous(d * 2)
    elif advance == 'legacy_api':
        adj.apply(d * 2)
    else:
        adj.forward()
    for a, b in zip(first, saved):
        assert torch.equal(a, b)
    with pytest.raises(RuntimeError, match='stale persistent forward'):
        torch.autograd.grad(first[-1].square().sum(), d)


def test_previous_mode_must_be_prepared_and_t0_rejected():
    s, dc = case(T=2)
    adj = PersistentAdjoint(s)
    assert adj.sx_previous is None
    assert len(adj.apply(dc)) == 5
    with pytest.raises(ValueError, match='previous_position=True'):
        adj.apply_with_previous(dc)
    s0 = replace(s, T=0)
    with pytest.raises(ValueError, match='T>=1'):
        warp_mpm_ext_with_previous(dc[:0], s0)
    with pytest.raises(ValueError, match='T>=1'):
        PersistentAdjoint(s0, previous_position=True)


def test_failed_input_copy_invalidates_previous_persistent_context():
    s, dc = case(T=2, layer=True)
    adj = PersistentAdjoint(s, previous_position=True)
    d = dc.requires_grad_()
    first = adj.apply_with_previous(d, u_t=torch.zeros(len(s.x0)))
    # dFc is already copied when the later invalid u shape fails, before rollout.
    with pytest.raises(RuntimeError):
        adj.apply_with_previous(d * 2, u_t=torch.zeros(len(s.x0) + 1))
    with pytest.raises(RuntimeError, match='stale persistent forward'):
        torch.autograd.grad(first[-1].square().sum(), d)
