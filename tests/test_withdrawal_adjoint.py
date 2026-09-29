"""Joint withdrawal ownership, boundary seeds and frozen-policy CPU contracts."""
from dataclasses import replace
from contextvars import Context, ContextVar

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.function import PersistentAdjoint
from physmorph.mpm.withdrawal import OwnedWithdrawal
from physmorph.mpm.withdrawal_adjoint import WithdrawalAdjoint
from test_position_sequence_bridge import case


def prepared(T=3):
    spec, dc, u, body = case(T=T, body=T > 1)
    nbr = spec.layer[2]
    rest = np.linalg.norm(spec.x0[:, None]-spec.x0[nbr], axis=-1).astype(np.float32)*.98
    # Nonzero F-layer coupling, permanent bonds, implicit support normalization.
    grad = np.random.default_rng(326).normal(0, .02, (*nbr.shape, 3)).astype(np.float32)
    layer = (*spec.layer[:5], grad, .8, spec.layer[7])
    prm = replace(spec.prm, gate_r_lo=.1, gate_r_hi=.8, gate_n0=0.)
    spec = replace(spec, prm=prm, layer=layer, bond_nbr=nbr.copy(), bond_rest=rest,
                   bond_frag=np.ones(len(spec.x0), np.float32), bond_threshold=100., pin_slip=True)
    return spec, (dc, u, body)


def control_leaves(values):
    return tuple(v.detach().clone().requires_grad_() if v is not None else None for v in values)


def states(tr, name):
    width = 9 if name in ('F', 'Fg') else 3
    return torch.stack([wp.to_torch(v).reshape(tr.N, width).clone() for v in getattr(tr, name)])


@pytest.mark.parametrize('capture', [False, True])
@pytest.mark.parametrize('T', [1, 3])
def test_complete_head_and_independent_zero_control_coast(capture, T):
    spec, controls = prepared(T)
    adj = WithdrawalAdjoint(spec, capture=capture)
    output = adj.apply(*controls)
    old = PersistentAdjoint(spec, position_sequence=True)
    baseline = old.apply_with_positions(controls[0], u_t=controls[1], body_t=controls[2])
    for actual, expected in zip(output[:6], baseline):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    coast = OwnedWithdrawal.capture(adj.head, T).trajectory()
    coast.rollout()
    for name in ('x', 'v', 'F', 'Fg'):
        actual = getattr(output, 'coast_'+name.upper() if name in ('x', 'v') else 'coast_'+name)
        torch.testing.assert_close(actual, states(coast, name), rtol=0, atol=0)
    for name in ('x', 'v', 'C', 'F', 'Fg'):
        assert getattr(adj.coast, name)[0] is getattr(adj.head, name)[T]
    assert adj.head.T == adj.coast.T == spec.T
    assert adj.coast.gate_n0 == adj.head.gate_n0 > 0
    assert adj.coast.body_control is None
    assert torch.count_nonzero(wp.to_torch(adj.coast._dfc(0))) == 0
    assert torch.count_nonzero(wp.to_torch(adj.coast.layer_u)) == 0
    for name in ('Fp', 'vol', 'pin', 'layer_ug', 'bond_frag'):
        assert getattr(adj.coast, name) is getattr(adj.head, name)
    # Incoming state is not reset to an identity/quiescent state.
    assert torch.count_nonzero(wp.to_torch(adj.head.C[T])) > 0
    assert torch.count_nonzero(output.v) > 0
    assert not torch.equal(output.F, torch.eye(3).reshape(1, 9).expand_as(output.F))


def test_affine_velocity_boundary_is_live_and_changes_coast():
    spec, controls = prepared()
    # This test specifically needs an active affine path. The separate full-
    # policy parity fixture retains the support gate and separating collider.
    spec = replace(spec, prm=replace(spec.prm, gate_r_lo=0., gate_r_hi=0.), pin_slip=False)
    leaves = control_leaves(controls)
    adj = WithdrawalAdjoint(spec, capture=False)
    output = adj.apply(*leaves)
    loss = output.coast_X[-1].square().sum()+.1*output.coast_V[-1].square().sum()
    grads = torch.autograd.grad(loss, leaves)
    assert all(torch.isfinite(g).all() and g.norm() > 1e-7 for g in grads)
    # Warp may consume intermediate adjoints during reverse assignment. Instead
    # construct an intentionally wrong C-only stop-gradient oracle with exactly
    # the same primal boundary, and require its control VJP to differ.
    broken = WithdrawalAdjoint(spec, capture=False)
    broken.coast.C[0] = wp.clone(adj.head.C[spec.T], requires_grad=False)
    broken_leaves = control_leaves(controls)
    wrong = broken.apply(*broken_leaves)
    assert torch.equal(wrong.coast_X, output.coast_X)
    assert torch.equal(wrong.coast_V, output.coast_V)
    wrong_loss = wrong.coast_X[-1].square().sum()+.1*wrong.coast_V[-1].square().sum()
    detached_grads = torch.autograd.grad(wrong_loss, broken_leaves)
    assert max(float((g-d).abs().max()) for g, d in zip(grads, detached_grads)) > 1e-6
    snapshot = OwnedWithdrawal.capture(adj.head, spec.T)
    arrays = snapshot.arrays()
    arrays['C0'].fill(0.)
    without_C = OwnedWithdrawal.from_arrays(arrays, snapshot.metadata()).trajectory()
    without_C.rollout()
    assert (states(without_C, 'x')-output.coast_X).abs().max() > 1e-6


def test_boundary_duplicate_seeds_equal_the_single_summed_seed():
    spec, controls = prepared()
    leaves = control_leaves(controls)
    adj = WithdrawalAdjoint(spec, capture=False)
    o = adj.apply(*leaves)
    # Exact binary weights exercise all four aliases without decimal sum noise.
    loss = (o.x+2*o.X[-1]+4*o.coast_X[0]).sum()
    loss += (o.v+2*o.V[-1]+4*o.coast_V[0]).sum()
    loss += (o.F+2*o.coast_F[0]).sum()+(o.Fg+4*o.coast_Fg[0]).sum()
    actual = torch.autograd.grad(loss, leaves)
    reference_leaves = control_leaves(controls)
    old = PersistentAdjoint(spec, position_sequence=True)
    r = old.apply_with_positions(reference_leaves[0], u_t=reference_leaves[1], body_t=reference_leaves[2])
    expected = torch.autograd.grad((7*r[0]+7*r[2]).sum()+3*r[1].sum()+5*r[3].sum(), reference_leaves)
    for g, want in zip(actual, expected):
        torch.testing.assert_close(g, want, rtol=0, atol=0)


def test_missing_repeated_and_all_zero_seeds_do_not_leak():
    spec, controls = prepared()
    leaves = control_leaves(controls)
    adj = WithdrawalAdjoint(spec, capture=False)
    o = adj.apply(*leaves)
    losses = [o.X[0].square().sum(), o.coast_F[-1].square().sum(),
              o.coast_V[1].square().sum(), o.coast_Fg[-1].square().sum(), o.x.square().sum()]
    first = torch.autograd.grad(losses[0], leaves, retain_graph=True)
    for loss in losses[1:]:
        gs = torch.autograd.grad(loss, leaves, retain_graph=True)
        assert all(torch.isfinite(g).all() for g in gs)
        assert sum(float(g.abs().sum()) for g in gs) > 1e-7
    again = torch.autograd.grad(losses[0], leaves, retain_graph=True)
    for a, b in zip(first, again):
        assert torch.equal(a, b)
    zero = torch.autograd.grad(o.coast_X.sum()*0., leaves)
    assert all(torch.count_nonzero(g) == 0 for g in zero)
    keys = [(str(a.device), a.ptr) for a in adj.grad_arrays]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize('replacement', ['apply', 'forward', 'bad_shape', 'bad_dtype', 'missing_body'])
def test_successful_or_failed_forward_invalidates_old_context(replacement):
    spec, controls = prepared()
    leaves = control_leaves(controls)
    adj = WithdrawalAdjoint(spec, capture=False)
    output = adj.apply(*leaves)
    saved = output.coast_X.clone()
    if replacement == 'apply':
        adj.apply(*controls)
    elif replacement == 'forward':
        adj.forward()
    else:
        values = list(controls)
        if replacement == 'bad_shape':
            values[1] = values[1][:-1]
        elif replacement == 'bad_dtype':
            values[0] = values[0].double()
        else:
            values[2] = None
        with pytest.raises(ValueError):
            adj.apply(*values)
    assert torch.equal(output.coast_X, saved)
    with pytest.raises(RuntimeError, match='stale'):
        torch.autograd.grad(output.coast_X.sum(), leaves)


def test_initial_inputs_and_returned_sequences_are_owned():
    spec, controls = prepared()
    adj = WithdrawalAdjoint(spec, capture=False)
    o = adj.apply(*controls)
    saved = [value.clone() for value in o]
    spec.x0.fill(3.)
    spec.C0.fill(0.)
    spec.Fp.fill(0.)
    spec.layer[7].fill(0.)
    spec.bond_frag.fill(0.)
    spec.prm.dt *= 2.
    repeated = adj.apply(*controls)
    assert all(torch.equal(a, b) for a, b in zip(repeated, saved))
    o.X.zero_()
    o.coast_X.fill_(9.)
    assert torch.equal(o.x, saved[0])
    assert torch.equal(repeated.X, saved[5])
    changed = adj.apply(controls[0]*1.3, controls[1], controls[2])
    assert not torch.equal(changed.X, saved[5])
    assert all(torch.equal(a, b) for a, b in zip(repeated, saved))


@pytest.mark.parametrize('field', ['x0', 'Fp', 'm', 'lam', 'vol0', 'layer'])
def test_differentiable_fixed_inputs_fail_closed(field):
    spec, _ = prepared()
    if field == 'layer':
        values = list(spec.layer)
        values[7] = torch.tensor(values[7], requires_grad=True)
        spec.layer = tuple(values)
    else:
        setattr(spec, field, torch.tensor(getattr(spec, field), dtype=torch.float32, requires_grad=True))
    with pytest.raises(ValueError, match='fixed initial/material/config'):
        WithdrawalAdjoint(spec, capture=False)


@pytest.mark.parametrize('field,value', [('vol0', None), ('T', 0), ('T', True), ('x0', np.zeros((0, 3), np.float32))])
def test_unsupported_dimensions_and_missing_source_volume_reject(field, value):
    spec, _ = prepared()
    setattr(spec, field, value)
    with pytest.raises(ValueError):
        WithdrawalAdjoint(spec, capture=False)


def test_no_layer_no_body_and_optional_u_preserve_gradient_contract():
    spec, controls = prepared(T=1)
    spec = replace(spec, layer=None)
    adj = WithdrawalAdjoint(spec, capture=False)
    d = controls[0].clone().requires_grad_()
    o = adj.apply(d)
    assert o.coast_X.shape == (2, len(spec.x0), 3)
    g, = torch.autograd.grad(o.coast_X[-1].square().sum(), d)
    assert torch.isfinite(g).all() and g.norm() > 1e-7
    with pytest.raises(ValueError, match='presence'):
        adj.apply(d, controls[1])


def test_optional_u_is_zero_with_layer_and_dtype_is_explicit():
    spec, controls = prepared(T=1)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        adj = WithdrawalAdjoint(spec, capture=False)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert adj.dc.dtype == adj.u.dtype == adj.body.dtype == torch.float32
    d = controls[0].clone().requires_grad_()
    missing = adj.apply(d)
    grad, = torch.autograd.grad(missing.coast_X[-1].square().sum(), d)
    assert torch.isfinite(grad).all() and grad.norm() > 1e-7
    explicit = adj.apply(controls[0], torch.zeros_like(controls[1]))
    assert all(torch.equal(a, b) for a, b in zip(missing, explicit))


def test_autograd_restores_independent_forward_context_copy_each_time(monkeypatch):
    spec, controls = prepared()
    leaves = control_leaves(controls)
    adj = WithdrawalAdjoint(spec, capture=False)
    marker = ContextVar('withdrawal_test_forward_context', default='outside')
    token = marker.set('forward')
    try:
        output = adj.apply(*leaves)
    finally:
        marker.reset(token)
    original = adj.backward
    seen = []
    def observed():
        seen.append(marker.get())
        assert marker.get() == 'forward'
        marker.set('backward_write')
        original()
    monkeypatch.setattr(adj, 'backward', observed)
    loss = output.coast_X[-1].square().sum()
    # A fresh Python context models the missing forward ContextVars. CUDA's
    # actual engine stream/device behavior remains a separate server gate.
    worker = Context()
    first = worker.run(lambda: torch.autograd.grad(loss, leaves, retain_graph=True))
    repeated = worker.run(lambda: torch.autograd.grad(loss, leaves))
    assert seen == ['forward', 'forward']
    assert marker.get() == 'outside' and worker.run(marker.get) == 'outside'
    for a, b in zip(first, repeated):
        assert torch.equal(a, b)
