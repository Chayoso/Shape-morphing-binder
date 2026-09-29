"""Same-forward two-tape handoff and independent control finite differences."""
from copy import deepcopy

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.post_assimilation_adjoint import PostAssimilationAdjoint
from physmorph.mpm.traj import Trajectory
from physmorph.plasticity.assimilation import assimilate_elastic
from test_withdrawal_adjoint_fd import coupled_case, mixed_coast_loss, direction_like


def case(*, new_pins=True):
    spec, controls = coupled_case()
    # A separating collider has additional branch surfaces; keep the existing
    # no-slip FD fixture and fixed fragment/layer policies for this first gate.
    spec.pin_slip = False
    pins = torch.as_tensor(spec.pin).bool().clone()
    if new_pins:
        pins[3] = True
    kwargs = dict(next_pins=pins, successor_layer=deepcopy(spec.layer),
                  successor_bonds=deepcopy(spec.bonds()), successor_eta=spec.eta.copy(),
                  eta=.5, isochoric=True, capture=False)
    return spec, controls, kwargs


def _tensor(array):
    return wp.to_torch(array).detach().clone()


def test_forward_matches_ordinary_assimilation_and_fresh_coast():
    spec, controls, kwargs = case()
    model = PostAssimilationAdjoint(spec, **kwargs)
    out = model.apply(*controls)
    head, coast = model.head, model.coast
    F, P = _tensor(head.F[spec.T]).numpy(), _tensor(head.Fp).numpy()
    old, pins = spec.pin.astype(bool), kwargs['next_pins'].numpy()
    new = pins & ~old
    first = assimilate_elastic(F, P, eta=.5, isochoric=True)
    first[old] = P[old]
    first[new] = assimilate_elastic(F[new], first[new], eta=1, isochoric=False)
    torch.testing.assert_close(_tensor(coast.Fp), torch.from_numpy(first), rtol=4e-6, atol=4e-6)
    v, C = _tensor(head.v[spec.T]), _tensor(head.C[spec.T])
    v[pins] = 0
    C[pins] = 0
    independent = Trajectory(x0=_tensor(head.x[spec.T]).numpy(),
        v0=v.numpy(), C0=C.numpy(), F0=F, Fg0=_tensor(head.Fg[spec.T]).numpy(),
        Fp=first, m=spec.m, lam=spec.lam, mu=spec.mu, vol0=spec.vol0,
        eta=kwargs['successor_eta'], pin=pins.astype(np.float32),
        layer=kwargs['successor_layer'], bonds=kwargs['successor_bonds'],
        prm=deepcopy(coast.prm), T=spec.T, device='cpu', requires_grad=False,
        track_geom=True, persistent=True)
    independent.rollout()
    for name, sequence in (('x', out.coast_X), ('v', out.coast_V),
                           ('F', out.coast_F), ('Fg', out.coast_Fg)):
        actual = torch.stack([_tensor(a).reshape(spec.x0.shape[0], -1) for a in getattr(independent, name)])
        torch.testing.assert_close(sequence, actual, rtol=2e-5, atol=3e-6)
    for name in ('x', 'v', 'C', 'F', 'Fg'):
        assert getattr(coast, name)[0].ptr != getattr(head, name)[spec.T].ptr
    assert coast.Fp.requires_grad and coast.Fp.grad is not None
    assert coast.Fp.ptr != head.Fp.ptr
    assert torch.equal(_tensor(coast.v[0])[pins], torch.zeros_like(v[pins]))
    assert torch.equal(_tensor(coast.C[0])[pins], torch.zeros_like(C[pins]))


@pytest.mark.parametrize('channel', [0, 1, 2], ids=['stress', 'surface_u', 'body'])
def test_future_control_finite_difference(channel):
    spec, controls, kwargs = case()
    model = PostAssimilationAdjoint(spec, **kwargs)
    out = model.apply(*controls)
    gradients = torch.autograd.grad(mixed_coast_loss(out), controls)
    assert all(torch.isfinite(g).all() and g.norm() > 1e-7 for g in gradients)
    direction = direction_like(controls[channel], channel)
    ad = float((gradients[channel].double()*direction.double()).sum())
    assert abs(ad) > 5e-5
    for epsilon in (1e-3, 5e-4):
        losses = []
        for sign in (-1, 1):
            perturbed = [v.detach().clone() for v in controls]
            perturbed[channel].add_(direction, alpha=sign*epsilon)
            losses.append(float(mixed_coast_loss(model.apply(*perturbed))))
        fd = (losses[1]-losses[0])/(2*epsilon)
        assert ad == pytest.approx(fd, rel=.02, abs=5e-6), (channel, epsilon, ad, fd)


def test_head_and_coast_seeds_add_and_repeated_zero_seed_does_not_leak():
    spec, controls, kwargs = case()
    model = PostAssimilationAdjoint(spec, **kwargs)
    out = model.apply(*controls)
    head = .01*sum(value.double().square().mean() for value in out[:6])
    coast = mixed_coast_loss(out) + .013*out.coast_X[0].sum() + .007*out.coast_F[0].sum()
    a = torch.autograd.grad(head, controls, retain_graph=True)
    b = torch.autograd.grad(coast, controls, retain_graph=True)
    combined = torch.autograd.grad(head+coast, controls, retain_graph=True)
    again = torch.autograd.grad(head+coast, controls, retain_graph=True)
    for x, y, z, repeat in zip(a, b, combined, again):
        torch.testing.assert_close(x+y, z, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(z, repeat, rtol=0, atol=0)
    zero, = torch.autograd.grad(out.coast_F.sum()*0, controls[:1])
    assert torch.count_nonzero(zero) == 0


def test_owned_policies_output_and_stale_generation():
    spec, controls, kwargs = case()
    model = PostAssimilationAdjoint(spec, **kwargs)
    out = model.apply(*controls)
    snapshot = tuple(value.detach().clone() for value in out)
    kwargs['next_pins'].fill_(True)
    kwargs['successor_eta'].fill(100.)
    kwargs['successor_layer'][0].fill(0.)
    model.apply(*(value.detach()*0 for value in controls))
    for value, owned in zip(out, snapshot):
        assert torch.equal(value, owned)
    assert int(model.next_pins.sum()) == 3
    assert torch.max(_tensor(model.coast.eta)) < 100
    assert torch.max(_tensor(model.coast.layer_mask)) > 0
    with pytest.raises(RuntimeError, match='stale'):
        torch.autograd.grad(out.coast_X.sum(), controls[0])
    out = model.apply(*controls)
    with pytest.raises(ValueError, match='float32'):
        model.apply(controls[0].double(), *controls[1:])
    with pytest.raises(RuntimeError, match='stale'):
        torch.autograd.grad(out.coast_X.sum(), controls[0])


def test_no_assimilation_no_pin_change_reduces_to_existing_bridge():
    from physmorph.mpm.withdrawal_adjoint import WithdrawalAdjoint
    spec, controls, kwargs = case(new_pins=False)
    # The ordinary runner still zeroes existing pins' carried v/C at handoff.
    spec.v0[spec.pin.astype(bool)] = 0
    spec.C0[spec.pin.astype(bool)] = 0
    kwargs.update(eta=0, settle_pin_assim=False)
    old = WithdrawalAdjoint(spec, capture=False)
    new = PostAssimilationAdjoint(spec, **kwargs)
    before = old.apply(*controls)
    after = new.apply(*controls)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    a = torch.autograd.grad(mixed_coast_loss(before), controls)
    b = torch.autograd.grad(mixed_coast_loss(after), controls)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=2e-5, atol=2e-6)


def test_released_pins_rejected():
    spec, _, kwargs = case()
    kwargs['next_pins'][0] = False
    with pytest.raises(RuntimeError, match='release'):
        PostAssimilationAdjoint(spec, **kwargs)


@pytest.mark.parametrize('drop', ['Fp', 'C'])
def test_omitted_boundary_paths_break_the_control_derivative(drop, monkeypatch):
    spec, controls, kwargs = case()
    model = PostAssimilationAdjoint(spec, **kwargs)
    def observation(out):
        if drop == 'C':
            return mixed_coast_loss(out)
        ids = torch.arange(out.coast_V.shape[1]*3, dtype=torch.float64).reshape(-1, 3)
        return ((out.coast_V[1].double()-out.coast_V[0].double())*(ids*.43+.2).sin()).mean()/spec.prm.dt
    full = torch.autograd.grad(observation(model.apply(*controls)), controls)
    assert _tensor(model.coast.Fp.grad)[model.new_pins].norm() > 1e-7
    with monkeypatch.context() as patch:
        if drop == 'Fp':
            patch.setattr(model, '_boundary_vjp', lambda: torch.zeros_like(model.boundary_F))
        else:
            combine = model._head_covectors
            def without_C(fp_to_F):
                model.coast.C[0].grad.zero_()
                combine(fp_to_F)
            patch.setattr(model, '_head_covectors', without_C)
        broken = torch.autograd.grad(observation(model.apply(*controls)), controls)
    witnesses = []
    for channel, (actual, incomplete) in enumerate(zip(full, broken)):
        direction = direction_like(controls[channel], channel)
        ad = float((actual.double()*direction.double()).sum())
        bad = float((incomplete.double()*direction.double()).sum())
        step = 5e-4
        losses = []
        for sign in (-1, 1):
            shifted = [value.detach().clone() for value in controls]
            shifted[channel].add_(direction, alpha=sign*step)
            losses.append(float(observation(model.apply(*shifted))))
        fd = (losses[1]-losses[0])/(2*step)
        allowance = max(.02*abs(fd), 5e-6)
        assert abs(ad-fd) <= allowance
        print(dict(drop=drop, channel=channel, full=ad, omitted=bad, fd=fd))
        witnesses.append(abs(bad-fd) > allowance)
    assert any(witnesses), f'{drop} omission did not break an independently measured direction'
