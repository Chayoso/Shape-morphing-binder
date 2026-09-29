"""Independent directional derivatives of the complete head-to-coast path."""
from copy import deepcopy

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.function import RolloutSpec
from test_boundary_packet import _trajectory


def coupled_case():
    """Prepared CPU fixture; numerical CUDA tests upload it explicitly."""
    tr = _trajectory(pin_slip=True)
    def own(value):
        return wp.to_torch(value).detach().clone().numpy().copy()
    layer = (own(tr.layer_mask), own(tr.layer_nrm), own(tr.layer_nbr).reshape(tr.N, -1),
             own(tr.layer_w).reshape(tr.N, -1), tr.layer_frac,
             own(tr.layer_g).reshape(tr.N, tr.layer_K, 3), 1. / tr.layer_inv_depth,
             own(tr.layer_ug))
    prm = deepcopy(tr.prm)
    prm.gate_n0 = tr.gate_n0
    spec = RolloutSpec(x0=own(tr.x[0]), m=own(tr.m), lam=own(tr.lam), mu=own(tr.mu),
        prm=prm, T=tr.T, Fp=own(tr.Fp), v0=own(tr.v[0]), F0=own(tr.F[0]), C0=own(tr.C[0]),
        device='cpu', vol0=own(tr.vol), Fg0=own(tr.Fg[0]),
        bond_nbr=own(tr.bond_nbr).reshape(tr.N, -1), bond_rest=own(tr.bond_rest).reshape(tr.N, -1),
        bond_frag=own(tr.bond_frag), bond_threshold=tr.frag_thr,
        layer=layer, eta=own(tr.eta), pin=own(tr.pin), pin_slip=True, body_ctrl=True, body_modes=2)
    controls = (torch.stack([wp.to_torch(v).clone() for v in tr.dFc_seq]),
                wp.to_torch(tr.layer_u).clone(), wp.to_torch(tr.body_control).clone())
    return spec, tuple(value.detach().requires_grad_() for value in controls)


def mixed_coast_loss(output):
    """Future-only observable; every derivative must cross the shared boundary."""
    n = output.coast_X.shape[1]
    device = output.coast_X.device
    idx = torch.arange(n * 3, device=device, dtype=torch.float64).reshape(n, 3)
    weights = (idx * .43 + .2).sin()
    positions = output.coast_X.double()
    loss = ((positions[-1] - positions[0]) * weights).sum() / n
    loss = loss + .007 * (output.coast_V[-1].double() * (weights + .3)).sum() / n
    fm = torch.arange(n * 9, device=device, dtype=torch.float64).reshape(n, 9)
    loss = loss + .011 * (output.coast_F[-1].double() * (fm * .23).cos()).sum() / n
    loss = loss + .013 * (output.coast_Fg[-1].double() * (fm * .17).sin()).sum() / n
    return loss


def direction_like(value, channel):
    # Deterministic without modifying a global RNG or picking an AD-dependent direction.
    idx = torch.arange(value.numel(), device=value.device, dtype=torch.float64).reshape(value.shape)
    direction = ((idx + 1) * (.29 + channel * .13)).sin().to(value.dtype)
    return direction / direction.square().mean().sqrt()


@pytest.mark.parametrize('channel', [0, 1, 2], ids=['stress', 'surface_u', 'body'])
def test_future_state_directional_derivative_crosses_every_control_channel(channel):
    from physmorph.mpm.withdrawal_adjoint import WithdrawalAdjoint
    spec, controls = coupled_case()
    model = WithdrawalAdjoint(spec, capture=False)
    output = model.apply(*controls)
    derivatives = torch.autograd.grad(mixed_coast_loss(output), controls)
    assert all(torch.isfinite(g).all() and g.norm() > 1e-7 for g in derivatives)
    direction = direction_like(controls[channel], channel)
    analytical = (derivatives[channel].double() * direction.double()).sum().item()
    # Contact body-control bracket was refined after the initial 1e-3/5e-4
    # finite-radius discrepancy (active-set changes are not measured). Both radii pass the same
    # tolerance; preserve the large-stencil no-slip control below (P326 log).
    radii = (1e-4, 5e-5) if channel == 2 else (1e-3, 5e-4)
    for epsilon in radii:
        losses = []
        for sign in (-1., 1.):
            perturbed = [value.detach().clone() for value in controls]
            perturbed[channel].add_(direction, alpha=sign * epsilon)
            losses.append(mixed_coast_loss(model.apply(*perturbed)).item())
        observed = (losses[1] - losses[0]) / (2 * epsilon)
        assert analytical == pytest.approx(observed, rel=.02, abs=5e-4), (channel, epsilon, analytical, observed)


def test_body_large_stencil_without_pinned_separating_collider():
    from physmorph.mpm.withdrawal_adjoint import WithdrawalAdjoint
    spec, controls = coupled_case()
    spec.pin_slip = False
    model = WithdrawalAdjoint(spec, capture=False)
    derivative, = torch.autograd.grad(mixed_coast_loss(model.apply(*controls)), controls[2])
    direction = direction_like(controls[2], 2)
    analytical = (derivative.double() * direction.double()).sum().item()
    for epsilon in (1e-3, 5e-4):
        losses = []
        for sign in (-1., 1.):
            perturbed = [value.detach().clone() for value in controls]
            perturbed[2].add_(direction, alpha=sign * epsilon)
            losses.append(mixed_coast_loss(model.apply(*perturbed)).item())
        assert analytical == pytest.approx((losses[1] - losses[0]) / (2*epsilon), rel=.02, abs=5e-4)


def test_head_only_derivative_is_unchanged_by_attaching_coast():
    from physmorph.mpm.function import PersistentAdjoint
    from physmorph.mpm.withdrawal_adjoint import WithdrawalAdjoint
    spec, controls = coupled_case()
    old = PersistentAdjoint(spec, position_sequence=True)
    old_out = old.apply_with_positions(*controls)
    old_loss = sum(value.double().square().mean() * (i + 1) * .01 for i, value in enumerate(old_out))
    want = torch.autograd.grad(old_loss, controls)
    new = WithdrawalAdjoint(spec, capture=False)
    new_out = new.apply(*controls)
    new_loss = sum(value.double().square().mean() * (i + 1) * .01 for i, value in enumerate(new_out[:6]))
    actual = torch.autograd.grad(new_loss, controls)
    for got, expected in zip(actual, want):
        torch.testing.assert_close(got, expected, rtol=3e-6, atol=3e-7)


def test_combined_boundary_seeds_equal_one_seed_per_shared_state():
    from physmorph.mpm.withdrawal_adjoint import WithdrawalAdjoint
    spec, controls = coupled_case()
    model = WithdrawalAdjoint(spec, capture=False)
    out = model.apply(*controls)
    # Binary-exact coefficients isolate duplicate-seed logic from the different
    # decimal rounding in (a+b+c) versus a single precomputed coefficient.
    multi = (.25 * out.x.sum() + .25 * out.X[-1].sum() + .5 * out.coast_X[0].sum()
             + .25 * out.v.sum() + .5 * out.V[-1].sum() + 1. * out.coast_V[0].sum()
             + .75 * out.F.sum() + 1. * out.coast_F[0].sum()
             + 1. * out.Fg.sum() + 1.25 * out.coast_Fg[0].sum())
    combined = torch.autograd.grad(multi, controls, retain_graph=True)
    single = out.x.sum() + 1.75 * out.v.sum() + 1.75 * out.F.sum() + 2.25 * out.Fg.sum()
    expected = torch.autograd.grad(single, controls, retain_graph=True)
    # A third seed set must not contain the prior simultaneous seeds.
    again = torch.autograd.grad(single, controls)
    for got, want, repeated in zip(combined, expected, again):
        torch.testing.assert_close(got, want, rtol=3e-6, atol=3e-7)
        torch.testing.assert_close(repeated, want, rtol=0, atol=0)
