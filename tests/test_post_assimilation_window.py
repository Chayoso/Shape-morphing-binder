"""Reduced post-assimilation adapter: CPU MPM and independent handoff evidence."""
from copy import deepcopy

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.traj import Trajectory
from physmorph.mpm.withdrawal import OwnedWithdrawal
from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.post_assimilation_window import PostAssimilationWindow
from physmorph.plasticity.assimilation import assimilate_elastic
from test_frozen_withdrawal_window import make_owner, coast_observable
from test_withdrawal_adjoint_fd import direction_like


def make_case():
    """N27/T20/dt.002/dx.5/grid16^3, no-slip/gate-off, actual new pin3."""
    owner = make_owner()
    spec = owner.spec
    spec.pin_slip = False
    spec.prm.gate_r_lo = spec.prm.gate_r_hi = 0.
    spec.layer = (*spec.layer[:5], None, 0., spec.layer[7])
    cfg = PipelineConfig(T=spec.T, body_ctrl=True, body_terminal_ctrl=True,
                         phys_loss='ot_pace', loss_units='density', device='cpu',
                         assim=.5, assim_iso=True, settle_pin_assim=True)
    out = owner.evaluate(owner.coefficients[:, 3:])
    head = owner.adjoint.traj
    F, P = out['F'].detach().reshape(-1, 3, 3).numpy(), spec.Fp.copy()
    old = spec.pin.astype(bool)
    pins = old.copy(); pins[3] = True
    Fp = assimilate_elastic(F, P, eta=cfg.assim, isochoric=cfg.assim_iso)
    Fp[old] = P[old]
    Fp[pins & ~old] = assimilate_elastic(F[pins & ~old], Fp[pins & ~old], eta=1., isochoric=False)
    v, C = out['v'].detach().numpy().copy(), out['C'].detach().numpy().copy()
    v[pins], C[pins] = 0., 0.
    layer = deepcopy(spec.layer)
    layer[0][3] = 0.  # Actual successor preparation is supplied, not recomputed.
    successor = Trajectory(x0=out['x'].detach().numpy(), v0=v, C0=C, F0=F, Fp=Fp,
        Fg0=wp.to_torch(head.Fg[spec.T]).detach().numpy(), m=spec.m, lam=spec.lam, mu=spec.mu,
        vol0=spec.vol0, eta=spec.eta*1.1, pin=pins.astype(np.float32),
        layer=layer, bonds=spec.bonds(), prm=deepcopy(spec.prm), T=spec.T,
        track_geom=True, persistent=True, requires_grad=False, device='cpu')
    return owner, OwnedWithdrawal.capture(successor, 0), cfg


def future_velocity(values):
    """Future velocity avoids subtracting almost equal FP32 position snapshots."""
    velocity = values['coast_V'][-1]
    ids = torch.arange(velocity.numel(), device=velocity.device, dtype=torch.float64).reshape_as(velocity)
    return (velocity.double()*((ids*.43+.2).sin()+.3)).sum()/len(velocity)


def test_original_head_basis_and_actual_subset_assimilation_coast_closure():
    owner, successor, cfg = make_case()
    model = PostAssimilationWindow(owner, successor, cfg, capture=False)
    values = model.evaluate(owner.coefficients[:, 3:])
    original = owner.evaluate(owner.coefficients[:, 3:])
    assert values['valid'] and values['pins_exact'] and values['health']['same_forward']
    for key in ('x', 'F', 'C', 'v', 'positions', 'V', 'body_energy'):
        assert torch.equal(values[key], original[key]), key
    a = successor.arrays()
    torch.testing.assert_close(values['coast_Fp'], torch.from_numpy(a['Fp']), rtol=4e-6, atol=4e-6)
    assert not values['coast_Fp'].requires_grad and not values['coast_pins'].requires_grad
    assert torch.equal(values['coast_pins'], torch.from_numpy(a['pin']).bool())
    assert torch.equal(values['coast_X'][0], values['x'])
    assert torch.max(torch.abs(values['x'][3]-torch.from_numpy(owner.spec.x0[3]))) > 1e-6
    assert torch.equal(values['coast_X'][:, 3], values['x'][3].expand(cfg.T+1, -1))
    assert torch.count_nonzero(values['v'][3]) > 0
    assert torch.count_nonzero(values['coast_V'][:, 3]) == 0
    assert torch.count_nonzero(values['coast_C'][:, 3]) == 0
    independent = successor.trajectory(persistent=True)
    independent.rollout()
    for name, key in (('x', 'coast_X'), ('v', 'coast_V'), ('F', 'coast_F'), ('Fg', 'coast_Fg'), ('C', 'coast_C')):
        want = torch.stack([wp.to_torch(value).clone() for value in getattr(independent, name)]).reshape_as(values[key])
        torch.testing.assert_close(values[key], want, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize('mode', [0, 1], ids=['displacement', 'terminal'])
def test_both_reduced_modes_have_resolved_post_assimilation_fd(mode):
    owner, successor, cfg = make_case()
    model = PostAssimilationWindow(owner, successor, cfg, capture=False)
    leaves = [owner.coefficients[:, :3].clone().requires_grad_(), owner.coefficients[:, 3:].clone().requires_grad_()]
    out = model.evaluate(leaves[1], leaves[0])
    gradients = torch.autograd.grad(future_velocity(out), leaves)
    assert all(torch.isfinite(value).all() and value.norm() > 1e-5 for value in gradients)
    direction = direction_like(leaves[mode], mode)
    ad = float((gradients[mode].double()*direction.double()).sum())
    assert abs(ad) > 5e-5
    for epsilon in (1e-3, 5e-4):
        losses = []
        for sign in (-1, 1):
            candidate = [value.detach().clone() for value in leaves]
            candidate[mode].add_(direction, alpha=sign*epsilon)
            value = model.evaluate(candidate[1], candidate[0])
            assert value['valid']
            losses.append(float(future_velocity(value)))
        fd = (losses[1]-losses[0])/(2*epsilon)
        assert ad == pytest.approx(fd, rel=.02, abs=5e-6), (mode, epsilon, ad, fd)


def test_policy_ownership_and_changed_trials_do_not_require_saved_basepoint_equality():
    owner, successor, cfg = make_case()
    model = PostAssimilationWindow(owner, successor, cfg, capture=False)
    terminal = owner.coefficients[:, 3:].clone()
    first = model.evaluate(terminal)
    saved = {key: value.clone() for key, value in first.items() if torch.is_tensor(value)}
    successor._arrays['pin'].fill(0)
    successor._arrays['eta'].fill(100)
    successor._arrays['layer_mask'].fill(0)
    cfg.assim = 0
    owner.spec.x0.fill(4); owner.stress.fill_(1); owner.gate.zero_()
    repeated = model.evaluate(terminal)
    for key, value in saved.items():
        assert torch.equal(repeated[key], value), key
    changed = model.evaluate(terminal*1.5)
    assert changed['valid'] and not torch.equal(changed['coast_X'], saved['coast_X'])
    for key, value in saved.items():
        assert torch.equal(first[key], value), key
    first['coast_Fp'].zero_(); first['coast_pins'].zero_()
    assert torch.equal(model.evaluate(terminal)['coast_Fp'], saved['coast_Fp'])


@pytest.mark.parametrize('expiration', ['owner', 'adapter', 'replacement', 'invalid'])
@pytest.mark.parametrize('key', ['coast_X', 'body_energy'])
def test_gradient_and_evaluation_leases(expiration, key):
    owner, successor, cfg = make_case()
    model = PostAssimilationWindow(owner, successor, cfg, capture=False)
    terminal = owner.coefficients[:, 3:].clone().requires_grad_()
    out = model.evaluate(terminal)
    saved = out[key].detach().clone()
    if expiration in ('owner', 'adapter'):
        (owner if expiration == 'owner' else model).close()
        with pytest.raises(RuntimeError, match='expired'):
            model.evaluate(terminal)
    elif expiration == 'replacement':
        model.evaluate(terminal.detach()*1.1)
    else:
        with pytest.raises(ValueError):
            model.evaluate(terminal[:-1])
    with pytest.raises(RuntimeError, match='expired|stale'):
        torch.autograd.grad(out[key].sum(), terminal)
    assert torch.equal(out[key], saved)


@pytest.mark.parametrize('corruption', ['head_old_pin', 'coast_new_pin', 'coast_new_v0',
                                       'coast_new_C0', 'coast_Fp_nan', 'coast_Fp_flip'])
def test_health_separates_old_head_pins_and_new_coast_boundary(monkeypatch, corruption):
    owner, successor, cfg = make_case()
    model = PostAssimilationWindow(owner, successor, cfg, capture=False)
    terminal = owner.coefficients[:, 3:]
    assert model.evaluate(terminal)['valid']
    original = model.adjoint.apply
    def corrupt(*args):
        out = original(*args)
        with torch.no_grad():
            if corruption == 'head_old_pin': out.X[0, 0, 0] += .001
            elif corruption == 'coast_new_pin': out.coast_X[1, 3, 0] += .001
            elif corruption == 'coast_new_v0': out.coast_V[0, 3, 0] = .001
            elif corruption == 'coast_new_C0': wp.to_torch(model.adjoint.coast.C[0])[3, 0, 0] = .001
            elif corruption == 'coast_Fp_nan': wp.to_torch(model.adjoint.coast.Fp)[3, 0, 0] = float('nan')
            else: wp.to_torch(model.adjoint.coast.Fp)[3] = torch.diag(torch.tensor([-1., 1., 1.]))
        return out
    monkeypatch.setattr(model.adjoint, 'apply', corrupt)
    out = model.evaluate(terminal)
    assert not out['valid']
    if corruption == 'head_old_pin':
        assert not out['health']['head_valid'] and out['health']['coast_valid']
    else:
        assert out['health']['head_valid'] and not out['health']['coast_valid']


@pytest.mark.parametrize('mismatch', ['m', 'lam', 'mu', 'vol', 'step', 'T', 'prm',
                                    'pin_mode', 'release', 'nonbinary', 'layer_F', 'gate'])
def test_incompatible_successor_rejected(mismatch):
    owner, successor, cfg = make_case()
    arrays, meta = successor.arrays(), successor.metadata()
    if mismatch in ('m', 'lam', 'mu', 'vol'): arrays[mismatch][2] *= 1.01
    elif mismatch == 'step': meta['step'] = 1
    elif mismatch == 'T': meta['T'] += 1
    elif mismatch == 'prm': meta['prm']['dt'] *= 2
    elif mismatch == 'pin_mode': meta['pin_mode'] = 1
    elif mismatch == 'release': arrays['pin'][0] = 0
    elif mismatch == 'nonbinary': arrays['pin'][3] = .5
    elif mismatch == 'layer_F':
        meta.update(layer_F=True, layer_inv_depth=1.)
        arrays['layer_g'] = np.zeros((meta['N'], meta['layer_K'], 3), np.float32)
    else:
        owner.spec.prm.gate_r_hi = meta['prm']['gate_r_hi'] = .8
        meta.update(gate=True, gate_n0=1.)
    bad = OwnedWithdrawal.from_arrays(arrays, meta)
    with pytest.raises(ValueError):
        PostAssimilationWindow(owner, bad, cfg, capture=False)


@pytest.mark.parametrize('policy', ['w_grow', 'assim_consensus', 'freeze_arrived', 'layer_F',
                                  'settle_pin_follow', 'settle_pin_yield', 'settle_pin_kkt',
                                  'commit_pic', 'shift_sub', 'rest_commit', 'settle_commit'])
def test_unsupported_commit_policies_rejected(policy):
    owner, successor, cfg = make_case()
    setattr(cfg, policy, True)
    with pytest.raises(ValueError, match='support'):
        PostAssimilationWindow(owner, successor, cfg, capture=False)


@pytest.mark.parametrize('name', ['m', 'lam', 'mu'])
def test_scalar_material_matches_the_actual_broadcast_and_still_rejects_changes(name):
    owner, successor, cfg = make_case()
    arrays, meta = successor.arrays(), successor.metadata()
    value = float(arrays[name][0])
    arrays[name].fill(value)
    setattr(owner.spec, name, value)
    successor = OwnedWithdrawal.from_arrays(arrays, meta)
    model = PostAssimilationWindow(owner, successor, cfg, capture=False)
    assert model.evaluate(owner.coefficients[:, 3:])['valid']
    setattr(owner.spec, name, value*1.01)
    with pytest.raises(ValueError, match='material'):
        PostAssimilationWindow(owner, successor, cfg, capture=False)
    setattr(owner.spec, name, np.ones((len(owner.idx), 1), np.float32))
    with pytest.raises(ValueError, match='material shape'):
        PostAssimilationWindow(owner, successor, cfg, capture=False)


@pytest.mark.parametrize('key,value', [('layer_frac_u', .2), ('layer_inv_depth', 1.)])
def test_unreconstructed_successor_layer_parameters_rejected(key, value):
    owner, successor, cfg = make_case()
    meta = successor.metadata(); meta[key] = value
    successor = OwnedWithdrawal.from_arrays(successor.arrays(), meta)
    with pytest.raises(ValueError, match='fraction/inverse depth'):
        PostAssimilationWindow(owner, successor, cfg, capture=False)
