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
from pin_collider_reference import first_step_pin_velocity


def make_case(*, pin_slip=False, fp64=False):
    """N27/T20/dt.002/dx.5/grid16^3, gate-off, actual new pin3."""
    owner = make_owner()
    spec = owner.spec
    spec.pin_slip = pin_slip
    spec.prm.gate_r_lo = spec.prm.gate_r_hi = 0.
    spec.layer = (*spec.layer[:5], None, 0., spec.layer[7])
    cfg = PipelineConfig(T=spec.T, body_ctrl=True, body_terminal_ctrl=True,
                         phys_loss='ot_pace', loss_units='density', device='cpu',
                         assim=.5, assim_iso=True, assim_fp64=fp64, settle_pin_assim=True,
                         settle_pin_slip=pin_slip)
    out = owner.evaluate(owner.coefficients[:, 3:])
    head = owner.adjoint.traj
    F, P = out['F'].detach().reshape(-1, 3, 3).numpy(), spec.Fp.copy()
    old = spec.pin.astype(bool)
    pins = old.copy(); pins[3] = True
    Fp = assimilate_elastic(F, P, eta=cfg.assim, isochoric=cfg.assim_iso, fp64=fp64)
    Fp[old] = P[old]
    Fp[pins & ~old] = assimilate_elastic(F[pins & ~old], Fp[pins & ~old], eta=1., isochoric=False, fp64=fp64)
    v, C = out['v'].detach().numpy().copy(), out['C'].detach().numpy().copy()
    v[pins], C[pins] = 0., 0.
    layer = deepcopy(spec.layer)
    layer[0][3] = 0.  # Actual successor preparation is supplied, not recomputed.
    successor = Trajectory(x0=out['x'].detach().numpy(), v0=v, C0=C, F0=F, Fp=Fp,
        Fg0=wp.to_torch(head.Fg[spec.T]).detach().numpy(), m=spec.m, lam=spec.lam, mu=spec.mu,
        vol0=spec.vol0, eta=spec.eta*1.1, pin=pins.astype(np.float32),
        layer=layer, bonds=spec.bonds(), prm=deepcopy(spec.prm), T=spec.T,
        track_geom=True, persistent=True, requires_grad=False, device='cpu', pin_slip=pin_slip)
    return owner, OwnedWithdrawal.capture(successor, 0), cfg


def future_velocity(values):
    """Future velocity avoids subtracting almost equal FP32 position snapshots."""
    velocity = values['coast_V'][-1]
    ids = torch.arange(velocity.numel(), device=velocity.device, dtype=torch.float64).reshape_as(velocity)
    return (velocity.double()*((ids*.43+.2).sin()+.3)).sum()/len(velocity)


@pytest.mark.parametrize('fp64', [False, True])
def test_original_head_basis_and_actual_subset_assimilation_coast_closure(fp64):
    owner, successor, cfg = make_case(fp64=fp64)
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
@pytest.mark.parametrize('fp64', [False, True])
def test_both_reduced_modes_have_resolved_post_assimilation_fd(mode, fp64):
    owner, successor, cfg = make_case(fp64=fp64)
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


@pytest.mark.parametrize('mode', [0, 1], ids=['displacement', 'terminal'])
@pytest.mark.parametrize('fp64', [False, True])
def test_slip_reduced_mode_fd_at_refined_radii(mode, fp64):
    owner, successor, cfg = make_case(pin_slip=True, fp64=fp64)
    model = PostAssimilationWindow(owner, successor, cfg, capture=False)
    leaves = [owner.coefficients[:, :3].clone().requires_grad_(),
              owner.coefficients[:, 3:].clone().requires_grad_()]
    out = model.evaluate(leaves[1], leaves[0])
    full = torch.autograd.grad(future_velocity(out), leaves)
    direction = direction_like(leaves[mode], mode)
    ad = float((full[mode].double()*direction.double()).sum())
    assert abs(ad) > 5e-5
    # The registered 1e-3 slip brackets failed in the first CUDA run and the
    # independent CPU audit, even after restoring gmpin's gradient. Refine the
    # radius, retaining the tolerance; no fixed contact active-set claim follows.
    # Original no-slip tests above keep their 1e-3/5e-4 brackets unchanged.
    for epsilon in (1e-4, 5e-5):
        losses = []
        for sign in (-1, 1):
            shifted = [value.detach().clone() for value in leaves]
            shifted[mode].add_(direction, alpha=sign*epsilon)
            values = model.evaluate(shifted[1], shifted[0])
            assert values['valid']
            losses.append(float(future_velocity(values)))
        fd = (losses[1]-losses[0])/(2*epsilon)
        print(dict(mode=mode, epsilon=epsilon, full=ad, fd=fd))
        assert ad == pytest.approx(fd, rel=.02, abs=5e-6)


def test_new_pin_collider_anchor_derivative_and_detached_mass_negative_control():
    owner, successor, cfg = make_case(pin_slip=True)
    assert successor.metadata()['pin_mode'] == 1
    model = PostAssimilationWindow(owner, successor, cfg, capture=False)
    broken = PostAssimilationWindow(owner, successor, cfg, capture=False)
    broken.adjoint = broken._make_adjoint()
    # Reproduce only the old missing edge; keep every primal policy/buffer and
    # the Fp/v/C handoff intact. The ordinary Trajectory allocation is detached.
    broken.adjoint.coast.gmpin = wp.clone(broken.adjoint.coast.gmpin, requires_grad=False)
    broken.adjoint._configure_seeds()
    leaves = [owner.coefficients[:, :3].clone().requires_grad_(),
              owner.coefficients[:, 3:].clone().requires_grad_()]
    actual = model.evaluate(leaves[1], leaves[0])
    incomplete = broken.evaluate(leaves[1], leaves[0])
    assert actual['valid'] and incomplete['valid']
    assert model.adjoint.coast.gmpin.requires_grad
    assert not broken.adjoint.coast.gmpin.requires_grad
    for key, value in actual.items():
        if torch.is_tensor(value):
            assert torch.equal(value, incomplete[key]), key
    # Independently verify this is the real slip successor, including moved
    # new-pin anchors, rather than a no-slip handoff with relabeled metadata.
    independent = successor.trajectory(persistent=True)
    independent.rollout()
    expected = torch.stack([wp.to_torch(value).clone() for value in independent.v])
    torch.testing.assert_close(actual['coast_V'], expected, rtol=2e-5, atol=3e-6)
    assert torch.max(torch.abs(actual['x'][3]-torch.from_numpy(owner.spec.x0[3]))) > 1e-6
    weights = (torch.arange(len(owner.idx)*3, dtype=torch.float64).reshape(-1, 3)*.43+.2).sin()
    def acceleration(values):
        return ((values['coast_V'][1].double()-values['coast_V'][0].double())*weights).mean()/owner.spec.prm.dt
    torch.autograd.grad(acceleration(actual), leaves, retain_graph=True)
    full = wp.to_torch(model.adjoint.coast.x[0].grad).clone()
    pin_mass_covector = wp.to_torch(model.adjoint.coast.gmpin.grad).clone()
    assert float(pin_mass_covector.norm()) > 1e-7
    torch.autograd.grad(acceleration(actual), leaves, retain_graph=True)
    assert torch.equal(wp.to_torch(model.adjoint.coast.gmpin.grad), pin_mass_covector)
    torch.autograd.grad(actual['coast_V'].sum()*0, leaves)
    assert torch.count_nonzero(wp.to_torch(model.adjoint.coast.gmpin.grad)) == 0
    torch.autograd.grad(acceleration(incomplete), leaves)
    missing = wp.to_torch(broken.adjoint.coast.x[0].grad).clone()
    # This is a conditional boundary-x derivative, not a reduced-body negative:
    # perturb only the new pin's anchor while holding all other successor state
    # fixed. Its first-step free velocity path passes through the collider field.
    direction = torch.zeros_like(actual['x'])
    direction[3] = torch.tensor([.3, -.7, .5])
    direction /= direction.norm()
    ad = float((full.double()*direction.double()).sum())
    bad = float((missing.double()*direction.double()).sum())
    assert abs(ad) > 5e-5 and bad == 0.
    initial_x = wp.to_torch(independent.x[0]).clone().double()
    initial_v = wp.to_torch(independent.v[0]).clone().double()
    independent.step(0)
    mass = wp.to_torch(independent.m).clone()
    grid_mass = wp.to_torch(independent.gm[0]).clone()
    grid_momentum = wp.to_torch(independent.gmom[0]).clone()
    pins = actual['coast_pins']
    def reference(x):
        return first_step_pin_velocity(x, pins, mass, grid_mass, grid_momentum, owner.spec.prm)
    def reference_loss(value):
        return ((value['velocity']-initial_v)*weights).mean()/owner.spec.prm.dt
    base_x = initial_x.clone().requires_grad_()
    base = reference(base_x)
    # The FP64 reference must first reconstruct the actual unshifted collider
    # and first-step free velocity before it supplies a derivative oracle.
    for key, source in (('pin_mass', independent.gmpin), ('grid_velocity', independent.gvel[0]),
                        ('velocity', independent.v[1])):
        torch.testing.assert_close(base[key], wp.to_torch(source).double(), rtol=2e-5, atol=3e-6)
    reference_gradient, = torch.autograd.grad(reference_loss(base), base_x)
    reference_ad = float((reference_gradient*direction.double()).sum())
    assert ad == pytest.approx(reference_ad, rel=.02, abs=5e-6)
    differences = []
    # Raw FP32 first-step differences failed at 5e-5 on CPU and at 1e-4 on
    # CUDA. Keep those receipts; this independent local map uses FP64 at all
    # both registered radii, with stable reference branches and FD convergence.
    radii = (1e-4, 5e-5)
    for epsilon in radii:
        losses = []
        for sign in (-1, 1):
            x = initial_x+sign*epsilon*direction.double()
            value = reference(x)
            for key, branch in base['branches'].items():
                assert torch.equal(value['branches'][key], branch), (epsilon, sign, key)
            losses.append(float(reference_loss(value)))
            # New slip anchors cannot change the fixed P2G inputs used here.
            wp.to_torch(independent.x[0]).copy_(x.float())
            independent.step(0)
            assert torch.equal(wp.to_torch(independent.gm[0]), grid_mass)
            assert torch.equal(wp.to_torch(independent.gmom[0]), grid_momentum)
        fd = (losses[1]-losses[0])/(2*epsilon)
        allowance = max(.02*abs(fd), 5e-6)
        print(dict(scope='fp64_fixed_successor_new_pin_anchor', epsilon=epsilon,
                   full=ad, reference_ad=reference_ad, detached_gmpin=bad, fd=fd))
        assert abs(ad-fd) <= allowance and abs(bad-fd) > allowance
        differences.append(fd)
    rounding_floor = 64*torch.finfo(torch.float64).eps*max(1., abs(float(reference_loss(base).detach())))/min(radii)
    assert abs(differences[-1]-differences[-2]) <= max(1e-6*abs(differences[-1]), rounding_floor)
