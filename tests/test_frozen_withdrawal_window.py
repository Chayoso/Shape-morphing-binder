"""Reduced-body withdrawal adapter; CPU MPM, no production policy change."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.withdrawal import OwnedWithdrawal
from physmorph.pipeline.frozen_body_window import FrozenBodyWindow
from physmorph.pipeline.frozen_withdrawal_window import FrozenWithdrawalWindow
from test_withdrawal_adjoint_fd import coupled_case, direction_like


def make_owner():
    """N27/T20 nonzero F/Fp/v/C, layer/bonds/pins; reusable CUDA upload fixture."""
    spec, controls = coupled_case()
    n, m = len(spec.x0), 9
    idx = torch.stack((torch.arange(n) % m, (torch.arange(n)+1) % m), 1)
    weights = torch.tensor([.25, .75]).expand(n, -1).clone()
    gate = torch.linspace(.4, .9, n)[:, None]
    coefficients = .003*torch.randn(m, 6, generator=torch.Generator().manual_seed(328))
    return FrozenBodyWindow(spec, SimpleNamespace(idx=idx, weights=weights), gate,
                            coefficients, controls[0], controls[1])


def coast_observable(values):
    """Future-only linear state probe, independent of the adapter's energy."""
    n = values['coast_X'].shape[1]
    device = values['coast_X'].device
    q = torch.arange(n*3, dtype=torch.float64, device=device).reshape(n, 3)
    weights = (.43*q+.2).sin()
    loss = ((values['coast_X'][-1].double()-values['coast_X'][0].double())*weights).sum()/n
    return loss+.007*(values['coast_V'][-1].double()*(weights+.3)).sum()/n


def test_exact_head_basis_energy_and_independent_complete_coast():
    owner = make_owner()
    model = FrozenWithdrawalWindow(owner, capture=False)
    terminal, displacement = owner.coefficients[:, 3:], owner.coefficients[:, :3]
    baseline = owner.evaluate(terminal, displacement, retain_full_state=True)
    values = model.evaluate(terminal, displacement)
    assert values['valid'] and values['pins_exact'] and values['health']['same_forward']
    assert values['scope'] == 'pre_assimilation_frozen_policy_withdrawal'
    for key in ('x', 'F', 'v', 'C', 'V', 'positions', 'body_energy', 'F_sequence', 'F_initial'):
        assert torch.equal(values[key], baseline[key]), key
    assert values['min_det'] == baseline['min_det']
    # A dense operator is independent of the indexed gather/reduction expression.
    dense = torch.zeros(len(owner.idx), len(owner.coefficients))
    dense.scatter_add_(1, owner.idx, owner.weights)
    expected_field = owner.spec.prm.dx*(dense@owner.coefficients)*owner.gate
    actual_field = model.adjoint.body.reshape(2, len(owner.idx), 3).permute(1, 0, 2).reshape(-1, 6)
    torch.testing.assert_close(actual_field, expected_field, rtol=2e-7, atol=1e-9)
    expected_energy = ((dense.double()@owner.coefficients.double())*owner.gate.double()).square().sum(1).mean()
    assert float(values['body_energy']) == pytest.approx(float(expected_energy), rel=3e-7)
    independent = OwnedWithdrawal.capture(model.adjoint.head, owner.spec.T).trajectory()
    independent.rollout()
    for name, key in (('x', 'coast_X'), ('v', 'coast_V'), ('F', 'coast_F'), ('Fg', 'coast_Fg'), ('C', 'coast_C')):
        shape = values[key].shape
        expected = torch.stack([wp.to_torch(a).clone() for a in getattr(independent, name)]).reshape(shape)
        assert torch.equal(values[key], expected), key
    for name in ('F', 'Fg', 'C'):
        sequence = values[name+'_sequence']
        assert sequence.shape == (20, 27, 3, 3) and not sequence.requires_grad
        assert torch.equal(sequence[-1].reshape_as(values[name]), values[name])
        assert torch.equal(sequence[0], wp.to_torch(getattr(model.adjoint.head, name)[1]))
    assert torch.count_nonzero(values['C']) > 0


def test_inputs_outputs_are_owned_and_later_forward_cannot_overwrite_evidence():
    owner = make_owner()
    model = FrozenWithdrawalWindow(owner, capture=False)
    terminal = owner.coefficients[:, 3:].clone()
    first = model.evaluate(terminal)
    saved = {k: v.clone() for k, v in first.items() if torch.is_tensor(v)}
    owner.coefficients.zero_(); owner.gate.zero_(); owner.weights.zero_()
    owner.stress.fill_(1.); owner.surface_u.fill_(1.)
    owner.spec.x0.fill(4.); owner.spec.C0.fill(1.); owner.spec.Fp.fill(0.)
    owner.spec.layer[7].fill(0.); owner.spec.prm.dt *= 2
    repeated = model.evaluate(terminal)
    for key, want in saved.items():
        assert torch.equal(repeated[key], want), key
    changed = model.evaluate(terminal*1.5)
    assert not torch.equal(changed['coast_X'], saved['coast_X'])
    for key, want in saved.items():
        assert torch.equal(first[key], want), key
    first['C_sequence'].zero_(); first['F_sequence'].zero_(); first['coast_C'].zero_()
    assert torch.equal(first['C'], saved['C'])
    assert torch.equal(first['F'], saved['F'])
    assert torch.equal(repeated['coast_C'], saved['coast_C'])


@pytest.mark.parametrize('closer', ['owner', 'adapter'])
def test_expiration_rejects_evaluation_and_backward_but_preserves_saved_values(closer):
    owner = make_owner()
    model = FrozenWithdrawalWindow(owner, capture=False)
    terminal = owner.coefficients[:, 3:].clone().requires_grad_()
    values = model.evaluate(terminal)
    saved = values['coast_X'].clone()
    (owner if closer == 'owner' else model).close()
    with pytest.raises(RuntimeError, match='expired'):
        model.evaluate(terminal)
    with pytest.raises(RuntimeError, match='expired'):
        torch.autograd.grad(coast_observable(values), terminal)
    assert torch.equal(values['coast_X'], saved)
    if closer == 'owner':
        with pytest.raises(ValueError, match='live'):
            FrozenWithdrawalWindow(owner)


@pytest.mark.parametrize('mode', [0, 1], ids=['displacement', 'terminal'])
def test_both_reduced_modes_have_resolved_future_derivatives(mode):
    owner = make_owner()
    # Preserve real pins but remove the separating contact branch for a smooth FD bracket.
    owner.spec.pin_slip = False
    model = FrozenWithdrawalWindow(owner, capture=False)
    leaves = [owner.coefficients[:, :3].clone().requires_grad_(),
              owner.coefficients[:, 3:].clone().requires_grad_()]
    output = model.evaluate(leaves[1], leaves[0])
    gradients = torch.autograd.grad(coast_observable(output), leaves)
    assert all(torch.isfinite(g).all() and g.norm() > 1e-5 for g in gradients)
    direction = direction_like(leaves[mode], mode)
    ad = float((gradients[mode].double()*direction.double()).sum())
    assert abs(ad) > 5e-5
    for epsilon in (1e-3, 5e-4):
        observed = []
        for sign in (-1, 1):
            candidate = [v.detach().clone() for v in leaves]
            candidate[mode].add_(direction, alpha=sign*epsilon)
            value = model.evaluate(candidate[1], candidate[0])
            assert value['valid']
            observed.append(float(coast_observable(value)))
        fd = (observed[1]-observed[0])/(2*epsilon)
        print(dict(mode=mode, epsilon=epsilon, analytical=ad, finite_difference=fd))
        assert ad == pytest.approx(fd, rel=.02, abs=5e-6)


@pytest.mark.parametrize('corruption', ['nonfinite_C', 'coast_orientation', 'coast_bounds', 'coast_pin',
                                       'coast_pin_v', 'coast_pin_C'])
def test_health_covers_coast_not_only_valid_head(monkeypatch, corruption):
    owner = make_owner()
    model = FrozenWithdrawalWindow(owner, capture=False)
    terminal = owner.coefficients[:, 3:]
    assert model.evaluate(terminal)['valid']
    original = model.adjoint.apply
    def corrupt(*args):
        out = original(*args)
        with torch.no_grad():
            if corruption == 'nonfinite_C':
                wp.to_torch(model.adjoint.coast.C[1])[3, 0, 0] = float('nan')
            elif corruption == 'coast_orientation':
                out.coast_F[1, 3] = torch.diag(torch.tensor([-1., 1., 1.])).reshape(9)
            elif corruption == 'coast_bounds':
                out.coast_X[1, 3, 0] = 100.
            elif corruption == 'coast_pin':
                out.coast_X[1, 0, 0] += .001
            elif corruption == 'coast_pin_v':
                out.coast_V[1, 0, 0] = .001
            else:
                wp.to_torch(model.adjoint.coast.C[1])[0, 0, 0] = .001
        return out
    monkeypatch.setattr(model.adjoint, 'apply', corrupt)
    values = model.evaluate(terminal)
    assert values['health']['head_valid'] and not values['health']['coast_valid']
    assert not values['valid']
    if corruption.startswith('coast_pin'):
        assert not values['pins_exact']


@pytest.mark.parametrize('field', ['v', 'C'])
def test_finite_nonzero_pinned_head_state_fails_health(monkeypatch, field):
    owner = make_owner()
    model = FrozenWithdrawalWindow(owner, capture=False)
    terminal = owner.coefficients[:, 3:]
    assert model.evaluate(terminal)['valid']
    original = model.adjoint.apply
    def corrupt(*args):
        out = original(*args)
        with torch.no_grad():
            if field == 'v':
                out.V[0, 0, 0] = .001
            else:
                wp.to_torch(model.adjoint.head.C[1])[0, 0, 0] = .001
        return out
    monkeypatch.setattr(model.adjoint, 'apply', corrupt)
    values = model.evaluate(terminal)
    assert values['health']['coast_valid'] and not values['health']['head_valid']
    assert not values['valid'] and not values['pins_exact']


def test_preflight_rejects_before_constructor_clones_or_evaluation_reductions(monkeypatch):
    import physmorph.pipeline.frozen_withdrawal_window as module
    owner = make_owner()
    model = FrozenWithdrawalWindow(owner, capture=False)
    terminal = owner.coefficients[:, 3:].clone()
    def no_numerics(*args, **kwargs):
        pytest.fail('Numerical operation ran before context rejection')
    monkeypatch.setattr(module, 'is_cuda_execution', lambda: True)  # CPU/context mismatch.
    monkeypatch.setattr(module, 'deepcopy', no_numerics)
    monkeypatch.setattr(torch, 'isfinite', no_numerics)
    with pytest.raises(RuntimeError, match='matching cuda_execution'):
        FrozenWithdrawalWindow(owner)
    with pytest.raises(RuntimeError, match='matching cuda_execution'):
        model.evaluate(terminal)
    assert model.adjoint is None


@pytest.mark.parametrize('invalid', ['shape', 'dtype', 'nonfinite', 'joint_radius'])
def test_bad_control_layout_and_original_joint_radius_reject(invalid):
    owner = make_owner()
    model = FrozenWithdrawalWindow(owner, capture=False)
    displacement = owner.coefficients[:, :3].clone()
    terminal = owner.coefficients[:, 3:].clone()
    if invalid == 'shape': terminal = terminal[:-1]
    if invalid == 'dtype': terminal = terminal.double()
    if invalid == 'nonfinite': terminal[0, 0] = float('nan')
    if invalid == 'joint_radius': displacement[0, 0] = .8; terminal[0, 0] = .8
    with pytest.raises(ValueError):
        model.evaluate(terminal, displacement)
    assert model.adjoint is None  # Validation did not launch any trajectory.


@pytest.mark.parametrize('replacement', ['valid', 'shape', 'joint_radius'])
@pytest.mark.parametrize('field', ['body_energy', 'coast_X'])
def test_successful_and_failed_replacements_expire_torch_and_warp_gradients(replacement, field):
    owner = make_owner()
    model = FrozenWithdrawalWindow(owner, capture=False)
    terminal = owner.coefficients[:, 3:].clone().requires_grad_()
    previous = model.evaluate(terminal)
    saved = previous[field].detach().clone()
    if replacement == 'valid':
        model.evaluate(terminal.detach()*1.1)
    else:
        invalid = terminal.detach()[:-1] if replacement == 'shape' else torch.ones_like(terminal)
        with pytest.raises(ValueError):
            model.evaluate(invalid)
    with pytest.raises(RuntimeError, match='stale'):
        torch.autograd.grad(previous[field].sum(), terminal)
    assert torch.equal(previous[field], saved)


def test_merit_callback_consumes_same_forward_head_without_objective_change(monkeypatch):
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig, runner
    source = np.random.default_rng(27).uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (source*[1.2, .85, 1.05]+[.1, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=2, animations=1, loss_res=12, render_views=2,
        render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
        outer_render_committed=True, body_ctrl=True, body_terminal_ctrl=True,
        lambda_auto=.3, w_kin=.2, w_kin_var=.3, w_ctrl=.001, w_jvol=.5, w_box=0.,
        max_ls_iters=1, adaptive_alpha=False, alpha=1e-4, replay_calibrate=False,
        phys_loss='ot_pace', loss_units='density', ot_samples=128, ot_iters=20,
        render_paced=True, layer_ctrl=True, layer_relax=True)
    original = runner.optimize_window
    observed = []
    def checkpoint(index, packet):
        owner = packet['rollout']
        terminal = owner.coefficients[:, 3:].clone().requires_grad_()
        baseline = owner.evaluate(terminal, retain_full_state=True)
        model = FrozenWithdrawalWindow(owner, capture=False)
        candidate = model.evaluate(terminal)
        assert candidate['valid']
        assert packet['evaluate_merit'](candidate) == packet['evaluate_merit'](baseline)
        gradient, = torch.autograd.grad(coast_observable(candidate), terminal)
        assert torch.isfinite(gradient).all() and gradient.norm() > 1e-8
        observed.append((model, terminal.detach()))
    def wrapped(*args, **kwargs):
        return original(*args, on_checkpoint=checkpoint, checkpoint_iterations=(1,),
                        checkpoint_rollout=True, checkpoint_merit=True, **kwargs)
    monkeypatch.setattr(runner, 'optimize_window', wrapped)
    result = runner.run_pipeline(source, target, MPMParams(dx=1., nx=32, ny=32, nz=32), cfg, log=lambda *_: None)
    assert observed and not any(result['guards'].values())
    for model, terminal in observed:
        with pytest.raises(RuntimeError, match='expired'):
            model.evaluate(terminal)
