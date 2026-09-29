"""Complete candidate-head covectors; real CPU rollouts, no adoption policy."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, runner
from physmorph.pipeline.frozen_withdrawal_window import FrozenWithdrawalWindow


def fixture():
    source = np.random.default_rng(27).uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (source*[1.2, .85, 1.05]+[.1, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=2, animations=1, loss_res=12, render_views=2,
        render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
        outer_render_committed=True, body_ctrl=True, body_terminal_ctrl=True,
        lambda_auto=.3, w_kin=.2, w_kin_var=.3, w_kin_running=.15,
        w_ctrl=.001, w_jvol=.5, w_box=0., max_ls_iters=1,
        adaptive_alpha=False, alpha=1e-4, replay_calibrate=False,
        phys_loss='ot_pace', loss_units='density', ot_samples=128, ot_iters=20,
        render_paced=True, layer_ctrl=True, layer_relax=True)
    return source, target, MPMParams(dx=1., nx=32, ny=32, nz=32), cfg


def run_observed(monkeypatch, callback, *, cfg=None, iterations=(1,)):
    source, target, prm, default = fixture()
    original = runner.optimize_window
    packets = []
    def observe(index, packet):
        callback(packet)
        packets.append(packet)
    def wrapped(*args, **kwargs):
        return original(*args, on_checkpoint=observe, checkpoint_iterations=iterations,
                        checkpoint_rollout=True, checkpoint_merit=True, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(runner, 'optimize_window', wrapped)
        result = runner.run_pipeline(source, target, prm, deepcopy(default if cfg is None else cfg), log=lambda *_: None)
    assert len(packets) == len(iterations)
    assert all(p['optimizer_state_exact'] and p['optimizer_state_after_callback_exact'] for p in packets)
    assert not any(result['guards'].values())
    return result, packets


def independent_inputs(values):
    """Independent endpoint/path leaves for analytic partial-derivative checks."""
    copied = {k: v.detach().clone() if torch.is_tensor(v) else v for k, v in values.items()}
    copied['F'] = (copied['F']*1.04).requires_grad_()
    v = copied['V']
    v = v + torch.arange(v.shape[0], dtype=v.dtype)[:, None, None]*torch.tensor([.2, -.1, .3])
    copied['V'] = v.requires_grad_()
    copied['v'] = copied['V'][-1]
    copied['body_energy'] = torch.tensor(.2, requires_grad=True)
    return copied


@pytest.mark.parametrize('layer,pbr', [(False, 0.), (True, 0.), (True, .2)])
def test_terms_scalar_parity_and_original_solver_isolation(monkeypatch, layer, pbr):
    source, target, prm, cfg = fixture()
    cfg.layer_ctrl = cfg.layer_relax = layer
    cfg.w_pbr = pbr
    baseline = runner.run_pipeline(source, target, prm, deepcopy(cfg), log=lambda *_: None)
    prior = []
    def check(packet):
        owner = packet['rollout']
        terminal = owner.coefficients[:, 3:].clone().requires_grad_()
        model = FrozenWithdrawalWindow(owner, capture=False)
        values = model.evaluate(terminal)
        evaluate = packet['evaluate_merit']
        for expired in prior:
            with pytest.raises(RuntimeError, match='expired'):
                expired.terms(values)
        binding = evaluate.binding_digest()
        scalar = evaluate(values)
        terms = evaluate.terms(values)
        assert scalar['unit_weight'] != 1.
        assert scalar['merit'] == scalar['physical']+scalar['lambda_render']*scalar['render']
        for key in scalar.keys()-{'merit'}:
            assert float(terms[key]) == scalar[key], key
        assert torch.equal(terms['merit'], terms['physical']+scalar['lambda_render']*terms['render'])
        for key in ('merit', 'physical', 'render', 'volume', 'stored_variance', 'body_energy'):
            assert terms[key].requires_grad, key
        if pbr:
            assert terms['pbr'].requires_grad and float(terms['pbr']) > 0
        gradient, = torch.autograd.grad(terms['merit'], terminal)
        assert torch.isfinite(gradient).all() and gradient.norm() > 1e-8
        assert evaluate.binding_digest() == binding and evaluate(values) == scalar
        prior.append(evaluate)
    observed, _ = run_observed(monkeypatch, check, cfg=cfg, iterations=(1, 2))
    np.testing.assert_array_equal(observed['frames'], baseline['frames'])
    np.testing.assert_array_equal(observed['F_frames'], baseline['F_frames'])
    for actual, expected in zip(observed['history'], baseline['history']):
        for key in ('loss', 'lambda', 'd_vol', 'd_sil', 'frame_end', 'render_influence_steps'):
            assert actual.get(key) == expected.get(key), key


def test_F_body_energy_and_full_V_partials_match_independent_analytic_formula(monkeypatch):
    _, _, _, cfg = fixture()
    def check(packet):
        owner = packet['rollout']
        base = owner.evaluate(owner.coefficients[:, 3:])
        values = independent_inputs(base)
        terms = packet['evaluate_merit'].terms(values)
        F, V, energy = values['F'], values['V'], values['body_energy']
        grad_F, grad_V, grad_energy = torch.autograd.grad(terms['merit'], (F, V, energy))
        T, N = V.shape[:2]
        wu = terms['unit_weight']
        matrix = F.detach().double().reshape(-1, 3, 3)
        J = torch.linalg.det(matrix)
        expected_F = (wu*cfg.w_jvol/N*(J.log()+(J-1)/J)*J)[:, None, None]*torch.linalg.inv(matrix).transpose(-1, -2)
        velocity = V.detach().double()
        expected_V = 2*wu/(T*N)*(cfg.w_kin_running*velocity+cfg.w_kin_var*(velocity-velocity.mean(0)))
        expected_V[-1] += 2*wu*cfg.w_kin/N*velocity[-1]
        torch.testing.assert_close(grad_F.double().reshape_as(expected_F), expected_F, rtol=2e-5, atol=1e-9)
        torch.testing.assert_close(grad_V.double(), expected_V, rtol=2e-5, atol=1e-9)
        assert float(grad_energy) == pytest.approx(wu*cfg.w_ctrl, rel=2e-7)
        assert grad_F.norm() > 0 and grad_V.norm() > 0 and grad_energy > 0
        assert float(terms['stored_variance']) == pytest.approx(float(velocity.var(0, correction=0).sum(-1).mean()), rel=3e-7)
    run_observed(monkeypatch, check)


def check_control_derivative(packet, mode, *, capture=False):
    """Identical FD witness/tolerances for CPU and the opt-in server CUDA gate."""
    owner = packet['rollout']
    model = FrozenWithdrawalWindow(owner, capture=capture)
    leaves = [owner.coefficients[:, :3].clone().requires_grad_(),
              owner.coefficients[:, 3:].clone().requires_grad_()]
    values = model.evaluate(leaves[1], leaves[0])
    evaluate = packet['evaluate_merit']
    terms = evaluate.terms(values)
    gradients = torch.autograd.grad(terms['merit'], leaves)
    assert all(torch.isfinite(g).all() and g.norm() > 1e-8 for g in gradients)
    if capture and values['x'].is_cuda:
        assert model.adjoint.g_fwd is not None and model.adjoint.g_bwd is not None
    direction = gradients[mode].detach()/gradients[mode].norm()
    ad = float((gradients[mode].double()*direction.double()).sum())
    assert abs(ad) > 50*1e-6  # Resolve the unchanged absolute FD allowance.
    for epsilon in (.003, .0015):
        samples = []
        for sign in (-1, 1):
            candidate = [x.detach().clone() for x in leaves]
            candidate[mode].add_(direction, alpha=sign*epsilon)
            values = model.evaluate(candidate[1], candidate[0])
            assert values['valid']
            samples.append(evaluate(values)['merit'])
        fd = (samples[1]-samples[0])/(2*epsilon)
        print(dict(mode=mode, epsilon=epsilon, ad=ad, fd=fd))
        assert ad == pytest.approx(fd, rel=.03, abs=1e-6)
    # A fresh graph remains unused until the source callback expires. This
    # tests the merit lease itself, including its direct body-energy path.
    independent = dict(values, body_energy=values['body_energy'].detach().clone().requires_grad_())
    retained = evaluate.terms(independent)
    return evaluate, retained, independent['body_energy']


@pytest.mark.parametrize('mode', [0, 1], ids=['displacement', 'terminal'])
def test_complete_merit_control_derivative_through_actual_joint_adapter(monkeypatch, mode):
    run_observed(monkeypatch, lambda packet: check_control_derivative(packet, mode))


def test_cache_restoration_invalid_inputs_and_nonfinite_terms(monkeypatch):
    import physmorph.pipeline.optimizer as module
    def check(packet):
        owner = packet['rollout']
        values = independent_inputs(owner.evaluate(owner.coefficients[:, 3:]))
        evaluate = packet['evaluate_merit']
        scalar, binding = evaluate(values), evaluate.binding_digest()
        for changed, message in ((dict(values, valid=False), 'exact-pin'),
                (dict(values, C=values['C'][:-1]), 'field: C'),
                (dict(values, v=values['v']+.1), 'Inconsistent'),
                (dict(values, body_energy=torch.tensor(-1.)), 'Inconsistent'),
                (dict(values, body_energy=torch.tensor(float('nan'))), 'body_energy')):
            with pytest.raises(ValueError, match=message):
                evaluate.terms(changed)
        def failed(*args, **kwargs):
            raise RuntimeError('injected render failure')
        with monkeypatch.context() as patch:
            patch.setattr(module, 'd_render', failed)
            with pytest.raises(RuntimeError, match='injected render failure'):
                evaluate.terms(values)
        assert evaluate(values) == scalar and evaluate.binding_digest() == binding
        with monkeypatch.context() as patch:
            patch.setattr(module, 'd_render', lambda x, *args, **kwargs: x.sum()*float('nan'))
            with pytest.raises(ValueError, match='Nonfinite'):
                evaluate.terms(values)
        assert evaluate(values) == scalar and evaluate.binding_digest() == binding
        # The returned leaf-energy observation owns storage; no hook is attached
        # to the caller's original independent leaf.
        terms = evaluate.terms(values)
        assert terms['body_energy'].data_ptr() != values['body_energy'].data_ptr()
        terms['body_energy'].detach().zero_()
        assert float(values['body_energy']) == pytest.approx(.2)
    run_observed(monkeypatch, check)


@pytest.mark.parametrize('key', ['merit', 'physical', 'body_energy', 'stored_variance'])
def test_gradient_lease_expires_after_callback_without_hooking_caller_leaves(monkeypatch, key):
    retained = []
    def check(packet):
        owner = packet['rollout']
        values = independent_inputs(owner.evaluate(owner.coefficients[:, 3:]))
        evaluate = packet['evaluate_merit']
        terms = evaluate.terms(values)
        retained.append((evaluate, terms, values))
    run_observed(monkeypatch, check)
    evaluate, terms, values = retained[0]
    with pytest.raises(RuntimeError, match='expired'):
        evaluate.terms(values)
    with pytest.raises(RuntimeError, match='gradient expired'):
        torch.autograd.grad(terms[key], (values['F'], values['V'], values['body_energy']), allow_unused=True)
    assert torch.autograd.grad(values['body_energy']*2, values['body_energy'])[0] == 2
