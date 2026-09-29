"""Real multiwindow selection and ordinary handoff; no quality-policy promotion."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from physmorph.mpm.withdrawal import OwnedWithdrawal
from physmorph.pipeline import optimizer, runner
from test_checkpoint_merit_terms import fixture
from test_window_selection import equal


def recipe():
    source, target, prm, cfg = fixture()
    cfg.animations = 4
    cfg.layer_k = 8
    cfg.bonds = True
    cfg.ctrl_rprop = True
    cfg.settle_pin = cfg.settle_pin_still = cfg.settle_pin_assim = True
    cfg.settle_pin_confirm = True
    cfg.plan_sticky = True
    cfg.outer_merit = True
    return source, target, prm, cfg


def observed_run(monkeypatch, selector=None, cfg=None):
    source, target, prm, default = recipe()
    prepared, outputs, input_controls = [], [], []
    original, constructor = runner.optimize_window, optimizer.Trajectory

    def observed(*args, **kwargs):
        trajectories = []
        def remember(*pos, **kw):
            tr = constructor(*pos, **kw)
            trajectories.append(tr)
            return tr
        input_controls.append(deepcopy((kwargs.get('dfc_init'), kwargs.get('mom_init'))))
        with monkeypatch.context() as patch:
            patch.setattr(optimizer, 'Trajectory', remember)
            result = original(*args, **kwargs)
        assert len(trajectories) == 1
        prepared.append(OwnedWithdrawal.capture(trajectories[0], 0).arrays())
        # Do not deepcopy the opaque context: the original six-tuple is enough.
        outputs.append(deepcopy((*result[:5], {k: v for k, v in result[5].items()
                                             if k != '_window_selection'})))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(runner, 'optimize_window', observed)
        result = runner.run_pipeline(source, target, prm, deepcopy(default if cfg is None else cfg),
                                     log=lambda *_: None, select_window=selector)
    return result, prepared, outputs, input_controls


@pytest.mark.parametrize('explicit', [False, True])
def test_identity_preserves_real_successor_preparation_and_new_pins(monkeypatch, explicit):
    baseline, prepared0, outputs0, _ = observed_run(monkeypatch)
    contexts = []
    def select(ctx):
        choice = ctx.original()
        inspection = ctx.inspect(choice)
        inspection['values']['positions'].zero_()
        inspection['values']['F_sequence'].zero_()
        inspection['coefficients'].fill_(99.)
        contexts.append(ctx)
        return choice if explicit else None
    actual, prepared, outputs, _ = observed_run(monkeypatch, select)
    assert len(contexts) >= 2 and all(ctx.closed for ctx in contexts)
    assert len(prepared) == len(prepared0) >= 3
    equal(prepared, prepared0)
    for output, baseline_output in zip(outputs, outputs0):
        stripped = (*output[:5], {k: v for k, v in output[5].items() if k != 'window_selection'})
        equal(stripped, baseline_output)
    for key in ('frames', 'F_frames', 'Fp', 'guards', 'pinned', 'pinned_at', 'stuck', 'termination'):
        equal(actual[key], baseline[key])
    assert not any(actual['guards'].values())
    assert np.any(prepared[2]['pin'] > .5), 'Fixture must exercise actual new pin admission'
    pinned = prepared[2]['pin'] > .5
    assert not np.any(prepared[2]['v0'][pinned]) and not np.any(prepared[2]['C0'][pinned])
    assert np.any(prepared[1]['Fp'] != prepared[0]['Fp']), 'Assimilation must be active'
    for a, b in zip(actual['history'], baseline['history']):
        stripped = {k: v for k, v in a.items() if k != 'window_selection'}
        equal(stripped, b)


@pytest.mark.parametrize('certified', [False, True])
def test_actual_private_forward_is_selected_as_whole_state(monkeypatch, certified):
    source, target, prm, cfg = fixture()
    cfg.animations = 2
    cfg.layer_k = 8
    cfg.w_pbr = .2
    cfg.settle_pin = False
    observations, contexts = [], []
    def select(ctx):
        contexts.append(ctx)
        info = ctx.inspect(ctx.original())
        # A tiny perturbation at the real final donor, not an endpoint edit.
        coeff = info['coefficients'] * (1 + 1e-5)
        choice = ctx.evaluate(coeff, 'fixture_changed_body')
        observed = ctx.inspect(choice)
        assert observed['eligible'], observed['failures']
        assert not torch.equal(observed['values']['positions'], info['values']['positions'])
        if certified:
            # API fixture only. This is deliberately NOT a production raw-quality certificate.
            ctx.certify(choice, dict(passed=True, scope='CPU integration fixture only'))
        observations.append(observed)
        return choice
    result, prepared, outputs, _ = observed_run(monkeypatch, select, cfg)
    assert len(observations) == 2 and all(ctx.closed for ctx in contexts)
    assert not any(result['guards'].values())
    for index, obs in enumerate(observations):
        row = result['history'][index]
        assert row['window_selection']['selected'] is certified
        assert row['accepted'] == outputs[index][5]['accepted']
        assert row['iters'] == len(outputs[index][4])
        equal(row['render_influence_steps'], outputs[index][5]['render_influence_steps'])
        expected = obs['values']['positions'].numpy() if certified else np.stack(outputs[index][0][1:])
        np.testing.assert_array_equal(np.stack(result['frames'][index*cfg.T+1:(index+1)*cfg.T+1]), expected)
        if certified:
            assert row['kin'] == obs['metrics']['stored_terminal']
            assert row['loss'] == obs['metrics']['merit']
            assert row['grad_norm'] is None and row['alpha_last'] is None
            assert row['F_kind'] == 'physics' and not row['commit_from_accepted']
    if certified:
        first = observations[0]['values']
        np.testing.assert_array_equal(prepared[1]['x0'], first['x'].numpy())
        np.testing.assert_array_equal(prepared[1]['v0'], first['v'].numpy())
        np.testing.assert_array_equal(prepared[1]['C0'], first['C'].numpy())
        np.testing.assert_array_equal(prepared[1]['F0'], first['F'].reshape(-1, 3, 3).numpy())


def test_selector_exception_closes_context_and_merit_lease(monkeypatch):
    contexts, evaluators = [], []
    def select(ctx):
        contexts.append(ctx)
        evaluators.append(ctx._evaluate_merit)
        raise RuntimeError('fixture selection interrupted')
    with pytest.raises(RuntimeError, match='fixture selection interrupted'):
        observed_run(monkeypatch, select)
    assert len(contexts) == 1 and contexts[0].closed
    with pytest.raises(RuntimeError, match='expired'):
        evaluators[0]({})


def test_null_donor_never_invokes_selection(monkeypatch):
    _, _, _, cfg = fixture()
    cfg.animations = 2
    monkeypatch.setattr(optimizer, '_state_ok', lambda *_: False)
    def forbidden(_):
        pytest.fail('Null donor cannot provide a selection context')
    result, _, outputs, _ = observed_run(monkeypatch, forbidden, cfg)
    assert outputs and all(not output[4] for output in outputs)
    assert all(row.get('null_commit') for row in result['history'])
    for frame in result['frames']:
        np.testing.assert_array_equal(frame, result['frames'][0])


def test_final_invalid_donor_cannot_resurrect_earlier_acceptance(monkeypatch):
    _, _, _, cfg = fixture()
    cfg.iters = cfg.animations = 1
    valid = optimizer._state_ok
    checks = []
    def final_invalid(state):
        checks.append(None)
        return valid(state) if len(checks) == 1 else False
    monkeypatch.setattr(optimizer, '_state_ok', final_invalid)
    result, _, outputs, _ = observed_run(monkeypatch, lambda _: pytest.fail('Invalid final donor'), cfg)
    assert len(checks) == 2 and not outputs[0][4]
    assert outputs[0][5]['accepted'] == 0
    assert result['history'][0]['null_commit'] == 1


def test_trailing_gradient_stop_keeps_the_accepted_merit_weight(monkeypatch):
    from physmorph.pipeline.render_loss import LambdaBalancer
    _, _, _, cfg = fixture()
    original, update = runner.optimize_window, LambdaBalancer.update
    updates, observations = [], []
    def record_lambda(self, *args, **kwargs):
        value = update(self, *args, **kwargs)
        updates.append(value)
        return value
    def stop_after_accept(x0, prm, inner_cfg, *args, **kwargs):
        def accepted(*_):
            inner_cfg.gd_tol = 1e6
        return original(x0, prm, inner_cfg, *args, **{**kwargs, 'on_iter': accepted})
    monkeypatch.setattr(LambdaBalancer, 'update', record_lambda)
    monkeypatch.setattr(runner, 'optimize_window', stop_after_accept)
    def select(ctx):
        observations.append(ctx.inspect(ctx.original()))
    result, _, outputs, _ = observed_run(monkeypatch, select, cfg)
    assert len(updates) == len(observations) == 1
    stats = outputs[0][5]
    assert stats['accepted'] == 1 and stats['commit_from_accepted']
    assert stats['grad_converged']
    assert observations[0]['metrics']['lambda'] == updates[0]
    assert stats['replay_diagnostics']['replay_lambda_final'] == updates[0]
    assert result['history'][0]['window_selection']['selected'] is False


def test_replayed_donor_keeps_ordinary_result_without_selection(monkeypatch):
    _, _, _, cfg = fixture()
    valid = optimizer._state_ok
    calls = []
    def reject_second_trial(state):
        calls.append(None)
        return False if len(calls) == 2 else valid(state)
    monkeypatch.setattr(optimizer, '_state_ok', reject_second_trial)
    result, _, outputs, _ = observed_run(monkeypatch, lambda _: pytest.fail('Replayed donor'), cfg)
    stats = outputs[0][5]
    assert stats['accepted'] == 1 and not stats['commit_from_accepted']
    assert stats['window_selection']['skipped'] == 'donor_replayed'
    assert not result['history'][0].get('null_commit')


def test_outer_rejection_of_selected_forward_rolls_back_and_cold_restarts(monkeypatch):
    _, _, _, cfg = fixture()
    cfg.animations = 3
    cfg.outer_merit = cfg.outer_reversal_always = True
    # Exercise the ordinary rejection branch for either displacement direction.
    cfg.outer_reversal_cos = 2.
    cfg.outer_reversal_gain = 1e6
    cfg.warm_start = True
    chosen = []
    def select(ctx):
        info = ctx.inspect(ctx.original())
        if info['window'] != 1:
            return None
        choice = ctx.evaluate(info['coefficients'] * (1 + 1e-5), 'will_be_rejected')
        inspected = ctx.inspect(choice)
        assert inspected['eligible'], inspected['failures']
        ctx.certify(choice, dict(passed=True, scope='API rejection fixture only'))
        chosen.append(inspected)
        return choice
    result, prepared, _, controls = observed_run(monkeypatch, select, cfg)
    assert len(chosen) == 1 and len(prepared) == 3
    rejected = result['history'][1]
    assert rejected['window_selection']['selected'] and rejected['outer_rejected']
    for key in ('x0', 'v0', 'C0', 'F0', 'Fp', 'pin'):
        np.testing.assert_array_equal(prepared[2][key], prepared[1][key])
    assert controls[1][0] is not None and controls[2] == (None, None)
    assert len(result['frames']) == cfg.T+1
    assert not np.array_equal(result['frames'][-1], chosen[0]['values']['x'].numpy())
