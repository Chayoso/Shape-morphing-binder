"""Global stop reasons are observations, not changes to solver decisions or rest."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, runner


def run_scripted(monkeypatch, outcomes, with_dressing=False, **overrides):
    """Exercise the actual runner with deterministic accepted/rejected candidates."""
    source = np.array([[-.5, -.5, -.5], [.5, .5, .5]], np.float32)
    gauss = SimpleNamespace(configure_source=lambda *a: None) if with_dressing else None
    monkeypatch.setattr(runner, 'build_target', lambda *a, **k: SimpleNamespace(
        gauss=gauss, dt3=None, tmass3=None, sils=None))
    calls, logs, callbacks = [], [], []

    def optimize(x, *args, **kwargs):
        index = len(calls)
        calls.append(x.copy())
        outcome = outcomes[index]
        if outcome == 'gradient':
            return [], [], {}, None, [], dict(grad_converged=True)
        if outcome == 'null':
            return [], [], {}, None, [], dict(ls_exhausted=1)
        loss, offset = outcome
        end = x + np.array([offset, 0., 0.], np.float32)
        identity = np.broadcast_to(np.eye(3, dtype=np.float32), (len(x), 3, 3)).copy()
        history = dict(loss=loss, d_vol=loss, kin=0., d_render=None, **{'lambda': 0.},
                       dfc_absmax=0., s_absmax=0.)
        return [x.copy(), end], [identity.copy(), identity.copy()], dict(
            F=identity, v=np.zeros_like(x), C=np.zeros_like(identity)), None, [history], dict(
                accepted=1, rejected=0)

    monkeypatch.setattr(runner, 'optimize_window', optimize)
    options = dict(device='cpu', animations=8, T=1, iters=1, patience=2,
                   hold_after_converge=True, persistent_rest_volume=False,
                   assim=0., w_kin=0., best_truncate=False)
    options.update(overrides)
    if with_dressing:
        from physmorph.pipeline import dressing
        monkeypatch.setattr(runner, '_surface_weights', lambda x, *a: np.ones(len(x), np.float32))
        monkeypatch.setattr(dressing, 'DressState', lambda *a: SimpleNamespace(
            cover_frames=lambda n: None, export=lambda: {}))
        options.update(local_dress_iters=1, gauss_children=2, render_surface_only=True,
                       surface_grad_frac=1.)
    cfg = PipelineConfig(**options)
    result = runner.run_pipeline(source, source.copy(), MPMParams(), cfg, log=logs.append,
        on_commit=lambda a, x, F, v, rec: callbacks.append((a, x.copy(), dict(rec))))
    return result, calls, logs, callbacks


def assert_reason(result, reason, count, frozen=True):
    termination = result['termination']
    assert result['converged'] is frozen
    assert termination['reason'] == reason
    assert termination['optimizer_attempts'] == count
    assert termination['attempt_index'] == (count - 1 if count else None)
    assert termination['attempt_number'] == (count if count else None)
    assert termination['optimization_stopped'] is True
    assert termination['stopping_rule_triggered'] is frozen
    assert termination['individual_rest'] == 'not_evaluated'
    json.dumps(termination, allow_nan=False)
    assert all(type(value) in (str, int, float, bool, type(None))
               for event in termination['triggers'] for value in event['evidence'].values())
    assert all('termination' not in h for h in result['history'])
    return termination


@pytest.mark.parametrize('hold', [False, True])
def test_window_start_gradient_stop_preserves_copied_hold(monkeypatch, hold):
    result, calls, logs, callbacks = run_scripted(monkeypatch, ['gradient'], hold_after_converge=hold)
    assert_reason(result, 'window_start_gradient_stop', 1)
    assert len(calls) == 1 and not callbacks
    assert len(result['frames']) == 1 + int(hold)
    assert result['history'] == [dict(animation=0, grad_converged=1, render_target_kind=None)] + (
        [dict(animation=1, held=1)] if hold else [])
    assert all(np.array_equal(x, calls[0]) for x in result['frames'])
    assert all('converged' not in line and 'at the optimum' not in line for line in logs)


def test_null_patience_keeps_null_frames_and_history(monkeypatch):
    result, calls, _, callbacks = run_scripted(monkeypatch, ['null', 'null'])
    assert_reason(result, 'null_commit_patience', 2)
    assert len(calls) == 2 and not callbacks
    assert len(result['frames']) == 4 and result['n_held'] == 1
    assert [h.get('null_commit', 0) for h in result['history']] == [1, 1, 0]
    assert all(np.array_equal(x, calls[0]) for x in result['frames'])


@pytest.mark.parametrize('hold', [False, True])
def test_dressing_suffix_does_not_replace_stop_attempt(monkeypatch, hold):
    result, calls, _, _ = run_scripted(monkeypatch, ['gradient'], with_dressing=True,
        animations=4, hold_after_converge=hold)
    assert_reason(result, 'window_start_gradient_stop', 1)
    assert len(calls) == 1 and result['n_held'] == 3 * int(hold)
    assert len(result['frames']) == 1 + 3 * int(hold)
    assert len(result['termination']['triggers']) == 1


def test_accepted_track_plateau_keeps_accepted_motion(monkeypatch):
    result, calls, logs, callbacks = run_scripted(monkeypatch, [(1., .1)] * 3)
    assert_reason(result, 'accepted_track_plateau', 3)
    assert len(calls) == len(callbacks) == 3
    assert [h.get('stale') for h in result['history']] == [0, 1, 2, None]
    assert [h.get('frame_end') for h in result['history']] == [2, 3, 4, None]
    assert all(np.array_equal(result['frames'][i + 1], callbacks[i][1]) for i in range(3))
    assert np.array_equal(result['frames'][-2], result['frames'][-1])
    assert not np.array_equal(result['frames'][0], result['frames'][-1])
    assert any('accepted-track plateau' in line for line in logs)


@pytest.mark.parametrize('patience,reject_stop,reason,triggers', [
    (1, 0, 'outer_rejection_patience', ['outer_rejection_patience']),
    (9, 2, 'outer_rejection_streak', ['outer_rejection_streak']),
    (1, 2, 'outer_rejection_patience', ['outer_rejection_patience', 'outer_rejection_streak']),
])
def test_outer_rejection_rules_keep_rollback_and_simultaneous_reasons(
        monkeypatch, patience, reject_stop, reason, triggers):
    # First catastrophic reject is free; the identical second reject consumes
    # patience. Neither rejected position enters the archive.
    result, calls, _, callbacks = run_scripted(monkeypatch,
        [(1., .1), (2., .5), (2., .5)], outer_merit=True,
        patience=patience, reject_stop=reject_stop)
    termination = assert_reason(result, reason, 3)
    assert [event['reason'] for event in termination['triggers']] == triggers
    assert all(event['attempt_index'] == 2 for event in termination['triggers'])
    assert len(calls) == len(callbacks) == 3
    assert np.array_equal(calls[1], calls[2])
    assert [h.get('outer_rejected', 0) for h in result['history']] == [0, 1, 1, 0]
    assert len(result['frames']) == 3
    assert np.array_equal(result['frames'][1], result['frames'][2])
    assert all(np.array_equal(callbacks[i][1], calls[1]) for i in (1, 2))


def test_cycle_rule_can_stop_despite_continuing_objective_progress(monkeypatch):
    result, calls, _, callbacks = run_scripted(monkeypatch,
        [(1., .1), (.9, -.1), (.8, .1), (.7, -.1)], stop_on_cycle=True)
    termination = assert_reason(result, 'displacement_cycle', 4)
    assert [h.get('stale') for h in result['history']] == [0, 0, 0, 0, None]
    assert termination['triggers'][0]['evidence']['cycle_stale'] == 2
    assert len(calls) == len(callbacks) == 4
    assert len(result['frames']) == 6 and result['n_held'] == 1


@pytest.mark.parametrize('animations,cap,count,reason', [
    (2, 0, 2, 'configured_window_budget'),
    (5, 2, 2, 'manual_window_cap'),
    (2, 2, 2, 'configured_window_budget'),
    (2, 7, 2, 'configured_window_budget'),
    (0, 0, 0, 'configured_window_budget'),
])
def test_budget_and_manual_cap_are_not_freeze_or_rest(monkeypatch, animations, cap, count, reason):
    result, calls, _, callbacks = run_scripted(monkeypatch, [(1., .1), (.8, .1)],
        animations=animations, stop_after_windows=cap, patience=9)
    termination = assert_reason(result, reason, count, frozen=False)
    assert termination['triggers'] == []
    assert termination['effective_window_limit'] == count
    assert len(calls) == len(callbacks) == count
    assert len(result['frames']) == 1 + count and result['n_held'] == 0


def test_freeze_on_last_budget_attempt_takes_precedence_without_extra_hold(monkeypatch):
    result, calls, _, _ = run_scripted(monkeypatch, ['null', 'null'], animations=2)
    assert_reason(result, 'null_commit_patience', 2)
    assert len(calls) == 2 and len(result['frames']) == 3 and result['n_held'] == 0
