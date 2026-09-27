"""CPU integration contracts for arrived/free geometric motion, not quality gates."""
import copy

import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, run_pipeline


def _fixture(**overrides):
    rng = np.random.default_rng(27)
    source = rng.uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (source * [1.2, .85, 1.05] + [.1, 0, 0]).astype(np.float32)
    options = dict(T=3, iters=2, animations=1, loss_res=12, render_views=2,
                   render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
                   phys_loss='ot_pace', ot_samples=64, ot_iters=4,
                   commit_pic=True, commit_pic_objective=True, geometric_rest=True,
                   body_ctrl=True, body_terminal_ctrl=True, lambda_auto=.3,
                   w_kin=.2, w_ctrl=0., w_box=0., max_ls_iters=1,
                   adaptive_alpha=False, alpha=1e-4)
    options.update(overrides)
    return source, target, MPMParams(dx=1., nx=32, ny=32, nz=32), PipelineConfig(**options)


def _measured(previous, raw, promoted, eligible, dt):
    # Use archive-space NumPy arithmetic independently of terminal_motion.
    a = (raw.astype(np.float64) - previous) / dt
    b = (promoted.astype(np.float64) - raw) / dt
    n = len(raw)
    raw_sq = np.square(a[eligible]).sum() / n
    remap_sq = np.square(b[eligible]).sum() / n
    return dict(raw_sq=raw_sq, remap_sq=remap_sq, total=raw_sq + remap_sq,
                delivered_sq=np.square((a+b)[eligible]).sum() / n,
                cross=2 * (a[eligible] * b[eligible]).sum() / n)


def _assert_telemetry(actual, expected, eligible, cfg, prm):
    assert actual['eligible_count'] == int(eligible.sum())
    assert actual['eligible_fraction'] == pytest.approx(eligible.mean())
    assert actual['normalization'] == 'all_particles'
    assert actual['weight'] == cfg.w_kin
    assert actual['unit_multiplier'] > 0 and np.isfinite(actual['unit_multiplier'])
    assert actual['effective_weight'] == pytest.approx(actual['unit_multiplier'] * cfg.w_kin)
    if cfg.loss_units == 'legacy':
        assert actual['unit_multiplier'] == 1.
    assert actual['dt'] == prm.dt
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, rel=3e-6, abs=1e-9)
    assert actual['delivered_sq'] == pytest.approx(actual['total'] + actual['cross'],
                                                rel=3e-6, abs=1e-9)
    if eligible.any():
        for key, value in expected.items():
            assert actual['conditional'][key] == pytest.approx(value / eligible.mean(),
                                                             rel=3e-6, abs=1e-9)
    else:
        assert actual['conditional'] is None


@pytest.mark.parametrize('reject_last_trial', [False, True])
@pytest.mark.parametrize('loss_units', ['legacy', 'density'])
def test_actual_rollout_geometry_agrees_through_acceptance_and_replay(monkeypatch, reject_last_trial, loss_units):
    from physmorph.pipeline import geometric_rest, optimizer, runner

    source, target, prm, cfg = _fixture(loss_units=loss_units)
    calls, previous_gradients, total_gradients = [], [], []
    original_motion = geometric_rest.terminal_motion

    def observe_motion(raw, previous, promoted, eligible, dt):
        before = [v.detach().clone() for v in (raw, previous, promoted, eligible)]
        motion = original_motion(raw, previous, promoted, eligible, dt)
        for value, saved in zip((raw, previous, promoted, eligible), before):
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
        if previous.requires_grad:
            previous.register_hook(lambda g: previous_gradients.append(g.detach().clone()) if g is not None else None)
            motion['total'].register_hook(lambda g: total_gradients.append(float(g)))
        calls.append((raw.requires_grad, previous.requires_grad))
        return motion

    monkeypatch.setattr(geometric_rest, 'terminal_motion', observe_motion)
    state_calls = []
    original_state_ok = optimizer._state_ok

    def state_ok(state):
        state_calls.append(None)
        if reject_last_trial and len(state_calls) == 2:
            return False  # A real second rollout already overwrote the evaluation buffer.
        return original_state_ok(state)

    monkeypatch.setattr(optimizer, '_state_ok', state_ok)
    observed, callbacks, commits = [], [], []
    original_optimize = runner.optimize_window

    def capture(*args, **kwargs):
        result = original_optimize(*args, **kwargs)
        frames, _, end, _, history, stats = result
        assert history, 'fixture must accept at least one actual optimization step'
        package = stats['owned_endpoint']
        unit_multiplier = 1. / args[3].unit_ratio
        assert stats['geometric_rest']['unit_multiplier'] == pytest.approx(unit_multiplier)
        if loss_units == 'density':
            assert not np.isclose(unit_multiplier, 1.), 'fixture must exercise unit conversion'
        eligible = stats['geometric_rest_mask'].numpy().copy()
        np.testing.assert_array_equal(eligible, stats['arrived_mask'] & ~package.pin.numpy())
        assert eligible.any()
        assert package.previous is not None and not package.previous.requires_grad
        np.testing.assert_array_equal(package.previous.numpy(), frames[-2])
        np.testing.assert_array_equal(package.raw.numpy(), frames[-1])
        expected = _measured(frames[-2], frames[-1], package.promoted.numpy(), eligible, prm.dt)
        for telemetry in (stats['geometric_rest'], stats['replay_diagnostics']['geometric_rest'],
                          history[-1]['geometric_rest'], callbacks[-1]['geometric_rest']):
            _assert_telemetry(telemetry, expected, eligible, cfg, prm)
        assert expected['total'] > 0 and expected['remap_sq'] > 0
        assert stats['commit_from_accepted'] is (not reject_last_trial)
        assert package.source == ('replay' if reject_last_trial else 'accepted_buffer')
        for row in history:
            # Check the term participates in accepted merit, beyond being logged.
            expected_loss = (row['d_vol'] + row['geometric_rest']['effective_weight']
                             * (row['kin'] + row['geometric_rest']['total'])
                             + row['lambda'] * row['d_render'])
            assert row['loss'] == pytest.approx(expected_loss, rel=3e-6, abs=1e-8)
        if reject_last_trial:
            replay = stats['replay_diagnostics']
            assert replay['replay_previous_max'] == 0.
            assert replay['replay_E_final'] <= replay['replay_E_accepted'] + replay['replay_E_tol']
            last = history[-1]
            expected_replay = (last['d_vol'] + last['geometric_rest']['effective_weight']
                               * (last['kin'] + expected['total'])
                               + replay['replay_lambda_final'] * last['d_render'])
            assert replay['replay_E_final'] == pytest.approx(expected_replay, rel=3e-6, abs=1e-8)
        observed.append(dict(expected=expected, mask=eligible, promoted=package.promoted.numpy().copy(),
                             F=end['F'].copy(), v=end['v'].copy(), history=copy.deepcopy(history)))
        return result

    monkeypatch.setattr(runner, 'optimize_window', capture)
    result = run_pipeline(source, target, prm, cfg, log=lambda *_: None,
                          on_iter=lambda _i, _x, _F, tele: callbacks.append(copy.deepcopy(tele)),
                          on_commit=lambda _i, _x, F, v, _rec: commits.append((F.copy(), v.copy())))
    rows = [r for r in result['history'] if r.get('frame_end')]
    assert len(rows) == len(observed) == len(commits) == 1
    assert not any(result['guards'].values())
    assert rows[0]['lambda'] > 0 and callbacks[-1]['d_render'] is not None
    assert len(observed[0]['history']) == (1 if reject_last_trial else 2)
    item = observed[0]
    _assert_telemetry(rows[0]['geometric_rest'], item['expected'], item['mask'], cfg, prm)
    np.testing.assert_array_equal(result['frames'][rows[0]['frame_end']-1], item['promoted'])
    np.testing.assert_array_equal(commits[0][0].reshape(-1, 9), item['F'].reshape(-1, 9))
    np.testing.assert_array_equal(commits[0][1], item['v'])
    assert any(raw_grad and previous_grad for raw_grad, previous_grad in calls)
    assert any(not raw_grad and not previous_grad for raw_grad, previous_grad in calls)
    assert all(torch.isfinite(g).all() for g in previous_gradients)
    assert any(float(g.abs().max()) > 0 for g in previous_gradients)
    assert any(g == pytest.approx(rows[0]['geometric_rest']['effective_weight']) for g in total_gradients)


def test_empty_arrived_cohort_runs_render_and_reports_zero_geometry(monkeypatch):
    from physmorph.pipeline import runner

    source, _, prm, cfg = _fixture()
    target = source + np.array([5., 0., 0.], np.float32)
    captured = []
    original = runner.optimize_window

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        stats = result[-1]
        assert not stats['geometric_rest_mask'].any()
        captured.append(copy.deepcopy(stats['geometric_rest']))
        return result

    monkeypatch.setattr(runner, 'optimize_window', capture)
    result = run_pipeline(source, target, prm, cfg, log=lambda *_: None)
    rows = [r for r in result['history'] if r.get('frame_end')]
    assert len(rows) == len(captured) == 1
    assert not any(result['guards'].values())
    assert rows[0]['lambda'] > 0
    expected = dict.fromkeys(('raw_sq', 'remap_sq', 'delivered_sq', 'cross', 'total'), 0.)
    for telemetry in (captured[0], rows[0]['geometric_rest']):
        _assert_telemetry(telemetry, expected, np.zeros(len(source), bool), cfg, prm)


def test_runner_rejects_stale_previous_rollout_position(monkeypatch):
    from physmorph.pipeline import runner

    source, target, prm, cfg = _fixture()
    original = runner.optimize_window

    def corrupt_previous(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result[4]
        package = result[-1]['owned_endpoint']
        saved = package.previous.clone()
        result[0][-2][0, 0] += .01
        torch.testing.assert_close(package.previous, saved, rtol=0, atol=0)
        return result

    monkeypatch.setattr(runner, 'optimize_window', corrupt_previous)
    with pytest.raises(ValueError, match='previous position no longer matches'):
        run_pipeline(source, target, prm, cfg, log=lambda *_: None)


@pytest.mark.parametrize('weight', [0., -1., float('nan'), float('inf')])
def test_geometric_rest_rejects_invalid_kinetic_weight_before_rollout(monkeypatch, weight):
    from physmorph.pipeline import runner

    source, target, prm, cfg = _fixture(w_kin=weight)
    monkeypatch.setattr(runner, 'optimize_window', lambda *_a, **_k: pytest.fail('invalid config reached rollout'))
    with pytest.raises(ValueError, match='positive finite w_kin'):
        run_pipeline(source, target, prm, cfg, log=lambda *_: None)


def test_optimizer_missing_arrival_cohort_fails_closed():
    from physmorph.pipeline.optimizer import optimize_window
    from physmorph.pipeline.render_loss import LambdaBalancer
    from physmorph.pipeline.runner import build_target

    source, target, prm, cfg = _fixture(phys_loss='auto')
    # A caller that bypasses the runner has not resolved auto to an arrival mode.
    # It must not silently penalize all particles or invent an empty cohort.
    pack = build_target(target, prm, cfg)
    with pytest.raises(ValueError, match='frozen full-plan arrival mask'):
        optimize_window(source, prm, cfg, pack, LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema),
                        log=lambda *_: None)


def test_existing_pins_are_excluded_without_admitting_new_pins():
    from physmorph.pipeline.optimizer import optimize_window
    from physmorph.pipeline.render_loss import LambdaBalancer
    from physmorph.pipeline.runner import build_target

    source, target, prm, cfg = _fixture()
    pin = np.zeros(len(source), np.float32)
    pin[::7] = 1.
    original_pin = pin.copy()
    pack = build_target(target, prm, cfg)
    frames, _, _, _, history, stats = optimize_window(
        source, prm, cfg, pack, LambdaBalancer(cfg.lambda_auto, cfg.lambda_ema),
        pin_init=pin, log=lambda *_: None)
    assert history
    package = stats['owned_endpoint']
    expected_mask = stats['arrived_mask'] & (original_pin == 0)
    np.testing.assert_array_equal(pin, original_pin)
    np.testing.assert_array_equal(package.pin.numpy(), original_pin > .5)
    np.testing.assert_array_equal(stats['geometric_rest_mask'].numpy(), expected_mask)
    assert 0 < expected_mask.sum() < len(source)
    np.testing.assert_array_equal(package.promoted.numpy()[pin > .5], source[pin > .5])
    expected = _measured(frames[-2], frames[-1], package.promoted.numpy(), expected_mask, prm.dt)
    _assert_telemetry(stats['geometric_rest'], expected, expected_mask, cfg, prm)
    # Diagnostic ownership must survive mutation of caller-owned input/arrival arrays.
    pin[:] = 0.
    stats['arrived_mask'][:] = False
    np.testing.assert_array_equal(package.pin.numpy(), original_pin > .5)
    np.testing.assert_array_equal(stats['geometric_rest_mask'].numpy(), expected_mask)
