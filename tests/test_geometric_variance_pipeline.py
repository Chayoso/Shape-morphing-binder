"""Check the optional saved-path objective against independent archive arithmetic."""
import copy
import numpy as np
import pytest
import torch

from physmorph.pipeline import PipelineConfig, run_pipeline
from physmorph.mpm.state import MPMParams


@pytest.mark.parametrize('reject_last', [False, True])
@pytest.mark.parametrize('units', ['legacy', 'density'])
def test_path_variance_matches_archive_and_merit(monkeypatch, reject_last, units):
    from physmorph.pipeline import runner, optimizer, geometric_variance
    source = np.random.default_rng(27).uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (source * [1.2, .85, 1.05] + [.1, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=2, animations=1, loss_res=12, render_views=2,
                         render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
                         commit_pic=True, commit_pic_objective=True, geometric_variance=True,
                         body_ctrl=True, body_terminal_ctrl=True, lambda_auto=.3,
                         w_kin=.2, w_kin_var=.7, w_ctrl=0., w_box=0., max_ls_iters=1,
                         adaptive_alpha=False, alpha=1e-4, loss_units=units)
    prm = MPMParams(dx=1., nx=32, ny=32, nz=32)
    state_calls = []
    state_ok = optimizer._state_ok
    def controlled_check(state):
        state_calls.append(None)
        return False if reject_last and len(state_calls) == 2 else state_ok(state)
    monkeypatch.setattr(optimizer, '_state_ok', controlled_check)
    gradients, weight_seeds = [], []
    original_rates = geometric_variance.path_rates
    original_var = geometric_variance.temporal_variance
    def rates(start, path, end, dt):
        if path.requires_grad:
            path.register_hook(lambda g: gradients.append(g.detach().clone()) if g is not None else None)
        return original_rates(start, path, end, dt)
    def variance(v):
        result = original_var(v)
        if result.requires_grad:
            result.register_hook(lambda g: weight_seeds.append(float(g)))
        return result
    monkeypatch.setattr(geometric_variance, 'path_rates', rates)
    monkeypatch.setattr(geometric_variance, 'temporal_variance', variance)
    callbacks, records = [], []
    original = runner.optimize_window
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        frames, _, end, _, history, stats = result
        assert history
        raw = np.stack(frames).astype(np.float64)
        saved = raw.copy(); saved[-1] = stats['owned_endpoint'].promoted.numpy()
        observed_rates = np.diff(saved, axis=0)/prm.dt
        expected = np.square(observed_rates-observed_rates.mean(0)).sum(-1).mean()
        effective = cfg.w_kin_var/args[3].unit_ratio
        for report in (stats['geometric_variance'], stats['replay_diagnostics']['geometric_variance'],
                       history[-1]['geometric_variance'], callbacks[-1]['geometric_variance']):
            assert report['geometric_variance'] == pytest.approx(expected, rel=4e-6, abs=1e-9)
            assert report['effective_weight'] == pytest.approx(effective)
            assert report['saved_phase_rms_wu'] == pytest.approx(
                np.sqrt(np.square(np.diff(saved, axis=0)).sum(-1).mean(-1)), rel=4e-6, abs=1e-9)
        for row in history:
            expected_loss = row['d_vol']+(cfg.w_kin*row['kin']+cfg.w_kin_var*row['kin_var'])/args[3].unit_ratio
            expected_loss += row['lambda']*row['d_render']
            assert row['loss'] == pytest.approx(expected_loss, rel=4e-6, abs=1e-8)
            assert row['kin_var'] == row['geometric_variance']['geometric_variance']
        assert stats['commit_from_accepted'] is not reject_last
        if reject_last:
            replay = stats['replay_diagnostics']
            assert replay['replay_E_final'] <= replay['replay_E_accepted']+replay['replay_E_tol']
            assert replay['replay_E_final'] == pytest.approx(history[-1]['loss'], rel=4e-6, abs=1e-8)
        records.append((saved[-1].copy(), copy.deepcopy(stats['geometric_variance'])))
        return result
    monkeypatch.setattr(runner, 'optimize_window', capture)
    result = run_pipeline(source, target, prm, cfg, log=lambda *_: None,
                          on_iter=lambda _i, _x, _F, r: callbacks.append(copy.deepcopy(r)))
    accepted = [row for row in result['history'] if row.get('frame_end')]
    assert len(accepted) == len(records) == 1 and not any(result['guards'].values())
    np.testing.assert_array_equal(result['frames'][accepted[0]['frame_end']-1], records[0][0])
    assert accepted[0]['geometric_variance'] == records[0][1]
    assert any(g[:-1].abs().max() > 0 for g in gradients)
    assert any(s == pytest.approx(records[0][1]['effective_weight']) for s in weight_seeds)


@pytest.mark.parametrize('overrides', [dict(commit_pic_objective=False), dict(geometric_rest=True),
    dict(w_kin_var=0.), dict(w_kin_var=float('nan')), dict(w_kin_var=float('inf')), dict(T=1),
    dict(settle_pin_kkt=True)])
def test_invalid_variance_scope_rejected_before_rollout(overrides):
    from physmorph.pipeline.endpoint_contract import validate_endpoint_config
    options = dict(commit_pic=True, commit_pic_objective=True, geometric_variance=True, w_kin_var=1.)
    options.update(overrides)
    with pytest.raises(ValueError, match='geometric_variance'):
        validate_endpoint_config(PipelineConfig(**options))
