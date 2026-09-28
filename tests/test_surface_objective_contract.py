"""Actual CPU MPM integration; only the CUDA surface raster is a test double.

These tests validate objective plumbing, not the Gaussian operator or its CUDA
adjoint. Production's CUDA-only validation remains tested without a bypass.
"""
from copy import deepcopy

import numpy as np
import pytest
import torch

from physmorph.mpm.state import MPMParams
from physmorph.pipeline import PipelineConfig, run_pipeline


def _recipe(**overrides):
    source = np.random.default_rng(27).uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (source * [1.2, .85, 1.05] + [.1, 0, 0]).astype(np.float32)
    options = dict(T=3, iters=2, animations=1, loss_res=12, render_views=2,
        render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
        commit_pic=True, commit_pic_objective=True, outer_render_committed=True,
        body_ctrl=True, body_terminal_ctrl=True, lambda_auto=.3,
        w_kin=.2, w_kin_var=0., w_ctrl=0., w_box=0., max_ls_iters=1,
        adaptive_alpha=False, alpha=1e-4, replay_calibrate=False,
        surface_gs_loss=True, surface_gs_weight=.7)
    options.update(overrides)
    return source, target, MPMParams(dx=1., nx=32, ny=32, nz=32), PipelineConfig(**options)


@pytest.fixture
def mocked_surface(monkeypatch):
    from physmorph.pipeline import endpoint_contract, surface_render_loss
    validate = endpoint_contract.validate_endpoint_config

    def allow_test_raster_on_cpu(cfg):
        # Bypass ONLY the new raster's CUDA requirement. All other production
        # endpoint restrictions still run, without modifying the live config.
        if cfg.surface_gs_loss:
            shadow = deepcopy(cfg)
            shadow.device, shadow.compute_backend = 'cuda', 'cuda'
            validate(shadow)
        else:
            validate(cfg)

    monkeypatch.setattr(endpoint_contract, 'validate_endpoint_config', allow_test_raster_on_cpu)
    bundles = []

    class Window:
        def __init__(self, reference, kind):
            self.reference = reference.detach().clone()
            self.kind, self.calls, self.seeds = kind, [], []

        def __call__(self, x):
            assert bool(torch.isfinite(x).all()), 'unsafe candidate reached the GS raster'
            self.calls.append((x.detach().clone(), bool(x.requires_grad)))
            # A genuinely position-dependent, independently reconstructible
            # scalar. No shared physics/grid/silhouette operator is used here.
            value = (x-self.reference).square().sum(1).mean()
            if value.requires_grad:
                value.register_hook(lambda seed: self.seeds.append(float(seed)))
            return value, dict(coverage=value, detail_coverage=value*0, detail_edge=value*0)

        def metadata(self):
            return dict(model='CPU_TEST_QUADRATIC_NOT_GAUSSIAN', target_kind=self.kind)

    class Views:
        def __init__(self, reference, views, **kwargs):
            self.reference = reference.detach().clone()
            self.windows = []
            bundles.append(self)

        def prepare(self, reference, kind):
            window = Window(reference, kind)
            self.windows.append(window)
            return window

    monkeypatch.setattr(surface_render_loss, 'SurfaceRenderViews', Views)
    return bundles


@pytest.mark.parametrize('reject_last', [False, True])
@pytest.mark.parametrize('units', ['legacy', 'density'])
def test_surface_term_reaches_gradient_accepted_merit_and_replay(monkeypatch, mocked_surface,
                                                              reject_last, units):
    from physmorph.pipeline import runner, optimizer
    from physmorph.pipeline.render_loss import d_render
    source, target, prm, cfg = _recipe(loss_units=units)
    original_state_ok = optimizer._state_ok
    checks = []

    def state_ok(state):
        checks.append(None)
        # First inner step accepted; second candidate overwrites the buffer but
        # is rejected, forcing the actual restored-control final replay path.
        return False if reject_last and len(checks) == 2 else original_state_ok(state)

    monkeypatch.setattr(optimizer, '_state_ok', state_ok)
    original = runner.optimize_window
    captured, callbacks = [], []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        frames, _, _, _, history, stats = result
        assert len(history) == (1 if reject_last else 2)
        pack, owned = args[3], stats['owned_endpoint']
        assert owned is not None
        gs = np.square(owned.promoted.numpy().astype(np.float64)-target).sum(1).mean()
        cic = float(d_render(owned.promoted, pack.sils, pack.views, cfg.render_res,
            pack.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray))
        assert history[-1]['surface_render']['total'] == pytest.approx(gs, rel=3e-6, abs=1e-9)
        assert history[-1]['d_render'] == pytest.approx(cic, rel=3e-6, abs=1e-9)
        assert history[-1]['d_sil'] == history[-1]['d_render']
        for row in history:
            expected = row['d_vol'] + cfg.w_kin*row['kin']/pack.unit_ratio
            expected += row['lambda'] * (row['d_sil'] + cfg.surface_gs_weight*row['surface_render']['total'])
            assert row['loss'] == pytest.approx(expected, rel=4e-6, abs=1e-9)
        assert stats['commit_from_accepted'] is not reject_last
        replay = stats['replay_diagnostics']
        if reject_last:
            assert replay['commit_source'] == 'replay'
            assert replay['replay_E_final'] == pytest.approx(history[-1]['loss'], rel=4e-6, abs=1e-8)
        else:
            assert replay['commit_source'] == 'accepted_buffer'
            assert replay['replay_E_final'] is None
        np.testing.assert_array_equal(frames[-1], owned.raw.numpy())
        captured.append((owned.promoted.numpy().copy(), history, stats))
        return result

    monkeypatch.setattr(runner, 'optimize_window', capture)
    result = run_pipeline(source, target, prm, cfg, log=lambda *_: None,
        on_iter=lambda _i, _x, _F, row: callbacks.append(deepcopy(row)))
    records = [r for r in result['history'] if r.get('frame_end')]
    assert len(records) == len(captured) == len(mocked_surface) == 1
    assert not any(result['guards'].values())
    window, = mocked_surface[0].windows
    # The first render-channel backward is unweighted by lambda, so this seed
    # independently verifies that the new coefficient participates in autograd.
    assert any(seed == pytest.approx(cfg.surface_gs_weight) for seed in window.seeds)
    assert all(np.isfinite(window.seeds))
    assert any(differentiable for _, differentiable in window.calls)
    assert any(not differentiable for _, differentiable in window.calls)
    np.testing.assert_array_equal(result['frames'][records[0]['frame_end']-1], captured[0][0])
    for callback, row in zip(callbacks, captured[0][1]):
        assert callback['d_render'] == pytest.approx(row['d_sil'])
    if reject_last:
        # Last raster call must be the restored accepted promoted state, not the
        # rejected candidate left in the evaluation scratch buffers.
        torch.testing.assert_close(window.calls[-1][0], torch.from_numpy(captured[0][0]), rtol=0, atol=0)


def test_zero_surface_weight_preserves_real_two_window_baseline(mocked_surface):
    source, target, prm, cfg = _recipe(animations=2, surface_gs_weight=0.)
    baseline_cfg = deepcopy(cfg); baseline_cfg.surface_gs_loss = False
    baseline = run_pipeline(source, target, prm, baseline_cfg, log=lambda *_: None)
    observed = run_pipeline(source, target, prm, cfg, log=lambda *_: None)
    np.testing.assert_array_equal(observed['frames'], baseline['frames'])
    np.testing.assert_array_equal(observed['F_frames'], baseline['F_frames'])
    assert observed['guards'] == baseline['guards']
    a = [r for r in baseline['history'] if r.get('frame_end')]
    b = [r for r in observed['history'] if r.get('frame_end')]
    assert len(a) == len(b) == 2
    for left, right in zip(a, b):
        for field in ('loss', 'lambda', 'accepted', 'body_accepted_alphas', 'd_render', 'd_sil'):
            assert left[field] == right[field]
        assert right['surface_render']['total'] > 0
    assert sum(len(bundle.windows) for bundle in mocked_surface) == 2
    assert not any(window.seeds for bundle in mocked_surface for window in bundle.windows)


@pytest.mark.parametrize('corruption', ['nan', 'outside'])
def test_unsafe_candidate_skips_surface_and_smaller_trial_can_succeed(monkeypatch,
                                                                   mocked_surface, corruption):
    from physmorph.pipeline import optimizer, runner
    from physmorph.mpm.endpoint_filter import FixedEndpointFilter
    source, target, prm, cfg = _recipe(iters=1, max_ls_iters=3)
    proposals, poisoned = [], []
    apply, endpoint = optimizer.apply_proposal, FixedEndpointFilter.endpoint
    pending = [False]

    def proposal(*args, **kwargs):
        result = apply(*args, **kwargs)
        proposals.append(kwargs['alpha'])
        pending[0] = len(proposals) == 1
        return result

    def poison(self, raw, pin_mask=None):
        mapped = endpoint(self, raw, pin_mask)
        if pending[0]:
            assert not torch.is_grad_enabled()
            pending[0] = False
            mapped = mapped.clone()
            mapped[0, 0] = float('nan') if corruption == 'nan' else 1000.
            poisoned.append(mapped.clone())
        return mapped

    captured, original = [], runner.optimize_window
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result
    monkeypatch.setattr(optimizer, 'apply_proposal', proposal)
    monkeypatch.setattr(FixedEndpointFilter, 'endpoint', poison)
    monkeypatch.setattr(runner, 'optimize_window', capture)
    result = run_pipeline(source, target, prm, cfg, log=lambda *_: None)
    assert len(poisoned) == 1 and len(proposals) >= 2
    assert proposals[1] == proposals[0]*.5
    assert len(captured[0][4]) == 1 and captured[0][5]['commit_from_accepted']
    assert not any(result['guards'].values())
    calls = mocked_surface[0].windows[0].calls
    assert all(bool(torch.isfinite(x).all()) and float(x.abs().max()) < 100 for x, _ in calls)
    assert captured[0][4][0]['render_influence']['backtracks'] >= 1


def test_surface_cpu_production_configuration_still_rejected():
    from physmorph.pipeline.endpoint_contract import validate_endpoint_config
    *_, cfg = _recipe()
    with pytest.raises(ValueError, match='CUDA'):
        validate_endpoint_config(cfg)
