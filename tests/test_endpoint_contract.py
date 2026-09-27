"""Shared-endpoint scope, ownership and real optimizer/runner closure on CPU."""
import numpy as np
import pytest
import torch

from physmorph.pipeline import PipelineConfig, run_pipeline
from physmorph.pipeline.endpoint_contract import OwnedEndpoint, validate_endpoint_config
from physmorph.mpm.state import MPMParams


@pytest.mark.parametrize('option,value', [('commit_pic', False), ('shift_sub', True),
    ('reattach', True), ('settle_commit', True), ('lg_sweeps', 1), ('use_gauss_loss', True),
    ('grad_dump', 'diagnostic'), ('assim_consensus', True), ('w_grow', 1.), ('w_corr', 1.)])
def test_shared_endpoint_rejects_unmodeled_paths(option, value):
    cfg = PipelineConfig(commit_pic=True, commit_pic_objective=True)
    setattr(cfg, option, value)
    with pytest.raises(ValueError, match='commit_pic_objective'):
        validate_endpoint_config(cfg)
    cfg.commit_pic_objective = False
    validate_endpoint_config(cfg)  # historical configuration still accepted here


def test_shared_endpoint_rejects_active_outside_h1_but_allows_inside_h1():
    cfg = PipelineConfig(commit_pic=True, commit_pic_objective=True, w_h1=1.)
    validate_endpoint_config(cfg)
    cfg.h1_outside = True
    with pytest.raises(ValueError, match='outside-H1'):
        validate_endpoint_config(cfg)


def test_owned_endpoint_rejects_stale_inputs_and_forbidden_repairs():
    start = torch.zeros(4, 3)
    pin = torch.tensor([True, False, False, False])
    raw = torch.ones_like(start) * .1
    promoted = raw.clone(); promoted[pin] = start[pin]
    bounds = (-torch.ones(3), torch.ones(3))
    item = OwnedEndpoint.capture(object(), start, pin, raw, promoted, 'accepted_buffer')
    result = item.promote(raw, start, pin, bounds)
    result.zero_()
    torch.testing.assert_close(item.promoted, promoted, rtol=0, atol=0)
    for args in ((raw+.01, start, pin), (raw, start+.01, pin), (raw, start, ~pin)):
        with pytest.raises(ValueError, match='no longer matches'):
            item.promote(*args, bounds)
    for bad in (torch.full_like(promoted, float('nan')), promoted+2):
        invalid = OwnedEndpoint.capture(object(), start, pin, raw, bad, 'replay')
        with pytest.raises(ValueError, match='repair is forbidden'):
            invalid.promote(raw, start, pin, bounds)
    invalid = OwnedEndpoint.capture(object(), start, pin, raw, raw, 'replay')
    with pytest.raises(ValueError, match='changed a window-start pin'):
        invalid.promote(raw, start, pin, bounds)


@pytest.mark.parametrize('corruption', ['raw_position', 'affine_momentum'])
def test_runner_fails_closed_instead_of_repairing_shared_endpoint(monkeypatch, corruption):
    import physmorph.pipeline.runner as runner
    rng = np.random.default_rng(27)
    src = rng.uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (src * [1.2, .85, 1.05]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=2, animations=1, loss_res=12, device='cpu',
                         commit_pic=True, commit_pic_objective=True)
    original = runner.optimize_window

    def corrupt(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result[4], 'test requires an actually accepted rollout'
        if corruption == 'raw_position':
            result[0][-1][0] = 1000.
        else:
            result[2]['C'][0, 0, 0] = float('nan')
        return result

    monkeypatch.setattr(runner, 'optimize_window', corrupt)
    with pytest.raises(RuntimeError, match='forbids state repairs'):
        run_pipeline(src, target, MPMParams(dx=1., nx=32, ny=32, nz=32), cfg, log=lambda *_: None)


@pytest.mark.parametrize('body', [False, True])
def test_real_shared_pic_objective_equals_saved_endpoint(monkeypatch, body):
    import physmorph.pipeline.runner as runner
    from physmorph.losses.volumetric import d_vol
    from physmorph.pipeline.render_loss import d_render

    rng = np.random.default_rng(27)
    src = rng.uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (src * [1.2, .85, 1.05] + [.1, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=3, animations=2, loss_res=12, render_views=2,
                         render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
                         commit_pic=True, commit_pic_objective=True, body_ctrl=body,
                         lambda_auto=.3, w_kin=0., w_ctrl=0., w_box=0.)
    prm = MPMParams(dx=1., nx=32, ny=32, nz=32)
    observed = []
    original = runner.optimize_window

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        frames, _, _, _, history, stats = result
        item = stats['owned_endpoint']
        if history:
            pack = args[3]
            assert item is not None
            lv = float(d_vol(item.promoted, pack.m, pack.grid, pack.lgmin, pack.ldx, pack.ldims))
            lr = float(d_render(item.promoted, pack.sils, pack.views, cfg.render_res,
                                pack.extent, cfg.sil_k, cfg.w_hole, cfg.w_spray))
            assert history[-1]['d_vol'] == pytest.approx(lv, rel=2e-6, abs=1e-9)
            assert history[-1]['d_sil'] == pytest.approx(lr, rel=2e-6, abs=1e-9)
            np.testing.assert_array_equal(frames[-1], item.raw.numpy())
            # Retain only owned values, not another live stencil between windows.
            observed.append((item.start.numpy().copy(), item.raw.numpy().copy(), item.promoted.numpy().copy()))
        return result

    monkeypatch.setattr(runner, 'optimize_window', capture)

    def hostile_callback(_a, x, F, v, _rec):
        x[:] = 1000.; F[:] = float('nan'); v[:] = 1000.

    result = run_pipeline(src, target, prm, cfg, log=lambda *_: None, on_commit=hostile_callback)
    accepted = [r for r in result['history'] if r.get('frame_end')]
    assert len(accepted) == len(observed) == 2
    assert not any(result['guards'].values())
    assert max(np.abs(raw-promoted).max() for _, raw, promoted in observed) > 1e-7
    for i, (row, (start, raw, promoted)) in enumerate(zip(accepted, observed)):
        np.testing.assert_array_equal(result['frames'][row['frame_end']-1], promoted)
        np.testing.assert_array_equal(start, src if i == 0 else observed[i-1][2])
        assert row['endpoint_contract']['objective_commit_max_wu'] == 0.
        assert row['replay_diagnostics']['position_space'] == 'promoted_xpic'
        assert row['endpoint_contract']['source'] == row['replay_diagnostics']['commit_source']
