from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from scripts.probes.control_withdrawal import (WithdrawalCapture, coast_arrays, coast_health,
                                              summarize_cohorts, bound_json, verify_files)
from scripts.probes.full_horizon import RestTrace


def test_capture_preserves_real_cpu_runner_and_exact_successor(tmp_path, monkeypatch):
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig, runner
    source = np.random.default_rng(27).uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (source * [1.2, .85, 1.05] + [.1, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=2, animations=2, loss_res=12, render_views=2,
        render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
        outer_render_committed=True, body_ctrl=True, body_terminal_ctrl=True,
        lambda_auto=.3, w_kin=.2, w_kin_var=.3, w_ctrl=.001, w_jvol=.5, w_box=0., max_ls_iters=1,
        adaptive_alpha=False, alpha=1e-4, replay_calibrate=False, phys_loss='ot_pace',
        loss_units='density', ot_samples=128, ot_iters=20, render_paced=True,
        commit_pic=False, shift_sub=False, assim=.5, layer_ctrl=True, layer_relax=True, layer_k=8)
    prm = MPMParams(dx=1., nx=32, ny=32, nz=32)
    baseline = runner.run_pipeline(source, target, prm, deepcopy(cfg), log=lambda *_: None)
    capture = WithdrawalCapture(tmp_path / 'states', attempts=(0, 1))
    trace = RestTrace(tmp_path / 'trace')
    monkeypatch.setattr(runner, 'optimize_window', trace.wrap(capture.wrap(runner.optimize_window)))
    observed = runner.run_pipeline(source, target, prm, deepcopy(cfg), log=lambda *_: None)
    np.testing.assert_array_equal(observed['frames'], baseline['frames'])
    np.testing.assert_array_equal(observed['F_frames'], baseline['F_frames'])
    np.testing.assert_array_equal(observed['Fp'], baseline['Fp'])
    assert observed['guards'] == baseline['guards']
    for a, b in zip(observed['history'], baseline['history']):
        for key in ('loss', 'lambda', 'd_vol', 'd_sil', 'frame_end'):
            assert a.get(key) == b.get(key)
    report = capture.finish(trace.finish(observed))
    assert len(report['pairs']) == 1
    assert report['missing'] == [dict(source_attempt=1, head_committed=True,
                                      captured_head=True, captured_successor=False)]
    pair = report['pairs'][0]
    with np.load(capture.folder / pair['pre']['path']) as pre, \
            np.load(capture.folder / pair['post']['path']) as post:
        np.testing.assert_array_equal(pre['x0'], post['x0'])
        np.testing.assert_array_equal(pre['v0'], post['v0'])
        np.testing.assert_array_equal(pre['C0'], post['C0'])
        assert np.any(pre['Fp'] != post['Fp'])  # Actual assimilation was captured.
        assert 'layer_nrm' in pre.files and 'layer_nrm' in post.files


def test_rejected_successor_is_valid_but_rejected_head_is_not(tmp_path):
    observer = WithdrawalCapture(tmp_path / 'states', attempts=(0, 2))
    observer.records = {(i, role): dict(x0_sha256='same') for i, role in
                        ((0, 'pre'), (1, 'post'), (2, 'pre'), (3, 'post'))}
    trace = dict(attempts=[dict(animation=i, committed=i == 0, end_frame=3, start_frame=3,
                                x0_sha256='same', sidecar='test.npz', sha256='test') for i in range(4)],
                 actual_last_accepted=0, termination={'reason': 'test'})
    result = observer.finish(trace)
    assert len(result['pairs']) == 1 and not result['pairs'][0]['successor_committed']
    assert result['missing'][0]['source_attempt'] == 2
    trace['attempts'][1]['x0_sha256'] = 'changed'
    with pytest.raises(ValueError, match='changed across'):
        observer.finish(trace)


def test_passive_readout_distinguishes_drift_reversal_and_empty_cohorts():
    x = np.array([[[0, 0, 0], [0, 0, 0]], [[1, 0, 0], [1, 0, 0]],
                  [[2, 0, 0], [0, 0, 0]]], np.float32)
    identity = np.tile(np.eye(3, dtype=np.float32), (2, 1, 1))
    tr = SimpleNamespace(x=[wp.array(a, dtype=wp.vec3, device='cpu') for a in x],
        v=[wp.array(np.zeros((2, 3), np.float32), dtype=wp.vec3, device='cpu')] * 3,
        F=[wp.array(identity, dtype=wp.mat33, device='cpu')] * 3, prm=SimpleNamespace(dt=.5))
    fields, _ = coast_arrays(tr, np.ones((2, 3), np.float32), spacing=2.)
    np.testing.assert_array_equal(fields['net_squared'], [1., 0.])
    np.testing.assert_array_equal(fields['reversals'], [0, 1])
    np.testing.assert_array_equal(fields['geometric_speed_squared'], [4., 4.])
    result = summarize_cohorts(fields, dict(all=torch.tensor([True, True]),
                                          empty=torch.tensor([False, False])))
    assert result['empty'] == dict(count=0, values=None)
    assert result['all']['values']['net_rms_sp'] == pytest.approx(np.sqrt(.5))
    assert result['all']['values']['step_rms_sp'] == .5
    assert result['all']['values']['geometric_speed_rms_wu_s'] == 2.


def test_health_rejects_affine_nan_and_pin_motion():
    from test_boundary_packet import _trajectory
    tr = _trajectory(pin_slip=True)
    tr.run()
    assert coast_health(tr)['valid']
    wp.to_torch(tr.C[2])[3, 0, 0] = float('nan')
    bad = coast_health(tr)
    assert not bad['valid'] and bad['nonfinite_C'] == 1
    wp.to_torch(tr.C[2])[3, 0, 0] = 0.
    pins = wp.to_torch(tr.pin) > .5
    wp.to_torch(tr.x[2])[pins] += .01
    bad = coast_health(tr)
    assert not bad['valid'] and bad['pinned_position_changes'] > 0


def test_provenance_parses_exact_bytes_and_rejects_substitution(tmp_path):
    path = tmp_path / 'source.json'
    path.write_text('{"source": 1}')
    value, digest = bound_json(path)
    assert value['source'] == 1
    verify_files({str(path): digest})
    path.write_text('{"source": 2}')
    with pytest.raises(ValueError, match='source changed'):
        verify_files({str(path): digest})
