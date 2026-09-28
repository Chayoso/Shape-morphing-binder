import numpy as np
import pytest
import torch

from scripts.probes.pic_components import (Capture, QUALITY_DIRECTIONS,
                                           endpoint_admissibility, quality_contrast)


def test_quality_screen_does_not_hide_small_regression_or_missing_evidence():
    current = {key: 1. for key in QUALITY_DIRECTIONS}
    other = current.copy()
    other['sil_iou'] -= 1e-10
    contrast = quality_contrast(current, other)
    assert not contrast['all_observed_no_worse']
    assert not contrast['no_worse']['sil_iou']
    other = current.copy()
    other['top_target_near_frac'] = None
    assert quality_contrast(current, other)['no_worse']['top_target_near_frac'] is None
    assert not quality_contrast(current, other)['all_observed_no_worse']


def test_candidate_admissibility_rejects_pin_shape_nonfinite_and_bounds_errors():
    start = torch.zeros(3, 3)
    pin = torch.tensor([True, False, False])
    bounds = (torch.full((3,), -1.), torch.ones(3))
    assert endpoint_admissibility(start.clone(), start, pin, bounds)['passed']
    assert not endpoint_admissibility(start[:2], start, pin, bounds)['passed']
    for row, value in ((0, .1), (1, 2.), (2, float('nan'))):
        changed = start.clone(); changed[row, 0] = value
        assert not endpoint_admissibility(changed, start, pin, bounds)['passed']


def test_rejected_attempt_never_admits_diagnostic_geometry(tmp_path):
    c = Capture(tmp_path, None, (1,))
    c.pending = dict(attempt=1, selected=True)
    c.commit(0, None, None, None, {'outer_rejected': True})
    assert c.accepted == 0 and c.pending is None and c.rows == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('force_invalid', [False, True])
def test_real_accepted_endpoint_uses_owned_operator_and_committed_geometry(tmp_path, monkeypatch, force_invalid):
    from physmorph.pipeline import runner, PipelineConfig
    from physmorph.mpm.state import MPMParams
    from scripts.probes.pic_endpoint_quality import prepare_reference
    src = np.random.default_rng(27).uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (src * [1.2, .85, 1.05] + [.1, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=3, animations=2, loss_res=12, render_views=2,
                         render_elevs=(0., .5), render_res=24, device='cpu', patience=5,
                         commit_pic=True, commit_pic_objective=True, body_ctrl=True,
                         lambda_auto=.3, w_kin=0., w_ctrl=0., w_box=0.)
    capture = Capture(tmp_path, prepare_reference(src, target), (1,))
    monkeypatch.setattr(runner, 'optimize_window', capture.wrap(runner.optimize_window))
    if force_invalid:
        from scripts.probes import pic_component_metrics
        original = pic_component_metrics.decompose_pic
        def invalid(*args, **kwargs):
            part = original(*args, **kwargs)
            part['report']['valid'] = False
            return part
        monkeypatch.setattr(pic_component_metrics, 'decompose_pic', invalid)
        with pytest.raises(RuntimeError, match='Accepted decomposition failed'):
            runner.run_pipeline(src, target, MPMParams(dx=1., nx=32, ny=32, nz=32), cfg,
                                log=lambda *_: None, on_commit=capture.commit)
        import json
        saved = json.loads((tmp_path / 'endpoint_001.json').read_text())
        assert not saved['comparison_eligible']
        assert saved['comparison_ineligible_reason']
        assert all(value is None for value in saved['comparisons'].values())
        assert saved['geometry']['current'] is not None
        return
    result = runner.run_pipeline(src, target, MPMParams(dx=1., nx=32, ny=32, nz=32), cfg,
                                 log=lambda *_: None, on_commit=capture.commit)
    assert capture.accepted == 2 and len(capture.rows) == 1
    row = capture.rows[0]
    assert row['decomposition']['valid']
    assert row['admissible']['current']['passed']
    assert row['geometry']['current']['endpoint_n'] == 160
    with np.load(tmp_path / 'endpoint_001.npz') as archive:
        np.testing.assert_array_equal(archive['current'], result['frames'][3])
        np.testing.assert_array_equal(archive['start'], src)
