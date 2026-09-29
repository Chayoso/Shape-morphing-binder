"""The output cap must retain a failure receipt, without claiming a passed run."""
import json
import pytest
from scripts.probes.post_assimilation_window import Capture


def test_failure_reserve_survives_full_report_budget_exhaustion(tmp_path, monkeypatch):
    from scripts.probes import window_selection
    monkeypatch.setattr(window_selection, 'LIMIT', 70000)
    capture = Capture(tmp_path)
    capture.write_json('evidence.json', dict(text='x'*3000))
    outcome = capture.finish(dict(passed=True, failure=None, text='x'*10000))
    assert not outcome
    report = json.loads((tmp_path/'failure.json').read_text())
    assert report['passed'] is False
    assert report['finalization_failure']['type'] == 'ValueError'
    assert (tmp_path/'evidence.json').exists()
    assert not (tmp_path/'result.json').exists()


def test_original_failure_is_preserved_when_finalization_also_fails(tmp_path, monkeypatch):
    from scripts.probes import window_selection
    monkeypatch.setattr(window_selection, 'LIMIT', 70000)
    capture = Capture(tmp_path)
    failure = dict(type='ValueError', message='Actual handoff boundary differs')
    assert not capture.finish(dict(passed=False, failure=failure, text='x'*10000))
    report = json.loads((tmp_path/'failure.json').read_text())
    assert report['failure'] == failure


def test_small_complete_report_keeps_original_success_flag(tmp_path):
    capture = Capture(tmp_path)
    assert capture.finish(dict(passed=True, failure=None))
    assert json.loads((tmp_path/'result.json').read_text())['passed']
    assert not (tmp_path/'failure.json').exists()


@pytest.mark.parametrize('unsupported', ['assim_consensus', 'w_grow'])
def test_precision_recipe_rejects_unimplemented_maps_before_starting(unsupported):
    from physmorph.pipeline.config import PipelineConfig
    from physmorph.pipeline.runner import run_pipeline
    cfg = PipelineConfig(assim_fp64=True)
    setattr(cfg, unsupported, True)
    with pytest.raises(ValueError, match='consensus or growth'):
        run_pipeline(None, None, None, cfg)
