"""Driver lifecycle and evidence-budget gates; no CUDA simulation locally."""
from copy import deepcopy
import json

import numpy as np
import pytest

from physmorph.pipeline import runner
from scripts.probes import joint_withdrawal_search as driver
from test_checkpoint_merit_terms import fixture


class StubRaw:
    """Raw measurement is independently tested; this isolates driver ownership."""
    def __init__(self, *args):
        self.count = 0

    def observe(self, label, values, baseline=False):
        self.count += int(baseline)
        return dict(passed=True, baseline_ready=self.count == 3)

    def archive_state(self):
        return dict(baseline_count=np.array(self.count))


@pytest.mark.parametrize('archive_failure', [False, True])
def test_real_parent_and_search_preserve_original_outer_state(monkeypatch, tmp_path, archive_failure):
    source, target, prm, cfg = fixture()
    expected = runner.run_pipeline(source, target, prm, deepcopy(cfg), log=lambda *_: None)
    capture = driver.JointSearchCapture(tmp_path, .5, source, target, window=0, iteration=2)
    original_save = capture.save
    def save(name, arrays):
        if archive_failure and name == 'raw_baseline_envelope.npz':
            raise ValueError('injected raw-envelope budget failure')
        original_save(name, arrays)
    monkeypatch.setattr(capture, 'save', save)
    monkeypatch.setattr(driver, 'RawWithdrawalQuality', StubRaw)
    monkeypatch.setattr(runner, 'optimize_window', capture.wrap(runner.optimize_window))
    actual = runner.run_pipeline(source, target, prm, deepcopy(cfg), on_commit=capture.commit, log=lambda *_: None)
    assert capture.report['measurement_passed'] and capture.outer_accepted
    assert capture.report['optimizer_state_after_callback_exact']
    assert all(capture.report['original_return_preserved'].values())
    assert all(capture.report['committed_original_endpoint'].values())
    assert capture.search['merit_binding_unchanged'] and capture.packet is None
    assert len(capture.forward_records) >= 3
    if archive_failure:
        assert capture.search['status'] == 'search_failed'
        assert not capture.search['candidate_found']
        assert 'injected' in capture.search['archive_error']['message']
    else:
        assert 'raw_baseline_envelope.npz' in capture.sidecars
    for item in capture.forward_records:
        assert driver.identity(tmp_path/item['archive']) == item['binding']
    np.testing.assert_array_equal(actual['frames'], expected['frames'])
    np.testing.assert_array_equal(actual['F_frames'], expected['F_frames'])
    assert actual['guards'] == expected['guards']
    for a, b in zip(actual['history'], expected['history']):
        for key in ('loss', 'lambda', 'd_vol', 'd_sil', 'frame_end', 'render_influence_steps'):
            assert a.get(key) == b.get(key)
    assert json.loads((tmp_path/'search.json').read_text())['status'] == capture.search['status']


def test_json_reserves_actual_utf8_growth_and_allows_smaller_rewrite(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, 'LIMIT', 66_000)
    path = tmp_path/'result.json'
    driver.write_json(path, dict(value='가'*30))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='JSON evidence'):
        driver.write_json(path, dict(value='x'*1000))
    assert path.read_bytes() == before
    driver.write_json(path, dict(failed=True))
    assert len(path.read_bytes()) < len(before)


def test_failure_json_can_consume_reserve_but_never_exceed_absolute_limit(monkeypatch, tmp_path):
    payload = dict(candidate_found=False, failure='budget')
    size = len((json.dumps(payload, indent=2)+'\n').encode('utf-8'))
    monkeypatch.setattr(driver, 'LIMIT', size)
    path = tmp_path/'result.json'
    with pytest.raises(ValueError, match='JSON evidence'):
        driver.write_json(path, payload)
    driver.write_json(path, payload, failure_only=True)
    assert path.stat().st_size == size
    with pytest.raises(ValueError, match='JSON evidence'):
        driver.write_json(path, dict(payload, extra=1), failure_only=True)
    assert json.loads(path.read_text()) == payload
