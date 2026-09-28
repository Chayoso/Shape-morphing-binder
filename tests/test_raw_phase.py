"""Raw-phase clock remains exact when an unphysical null-held row is present."""
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import warp as wp


@pytest.fixture
def phase_probe(tmp_path, monkeypatch):
    previous = wp.config.kernel_cache_dir
    monkeypatch.setenv('WARP_CACHE_PATH', str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        'raw_phase', Path(__file__).parents[1]/'scripts/probes/raw_phase.py')
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        wp.config.kernel_cache_dir = previous
    return module


def test_phase_windows_keep_prior_accepted_endpoint_and_skip_null_hold(phase_probe):
    records = [dict(frame_end=3), dict(frame_end=6), dict(frame_end=8)]
    assert phase_probe.phase_frame_indices(records, 1, 3, 2) == [[2, 4, 5], [5, 6, 7]]


def test_prefix_phase_scope_preserves_window1_exclusion(phase_probe):
    records = [dict(frame_end=1+20*k) for k in range(1, 9)]
    rows = phase_probe.phase_frame_indices(records, 1, 8, 20)
    assert len(rows) == 7
    assert rows[0] == list(range(20, 41)) and rows[-1] == list(range(140, 161))
    assert all(0 not in row for row in rows)


def test_rms_reports_vector_and_component_magnitudes_without_changing_old_output(phase_probe):
    moves = np.array([[[3., 4., 0.]], [[5., 12., 0.]]])
    n, t1, t2 = np.eye(3)[:, None, :]
    old = phase_probe.motion_summary(moves, n, t1, t2, 2.)
    new = phase_probe.motion_summary(moves, n, t1, t2, 2., include_rms=True)
    assert new['displacement_sp']['rms'] == pytest.approx(np.sqrt(97)/2)
    assert new['signed_normal_sp']['rms'] == pytest.approx(np.sqrt(17)/2)
    assert new['tangent_length_sp']['rms'] == pytest.approx(np.sqrt(80)/2)
    for key in old:
        assert 'rms' not in old[key]
        assert {k: new[key][k] for k in old[key]} == old[key]
    assert phase_probe.distribution(np.empty(0), include_rms=True) is None


@pytest.mark.parametrize('sampled_count,reason', [(0, 'empty_common_free_cohort'),
                                                 (5, 'fewer_than_three_common_commits')])
def test_variance_phase_empty_or_short_cohort_returns_inconclusive_report(
        phase_probe, tmp_path, monkeypatch, sampled_count, reason):
    a, b = tmp_path/'baseline', tmp_path/'candidate'
    runs = [dict(prefix=str(path), meta=dict(provenance=dict(code_hash='frozen'))) for path in (a, b)]
    monkeypatch.setattr(phase_probe, 'checked_runs', lambda *_: (runs, {'geometric_variance': [False, True]}))
    # Prohibit any numerical backend entry: there is no eligible motion to measure.
    def forbidden(*args, **kwargs):
        raise AssertionError('Inconclusive cohort must not run a CUDA calculation')
    monkeypatch.setattr(phase_probe, 'cuda_execution', forbidden)
    quality_file = Path(phase_probe.sys.modules['scripts.probes.quality_compare'].__file__)
    report = dict(intervention='geometric_variance_prefix', code_hash='frozen',
                  probe_sha256=hashlib.sha256(quality_file.read_bytes()).hexdigest(),
                  baseline=dict(prefix=str(a)), candidate=dict(prefix=str(b)),
                  n=300000, T=20, native_spacing=.035, mpm=dict(dt=1/240),
                  analysis_scope=dict(kind='common accepted prefix'),
                  cohorts=dict(common_endpoint_free_both=dict(sampled_count=sampled_count,
                               eligible_count=sampled_count, ids_sha256='fixture',
                               arms=dict(baseline=None, candidate=None))))
    reference, out = tmp_path/'quality.json', tmp_path/'phase.json'
    reference.write_text(json.dumps(report))
    phase_probe.phase_audit(a, b, reference, out)
    result = json.loads(out.read_text())
    assert result['status'] == 'inconclusive' and result['reason'] == reason
    assert result['baseline'] is result['candidate'] is None
    assert 'no rest or repair conclusion' in result['scope']
