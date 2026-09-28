"""CPU reference tests for the archive-only saved-state supply diagnostic."""
import copy
from pathlib import Path

import numpy as np
import pytest

from scripts.probes import constitutive_quality as cq
from scripts.probes import transient_supply as probe


def quality_fixture():
    return dict(probe_sha256=probe.QUALITY_PROBE_SHA, requested_windows=24,
                n=300000, T=20, loss_res=36, config_changes={},
                config=dict(archive_stride=1, stop_after_windows=24, commit_pic=True,
                            commit_pic_objective=True, compute_backend='cuda', shift_sub=False),
                mpm=dict(dt=1/240, dx=.3062907543956724), native_spacing=.035,
                target_spacing=.035, density_radius=.07,
                cohorts=dict(fixed_source_upper_surface=dict(sampled_count=6712,
                    eligible_count=6712, ids_sha256=probe.SOURCE_IDS_SHA)))


def test_exact_reviewed_quality_binding():
    quality = quality_fixture()
    probe.validate_quality(quality, probe.QUALITY_PROBE_SHA)
    with pytest.raises(ValueError, match='probe'):
        probe.validate_quality(quality, '0'*64)


@pytest.mark.parametrize('location,key,value', [
    (None, 'requested_windows', 6), (None, 'T', 19),
    ('config', 'archive_stride', 2), ('config', 'shift_sub', True),
    ('config', 'commit_pic_objective', False), ('mpm', 'dt', 1/120),
    (None, 'density_radius', float('nan')),
])
def test_changed_quality_scope_rejected(location, key, value):
    quality = quality_fixture()
    target = quality[location] if location else quality
    target[key] = value
    with pytest.raises(ValueError):
        probe.validate_quality(quality, probe.QUALITY_PROBE_SHA)


def test_changed_fixed_ids_rejected():
    quality = quality_fixture()
    quality['cohorts']['fixed_source_upper_surface']['ids_sha256'] = '0'*64
    with pytest.raises(ValueError, match='cohort'):
        probe.validate_quality(quality, probe.QUALITY_PROBE_SHA)


def test_artifacts_bound_before_loading_and_mutation_fails(tmp_path):
    prefix = str(tmp_path/'run')
    paths = [prefix+suffix for suffix in ('.json', '.log', '_render_full_dt_iso_nn.npz')]
    for name in paths:
        Path(name).write_bytes(b'not parsed by the provenance check')
    entries = [cq.file_record(Path(name)) for name in paths]
    arm = dict(prefix=prefix, artifact_provenance=entries, run_json_sha256=entries[0]['sha256'])
    assert probe.verify_artifacts(arm, cq.file_record) == entries
    altered = copy.deepcopy(arm)
    altered['artifact_provenance'][2]['path'] = paths[1]
    with pytest.raises(ValueError, match='membership'):
        probe.verify_artifacts(altered, cq.file_record)
    Path(paths[2]).write_bytes(b'changed raw archive')
    with pytest.raises(ValueError, match='hash or identity'):
        probe.verify_artifacts(arm, cq.file_record)


def test_w1_previous_anchor_and_interior_holds_excluded():
    # T=3 for the small indexing fixture; rows4/5 are null holds.
    records = [dict(animation=0, frame_end=4), dict(animation=2, frame_end=9)]
    windows = probe.window_frames(records, [0, 1, 2, 3, 6, 7, 8], steps=3)
    assert windows == [dict(commit=1, attempt=1, frame_indices=[0, 1, 2, 3]),
                       dict(commit=2, attempt=3, frame_indices=[3, 6, 7, 8])]
    with pytest.raises(ValueError, match='map'):
        probe.window_frames(records, list(range(9)), steps=3)
    with pytest.raises(ValueError, match='Nonphysical'):
        probe.window_frames([dict(animation=0, frame_end=4, null_commit=True)], [0, 1, 2, 3], steps=3)


def test_real_cloud_interior_supply_loss_with_endpoint_recovery():
    # Eight neighbors support source ID0 at both endpoints. ID9 covers target1.
    cluster = np.array([[0., 0, 0], [.1, 0, 0], [-.1, 0, 0], [0, .1, 0],
                        [0, -.1, 0], [0, 0, .1], [0, 0, -.1], [.1, .1, 0], [-.1, -.1, 0],
                        [2., 0, 0]])
    interior = cluster.copy()
    interior[:9, 0] = np.arange(9)+5
    target = np.array([[0., 0, 0], [2., 0, 0]])
    source_ids = np.array([0])
    samples = [probe.frame_sample(x, source_ids, target, .25, .1)
               for x in (cluster, interior, cluster, cluster)]
    counts, gaps = map(np.stack, zip(*samples))
    original_counts, original_gaps = counts.copy(), gaps.copy()
    result, source_lost, target_lost, source_eligible, target_eligible = probe.window_summary(
        counts, gaps, dict(commit=1, attempt=1, frame_indices=[0, 1, 2, 3]))
    assert counts[:, 0].tolist() == [8, 0, 8, 8]
    assert result['source_endpoint_supported_loss']['eligible_ids'] == 1
    assert result['source_endpoint_supported_loss']['lost_observations'] == 1
    assert result['source_endpoint_supported_loss']['severity'] == dict(
        minimum_count=0, zero_count_observations=1, zero_count_union_ids=1)
    assert result['source_endpoint_supported_loss']['worst'] == dict(phase=1, frame=1, count=1, fraction=1.)
    assert result['target_endpoint_covered_loss']['eligible_ids'] == 2
    assert result['target_endpoint_covered_loss']['union_fraction'] == .5
    assert source_lost.tolist() == [True] and target_lost.tolist() == [True, False]
    assert source_eligible.all() and target_eligible.all()
    assert result['source_density_dip']['below_endpoint_floor'] == 1.
    assert result['target_coverage_dip']['below_endpoint_floor'] == .5
    np.testing.assert_array_equal(counts, original_counts)
    np.testing.assert_array_equal(gaps, original_gaps)


def test_threshold_inclusivity_and_unfinished_target_not_transient_hole():
    counts = np.array([[4, 3, 5], [3, 0, 4], [4, 4, 3]])
    gaps = np.array([[2., 3., 1.], [2.01, 9., 2.], [2., 2., 3.]])
    result, source_lost, target_lost, _, _ = probe.window_summary(
        counts, gaps, dict(commit=2, attempt=2, frame_indices=[20, 21, 22]))
    assert source_lost.tolist() == target_lost.tolist() == [True, False, False]
    for name in ('source_endpoint_supported_loss', 'target_endpoint_covered_loss'):
        assert result[name]['eligible_ids'] == result[name]['union_lost_ids'] == 1
    # Large loss on an ID that is missing at an endpoint is deliberately ineligible.
    assert result['frames'][1]['target_coverage'] == 1/3


def test_empty_and_no_eligible_denominators_are_null():
    window = dict(commit=1, attempt=1, frame_indices=[0, 1, 2])
    for counts, gaps in ((np.empty((3, 0)), np.empty((3, 0))),
                         (np.zeros((3, 2)), np.full((3, 2), 3.))):
        result, *masks = probe.window_summary(counts, gaps, window)
        for name in ('source_endpoint_supported_loss', 'target_endpoint_covered_loss'):
            value = result[name]
            assert value['eligible_ids'] == value['union_lost_ids'] == 0
            assert value['union_fraction'] is None and value['worst'] is None
            assert value['status'].startswith('inconclusive')
        assert result['source_endpoint_supported_loss']['severity'] == dict(
            minimum_count=None, zero_count_observations=0, zero_count_union_ids=0)
        assert result['target_endpoint_covered_loss']['severity'] == dict(maximum_gap_sp=None, p95_gap_sp=None)
        assert not any(mask.any() for mask in masks)


def test_stable_supported_cloud_reports_no_event():
    result, *masks = probe.window_summary(np.full((3, 2), 4), np.full((3, 2), 2.),
        dict(commit=1, attempt=1, frame_indices=[0, 1, 2]))
    assert result['source_endpoint_supported_loss']['union_fraction'] == 0.
    assert result['target_endpoint_covered_loss']['worst'] is None
    assert not masks[0].any() and not masks[1].any()
    assert result['source_density_dip']['below_endpoint_floor'] == 0.


def test_severity_uses_only_endpoint_eligible_loss_observations():
    # ID2 is never eligible: its zero counts/huge gaps must not enter severity.
    counts = np.array([[4, 4, 0], [0, 3, 0], [0, 4, 0], [4, 4, 0]])
    gaps = np.array([[2., 1., 50.], [2.1, 3., 1000.], [4., 1., 999.], [2., 1., 50.]])
    result, *_ = probe.window_summary(counts, gaps,
        dict(commit=1, attempt=1, frame_indices=[0, 1, 2, 3]))
    source = result['source_endpoint_supported_loss']
    target = result['target_endpoint_covered_loss']
    assert source['eligible_ids'] == target['eligible_ids'] == 2
    assert source['severity'] == dict(minimum_count=0, zero_count_observations=2, zero_count_union_ids=1)
    assert target['severity']['maximum_gap_sp'] == 4.
    assert target['severity']['p95_gap_sp'] == pytest.approx(3.9)
    # Without a zero, the loss minimum is3; unsupported endpoint zeros stay out.
    counts[1:3, 0] = 3
    result, *_ = probe.window_summary(counts, gaps,
        dict(commit=1, attempt=1, frame_indices=[0, 1, 2, 3]))
    assert result['source_endpoint_supported_loss']['severity'] == dict(
        minimum_count=3, zero_count_observations=0, zero_count_union_ids=0)
