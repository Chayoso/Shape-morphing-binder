"""Archive-clock regression for the P292 raw comparison; CPU metadata only."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import warp as wp


@pytest.fixture
def probe(tmp_path, monkeypatch):
    previous_cache = wp.config.kernel_cache_dir
    monkeypatch.setenv('WARP_CACHE_PATH', str(tmp_path))
    spec = importlib.util.spec_from_file_location(
        'quality_compare', Path(__file__).parents[1]/'scripts/probes/quality_compare.py')
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        wp.config.kernel_cache_dir = previous_cache
    return module


def test_null_hold_does_not_mask_physical_direction_reversal(probe):
    # Two T=2 rollouts separated by one line-search-null held frame.
    frames = np.array([0., 1., 2., 2., 1., 0.])
    records = [dict(animation=0, frame_end=3), dict(animation=2, frame_end=6)]
    indices = probe.accepted_raw_indices(records, steps=2)
    assert indices == [0, 1, 2, 4, 5]
    moves = np.diff(frames[indices])
    assert np.count_nonzero(moves[1:]*moves[:-1] < 0) == 1
    assert not np.any(moves == 0)


def test_rejected_candidate_without_archive_rows_does_not_create_gap(probe):
    records = [dict(animation=0, frame_end=3), dict(animation=3, frame_end=5)]
    assert probe.accepted_raw_indices(records, steps=2) == [0, 1, 2, 3, 4]


def test_inconsistent_accepted_spans_fail(probe):
    with pytest.raises(ValueError, match='overlap'):
        probe.accepted_raw_indices([dict(frame_end=3), dict(frame_end=4)], steps=2)


def test_stress_taper_scope_is_explicit_and_does_not_relax_body_default(probe):
    baseline = dict(body_rprop=False, ctrl_taper_sp=0., stop_after_windows=60, iters=8)
    candidate = dict(baseline, ctrl_taper_sp=2., stop_after_windows=8)
    with pytest.raises(ValueError, match='body_rprop'):
        probe.checked_config_changes(baseline, candidate, 'body_rprop')
    assert set(probe.checked_config_changes(baseline, candidate, 'stress_taper')) == {'ctrl_taper_sp', 'stop_after_windows'}
    for changed in (dict(candidate, body_rprop=True), dict(candidate, iters=7), dict(candidate, stop_after_windows=9)):
        with pytest.raises(ValueError, match='stress_taper'):
            probe.checked_config_changes(baseline, changed, 'stress_taper')


def test_commit_pic_prefix_scope_cannot_change_another_correction(probe):
    baseline = dict(commit_pic=True, shift_sub=True, body_rprop=False, stop_after_windows=60)
    candidate = dict(baseline, commit_pic=False, stop_after_windows=8)
    with pytest.raises(ValueError, match='body_rprop'):
        probe.checked_config_changes(baseline, candidate, 'body_rprop')
    assert set(probe.checked_config_changes(baseline, candidate, 'commit_pic_off')) == {'commit_pic', 'stop_after_windows'}
    for changed in (dict(candidate, shift_sub=False), dict(candidate, body_rprop=True),
                    dict(candidate, stop_after_windows=7), dict(candidate, commit_pic=True)):
        with pytest.raises(ValueError, match='commit_pic_off'):
            probe.checked_config_changes(baseline, changed, 'commit_pic_off')


def test_full_pic_ablation_requires_identical_full_horizons(probe):
    baseline = dict(commit_pic=True, shift_sub=True, body_rprop=False, stop_after_windows=60)
    candidate = dict(baseline, commit_pic=False)
    assert probe.checked_config_changes(baseline, candidate, 'commit_pic_off_full') == {'commit_pic': [True, False]}
    for changed in (dict(candidate, stop_after_windows=8), dict(candidate, body_rprop=True),
                    dict(candidate, shift_sub=False)):
        with pytest.raises(ValueError, match='commit_pic_off_full'):
            probe.checked_config_changes(baseline, changed, 'commit_pic_off_full')
    with pytest.raises(ValueError, match='commit_pic_off_full'):
        probe.checked_config_changes(dict(baseline, stop_after_windows=8),
                                     dict(candidate, stop_after_windows=8), 'commit_pic_off_full')


def test_full_pic_scope_preserves_different_complete_run_lengths(probe):
    runs = [dict(records=[dict(frame_end=21*(i+1)) for i in range(n)], delivered=n*21+1)
            for n in (32, 40)]
    scoped, scope = probe.scoped_runs(runs, 'commit_pic_off_full')
    assert scoped is runs
    assert [len(run['records']) for run in scoped] == [32, 40]
    assert [run['delivered'] for run in scoped] == [673, 841]
    assert scope['kind'] == 'full runs'


def test_render_handoff_requires_single_flag_and_active_committed_render(probe):
    baseline = dict(commit_pic=False, outer_render_committed=True, render_paced=True,
                    render_paced_arrived=False, lambda_auto=.5, stop_after_windows=60)
    candidate = dict(baseline, render_paced_arrived=True)
    assert probe.checked_config_changes(baseline, candidate, 'render_arrival_handoff') == {
        'render_paced_arrived': [False, True]}
    for key, value in [('commit_pic', True), ('outer_render_committed', False),
                       ('render_paced', False), ('lambda_auto', 0.), ('stop_after_windows', 8)]:
        with pytest.raises(ValueError, match='render_arrival_handoff'):
            probe.checked_config_changes(dict(baseline, **{key: value}),
                                         dict(candidate, **{key: value}), 'render_arrival_handoff')
    with pytest.raises(ValueError, match='render_arrival_handoff'):
        probe.checked_config_changes(baseline, dict(candidate, settle_pin_confirm=True), 'render_arrival_handoff')
    runs = [dict(records=[1]*32), dict(records=[1]*40)]
    scoped, scope = probe.scoped_runs(runs, 'render_arrival_handoff')
    assert scoped is runs and scope['kind'] == 'full runs'


def test_reference_history_preserves_null_trials_and_separates_trimmed_acceptance(probe):
    history = [dict(animation=0, frame_end=21, render_target_kind='paced'),
               dict(animation=1, null_commit=1, render_target_kind='paced'),
               dict(animation=2, frame_end=41, render_target_kind='fixed'),
               dict(animation=3, frame_end=61, render_target_kind='fixed'),
               dict(animation=4, render_target_kind='fixed')]
    run = dict(records=[history[0], history[2]], delivered=41, arm=dict(history=history))
    history.insert(2, dict(animation=2, c2f_render_res=96))
    history.append(dict(animation=5, held=1))
    full = probe.render_reference_history(run, 'render_arrival_handoff')
    assert [row['attempt'] for row in full] == [1, 2, 3, 4, 5]
    assert not full[1]['accepted'] and full[1]['null_commit']
    assert full[3]['accepted'] and not full[3]['delivered']
    assert not full[4]['accepted']
    prefix = probe.render_reference_history(run, 'commit_pic_off')
    assert [row['attempt'] for row in prefix] == [1, 2, 3]
    assert [row['render_target_kind'] for row in prefix] == ['paced', 'paced', 'fixed']


def test_prefix_view_preserves_original_metadata(probe):
    records = [dict(animation=i, frame_end=1+2*(i+1)) for i in range(10)]
    original = dict(records=records, delivered=22, arm=dict(config=dict(T=2, stop_after_windows=60)))
    candidate = dict(records=records[:8], delivered=18, arm=dict(config=dict(T=2, stop_after_windows=8)))
    scoped, description = probe.scoped_runs([original, candidate], 'stress_taper')
    assert [len(run['records']) for run in scoped] == [8, 8]
    assert [run['delivered'] for run in scoped] == [17, 17]
    assert description['original_commits'] == [10, 8]
    assert len(original['records']) == 10 and original['delivered'] == 22
    assert scoped[0]['arm']['config']['stop_after_windows'] == 60
