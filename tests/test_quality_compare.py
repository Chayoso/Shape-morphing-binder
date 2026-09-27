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


def test_handoff_bands_split_after_trigger_without_including_later_paced_solve(probe):
    assert probe.handoff_intervals(26, 39) == {'pre_trigger': [16, 26], 'post_trigger': [26, 36]}
    assert probe.handoff_intervals(26, 29) == {'pre_trigger': [16, 26], 'post_trigger': [26, 29]}
    assert probe.handoff_intervals(26, 26) == {'pre_trigger': [16, 26]}
    assert probe.handoff_intervals(26, 25) == {}
    assert probe.handoff_intervals(1, 5) == {'post_trigger': [1, 5]}


def test_shared_pic_prefix_requires_one_objective_flag_and_fixed_contract(probe):
    baseline = dict(commit_pic=True, shift_sub=False, outer_render_committed=True,
                    lambda_auto=.5, stop_after_windows=8, T=20)
    candidate = dict(baseline, commit_pic_objective=True)
    mode = 'commit_pic_objective_prefix'
    assert probe.checked_config_changes(baseline, candidate, mode) == {
        'commit_pic_objective': [None, True]}
    assert probe.checked_config_changes(dict(baseline, commit_pic_objective=False), candidate, mode) == {
        'commit_pic_objective': [False, True]}
    for key, value in [('commit_pic', False), ('shift_sub', True),
                       ('outer_render_committed', False), ('lambda_auto', 0.),
                       ('stop_after_windows', 60), ('render_until', 7)]:
        with pytest.raises(ValueError, match=mode):
            probe.checked_config_changes(dict(baseline, **{key: value}),
                                         dict(candidate, **{key: value}), mode)
    for changed in (dict(candidate, T=10), dict(candidate, stop_after_windows=7),
                    dict(candidate, commit_pic_objective=False)):
        with pytest.raises(ValueError, match=mode):
            probe.checked_config_changes(baseline, changed, mode)


def test_shared_pic_audit_uses_only_common_prefix_and_preserves_endpoint_record(probe):
    records = [dict(animation=i, frame_end=1+20*(i+1),
                    endpoint_contract=dict(space='promoted_xpic', objective_commit_max_wu=0.))
               for i in range(8)]
    runs = [dict(records=records[:n], delivered=1+20*n,
                 arm=dict(config=dict(T=20, stop_after_windows=8))) for n in (8, 6)]
    scoped, description = probe.scoped_runs(runs, 'commit_pic_objective_prefix')
    assert description['kind'] == 'common accepted prefix'
    assert description['requested_commits'] == 8 and description['analyzed_commits'] == 6
    assert description['endpoint_claim'] == 'prefix endpoint only; no final quality or convergence inference'
    assert [len(run['records']) for run in scoped] == [6, 6]
    assert [run['delivered'] for run in scoped] == [121, 121]
    assert scoped[1]['records'][-1]['endpoint_contract'] == records[5]['endpoint_contract']
    assert len(runs[0]['records']) == 8 and runs[0]['delivered'] == 161


def test_geometric_rest_prefix_changes_only_the_arrived_motion_penalty(probe):
    baseline = dict(commit_pic=True, commit_pic_objective=True, shift_sub=False,
                    outer_render_committed=True, lambda_auto=.5, stop_after_windows=8,
                    T=20, w_kin=.5, phys_loss='ot_pace')
    candidate = dict(baseline, geometric_rest=True)
    mode = 'geometric_rest_prefix'
    assert mode in probe.PREFIX_INTERVENTIONS and mode not in probe.FULL_INTERVENTIONS
    for base_value in (None, False):
        original = dict(baseline)
        if base_value is not None:
            original['geometric_rest'] = base_value
        assert probe.checked_config_changes(original, candidate, mode) == {
            'geometric_rest': [base_value, True]}
    for key, value in [('commit_pic', False), ('commit_pic_objective', False), ('shift_sub', True),
                       ('outer_render_committed', False), ('lambda_auto', 0.),
                       ('stop_after_windows', 60), ('T', 10), ('render_until', 7),
                       ('w_kin', 0.), ('w_kin', -1.), ('w_kin', float('inf')),
                       ('phys_loss', 'ot'), ('phys_loss', 'auto')]:
        with pytest.raises(ValueError, match=mode):
            probe.checked_config_changes(dict(baseline, **{key: value}),
                                         dict(candidate, **{key: value}), mode)
    for changed in (dict(candidate, w_kin=1.), dict(candidate, body_rprop=True),
                    dict(candidate, settle_pin_still=True), dict(candidate, geometric_rest=False)):
        with pytest.raises(ValueError, match=mode):
            probe.checked_config_changes(baseline, changed, mode)
    with pytest.raises(ValueError, match=mode):
        probe.checked_config_changes(dict(baseline, geometric_rest=None), candidate, mode)


def test_geometric_rest_prefix_fixes_the_motion_time_unit(probe):
    mode = 'geometric_rest_prefix'
    mpm = dict(dt=1/240, dx=.3062907544, nx=36, ny=36, nz=36)
    probe.checked_mpm_parameters(mpm, dict(mpm), mode)
    with pytest.raises(ValueError, match='discretisation mismatch'):
        probe.checked_mpm_parameters(mpm, dict(mpm, dt=1/120), mode)
    for dt in (1/120, 0., float('inf'), None):
        wrong = dict(mpm, dt=dt)
        with pytest.raises(ValueError, match='requires dt=1/240'):
            probe.checked_mpm_parameters(wrong, dict(wrong), mode)
    # Older reviewed comparisons retain their own exact matched discretisation.
    wrong = dict(mpm, dt=1/120)
    probe.checked_mpm_parameters(wrong, dict(wrong), 'body_rprop')


def test_geometric_rest_prefix_preserves_telemetry_without_final_rest_inference(probe):
    records = [dict(animation=i, frame_end=1+20*(i+1),
                    geometric_rest=dict(eligible_count=7+i, raw_sq=.1, remap_sq=.2, total=.3,
                                        weight=.5, effective_weight=.25, unit_multiplier=.5))
               for i in range(8)]
    runs = [dict(records=records[:n], delivered=1+20*n,
                 arm=dict(config=dict(T=20, stop_after_windows=8))) for n in (8, 5)]
    scoped, description = probe.scoped_runs(runs, 'geometric_rest_prefix')
    assert description['kind'] == 'common accepted prefix' and description['analyzed_commits'] == 5
    assert description['endpoint_claim'] == 'prefix endpoint only; no final quality or convergence inference'
    assert [run['delivered'] for run in scoped] == [101, 101]
    assert scoped[0]['records'][-1]['geometric_rest'] == records[4]['geometric_rest']
    assert len(runs[0]['records']) == 8 and runs[0]['delivered'] == 161
