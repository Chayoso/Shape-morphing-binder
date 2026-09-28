"""Archive-clock regression for the P292 raw comparison; CPU metadata only."""
import importlib.util
import json
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
                       ('phys_loss', 'ot'), ('phys_loss', 'density')]:
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


def test_actual_serialized_p294_auto_configs_require_resolution_and_history(probe, tmp_path):
    fixture = json.loads((Path(__file__).parent/'fixtures/p294_auto_arrival.json').read_text())
    mode = 'geometric_rest_prefix'
    a, b = fixture['baseline'], fixture['candidate']
    assert a['config']['phys_loss'] == b['config']['phys_loss'] == 'auto'
    assert probe.checked_config_changes(a['config'], b['config'], mode) == {'geometric_rest': [None, True]}
    for name, original in (('baseline', a), ('candidate', b)):
        prefix = tmp_path/name
        log = Path(str(prefix)+'.log')
        log.write_text(original['resolver_line']+'\n', encoding='utf-8')
        run = dict(prefix=prefix, arm=dict(config=original['config']), records=original['records'])
        result = probe.checked_arrival_evidence(run)
        assert result['resolved_mode'] == 'ot_pace' and result['accepted_commits_checked'] == 8
        assert result['log_sha256'] == probe.hashlib.sha256(log.read_bytes()).hexdigest()
        for contents in ('', original['resolver_line']+'\n'+original['resolver_line'],
                         original['resolver_line'].replace('ot_pace + cell-wise hand-off', 'ot')):
            log.write_text(contents, encoding='utf-8')
            with pytest.raises(ValueError, match='Serialized auto'):
                probe.checked_arrival_evidence(run)
        log.write_text(original['resolver_line']+'\n', encoding='utf-8')
        for wrong in (dict(original['records'][0], pin_arrival_evidence='legacy_no_arrival_contract'),
                      dict(original['records'][0], arrived_end_frac=None),
                      dict(original['records'][0], arrived_end_frac=float('nan'))):
            with pytest.raises(ValueError, match='Every accepted commit'):
                probe.checked_arrival_evidence(dict(run, records=[wrong]+original['records'][1:]))


def test_full_geometric_rest_retains_strict_contract_at_cap60(probe):
    fixture = json.loads((Path(__file__).parent/'fixtures/p294_auto_arrival.json').read_text())
    a = dict(fixture['baseline']['config'], stop_after_windows=60)
    b = dict(fixture['candidate']['config'], stop_after_windows=60)
    mode = 'geometric_rest_full'
    assert mode in probe.FULL_INTERVENTIONS and mode not in probe.PREFIX_INTERVENTIONS
    assert probe.checked_config_changes(a, b, mode) == {'geometric_rest': [None, True]}
    for key, value in [('commit_pic', False), ('commit_pic_objective', False), ('shift_sub', True),
                       ('outer_render_committed', False), ('T', 10), ('w_kin', 0.),
                       ('stop_after_windows', 8), ('render_until', 8), ('lambda_auto', 0.)]:
        with pytest.raises(ValueError, match=mode):
            probe.checked_config_changes(dict(a, **{key: value}), dict(b, **{key: value}), mode)
    for changed in (dict(b, stop_after_windows=8), dict(b, w_kin=10.), dict(b, body_rprop=True)):
        with pytest.raises(ValueError, match=mode):
            probe.checked_config_changes(a, changed, mode)
    mpm = dict(dt=1/240, dx=.3062907544)
    probe.checked_mpm_parameters(mpm, dict(mpm), mode)
    with pytest.raises(ValueError, match='requires dt=1/240'):
        probe.checked_mpm_parameters(dict(mpm, dt=1/120), dict(mpm, dt=1/120), mode)


def test_full_geometric_rest_uses_complete_unequal_runs(probe):
    runs = [dict(records=[dict(frame_end=1+20*(i+1)) for i in range(n)], delivered=1+20*n)
            for n in (32, 40)]
    scoped, description = probe.scoped_runs(runs, 'geometric_rest_full')
    assert scoped is runs
    assert [len(run['records']) for run in scoped] == [32, 40]
    assert [run['delivered'] for run in scoped] == [641, 801]
    assert description['kind'] == 'full runs'


def test_serialized_auto_pair_cannot_resolve_to_different_ot_modes(probe, tmp_path):
    fixture = json.loads((Path(__file__).parent/'fixtures/p294_auto_arrival.json').read_text())
    runs = []
    for name in ('baseline', 'candidate'):
        original = fixture[name]
        prefix = tmp_path/name
        Path(str(prefix)+'.log').write_text(original['resolver_line']+'\n', encoding='utf-8')
        runs.append(dict(prefix=prefix, arm=dict(config=original['config']), records=original['records']))
    assert [v['resolved_mode'] for v in probe.checked_arrival_modes(runs)] == ['ot_pace', 'ot_pace']
    Path(str(runs[1]['prefix'])+'.log').write_text(
        fixture['candidate']['resolver_line'].replace('ot_pace', 'ot_shape')+'\n', encoding='utf-8')
    with pytest.raises(ValueError, match='same full-plan OT arrival mode'):
        probe.checked_arrival_modes(runs)


@pytest.mark.parametrize('mode,cap', [('geometric_variance_prefix', 8),
                                     ('geometric_variance_full', 60)])
def test_geometric_variance_requires_exact_policy_and_diagnostic_flags(probe, mode, cap):
    baseline = dict(stop_after_windows=cap, T=20, commit_pic=True, commit_pic_objective=True,
                    shift_sub=False, outer_render_committed=True, geometric_rest=False,
                    motion_accounting=True, w_kin_var=200., lambda_auto=.5, phys_loss='ot_pace')
    candidate = dict(baseline, geometric_variance=True)
    assert (mode in probe.PREFIX_INTERVENTIONS) == (cap == 8)
    assert (mode in probe.FULL_INTERVENTIONS) == (cap == 60)
    assert probe.checked_config_changes(baseline, candidate, mode) == {'geometric_variance': [None, True]}
    assert probe.checked_config_changes(dict(baseline, geometric_variance=False), candidate, mode) == {
        'geometric_variance': [False, True]}
    for key, value in [('stop_after_windows', 60 if cap == 8 else 8), ('T', 10), ('commit_pic', False),
                       ('commit_pic_objective', False), ('shift_sub', True),
                       ('outer_render_committed', False), ('geometric_rest', True),
                       ('motion_accounting', False), ('render_until', cap-1), ('phys_loss', 'density'),
                       ('w_kin_var', 0.), ('w_kin_var', -1.), ('w_kin_var', float('inf')),
                       ('w_kin_var', float('nan')), ('w_kin_var', None), ('w_kin_var', True),
                       ('lambda_auto', 0.), ('lambda_auto', float('inf'))]:
        with pytest.raises(ValueError, match=mode):
            probe.checked_config_changes(dict(baseline, **{key: value}), dict(candidate, **{key: value}), mode)
    for changed in (dict(candidate, w_kin_var=100.), dict(candidate, layer_relax=False),
                    dict(candidate, body_rprop=True), dict(candidate, geometric_variance=False)):
        with pytest.raises(ValueError, match=mode):
            probe.checked_config_changes(baseline, changed, mode)
    with pytest.raises(ValueError, match=mode):
        probe.checked_config_changes(dict(baseline, geometric_variance=None), candidate, mode)


@pytest.mark.parametrize('mode', ['geometric_variance_prefix', 'geometric_variance_full'])
def test_geometric_variance_requires_original_time_step(probe, mode):
    mpm = dict(dt=1/240, dx=.3062907543956724, nx=36, ny=36, nz=36)
    probe.checked_mpm_parameters(mpm, dict(mpm), mode)
    for dt in (1/120, 0., None, float('nan')):
        wrong = dict(mpm, dt=dt)
        with pytest.raises(ValueError, match='requires dt=1/240'):
            probe.checked_mpm_parameters(wrong, dict(wrong), mode)
    with pytest.raises(ValueError, match='discretisation mismatch'):
        probe.checked_mpm_parameters(mpm, dict(mpm, dx=.3), mode)


def test_full_geometric_variance_retains_each_unequal_endpoint_and_full_history(probe):
    runs = [dict(records=[dict(animation=i, frame_end=1+20*(i+1)) for i in range(n)],
                 delivered=1+20*n) for n in (32, 40)]
    scoped, description = probe.scoped_runs(runs, 'geometric_variance_full')
    assert scoped is runs and description['kind'] == 'full runs'
    assert [len(run['records']) for run in scoped] == [32, 40]
    assert [run['delivered'] for run in scoped] == [641, 801]
    runs[0]['arm'] = dict(history=runs[0]['records']+[
        dict(animation=32, null_commit=True), dict(animation=33, outer_accepted=False)])
    history = probe.render_reference_history(runs[0], 'geometric_variance_full')
    assert history[-1]['attempt'] == 34 and history[-2]['null_commit'] is True
    assert not history[-1]['accepted'] and history[31]['delivered'] is True


def test_late_common_material_motion_uses_last_ten_windows_not_first_eight(probe):
    source = np.random.default_rng(91).normal(size=(35, 3))
    # A large early excursion is outside the late interval, which translates uniformly.
    offsets = np.arange(801, dtype=float)*.01
    offsets[:400] += 100.
    frames = source[None]+offsets[:, None, None]*np.array([1., 0., 0.])
    records = [dict(animation=i, frame_end=1+20*(i+1)) for i in range(40)]
    run = dict(source=source, frames=frames, records=records,
               pins=np.zeros(35, bool), pin_at=np.zeros(35, int),
               curve=[dict(commit=i+1, attempt=i+1, frame=20*(i+1)) for i in range(40)],
               physical_indices=list(range(801)))
    value = probe.cohort_motion(run, np.array([0, 1]), common=32, spacing=.1, include_rms=True)
    assert value['commit_range'] == [22, 32] and value['frame_range'] == [440, 640]
    assert value['raw_physical_states'] == 201
    assert value['raw_step_sp']['rms'] == pytest.approx(.1)
    assert value['step_sp']['rms'] == pytest.approx(2.)
    assert value['net_over_path']['median'] == pytest.approx(1.)
    assert len(run['records']) == 40  # Candidate's later endpoint was not trimmed.


def test_fixed_source_density_keeps_ids_that_leave_the_moving_top(probe):
    x = np.array([[0., 2.5, 0.], [.125, 2.5, 0.], [5., 0., 0.]])
    ids = np.array([0, 1])
    before = probe.fixed_material_density(x, probe.KDTree(x), ids, .2)
    assert before == dict(particles=2, density=.125, under_half=1., y_gt_2_3_frac=1.)
    moved = x.copy()
    moved[1] = [0., 0., 0.]
    after = probe.fixed_material_density(moved, probe.KDTree(moved), ids, .2)
    assert after == dict(particles=2, density=0., under_half=1., y_gt_2_3_frac=.5)
    assert (moved[:, 1] > 2.3).sum() == 1  # No dynamic cohort reselection occurred.
    assert probe.fixed_material_density(moved, probe.KDTree(moved), ids[:0], .2) == dict(
        particles=0, density=None, under_half=None, y_gt_2_3_frac=None)
    np.testing.assert_array_equal(ids, [0, 1])


def test_new_motion_rms_is_optional_and_not_mean_or_signed_mean(probe):
    values = np.array([-3., 4.])
    legacy = probe.stats(values)
    updated = probe.stats(values, include_rms=True)
    assert set(legacy) == {'median', 'p95', 'max'}
    assert updated['rms'] == pytest.approx(np.sqrt(12.5))
    assert {k: updated[k] for k in legacy} == legacy
    assert probe.stats(values[:0], include_rms=True) is None
