"""Independently calculable CPU fixtures for the accepted-only observer."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.probes.arrival_history import (Accumulator, Capture, EvidenceWriter,
    arrival_mask, budget, LIMIT_BYTES, FLOAT_FIELDS, INT_FIELDS)


def points(xs):
    return torch.tensor([[float(x), 0., 0.] for x in xs], dtype=torch.float32)


def candidate(path_x, plan_x, *, attempt=1, radius=.5, promoted=None, pin=None, start_actual=None):
    path = torch.stack([points(xs) for xs in path_x])
    plan = points(plan_x)
    end = path[-1].clone() if promoted is None else points(promoted)
    start = arrival_mask(path[0], plan, radius)
    return dict(attempt=attempt, path=path, plan=plan, radius=radius, promoted=end,
                pin=torch.zeros(len(plan), dtype=torch.bool) if pin is None else torch.tensor(pin, dtype=torch.bool),
                T=len(path)-1, start_geometric=start,
                start_arrived=start.clone() if start_actual is None else torch.tensor(start_actual, dtype=torch.bool),
                end_arrived=arrival_mask(end, plan, radius))


def test_budget_bounds_complete_float64_counters_and_exact_sidecars():
    report = budget()
    assert report['per_window_array_bytes'] == 11_700_000
    assert report['final_counter_array_bytes'] == 51_900_000
    assert report['preflight_max_bytes'] == 346_300_000 < LIMIT_BYTES
    with pytest.raises(ValueError, match='350 MB'):
        budget(windows=25)


def test_initial_arrived_get_w1_motion_and_separate_endpoint_admission():
    acc = Accumulator(points([0., 2.]), 1., .1)
    row = acc.accept(candidate([[0., 2.], [1., 1.], [0., 0.]], [0., 0.]))
    assert acc.first_arrival.tolist() == [0, 1]
    assert acc.first_endpoint_arrival.tolist() == [1, 1]
    assert acc.first_start_arrived.tolist() == [1, 0]
    assert acc.banks['start_free']['steps'].tolist() == [2, 0]
    assert acc.banks['start_free']['path_wu'].tolist() == [2., 0.]
    assert acc.banks['start_free']['reversed_pairs'].tolist() == [1, 0]
    assert row['newly_arrived'] == 1 and row['initial_arrived'] == 1
    summary = acc.summary()
    assert summary['unobserved_after_first_arrival'] == 1
    assert summary['cohorts']['initial_arrived']['observed_ids'] == 1
    assert summary['cohorts']['later_endpoint_arrived']['unobserved_ids'] == 1
    assert summary['cohorts']['later_endpoint_arrived']['net_anchor_observed_ids'] == 0
    assert summary['cohorts']['later_endpoint_arrived']['net_anchor_rms_wu'] is None


def test_static_plan_reclassification_is_not_geometric_escape():
    acc = Accumulator(points([0.]), 1., .1)
    acc.accept(candidate([[0.], [0.], [0.]], [0.]))
    row = acc.accept(candidate([[0.], [0.], [0.]], [3.], attempt=2))
    assert row['transitions']['plan_departures'] == 1
    assert row['transitions']['geometric_departures'] == 0
    row = acc.accept(candidate([[0.], [0.], [0.]], [0.], attempt=3))
    assert row['transitions']['plan_reentries'] == 1
    assert row['transitions']['geometric_reentries'] == 0
    assert acc.banks['start_free']['path_wu'].item() == 0.
    assert acc.anchor[0, 0].item() == 0.


def test_fixed_plan_escape_reentry_not_censored_and_boundary_reversal_counted():
    acc = Accumulator(points([0.]), 1., .1)
    first = acc.accept(candidate([[0.], [.4], [1.]], [0.]))
    second = acc.accept(candidate([[1.], [.5], [0.]], [0.], attempt=2))
    assert first['transitions']['geometric_departures'] == 1
    assert second['transitions']['geometric_reentries'] == 1
    assert second['previously_arrived'] == 1
    bank = acc.banks['start_free']
    assert bank['steps'].item() == 4
    assert bank['path_wu'].item() == pytest.approx(2.)
    assert bank['step_square_wu2'].item() == pytest.approx(.16+.36+.25+.25)
    assert bank['max_excursion_wu'].item() == 1.
    # The negative third displacement reverses the preceding window's final +.6.
    assert bank['eligible_reversal_pairs'].item() == 3
    assert bank['reversed_pairs'].item() == 1
    assert second['groups']['start_free']['reversed_pairs'] == 1


def test_start_pin_separates_zero_motion_without_diluting_free_denominator():
    acc = Accumulator(points([0., 0.]), 1., .1)
    acc.accept(candidate([[0., 0.], [.25, .25], [.5, .5]], [0., 0.]))
    row = acc.accept(candidate([[.5, .5], [.5, .25], [.5, 0.]], [0., 0.], attempt=2, pin=[True, False]))
    assert row['new_start_pins'] == 1
    assert row['groups']['start_pinned']['saved_steps'] == 2
    assert row['groups']['start_pinned']['saved_step_rms_wu'] == 0.
    assert row['groups']['start_free']['saved_step_rms_wu'] == .25
    assert acc.banks['start_free']['steps'].tolist() == [2, 4]
    assert acc.banks['start_pinned']['steps'].tolist() == [2, 0]
    with pytest.raises(ValueError, match='Pin release'):
        acc.accept(candidate([[.5, 0.], [.5, 0.]], [0., 0.], attempt=3))


def test_raw_last_pic_and_saved_last_are_distinct_quantities():
    acc = Accumulator(points([0.]), 1., .1)
    row = acc.accept(candidate([[0.], [1.], [3.]], [0.], radius=10., promoted=[0.]))
    group = row['groups']['start_free']
    assert group['raw_final']['rms_wu'] == 2.
    assert group['pic_jump']['rms_wu'] == 3.
    assert group['saved_final']['rms_wu'] == 1.
    bank = acc.banks['start_free']
    assert bank['path_wu'].item() == 2.
    assert bank['step_square_wu2'].item() == 2.
    assert bank['raw_final_square_wu2'].item() == 4.
    assert bank['pic_jump_square_wu2'].item() == 9.


def test_final_new_arrival_unobserved_never_arrived_and_zero_pair_denominators():
    acc = Accumulator(points([2., 5.]), 1., .1)
    acc.accept(candidate([[2., 5.], [1., 4.], [0., 3.]], [0., 0.]))
    summary = acc.summary()
    assert acc.first_arrival.tolist() == [1, -1]
    assert summary['unobserved_after_first_arrival'] == 1
    assert summary['never_arrived_count'] == 1
    assert summary['groups']['start_free']['step_rms_wu'] is None
    assert summary['groups']['start_free']['reversal_fraction'] is None


def test_actual_start_mask_is_separate_from_geometric_plan_predicate():
    acc = Accumulator(points([0.]), 1., .1)
    row = acc.accept(candidate([[0.], [0.]], [0.], start_actual=[False]))
    assert row['actual_vs_geometric_start_mismatch'] == 1
    assert acc.first_start_arrived.item() == 0
    assert acc.first_arrival.item() == 0


def prepared_observer(capture, data, *, broken=None):
    """Exercise the actual on_rollout and optimize_window-return interfaces."""
    import warp as wp
    wp.init()
    tr = SimpleNamespace(T=data['T'], x=[wp.array(x.numpy(), dtype=wp.vec3, device='cpu') for x in data['path']],
                         pin=wp.array(data['pin'].numpy().astype(np.float32), dtype=wp.float32, device='cpu'))
    owned = SimpleNamespace(start=data['path'][0].clone(), raw=data['path'][-1].clone(),
                            pin=data['pin'].clone(), promoted=data['promoted'].clone())
    if broken is not None:
        getattr(owned, broken).add_(1)
    plan = data['plan'].clone()
    def original(*, on_rollout):
        on_rollout(tr, data['promoted'], data['attempt']-1)
        return None, None, None, None, [], dict(owned_endpoint=owned, plan_img=plan,
            arrived_mask=data['start_arrived'], pace_r=data['radius'])
    capture.wrap(original)()
    return tr, owned, plan


def test_owned_rollout_plan_and_stats_survive_mutation_and_reject_reset(tmp_path):
    c = Capture(points([0.]), 1., .1, require_cuda=False)
    data = candidate([[0.], [1.], [0.]], [0.])
    tr, owned, plan = prepared_observer(c, data)
    import warp as wp
    wp.to_torch(tr.x[1]).fill_(99)
    owned.promoted.fill_(99)
    plan.fill_(99)
    assert c.pending['path'][1, 0, 0].item() == 1.
    assert c.pending['plan'][0, 0].item() == 0.
    c.commit(0, points([0.]), None, None, dict(frame_end=3, null_commit=1))
    assert c.accumulator.accepted == 0 and c.pending is None
    assert c.accumulator.first_arrival.item() == -1
    prepared_observer(c, candidate([[0.], [.25], [0.]], [0.], attempt=2))
    c.commit(1, points([0.]), None, None, dict(frame_end=6))
    assert c.accumulator.accepted == 1
    assert c.accumulator.banks['start_free']['path_wu'].item() == .5
    assert c.rows[0]['attempt'] == 2


@pytest.mark.parametrize('field', ['start', 'raw', 'promoted'])
def test_owned_endpoint_mismatch_rejected_and_pending_cleared(field):
    c = Capture(points([0.]), 1., .1, require_cuda=False)
    with pytest.raises(ValueError, match='Owned endpoint'):
        prepared_observer(c, candidate([[0.], [0.]], [0.]), broken=field)
    assert c.pending is None and c.accumulator.accepted == 0


def test_wrong_outer_attempt_or_positions_never_admitted():
    c = Capture(points([0.]), 1., .1, require_cuda=False)
    prepared_observer(c, candidate([[0.], [0.]], [0.]))
    with pytest.raises(ValueError, match='matching'):
        c.commit(1, points([0.]), None, None, dict(frame_end=2))
    with pytest.raises(ValueError, match='differs'):
        c.commit(0, points([1.]), None, None, dict(frame_end=2))
    assert c.accumulator.accepted == 0


def test_discontinuous_start_or_moved_pin_fails_before_admission():
    acc = Accumulator(points([0.]), 1., .1)
    with pytest.raises(ValueError, match='same endpoint'):
        acc.accept(candidate([[1.], [1.]], [0.]))
    with pytest.raises(ValueError, match='pin moved'):
        acc.accept(candidate([[0.], [1.]], [0.], pin=[True]))
    assert acc.accepted == 0 and acc.first_arrival.item() == -1


def test_exact_sidecar_and_counter_layout_roundtrip(tmp_path):
    limits = budget(n=2, windows=2)
    writer = EvidenceWriter(tmp_path/'evidence', limits)
    c = Capture(points([0., 2.]), 1., .1, writer=writer, require_cuda=False)
    prepared_observer(c, candidate([[0., 2.], [.25, 1.], [0., 0.]], [0., 0.]))
    c.commit(0, points([0., 0.]), None, None, dict(frame_end=3))
    with np.load(tmp_path/'evidence/accepted_001.npz', allow_pickle=False) as archive:
        assert set(archive.files) == {'__meta__', 'plan', 'raw', 'promoted', 'start_arrived', 'end_arrived', 'start_pin'}
        np.testing.assert_array_equal(archive['end_arrived'], [True, True])
        np.testing.assert_array_equal(archive['promoted'], points([0., 0.]).numpy())
    arrays = c.accumulator.arrays()
    arrays.update(final_pin=torch.zeros(2, dtype=torch.bool), final_pin_at_attempt=torch.zeros(2, dtype=torch.int32))
    assert sum(value.numel()*value.element_size() for value in arrays.values()) == limits['final_counter_array_bytes']
    writer.arrays('counters.npz', arrays, {})
    with np.load(tmp_path/'evidence/counters.npz', allow_pickle=False) as archive:
        assert archive['first_arrival_commit'].tolist() == [0, 1]
        for field in FLOAT_FIELDS:
            assert archive['start_free__'+field].dtype == np.float64
        for field in INT_FIELDS:
            assert archive['start_free__'+field].dtype == np.int32
    with pytest.raises(FileExistsError):
        writer.arrays('counters.npz', arrays, {})


def test_actual_two_window_runner_observer_is_neutral(monkeypatch):
    from physmorph.pipeline import runner, PipelineConfig
    from physmorph.mpm.state import MPMParams
    source = np.random.default_rng(27).uniform(-1.5, 1.5, (160, 3)).astype(np.float32)
    target = (source*[1.2, .85, 1.05]+[.1, 0, 0]).astype(np.float32)
    cfg = PipelineConfig(T=3, iters=3, animations=2, stop_after_windows=2,
        loss_res=12, render_views=2, render_elevs=(0., .5), render_res=24,
        device='cpu', patience=5, phys_loss='ot_pace', commit_pic=True,
        commit_pic_objective=True, outer_render_committed=True, body_ctrl=True,
        shift_sub=False, motion_accounting=True, warm_start=True,
        lambda_auto=.3, w_kin=0., w_ctrl=0., w_box=0.)
    prm = MPMParams(dx=1., nx=32, ny=32, nz=32)
    original = runner.optimize_window
    controls = []

    def retain_controls(function):
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            value = result[-1]['dfc']
            assert value is not None
            controls.append(value.copy())
            return result
        return wrapper

    monkeypatch.setattr(runner, 'optimize_window', retain_controls(original))
    plain = runner.run_pipeline(source, target, prm, cfg, log=lambda *_: None)
    plain_controls, controls = controls, []
    capture = Capture(torch.from_numpy(source), 1., prm.dt, require_cuda=False, max_windows=2)
    monkeypatch.setattr(runner, 'optimize_window', retain_controls(capture.wrap(original)))
    observed = runner.run_pipeline(source, target, prm, cfg, log=lambda *_: None, on_commit=capture.commit)
    assert capture.accumulator.accepted == len(capture.rows) == 2
    assert len(plain_controls) == len(controls) == 2
    for a, b in zip(plain['frames'], observed['frames']):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(plain_controls, controls):
        np.testing.assert_array_equal(a, b)
    assert len(plain['frames']) == len(observed['frames'])
    for a, b in zip(plain['history'], observed['history']):
        assert a.get('lambda') == b.get('lambda')
        assert a.get('body_accepted_alphas') == b.get('body_accepted_alphas')
        assert a.get('frame_end') == b.get('frame_end')
        if a.get('frame_end') and not a.get('null_commit') and not a.get('held'):
            assert a['body_accepted_alphas']
            assert a['body_update_modes_rms'] == b['body_update_modes_rms']
            assert a['body_rms_wu'] == b['body_rms_wu']
    assert plain['guards'] == observed['guards']
    assert all(row['endpoint_exact'] and row['pins_exact'] for row in capture.rows)
