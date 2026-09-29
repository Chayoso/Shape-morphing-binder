"""CPU parity and ownership gates for shared current-prefix admission."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from physmorph.pipeline.config import PipelineConfig
from physmorph.pipeline.preparation_admission import (
    AdmissionHistory, accumulate_pins, exclude_transit_rays, preview_active_admission,
    reversal_candidates, smooth_material, update_reversal, validate_preview_config)
from physmorph.pipeline.settlement import pin_arrival_evidence


def config():
    return PipelineConfig(body_ctrl=True, body_terminal_ctrl=True, phys_loss='ot_pace',
        loss_units='density', ctrl_rprop=True, ctrl_rprop_smooth=True, ctrl_rprop_k=8,
        ctrl_rprop_arrived=True, settle_pin=True, settle_pin_ray=True,
        settle_pin_assim=True, settle_pin_slip=True, u_rprop=True)


def case():
    n = 12
    end = np.zeros((n, 3), np.float32)
    end[:, 0] = np.arange(n)*10
    end[0] = [0, 0, 0]; end[1] = [4, 0, 0]; end[2] = [4, 4, 0]
    start = end.copy(); start[:, 0] += 1
    plan = end.copy(); plan[0, 0] = 8; plan[5, 1] += 3
    arrived = np.ones(n, bool); arrived[0] = False; arrived[4] = False
    pins = np.zeros(n, np.float32); pins[3] = 1
    previous = np.zeros_like(end); previous[:, 0] = 1
    counts = np.zeros(n, np.int32); counts[[1, 2, 5]] = 1; counts[4] = 2
    arrays = dict(scale=np.ones(n, np.float32), previous=previous, reversals=counts,
        frozen=np.zeros(n, bool), settled=pins > .5, settled_at=np.full(n, -1, np.int32),
        pins=pins, neighbors=np.repeat(np.arange(n)[:, None], 8, axis=1))
    arrays['scale'][3] = 0
    start[3] = end[3]; previous[3] = 0; arrays['settled_at'][3] = 1
    return arrays, start, end, plan, arrived


def preview(arrays=None, **changes):
    original, start, end, plan, arrived = case()
    args = dict(history=AdmissionHistory(**(original if arrays is None else arrays)),
                x_start=start, x_accepted=end, plan_img=plan, pace_r=.25,
                arrived_start=arrived, cfg=config(), dx=.5, window_index=19)
    args.update(changes)
    return preview_active_admission(**args)


def test_endpoint_arrival_reversal_ray_and_old_pins():
    row = preview()
    assert np.flatnonzero(row['newly']).tolist() == [2, 4]
    assert np.flatnonzero(row['pins']).tolist() == [2, 3, 4]
    assert row['reversals'][1:6].tolist() == [2, 2, 0, 2, 2]
    assert not row['arrival'].eligible[5]  # Start-arrived is insufficient at the endpoint.
    assert row['arrival'].eligible[4]  # Confirm is off: endpoint arrival can admit an old two-reversal ID.
    assert row['telemetry']['pin_ray_blocked_frac'] == pytest.approx(1/3)
    assert row['settled_at'][[2, 3, 4]].tolist() == [20, 1, 20]
    assert np.all(row['scale'][[2, 3, 4]] == 0)
    assert np.all(row['scale_apply'][[2, 3, 4]] == 0)


def test_preview_matches_shared_ordinary_admission_composition_exactly():
    arrays, start, end, plan, arrived = case()
    ordinary = update_reversal(end-start, arrays['previous'], arrays['scale'], arrays['reversals'],
        arrays['frozen'], arrays['neighbors'], arrived, .5)
    eligible = pin_arrival_evidence(end, plan, .25, arrived, require_start=False).eligible
    newly = reversal_candidates(eligible, ordinary['reversals'], arrays['settled'])
    rays = exclude_transit_rays(end, plan, eligible, newly, 1.)
    settled, at = accumulate_pins(arrays['settled'], arrays['settled_at'], rays['newly'], 19)
    ordinary['scale'][settled] = 0
    ordinary['scale_apply'][settled] = 0
    row = preview(arrays)
    for key, value in ordinary.items():
        assert np.array_equal(row[key], value), key
    assert np.array_equal(row['settled'], settled)
    assert np.array_equal(row['settled_at'], at)
    assert np.array_equal(row['ray_samples'], rays['samples'])


def test_owned_history_and_results_cannot_mutate_future_preview():
    arrays, start, end, plan, arrived = case()
    history = AdmissionHistory(**arrays)
    args = (history, start, end, plan, .25, arrived)
    first = preview_active_admission(*args, cfg=config(), dx=.5, window_index=19)
    arrays['reversals'][:] = 100
    exported = history.arrays(); exported['neighbors'][:] = 0; exported['settled'][:] = True
    first['pins'][:] = 0; first['reversals'][:] = 0
    repeated = preview_active_admission(*args, cfg=config(), dx=.5, window_index=19)
    assert np.flatnonzero(repeated['newly']).tolist() == [2, 4]
    assert repeated['reversals'][1] == 2


def test_source_smoothing_changes_reversal_witness():
    displacement = np.array([[-1., 0, 0], [3., 0, 0]], np.float32)
    previous = np.array([[1., 0, 0], [1., 0, 0]], np.float32)
    args = (displacement, previous, np.full(2, .5, np.float32), np.zeros(2, np.int32), np.zeros(2, bool))
    local = update_reversal(*args, None, np.ones(2, bool), 1.)
    smooth = update_reversal(*args, np.array([[1], [0]]), np.ones(2, bool), 1.)
    assert local['flip'].tolist() == [True, False]
    assert smooth['flip'].tolist() == [False, False]
    assert np.array_equal(smooth['scale'], np.full(2, .6, np.float32))


def test_tiny_displacement_does_not_increment_reversals():
    displacement = np.full((2, 3), 1e-7, np.float32)
    row = update_reversal(displacement, -displacement, np.ones(2, np.float32),
        np.ones(2, np.int32), np.zeros(2, bool), None, None, 1.)
    assert not row['active'].any() and np.all(row['reversals'] == 1)


@pytest.mark.parametrize('neighbors', [None, np.array([[1, 2], [2, 0], [0, 1]])])
def test_exact_frozen_legacy_reversal_arithmetic(neighbors):
    # Frozen pre-extraction sequence; guards operand order and float32 behavior.
    now = np.array([[-1, .25, 0], [1, .5, 0], [0, 0, 0]], np.float32)
    previous = np.array([[1, .5, 0], [1, .5, 0], [0, 0, 0]], np.float32)
    scale = np.array([.125, .7, .25], np.float32); counts = np.array([1, 4, 0], np.int32)
    arrived = np.array([True, False, True])
    def old_smooth(v):
        return v if neighbors is None else (v+v[neighbors].sum(1))/float(neighbors.shape[1]+1)
    a, b = old_smooth(now), old_smooth(previous)
    n0, n1 = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    active = (n0 > 1e-4*.5) & (n1 > 1e-4*.5)
    cosine = (a*b).sum(1)/np.maximum(n0*n1, 1e-30)
    flip = active & (cosine < 0) & arrived; same = active & (cosine >= 0)
    expected_scale, expected_counts = scale.copy(), counts.copy()
    expected_scale[flip] *= .5; expected_counts[flip] += 1
    expected_scale[same] = np.minimum(1., expected_scale[same]*1.2)
    row = update_reversal(now, previous, scale, counts, np.zeros(3, bool), neighbors, arrived, .5)
    assert np.array_equal(row['scale'], expected_scale)
    assert np.array_equal(row['scale_apply'], old_smooth(expected_scale))
    assert np.array_equal(row['reversals'], expected_counts)


@pytest.mark.parametrize('field', ['settle_pin_confirm', 'settle_pin_clear', 'settle_pin_still',
    'settle_pin_stuck', 'settle_pin_follow', 'settle_pin_yield', 'settle_pin_kkt', 'settle_eta',
    'freeze_arrived', 'body_rprop', 'assim_consensus', 'layer_F', 'commit_pic'])
def test_unsupported_preview_policies_fail_closed(field):
    cfg = config(); setattr(cfg, field, True)
    with pytest.raises(ValueError):
        preview(cfg=cfg)


@pytest.mark.parametrize('field,change', [
    ('settled', lambda a: a.astype(np.float32)),
    ('frozen', lambda a: a.astype(np.float32)),
    ('reversals', lambda a: a.astype(np.float32)),
    ('settled_at', lambda a: a.astype(np.float64)),
    ('previous', lambda a: np.full_like(a, np.nan)),
    ('scale', lambda a: np.full_like(a, np.inf)),
    ('pins', lambda a: np.zeros_like(a)),
    ('neighbors', lambda a: np.full_like(a, 99))])
def test_invalid_current_history_cannot_be_repaired(field, change):
    arrays = case()[0]; arrays[field] = change(arrays[field])
    with pytest.raises(ValueError):
        preview(arrays)


def test_native_recipe_is_supported_without_claiming_u_bound_preview():
    path = Path(__file__).resolve().parents[1]/'docs/evidence/p335/p335_native2.protocol.json'
    cfg = PipelineConfig(**json.loads(path.read_text())['effective_config'])
    assert cfg.phys_loss == 'auto'
    cfg.phys_loss = 'ot_pace'  # Actual runner resolves this before the live selection seam.
    validate_preview_config(cfg)
    assert cfg.u_rprop
    assert 'not previewed' in preview()['scope']
