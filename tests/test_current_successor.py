"""Current-owned passive preparation; small CPU boundary and ownership gates."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from physmorph.pipeline.current_successor import prepare_original_successor
from physmorph.pipeline.frozen_body_window import FrozenBodyWindow
from physmorph.pipeline.preparation_admission import AdmissionHistory
from physmorph.plasticity import assimilate_elastic
from test_post_assimilation_window import make_case
from test_post_selection_real import build_context
from test_window_selection import equal


def current_case():
    # N54/T20, dt.002, dx.5, grid16^3. Two translated copies exceed the
    # ordinary layer builder's32-neighbor minimum. Data merit is synthetic.
    owner, _, cfg = make_case(pin_slip=True, fp64=True)
    spec = deepcopy(owner.spec)
    n0 = len(spec.x0)
    for name in ('x0', 'm', 'lam', 'mu', 'Fp', 'v0', 'F0', 'C0', 'vol0',
                 'Fg0', 'bond_rest', 'bond_frag', 'eta', 'pin'):
        value = getattr(spec, name)
        setattr(spec, name, np.concatenate([value, value]).copy())
    spec.x0[n0:, 0] += 1.25
    spec.bond_nbr = np.concatenate([owner.spec.bond_nbr, owner.spec.bond_nbr+n0])
    layer = owner.spec.layer
    spec.layer = tuple(np.concatenate([v, v+n0 if i == 2 else v]).copy()
                       if isinstance(v, np.ndarray) else v for i, v in enumerate(layer))
    owner = FrozenBodyWindow(spec, SimpleNamespace(idx=owner.idx.repeat(2, 1),
        weights=owner.weights.repeat(2, 1)), owner.gate.repeat(2, 1), owner.coefficients,
        owner.stress.repeat(1, 2, 1, 1), owner.surface_u.repeat(2))
    cfg.layer_relax = cfg.layer_ctrl = True
    cfg.layer_k = 8
    cfg.bonds = True
    cfg.ctrl_rprop = cfg.ctrl_rprop_smooth = cfg.ctrl_rprop_arrived = True
    cfg.ctrl_rprop_k = 8
    cfg.settle_pin = cfg.settle_pin_ray = True
    cfg.u_rprop = True  # Its future nonzero-control bound is outside passive scope.
    ctx, original = build_context(owner, cfg)
    start, end = original[0][0], original[0][-1]
    ctx._original[5].update(plan_img=end.copy(), pace_r=.05)
    n = len(start)
    old = owner.spec.pin.astype(bool)
    neighbors = np.stack([(np.arange(n)+i+1) % n for i in range(8)], 1).astype(np.int32)
    reversals = np.zeros(n, np.int32)
    reversals[3] = 1
    history = AdmissionHistory(scale=np.where(old, 0., 1.).astype(np.float32),
        previous=-(end-start), reversals=reversals, frozen=np.zeros(n, bool),
        settled=old, settled_at=np.where(old, 0, -1).astype(np.int32),
        pins=old.astype(np.float32), neighbors=neighbors)
    return ctx, history


def test_current_preview_matches_ordinary_boundary_without_mutating_original():
    ctx, history = current_case()
    try:
        before = deepcopy(ctx._original)
        old_history = history.arrays()
        state, admission, report = prepare_original_successor(ctx, history)
        actual = state.arrays()
        old = old_history['pins'] > .5
        newly = admission['newly']
        pins = actual['pin'] > .5
        assert newly.sum() == 1 and newly[3]
        assert pins.sum() == old.sum()+1
        F, P = before[2]['F'], ctx._owner.spec.Fp
        ordinary = assimilate_elastic(F, P, eta=.5, isochoric=True, fp64=True)
        ordinary[old] = P[old]
        ordinary[newly] = assimilate_elastic(F[newly], ordinary[newly], eta=1.,
                                             isochoric=False, fp64=True)
        np.testing.assert_allclose(actual['Fp'], ordinary, rtol=2e-7, atol=2e-7)
        assert np.array_equal(actual['x0'], before[0][-1])
        assert np.array_equal(actual['F0'], before[2]['F'])
        assert np.array_equal(actual['v0'][~pins], before[2]['v'][~pins])
        assert np.array_equal(actual['C0'][~pins], before[2]['C'][~pins])
        assert not actual['v0'][pins].any() and not actual['C0'][pins].any()
        assert np.array_equal(actual['vol'], ctx._owner.spec.vol0)
        assert report['neutralized_policy_fields'] == ['layer_ug']
        assert report['next_controlled_optimizer_state'] == 'not previewed'
        equal(ctx._original, before)
        equal(history.arrays(), old_history)
        # Detachment survives later edits/lease closure; it is no selection receipt.
        ctx._owner.spec.vol0.fill(999.)
        admission['pins'].fill(0.)
        ctx.close()
        assert np.array_equal(state.arrays()['vol'], actual['vol'])
        assert np.array_equal(state.arrays()['pin'], actual['pin'])
    finally:
        ctx.close()


def test_current_preview_rejects_other_pin_history_and_expired_context():
    ctx, history = current_case()
    values = history.arrays()
    values['pins'][0] = 1-values['pins'][0]
    values['settled'] = values['pins'] > .5
    wrong = AdmissionHistory(**values)
    try:
        with pytest.raises(ValueError, match='does not match the head pin state'):
            prepare_original_successor(ctx, wrong)
    finally:
        ctx.close()
    with pytest.raises(RuntimeError, match='expired'):
        prepare_original_successor(ctx, history)


def test_actual_runner_preview_matches_its_own_next_preparation(monkeypatch):
    from test_window_selection_pipeline import recipe, observed_run
    _, _, _, cfg = recipe()
    cfg.ctrl_rprop_smooth = cfg.ctrl_rprop_arrived = True
    cfg.ctrl_rprop_k = 8
    cfg.settle_pin_ray = cfg.settle_pin_slip = True
    cfg.settle_pin_still = cfg.settle_pin_confirm = False
    cfg.assim_fp64 = True
    previews, histories = {}, {}
    def select(ctx):
        histories[ctx._window] = ctx.current_admission_history()
        if ctx._window > 0:
            previews[ctx._window] = ctx.preview_current_successor()
        return ctx.original()
    result, prepared, _, _ = observed_run(monkeypatch, select, cfg=cfg)
    assert not any(result['guards'].values())
    compared = 0
    for index, (preview, predicted, report) in previews.items():
        if index+1 not in histories:
            continue
        actual = prepared[index+1]
        expected = preview.arrays()
        assert actual.keys() == expected.keys()
        for key in expected:
            if key == 'layer_ug':
                continue  # Zero-u-only qualification tested separately on CUDA.
            if key == 'Fp':
                np.testing.assert_allclose(actual[key], expected[key], rtol=2e-7, atol=2e-7)
            else:
                assert np.array_equal(actual[key], expected[key]), key
        history = histories[index+1]
        for key in ('scale', 'reversals', 'frozen', 'settled', 'settled_at', 'pins'):
            assert np.array_equal(history[key], predicted[key]), key
        assert np.array_equal(history['previous'], predicted['displacement'])
        compared += 1
    assert compared >= 2
