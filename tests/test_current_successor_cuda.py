"""Explicit hyde06 current-owned preview gate: N54/T20/dt.002/dx.5/grid16^3.

This uses synthetic data merit and constructed reversal history. It validates
device preparation/ownership, not outer acceptance, native shape/rest or adjoints.
"""
from copy import deepcopy
import os
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from physmorph.compute import cuda_execution, to_array
from physmorph.pipeline.frozen_body_window import FrozenBodyWindow
from physmorph.pipeline.preparation_admission import AdmissionHistory
from physmorph.plasticity import assimilate_elastic
from test_current_successor import current_case
from test_frozen_withdrawal_window_cuda import upload_owner
from test_post_selection_real import build_context
from test_preparation_geometry_cuda import device_only, tensor


pytestmark = pytest.mark.skipif(
    os.environ.get('PHYSMORPH_PREPARATION_CUDA_TEST') != '1' or not torch.cuda.is_available(),
    reason='Explicit hyde06 current-successor CUDA gate')


def test_cuda_current_original_preview_owns_material_history_and_zero_controls(monkeypatch):
    cpu_context, cpu_history = current_case()
    try:
        source = cpu_context._owner
        owner = FrozenBodyWindow(source.spec, SimpleNamespace(idx=source.idx, weights=source.weights),
            source.gate, source.coefficients, source.stress, source.surface_u)
        cfg, history_inputs = deepcopy(cpu_context._cfg), cpu_history.arrays()
    finally:
        cpu_context.close()
    with cuda_execution('cuda:0', input_sizes=(54,)):
        owner = upload_owner(owner)
        cfg.device, cfg.compute_backend = 'cuda:0', 'cuda'
        ctx = None
        try:
            with device_only(monkeypatch):
                ctx, original = build_context(owner, cfg)
                ctx._original[5].update(plan_img=original[0][-1].copy(), pace_r=.05)
                current = {key: None if value is None else to_array(value, copy=True)
                           for key, value in history_inputs.items()}
                # Define the same reversal fixture against this actual GPU head,
                # rather than borrowing the CPU rollout's slightly different step.
                current['previous'] = -(original[0][-1]-original[0][0])
                history = AdmissionHistory(**current)
                ctx.bind_current_admission_history(history)
                old_history = ctx.current_admission_history()
                head = {key: tensor(original[2][key]).clone() for key in ('F', 'v', 'C')}
                head['x'] = tensor(original[0][-1]).clone()
                source_Fp = tensor(owner.spec.Fp).clone()
                state, admission, report = ctx.preview_current_successor()
                actual = state.arrays()
                assert all(hasattr(value, '__cuda_array_interface__') for value in actual.values())
                old = tensor(old_history['pins']) > .5
                newly, pins = tensor(admission['newly']), tensor(actual['pin']) > .5
                assert newly.sum() == 1 and newly[3]
                assert pins.sum() == old.sum()+1
                ordinary = assimilate_elastic(original[2]['F'], owner.spec.Fp,
                    eta=cfg.assim, isochoric=cfg.assim_iso, fp64=True)
                ordinary[to_array(old)] = owner.spec.Fp[to_array(old)]
                ordinary[to_array(newly)] = assimilate_elastic(original[2]['F'][to_array(newly)],
                    ordinary[to_array(newly)], eta=1., isochoric=False, fp64=True)
                torch.testing.assert_close(tensor(actual['Fp']), tensor(ordinary), rtol=2e-7, atol=2e-7)
                assert torch.equal(tensor(owner.spec.Fp), source_Fp)
                assert torch.equal(tensor(actual['x0']), head['x'])
                assert torch.equal(tensor(actual['F0']), head['F'])
                for name in ('v', 'C'):
                    assert torch.equal(tensor(actual[name+'0'])[~pins], head[name][~pins])
                    assert torch.count_nonzero(tensor(actual[name+'0'])[pins]) == 0
                    assert torch.equal(tensor(ctx._original[2][name]), head[name])
                assert torch.equal(tensor(ctx._original[0][-1]), head['x'])
                assert torch.equal(tensor(ctx._original[2]['F']), head['F'])
                for name in ('m', 'lam', 'mu', 'eta'):
                    assert torch.equal(tensor(actual[name]), tensor(getattr(owner.spec, name)))
                assert torch.equal(tensor(actual['vol']), tensor(owner.spec.vol0))
                for name, value in ctx.current_admission_history().items():
                    assert value is None if old_history[name] is None else torch.equal(tensor(value), tensor(old_history[name]))
                assert len(ctx._choices) == 1 and ctx._post_pins is None
                assert report['neutralized_policy_fields'] == ['layer_ug']
                assert report['next_controlled_optimizer_state'] == 'not previewed'
                passive = state.trajectory()
                assert passive.N == 54 and passive.T == 20 and not passive.layer_F
                assert passive.body_control is None and not passive.x[0].requires_grad
                assert torch.count_nonzero(wp.to_torch(passive._dfc(0))) == 0
                assert torch.count_nonzero(wp.to_torch(passive.layer_u)) == 0
                assert torch.equal(wp.to_torch(passive.C[0]), tensor(actual['C0']))
                saved_volume, saved_pin = tensor(actual['vol']).clone(), tensor(actual['pin']).clone()
                owner.spec.vol0.fill(999.)
                admission['pins'].fill(0.)
                ctx.close()
                assert torch.equal(tensor(state.arrays()['vol']), saved_volume)
                assert torch.equal(tensor(state.arrays()['pin']), saved_pin)
        finally:
            if ctx is not None:
                ctx.close()
            else:
                owner.close()
