"""Matched-origin and independent component-gate checks without simulation."""
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pytest
import torch

from scripts.probes import running_braking_repair as running
from scripts.probes.silhouette_braking_repair import silhouette_repair
from test_paired_braking import attach_merit
from test_remainder_braking import CurvedReference,make_case,synthetic_summary


@dataclass
class SplitReference(CurvedReference):
    pbr_weight: float = 1.
    coordinate: int = 0

    def terms(self,x):
        result = super().terms(x)
        result['silhouette'] = x[0,self.coordinate]
        result['pbr'] = result['render']-result['silhouette']
        return result


def split_case(coordinate=0):
    model,packet,original = make_case()
    packet['reference'] = SplitReference(coordinate=coordinate)
    data = packet['reference'].terms(packet['positions'][-1])
    packet['history'].update(d_render=float(data['silhouette']),d_pbr=float(data['pbr']))
    attach_merit(packet)
    return model,packet,original


def test_pair_uses_one_gradient_origin_and_component_gate_changes_acceptance(monkeypatch,tmp_path):
    model,packet,original = split_case()
    monkeypatch.setattr(running,'summarize',synthetic_summary)
    forward = model.evaluate
    grad_calls = []
    def observe(terminal,displacement):
        if displacement.requires_grad:
            grad_calls.append(1)
            assert len(grad_calls)==1,'An arm recalculated the common origin'
        return forward(terminal,displacement)
    monkeypatch.setattr(model,'evaluate',observe)
    report = silhouette_repair(model,packet,None,None,tmp_path)
    assert report['arm_order']==['aggregate','silhouette'] and len(grad_calls)==1
    assert torch.equal(model.coefficients,original)
    a,b = (report['arms'][k] for k in report['arm_order'])
    assert a['accepted_running_updates']==1 and a['repeated_feasible']
    assert not a['rows'][-1]['silhouette_ceiling_passed']
    assert not a['rows'][-1]['original_merit_nonincrease']  # Reporting does not select.
    assert b['accepted_running_updates']==0 and not b['repeated_feasible']
    assert b['trials'] and all(not t['running_update_accepted'] for t in b['trials'])
    for arm in (a,b):
        assert arm['rows'][:3]==report['baseline']['rows']
        assert arm['rows'][3]['running']==report['origin']['running']
        assert arm['shared_origin_sha256']==report['origin']['package_sha256']
        assert arm['running_replay_noise']==report['origin']['noise']
        for trial in arm['trials']:
            if 'running_origin' in trial:
                assert trial['running_origin']==report['origin']['running']
                assert trial['reduction_threshold']==report['origin']['threshold']
    with np.load(tmp_path/'aggregate/linearization1.npz') as x,np.load(tmp_path/'silhouette/linearization1.npz') as y:
        np.testing.assert_array_equal(x['running_gradient'],y['running_gradient'])
        np.testing.assert_array_equal(x['data_gradients'],y['data_gradients'][:2])
        np.testing.assert_array_equal(x['origin_data'],y['origin_data'][:2])


def test_shared_origin_owns_gradients_and_replays_enforce_actual_component(monkeypatch,tmp_path):
    model,packet,original = split_case(coordinate=2)
    monkeypatch.setattr(running,'summarize',synthetic_summary)
    for name in ('baseline','origin','arm'): (tmp_path/name).mkdir()
    baseline = running.repair(model,packet,None,None,tmp_path/'baseline',baseline_only=True)
    origin = running.repair(model,packet,None,None,tmp_path/'origin',shared_baseline=baseline,origin_only=True)
    digest = origin.digest()
    origin.tensor('gradient_render','cpu').zero_()
    data = origin.decode();data['threshold']=100.;data['terminal_record']['data']['silhouette']=-100.
    assert origin.digest()==digest
    assert float(origin.tensor('gradient_render','cpu').norm())>0
    forward = model.evaluate
    repeats = Counter()
    def perturb(terminal,displacement):
        values = forward(terminal,displacement)
        if not torch.equal(displacement,original[:,:3]):
            key = displacement.detach().numpy().tobytes()
            repeats[key]+=1
            if repeats[key]==3:  # Second fixed replay after its selection forward.
                x = values['x'].clone();x[0,2]=.21
                values.update(x=x,v=x*.5,positions=torch.stack((x*.5,x)),V=torch.stack((x*.5,x))*.5)
        return values
    monkeypatch.setattr(model,'evaluate',perturb)
    report = running.repair(model,packet,None,None,tmp_path/'arm',shared_baseline=baseline,
        shared_origin=origin,protect_silhouette=True,correction_rounds=2,quality_backtracking=True)
    assert report['accepted_running_updates']==1 and len(report['fixed_candidate_replays'])==3
    assert not report['repeated_feasible']
    witness = report['fixed_candidate_replays'][1]
    assert witness['feasible'] and witness['running_decrease_resolved']
    assert not witness['silhouette_ceiling_passed'] and not witness['full_search_passed']
    assert origin.digest()==digest and torch.equal(model.coefficients,original)
    packet['trial05_terminal']+=.001
    with pytest.raises(ValueError,match='coefficients'):
        running.repair(model,packet,None,None,tmp_path/'arm',shared_baseline=baseline,
            shared_origin=origin,quality_backtracking=True,correction_rounds=2)
