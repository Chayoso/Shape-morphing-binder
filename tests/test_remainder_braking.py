"""Analytic curved-data callback checks, without a simulation or renderer."""
from types import SimpleNamespace
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from physmorph.pipeline.affine_braking import observed_remainder
from scripts.probes import running_braking_repair as probe


def test_remainder_uses_actual_projected_step_and_frozen_origin():
    G = torch.tensor([[1.,0.],[0.,2.]],dtype=torch.float64)
    origin = torch.tensor([2.,3.],dtype=torch.float64)
    delta = torch.tensor([.25,-.5],dtype=torch.float32)
    actual = origin+G@delta.double()+torch.tensor([.0625,.25])
    result = observed_remainder(actual,origin,G,delta)
    torch.testing.assert_close(result,torch.tensor([.0625,.25],dtype=torch.float64),rtol=0,atol=0)
    with pytest.raises(ValueError,match='Nonfinite'):
        observed_remainder(actual*float('inf'),origin,G,delta)


@dataclass
class CurvedReference:
    pbr_weight: float = 0.

    def terms(self,x):
        a,b,c = x[0]
        volume = 1-a+.1*b*b-c
        render = 2-a-b-c
        return dict(volume=volume,render=render,silhouette=render,pbr=render*0)


class CurvedModel:
    def __init__(self,fail_replay=False):
        self.coefficients = torch.tensor([[.2,.6,0.,.2,0.,0.]],dtype=torch.float32)
        self.spec = SimpleNamespace(T=2,prm=SimpleNamespace(dx=1.))
        self.calls = []
        self.fail_replay = fail_replay

    def evaluate(self,terminal,displacement):
        self.calls.append((displacement.detach().clone(),terminal.detach().clone()))
        x = torch.stack((displacement[0,0],displacement[0,1],terminal[0,0]))[None]
        if self.fail_replay and len(self.calls)==12:
            # Fixed coefficients, perturbed second held-out forward witness.
            x = x-torch.tensor([[.1,0.,0.]])
        positions = torch.stack((x*.5,x))
        return dict(x=x,v=x*.5,positions=positions,V=positions*.5,
            F=torch.eye(3).repeat(2,1,1,1),C=torch.zeros(2,1,3,3),
            body_energy=(displacement.square().sum()+terminal.square().sum()),
            valid=True,pins_exact=True,min_det=1.)


def synthetic_summary(capture,source,target):
    packet = capture.packets[8]
    length = float(packet['positions'][-1].norm())
    movement = dict(net_rms_sp=length,step_rms_sp=length/2,path_mean_sp=length,
                    physical_terminal_rms_wu_s=length/2,geometric_terminal_rms_wu_s=length/2)
    geometry = dict(sil_iou=1.,upper_target_near_frac=1.,target_near_frac=1.,tip_n=1,
                    fixed_source_upper_density=1.,chamfer=0.)
    return dict(rows=[dict(motion=dict(start_free=movement,start_arrived_free=movement),geometry=geometry)],
                source_upper_ids_sha256='synthetic')


def make_case(fail_replay=False):
    model = CurvedModel(fail_replay)
    before = model.coefficients.clone()
    reference = CurvedReference()
    baseline = model.evaluate(before[:,3:],before[:,:3])
    data = reference.terms(baseline['x'])
    packet = dict(reference=reference,positions=baseline['positions'],V=baseline['V'],F=baseline['F'],
        x0=torch.zeros(1,3),dt=1.,start_arrived=torch.ones(1,dtype=torch.bool),
        pins=torch.zeros(1,dtype=torch.bool),lambda_render=.5,
        trial05_terminal=torch.tensor([[.15,0.,0.]]),
        history=dict(d_vol=float(data['volume']),d_render=float(data['render']),d_pbr=0.,
                     body_update_modes_rms=[.3,.05]))
    return model,packet,before


@pytest.mark.parametrize('fail_replay',[False,True])
def test_curved_callback_replaces_model_error_and_preserves_origin(monkeypatch,tmp_path,fail_replay):
    model,packet,before = make_case(fail_replay)
    monkeypatch.setattr(probe,'summarize',synthetic_summary)
    result = probe.repair(model,packet,None,None,tmp_path,correction_rounds=2)
    assert torch.equal(model.coefficients,before)
    trials = result['trials']
    corrected = [t for t in trials if t['correction']>0]
    assert corrected, 'Curved constraint must require actual-model correction'
    for i,t in enumerate(trials):
        if t['correction']==0:
            assert t['model_remainder_used']==[0.,0.]
        else:
            assert t['model_remainder_used']==trials[i-1]['observed_model_remainder']
        if not t.get('valid'): continue
        path = tmp_path/f"linearization{t['iteration']}.npz"
        with np.load(path) as data:
            origin_data = data['origin_data']
            original_bounds = data['bounds']
        # Origin data and original bounds remain the original ceiling throughout corrections.
        np.testing.assert_allclose(origin_data+original_bounds,
            [result['data_ceilings']['volume'],result['data_ceilings']['render']],rtol=0,atol=1e-15)
        np.testing.assert_allclose(t['shifted_bounds'],original_bounds-np.array(t['model_remainder_used']),rtol=0,atol=0)
        if t['running_update_accepted']:
            assert all(t['data'][k]<=result['data_ceilings'][k] for k in ('volume','render'))
    assert result['accepted_running_updates']==1 and len(trials)==2
    assert len(result['fixed_candidate_replays'])==3 and len(model.calls)==13
    assert all(torch.equal(model.calls[-1][0],d) and torch.equal(model.calls[-1][1],b)
               for d,b in model.calls[-3:])
    assert result['fixed_candidate_replays'][0]['feasible']
    assert result['fixed_candidate_replays'][2]['feasible']
    assert result['repeated_feasible'] is (not fail_replay)
    assert result['fixed_candidate_replays'][1]['feasible'] is (not fail_replay)


def test_raw_quality_rejection_keeps_origin_and_backtracks(monkeypatch,tmp_path):
    model,packet,before = make_case()
    def quality_summary(capture,source,target):
        result = synthetic_summary(capture,source,target)
        # The longest proposed correction moves past this independent quality boundary.
        if float(capture.packets[8]['positions'][-1].norm())<.625:
            result['rows'][0]['geometry']['upper_target_near_frac'] = .9
        return result
    monkeypatch.setattr(probe,'summarize',quality_summary)
    result = probe.repair(model,packet,None,None,tmp_path,correction_rounds=2,quality_backtracking=True)
    assert torch.equal(model.coefficients,before)
    rejected = [r for r in result['trials'] if r.get('quality_backtracking_rejected')]
    assert rejected and all(r['data_restored'] and r['data_running_passed'] for r in rejected)
    assert all(not r['running_update_accepted'] for r in rejected)
    assert result['accepted_running_updates']==1 and result['repeated_feasible']
    accepted = [r for r in result['trials'] if r['running_update_accepted']]
    assert len(accepted)==1 and accepted[0]['halvings']>rejected[0]['halvings']
    assert all(r['iteration']==1 for r in result['trials'])
    assert len(list(tmp_path.glob('linearization*.npz')))==1
    assert len(result['fixed_candidate_replays'])==3
    with np.load(tmp_path/'linearization1.npz') as archive:
        np.testing.assert_array_equal(archive['displacement'],before[:,:3].numpy())
