"""Common-origin integration with analytic physical-response stand-ins."""
from collections import Counter

import numpy as np
import pytest
import torch

from scripts.probes import running_braking_repair as running
from scripts.probes.support_braking_repair import support_repair
from test_silhouette_braking import split_case
from test_remainder_braking import synthetic_summary


def test_pair_has_one_origin_and_support_can_reject_data_feasible_candidate(monkeypatch,tmp_path):
    model,packet,original=split_case(coordinate=2)
    target=np.array([[.28,.6,.4],[.3925,.6,.4]],dtype=np.float32)
    monkeypatch.setattr(running,'summarize',synthetic_summary)
    forward=model.evaluate;calls=[]
    def observe(terminal,displacement):
        if displacement.requires_grad: calls.append(1)
        return forward(terminal,displacement)
    monkeypatch.setattr(model,'evaluate',observe)
    report=support_repair(model,packet,None,target,tmp_path)
    assert len(calls)==1 and report['search_status']=='ready'
    assert report['arm_order']==['silhouette','support']
    assert report['origin']['support_metadata']['count']==1
    a,b=(report['arms'][k] for k in report['arm_order'])
    assert a['accepted_running_updates']==1 and a['repeated_feasible']
    assert not a['rows'][-1]['support_constraints_passed']
    assert b['accepted_running_updates']==0
    with np.load(tmp_path/'silhouette/linearization1.npz') as x,np.load(tmp_path/'support/linearization1.npz') as y:
        np.testing.assert_array_equal(x['data_gradients'],y['data_gradients'][:3])
        np.testing.assert_array_equal(x['running_gradient'],y['running_gradient'])
        np.testing.assert_array_equal(x['origin_data'],y['origin_data'][:3])
        assert y['data_gradients'].shape[0]==4
    for label,arm in report['arms'].items():
        valid=[t for t in arm['trials'] if t.get('valid')]
        assert len(list((tmp_path/label).glob('*_endpoint.npz')))==len(valid)
        for row in valid:
            assert len(row['support']['lost_ids'])==len(row['support']['gained_ids'])==3
            with np.load(tmp_path/label/(row['label']+'_endpoint.npz')) as f:
                np.testing.assert_array_equal(f['support_values'],row['support']['values'])
    assert torch.equal(model.coefficients,original)


def test_actual_support_is_checked_again_on_fixed_repeats(monkeypatch,tmp_path):
    model,packet,original=split_case(coordinate=2)
    target=np.array([[.38,.6,.25],[.48,.6,.25]],dtype=np.float32)
    monkeypatch.setattr(running,'summarize',synthetic_summary)
    # Build immutable packages separately so only the treatment's repeats are perturbed.
    for label in ('baseline','origin','arm'): (tmp_path/label).mkdir()
    baseline=running.repair(model,packet,None,target,tmp_path/'baseline',baseline_only=True,prepare_support=True)
    origin=running.repair(model,packet,None,target,tmp_path/'origin',shared_baseline=baseline,
                          origin_only=True,prepare_support=True)
    assert origin.decode()['support_metadata']['status']=='ready'
    before=origin.digest(); base_before=baseline.digest()
    origin.tensor('support_targets','cpu').fill_(123.)
    baseline.tensor('endpoints','cpu').zero_()
    origin.decode()['support_metadata']['radius']=100.
    assert origin.digest()==before and baseline.digest()==base_before
    forward=model.evaluate;repeats=Counter()
    def perturb(terminal,displacement):
        value=forward(terminal,displacement)
        if not torch.equal(displacement,original[:,:3]):
            key=displacement.detach().numpy().tobytes();repeats[key]+=1
            if repeats[key]==3:
                # Clear the original data ceilings while losing the protected target.
                x=torch.tensor([[.551,.30,.15]],dtype=displacement.dtype)
                value.update(x=x,v=x*.5,positions=torch.stack((x*.5,x)),V=torch.stack((x*.5,x))*.5)
        return value
    monkeypatch.setattr(model,'evaluate',perturb)
    report=running.repair(model,packet,None,target,tmp_path/'arm',shared_baseline=baseline,
        shared_origin=origin,protect_silhouette=True,protect_support=True,save_search_endpoints=True,
        quality_backtracking=True,correction_rounds=2)
    assert report['accepted_running_updates']==1 and len(report['fixed_candidate_replays'])==3
    second=report['fixed_candidate_replays'][1]
    assert second['feasible'] and second['running_decrease_resolved'] and second['silhouette_ceiling_passed']
    assert not second['support_constraints_passed'] and not second['full_search_passed']
    assert not report['repeated_feasible']
    assert origin.digest()==before and baseline.digest()==base_before and torch.equal(model.coefficients,original)


@pytest.mark.parametrize('over_budget',[False,True])
def test_no_intervention_and_over_budget_skip_both_searches(monkeypatch,tmp_path,over_budget):
    model,packet,original=split_case(coordinate=2)
    if over_budget:
        angles=np.arange(6)*np.pi/3
        target=np.array([.2,.6,.2])+np.stack((.01*np.cos(angles),.01*np.sin(angles),np.zeros(6)),axis=1)
    else:
        target=np.array([[0.,0.,0.],[10.,0.,0.]])
    target=target.astype(np.float32)
    monkeypatch.setattr(running,'summarize',synthetic_summary)
    forward=model.evaluate;gradient_calls=[]
    def unchanged(terminal,displacement):
        assert torch.equal(displacement,original[:,:3]),'A skipped experiment searched a control'
        if displacement.requires_grad: gradient_calls.append(1)
        return forward(terminal,displacement)
    monkeypatch.setattr(model,'evaluate',unchanged)
    report=support_repair(model,packet,None,target,tmp_path)
    assert report['search_status']==('over_budget' if over_budget else 'no_intervention')
    assert report['origin']['support_metadata']['count']==(6 if over_budget else 0)
    assert not report['arms'] and not report['arm_order'] and len(gradient_calls)==1
    assert torch.equal(model.coefficients,original)


def test_uncertified_witness_skips_both_searches(monkeypatch,tmp_path):
    model,packet,original=split_case(coordinate=2)
    target=np.array([[.28,.6,.4],[.3925,.6,.4]],dtype=np.float32)
    monkeypatch.setattr(running,'summarize',synthetic_summary)
    select=running.select_support
    def failed_certificate(*args,**kwargs):
        result=select(*args,**kwargs)
        assert result['status']=='ready'
        # The helper's real uncertified-witness geometry has a separate unit case.
        # This injects its rejection status to test orchestration only.
        return dict(result,status='uncertified_witness')
    monkeypatch.setattr(running,'select_support',failed_certificate)
    forward=model.evaluate
    def unchanged(terminal,displacement):
        assert torch.equal(displacement,original[:,:3])
        return forward(terminal,displacement)
    monkeypatch.setattr(model,'evaluate',unchanged)
    report=support_repair(model,packet,None,target,tmp_path)
    assert report['search_status']=='uncertified_witness' and not report['arms'] and not report['arm_order']
    assert torch.equal(model.coefficients,original)
