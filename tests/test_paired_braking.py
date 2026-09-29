"""Paired diagnostic isolation and report-only merit using an analytic forward."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from physmorph.pipeline.diagnostic_binding import content_digest
from scripts.probes import paired_braking_repair as paired
from scripts.probes import running_braking_repair as running
from test_remainder_braking import make_case,synthetic_summary


def attach_merit(packet):
    def evaluate(values):
        data = packet['reference'].terms(values['x'])
        # Deliberately worsens a separate reported merit when braking succeeds.
        physical = float(data['volume']+10*(.2-values['x'][0,2]))
        render = float(data['render']);weight = packet['lambda_render']
        return dict(merit=physical+weight*render,physical=physical,volume=float(data['volume']),
            render=render,lambda_render=weight,unit_weight=.01,
            silhouette=float(data['silhouette']),pbr=float(data['pbr']))
    evaluate.binding_digest = lambda: content_digest(dict(reference=packet['reference'],weight=packet['lambda_render']))
    packet['evaluate_merit'] = evaluate
    packet['history']['loss'] = evaluate(dict(x=packet['positions'][-1]))['merit']


def test_shared_package_owns_immutable_rows_and_tensor_bytes(monkeypatch,tmp_path):
    model,packet,_ = make_case();attach_merit(packet)
    monkeypatch.setattr(running,'summarize',synthetic_summary)
    baseline = running.repair(model,packet,None,None,tmp_path,baseline_only=True)
    digest = baseline.digest();data = baseline.decode()
    data['ceilings']['volume']+=100
    data['rows'][0]['data']['volume']+=100
    baseline.tensor('C','cpu').fill_(10)
    baseline.tensor('coefficients','cpu').fill_(10)
    assert baseline.digest()==digest and baseline.decode()['ceilings']['volume']<10
    assert torch.equal(baseline.tensor('coefficients','cpu'),model.coefficients)
    with pytest.raises(FrozenInstanceError): baseline.payload=b'changed'
    packet['pins'][0] = True
    with pytest.raises(ValueError,match='cohort|context'):
        running.repair(model,packet,None,None,tmp_path,shared_baseline=baseline)


@pytest.mark.parametrize('invalid_first',[False,True])
def test_both_arms_share_one_baseline_and_merit_does_not_select(monkeypatch,tmp_path,invalid_first):
    model,packet,original = make_case();attach_merit(packet)
    monkeypatch.setattr(running,'summarize',synthetic_summary)
    if invalid_first:
        forward = model.evaluate
        def invalid(terminal,displacement):
            values = forward(terminal,displacement)
            if float(terminal[0,0])<.16: values['valid']=False
            return values
        monkeypatch.setattr(model,'evaluate',invalid)
    np.savez(tmp_path/'trial025.npz',terminal=np.array([[.175,0.,0.]],dtype=np.float32))
    out = tmp_path/'paired';out.mkdir()
    report = paired.paired_repair(model,packet,None,None,out)
    assert report['arm_order']==['terminal05','terminal025']
    assert set(report['arms'])=={'terminal05','terminal025'}
    assert torch.equal(model.coefficients,original)
    # One manual fixture forward precedes exactly three common original repeats.
    assert sum(torch.equal(b,original[:,3:]) for _,b in model.calls[1:])==3
    for label,arm in report['arms'].items():
        assert arm['rows'][:3]==report['baseline']['rows']
        assert arm['data_ceilings']==report['baseline']['ceilings']
        assert arm['original_merit_ceiling']==report['baseline']['merit_ceiling']
        assert arm['shared_baseline_sha256']==report['baseline']['package_sha256']
        assert arm['terminal_label']==label and arm['shared_context_before_after_exact']
    first = report['arms']['terminal05']
    if invalid_first:
        assert first['invalid_selected_terminal'] and first['accepted_running_updates']==0
    else:
        assert first['accepted_running_updates']==1 and first['repeated_feasible']
        assert not first['rows'][-1]['original_merit_nonincrease']
        assert len(first['fixed_candidate_replays'])==3
    assert report['arms']['terminal025']['trials']


def test_content_binding_detects_value_shape_dtype_and_nested_inputs():
    value = dict(spec=SimpleNamespace(dt=.1),tensor=torch.zeros(3),ref=[np.ones((2,3))])
    original = content_digest(value)
    value['spec'].dt=.2
    assert content_digest(value)!=original
    assert content_digest(torch.tensor(1.))!=content_digest(torch.tensor([1.]))
    assert content_digest(torch.ones(3))!=content_digest(torch.ones(3,dtype=torch.float64))
