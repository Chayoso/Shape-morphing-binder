"""Conservative fixed support, exact boundary and actual supplier replacement."""
import pytest
import torch

from physmorph.pipeline.support_braking import select_support, support_values, support_report, target_neighbors


def case(n=4):
    q = torch.zeros(n,3,dtype=torch.float64);q[:,0] = torch.arange(n)*10.
    x = q.clone();x[:,1] = .4
    base = x[None].repeat(3,1,1)
    origin = x.clone();origin[:2,1] = .7
    return q,base,origin


def test_selection_includes_all_stable_losses_and_owns_ids():
    q,base,origin = case()
    selected = select_support(base,origin,q,.5)
    assert selected['status']=='ready' and selected['count']==2
    assert selected['target_ids'].tolist()==[0,1] and selected['witnesses'].tolist()==[0,1]
    assert selected['witness_values'].dtype==torch.float64
    assert bool((selected['witness_values'][:3]<=0).all())
    q.fill_(99.)
    assert selected['targets'][0].tolist()==[0.,0.,0.]


def test_cap_is_abort_not_truncation_and_no_intervention_is_explicit():
    q,base,origin = case(6);origin[:,1] = .7
    selected = select_support(base,origin,q,.5)
    assert selected['status']=='over_budget' and selected['count']==6
    assert selected['target_ids'].numel()==6
    assert select_support(base,base[0],q,.5)['status']=='no_intervention'


def test_stable_coverage_does_not_imply_a_common_witness():
    q = torch.tensor([[0.,0.,0.]],dtype=torch.float64)
    base = torch.tensor([[[.4,0,0],[.8,0,0]], [[.8,0,0],[.4,0,0]], [[.4,0,0],[.8,0,0]]],dtype=torch.float64)
    selected = select_support(base,base[0]+torch.tensor([0.,1.,0.]),q,.5)
    assert selected['target_ids'].tolist()==[0] and selected['status']=='uncertified_witness'


def test_actual_support_rejects_witness_even_when_reassigned_supplier_covers():
    q,base,origin = case()
    selected = select_support(base,origin,q,.5)
    x = base[0].clone()
    x[0,1]=.7;x[2]=q[0]  # Different material covers target0; target2 now loses support.
    report = support_report(x,q,selected)
    assert report['protected_raw_covered']==[True,True]
    assert not report['passed'] and report['values'][0]>0
    assert report['stable_lost_ids']==[2] and report['lost_ids']==[[2]]*3
    assert report['protected_nearest_ids'][0]==2


def test_fp64_boundary_and_directional_derivative():
    q,base,origin = case()
    selected = select_support(base,origin,q,.5)
    x = base[0].clone().requires_grad_()
    values = support_values(x,selected['targets'],selected['witnesses'],.5)
    grad, = torch.autograd.grad(values.sum(),x)
    direction = grad/grad.norm();h=1e-6
    fd = (support_values(x+h*direction,selected['targets'],selected['witnesses'],.5).sum()
          - support_values(x-h*direction,selected['targets'],selected['witnesses'],.5).sum())/(2*h)
    assert float(fd.detach())==pytest.approx(float(grad.norm()),rel=1e-9)
    edge = base[0].clone();edge[:2,1] = .5
    assert support_report(edge,q,selected)['passed']
    edge[0,1] = torch.nextafter(edge[0,1],torch.tensor(float('inf')))
    r = support_report(edge,q,selected)
    assert not r['passed'] and r['scalar_radius_agreement']


def test_repeat_ambiguity_is_not_silently_protected():
    q,base,origin = case();base[2,0,1] = .7
    selected = select_support(base,origin,q,.5)
    assert selected['target_ids'].tolist()==[1]
    report = support_report(origin,q,selected)
    assert report['baseline_ambiguous_ids']==[0]
    assert report['lost_ids']==[[0,1],[0,1],[1]]


@pytest.mark.parametrize('radius',[0.,1e-200,float('inf'),float('nan')])
def test_invalid_radius_is_rejected(radius):
    q,base,origin = case()
    with pytest.raises(ValueError,match='support radius'):
        support_values(origin,q,torch.arange(4),radius)


def test_nonfinite_empty_and_invalid_tree_outputs_rejected(monkeypatch):
    from physmorph.pipeline import support_braking as module
    q,base,origin = case()
    with pytest.raises(ValueError,match='neighbor inputs'): target_neighbors(origin,q[:0])
    origin[0,0] = float('nan')
    with pytest.raises(ValueError,match='neighbor inputs'): target_neighbors(origin,q)
    class InvalidTree:
        def __init__(self,x): pass
        def query(self,q): return torch.full((len(q),),float('inf')),torch.full((len(q),),4)
    monkeypatch.setattr(module,'KDTree',InvalidTree)
    with pytest.raises(ValueError,match='neighbor output'): target_neighbors(base[0],q)


def test_report_requires_frozen_target_identity_and_coverage_layout():
    q,base,origin = case(); selected=select_support(base,origin,q,.5)
    with pytest.raises(ValueError,match='target identity'): support_report(origin,q+.01,selected)
    selected['baseline_covered']=selected['baseline_covered'][:2]
    with pytest.raises(ValueError,match='coverage layout'): support_report(origin,q,selected)
