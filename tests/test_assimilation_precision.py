"""Per-call FP64 assimilation, with FP32 state stores and fixed pin branches.

CPU only by default. No production rollout is executed here.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from physmorph.plasticity import assimilation as ordinary
from physmorph.plasticity import assimilation_adjoint as adjoint
from scripts.probes.post_assimilation_closure_diagnosis import fp64_handoff


def cases():
    F=torch.diag_embed(torch.tensor([[1.3,.82,1.05],[1.,1.,1.],
        [2.,2.,.6],[12.,4.,1/48],[-1.,1.,1.],[0.,0.,0.],[4.,4.,5e-7]],dtype=torch.float32))
    F[0,0,1]=.17;F[2,1,2]=-.12
    P=torch.eye(3).repeat(len(F),1,1)
    P[0]=torch.tensor([[1.1,.13,.03],[0.,.91,.07],[.02,0.,1.04]])
    old=torch.tensor([True,False,False,False,False,False,False])
    new=torch.tensor([False,True,False,True,False,False,False])
    return F,P,old,new


@pytest.mark.parametrize('eta',[0.,.5,1.])
@pytest.mark.parametrize('iso',[False,True])
@pytest.mark.parametrize('settle',[False,True])
def test_each_entry_matches_independent_diagnostic_and_per_call_store(eta,iso,settle):
    F,P,old,new=cases();saved=(F.clone(),P.clone())
    cfg=SimpleNamespace(assim=eta,assim_iso=iso,assim_smin=.2,assim_smax=5.,settle_pin_assim=settle)
    expected=fp64_handoff(F,P,old,new,cfg)
    actual=adjoint.assimilate_handoff(F,P,old,new,eta=eta,isochoric=iso,
                                    settle_pin_assim=settle,fp64=True)
    first=ordinary.assimilate_elastic(F.numpy(),P.numpy(),eta=eta,isochoric=iso,fp64=True)
    assert isinstance(first,np.ndarray) and first.dtype==np.float32
    if settle:
        first[old.numpy()]=P[old].numpy()
        first[new.numpy()]=ordinary.assimilate_elastic(F[new].numpy(),first[new.numpy()],
                                                      eta=1.,isochoric=False,fp64=True)
    assert actual.dtype==torch.float32 and actual.device.type=='cpu'
    assert torch.equal(actual,expected) and np.array_equal(first,expected.numpy())
    assert torch.equal(F,saved[0]) and torch.equal(P,saved[1])
    if settle:assert torch.equal(actual[old],P[old])


def test_per_call_cast_is_distinct_and_differentiable():
    gen=torch.Generator().manual_seed(17)
    F=(torch.eye(3)[None]+.2*torch.randn(5,3,3,generator=gen)).requires_grad_()
    P=(torch.eye(3)[None]+.1*torch.randn(5,3,3,generator=gen)).requires_grad_()
    old,new=torch.zeros(5,dtype=torch.bool),torch.ones(5,dtype=torch.bool)
    actual=adjoint.assimilate_handoff(F,P,old,new,eta=.5,isochoric=True,fp64=True)
    once=adjoint.assimilate_handoff(F.double(),P.double(),old,new,eta=.5,isochoric=True).float()
    assert not torch.equal(actual,once), 'Fixture must detect a missing intermediate FP32 store'
    first=adjoint.assimilate_elastic_differentiable(F.double(),P.double(),eta=.5,isochoric=True).float()
    explicit=adjoint.assimilate_elastic_differentiable(F.double(),first.double(),eta=1.,isochoric=False).float()
    assert torch.equal(actual,explicit)
    weight=torch.linspace(-.7,.9,F.numel()).reshape_as(F)
    g_actual=torch.autograd.grad((actual*weight).sum(),(F,P),retain_graph=True)
    g_explicit=torch.autograd.grad((explicit*weight).sum(),(F,P))
    for a,b in zip(g_actual,g_explicit):
        assert torch.equal(a,b) and torch.isfinite(a).all() and float(a.norm())>.01


def test_double_internal_ops_and_fp32_public_store(monkeypatch):
    calls=[];svd=adjoint.svd3
    def record(value):
        calls.append(value.dtype)
        return svd(value)
    monkeypatch.setattr(adjoint,'svd3',record)
    F,P,_,_=cases()
    for enabled,dtype in ((False,torch.float32),(True,torch.float64)):
        calls.clear()
        actual=adjoint.assimilate_elastic_differentiable(F,P,fp64=enabled)
        assert actual.dtype==torch.float32 and calls==[dtype,dtype]
    calls.clear()
    ordinary.assimilate_elastic(F.numpy(),P.numpy(),fp64=True)
    assert calls==[torch.float64,torch.float64]


def test_default_ordinary_path_never_enters_new_shared_path(monkeypatch):
    F,P,_,_=cases()
    # Capture the existing CPU primitive before forbidding the new shared path.
    expected=ordinary.assimilate_elastic(F.numpy(),P.numpy(),eta=.4,isochoric=True)
    def forbidden(*args,**kwargs):raise AssertionError('Opt-in path used by default')
    monkeypatch.setattr(adjoint,'assimilate_elastic_differentiable',forbidden)
    assert np.array_equal(ordinary.assimilate_elastic(F.numpy(),P.numpy(),eta=.4,isochoric=True),expected)
    assert np.array_equal(ordinary.assimilate_elastic(F.numpy(),P.numpy(),eta=.4,isochoric=True,fp64=False),expected)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_default_adjoint_forward_and_gradient_are_identical_to_existing_composition(dtype):
    F,P,_,_=cases();F=F[:3].to(dtype).requires_grad_();P=P[:3].to(dtype).requires_grad_()
    inverse,info=torch.linalg.inv_ex(P,check_errors=False)
    assert not info.any()
    increment=adjoint._ElasticIncrement.apply(F@inverse,.4,True)
    expected=adjoint._CumulativeBand.apply(increment@P,.2,5.,True)
    actual=adjoint.assimilate_elastic_differentiable(F,P,eta=.4,isochoric=True)
    explicit_false=adjoint.assimilate_elastic_differentiable(F,P,eta=.4,isochoric=True,fp64=False)
    assert torch.equal(actual,expected) and torch.equal(actual,explicit_false)
    w=torch.arange(F.numel(),dtype=dtype).reshape_as(F)/17
    ga=torch.autograd.grad((actual*w).sum(),(F,P),retain_graph=True)
    ge=torch.autograd.grad((expected*w).sum(),(F,P))
    assert all(torch.equal(a,e) for a,e in zip(ga,ge))


@pytest.mark.parametrize('iso',[False,True])
def test_fp32_identity_pullback_matches_independent_linear_tangent(iso):
    F=torch.eye(3)[None].requires_grad_();P=torch.eye(3)[None].requires_grad_()
    w=torch.tensor([[[.2,.7,-.5],[-.3,1.2,.4],[.8,-.6,.1]]])
    out=adjoint.assimilate_elastic_differentiable(F,P,eta=.37,isochoric=iso,fp64=True)
    gF,gP=torch.autograd.grad((out*w).sum(),(F,P))
    sym=(w+w.transpose(1,2))*.5;skew=(w-w.transpose(1,2))*.5
    if iso:sym=sym-torch.eye(3)[None]*sym.diagonal(dim1=1,dim2=2).sum(1)[:,None,None]/3
    torch.testing.assert_close(gF,.37*sym,rtol=64*torch.finfo(torch.float32).eps,atol=64*torch.finfo(torch.float32).eps)
    torch.testing.assert_close(gP,.63*sym+skew,rtol=64*torch.finfo(torch.float32).eps,atol=64*torch.finfo(torch.float32).eps)


@pytest.mark.parametrize('iso',[False,True])
@pytest.mark.parametrize('channel',['F','Fp'])
def test_full_fp32_stored_handoff_gradient_against_resolved_centered_fd(iso,channel):
    # Smooth noncommuting row, repeated spectrum, active cumulative clamp, and
    # old/new/free pin branches. Perturbations stay away from branch boundaries.
    F,P,_,_=cases();F=F[:4].clone().requires_grad_();P=P[:4].clone().requires_grad_()
    old=torch.tensor([True,False,False,False]);new=torch.tensor([False,True,False,True])
    gen=torch.Generator().manual_seed(729)
    direction=torch.randn(F.shape,generator=gen);direction/=direction.norm()
    weight=torch.randn(F.shape,generator=gen,dtype=torch.float64)
    def objective(f,p):
        return (adjoint.assimilate_handoff(f,p,old,new,eta=.4,isochoric=iso,fp64=True).double()*weight).sum()
    gradients=torch.autograd.grad(objective(F,P),(F,P))
    gradient=gradients[0 if channel=='F' else 1]
    analytic=float((gradient.double()*direction.double()).sum())
    assert abs(analytic)>.05, 'FD must resolve a substantive derivative'
    assert torch.equal(gradients[0][old],torch.zeros_like(F[old]))
    torch.testing.assert_close(gradients[1][old],weight[old].float(),rtol=0,atol=0)
    for step in (1e-3,5e-4):
        with torch.no_grad():
            if channel=='F':plus,minus=objective(F+step*direction,P),objective(F-step*direction,P)
            else:plus,minus=objective(F,P+step*direction),objective(F,P-step*direction)
        measured=float((plus-minus)/(2*step))
        assert analytic==pytest.approx(measured,rel=2e-3,abs=3e-4),(iso,channel,step,analytic,measured)


def test_skipped_elastic_row_still_clamps_and_old_pins_block_f_gradient():
    P=torch.diag(torch.tensor([7.,1.3,.1]))[None].requires_grad_()
    F=(torch.diag(torch.tensor([-1.2,1.1,.9]))[None]@P.detach()).requires_grad_()
    value=adjoint.assimilate_elastic_differentiable(F,P,fp64=True)
    gF,gP=torch.autograd.grad(value.sum(),(F,P))
    assert torch.equal(gF,torch.zeros_like(F)) and torch.isfinite(gP).all()
    sv=torch.linalg.svdvals(value)
    assert float(sv.min())>=.2-1e-7 and float(sv.max())<=5.+1e-7


@pytest.mark.parametrize('eta',[0.,.5])
def test_unsupported_consensus_and_fp64_state_rejected_even_for_identity(eta):
    F,P,_,_=cases()
    with pytest.raises(ValueError,match='consensus'):
        ordinary.assimilate_elastic(F.numpy(),P.numpy(),eta=eta,Fe=F.numpy(),fp64=True)
    with pytest.raises(ValueError,match='FP32'):
        ordinary.assimilate_elastic(F.double().numpy(),P.double().numpy(),eta=eta,fp64=True)
    with pytest.raises(ValueError,match='FP32'):
        adjoint.assimilate_elastic_differentiable(F.double(),P.double(),eta=eta,fp64=True)


@pytest.mark.parametrize('flag',[None,1,'yes'])
def test_precision_metadata_is_boolean(flag):
    F,P,old,new=cases()
    for fn in (lambda:ordinary.assimilate_elastic(F.numpy(),P.numpy(),fp64=flag),
               lambda:adjoint.assimilate_elastic_differentiable(F,P,fp64=flag),
               lambda:adjoint.assimilate_handoff(F,P,old,new,fp64=flag)):
        with pytest.raises(ValueError,match='boolean'):fn()


def test_fp64_ordinary_output_is_owned_and_no_input_graph_is_retained():
    F,P,_,_=cases();F.requires_grad_();P.requires_grad_()
    actual=ordinary.assimilate_elastic(F,P,fp64=True)
    assert isinstance(actual,np.ndarray) and actual.dtype==np.float32
    saved=P.detach().clone();actual[:]=99
    assert torch.equal(P.detach(),saved) and F.grad is None and P.grad is None
