"""Independent analytic and constrained-optimizer checks for the diagnostic step."""
import numpy as np
import pytest
import torch
from scipy.optimize import minimize

from physmorph.pipeline.affine_braking import affine_ball_step,projected_affine_check,geometric_running


def solve(g,G,b,r):
    tensors = [torch.tensor(x,dtype=torch.float64) for x in (g,G,b)]
    return affine_ball_step(*tensors,r)


def test_restore_deficit_and_use_remaining_trust_radius():
    step,info = solve([0.,1.,0.],[[1.,0.,0.],[0.,0.,1.]],[-.6,0.],1.)
    torch.testing.assert_close(step,torch.tensor([-.6,-.8,0.],dtype=torch.float64))
    assert info['status']=='linear_candidate'
    # Scaling a feasible restorative step fails the original affine constraint.
    check = projected_affine_check(step/2,torch.tensor([[1.,0.,0.],[0.,0.,1.]]),
                                   torch.tensor([-.6,0.]),torch.float32)
    assert not check['passed']


@pytest.mark.parametrize('G,b',[
    ([[1.,0.,0.],[0.,1.,0.]],[-.8,-.8]),
    ([[1.,0.,0.],[-1.,0.,0.]],[-.1,-.1]),
    ([[0.,0.,0.],[1.,0.,0.]],[-.1,1.]),
])
def test_infeasible_linear_problem_is_reported(G,b):
    step,_ = solve([1.,2.,3.],G,b,1.)
    assert step is None


def test_redundant_and_flat_constraints():
    step,_ = solve([0.,0.,0.],[[1.,0.,0.],[2.,0.,0.]],[-.6,-1.2],1.)
    torch.testing.assert_close(step,torch.tensor([-.6,0.,0.],dtype=torch.float64))
    zero,_ = solve([1.,2.,3.],[[0.,0.,0.],[1.,0.,0.]],[0.,1.],0.)
    assert torch.equal(zero,torch.zeros(3,dtype=torch.float64))


@pytest.mark.parametrize('angle',[1e-4,1e-6,1e-8])
def test_nearly_opposing_planes_retain_their_narrow_feasible_set(angle):
    G = [[1.,0.],[-1.,angle]]
    b = [-.6,.6-.8*angle]
    step,_ = solve([0.,-1.],G,b,1.)
    assert step is not None
    torch.testing.assert_close(step,torch.tensor([-.6,-.8],dtype=torch.float64),rtol=0,atol=2e-8)


@pytest.mark.parametrize('value',[float('inf'),-float('inf'),float('nan')])
def test_nonfinite_projected_constraints_never_pass(value):
    with pytest.raises(ValueError,match='Nonfinite'):
        projected_affine_check(torch.zeros(3),torch.ones(2,3),torch.tensor([value,0.]),torch.float32)


def test_geometric_running_counts_initial_step_and_fixed_cohort():
    x0 = torch.zeros(2,3)
    positions = torch.tensor([[[1.,0.,0.],[50.,0.,0.]]]*3,requires_grad=True)
    value = geometric_running(positions,x0,.5,torch.tensor([True,False]))
    assert float(value)==pytest.approx(4/3)
    value.backward()
    assert torch.equal(positions.grad[:,1],torch.zeros(3,3))


def test_malformed_constraint_layout_is_rejected():
    with pytest.raises(ValueError,match='layout'):
        projected_affine_check(torch.zeros(3),torch.ones(2,3),torch.zeros(2,1),torch.float32)


def test_random_feasible_problems_against_independent_slsqp():
    rng = np.random.default_rng(512)
    for _ in range(12):
        g,G = rng.normal(size=7),rng.normal(size=(2,7))
        interior = rng.normal(size=7); interior *= .4/np.linalg.norm(interior)
        b = G@interior+rng.uniform(.05,.4,size=2)
        expected = minimize(lambda x:g@x,interior,jac=lambda x:g,method='SLSQP',
            constraints=[{'type':'ineq','fun':lambda x:b-G@x,'jac':lambda x:-G},
                         {'type':'ineq','fun':lambda x:1-x@x,'jac':lambda x:-2*x}],
            options={'ftol':1e-10,'maxiter':200})
        assert expected.success,expected.message
        actual,info = solve(g,G,b,1.)
        assert actual is not None,info
        a = actual.numpy()
        assert np.linalg.norm(a)<=1+1e-10 and np.all(G@a<=b+1e-10)
        assert g@a == pytest.approx(expected.fun,rel=0,abs=2e-8)
