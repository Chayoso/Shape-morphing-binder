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


@pytest.mark.parametrize('planes',[2,3,4,8])
def test_random_feasible_problems_against_independent_slsqp(planes):
    rng = np.random.default_rng(512)
    for _ in range(12):
        g,G = rng.normal(size=7),rng.normal(size=(planes,7))
        interior = rng.normal(size=7); interior *= .4/np.linalg.norm(interior)
        b = G@interior+rng.uniform(.05,.4,size=planes)
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


def test_three_active_planes_leave_only_tangent_ball_descent():
    G = np.eye(4)[:3]
    b = np.array([-.2,-.3,-.4])
    expected = np.r_[b,-np.sqrt(1-b@b)]
    step,info = solve([0.,0.,0.,1.],G,b,1.)
    np.testing.assert_allclose(step.numpy(),expected,rtol=0,atol=1e-12)
    assert info['active']==[0,1,2]


@pytest.mark.parametrize('angle',[1e-4,1e-6,1e-8])
def test_scaled_nearly_dependent_three_planes(angle):
    # An almost opposing combination retains a narrow restorative intersection.
    G = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[-1.,-1.,angle,0.]])
    b = np.array([-.2,-.3,.5-.4*angle])
    scale = np.array([1e-5,1e3,1e-2]); G*=scale[:,None]; b*=scale
    step,info = solve([0.,0.,0.,1.],G,b,1.)
    assert step is not None,info
    np.testing.assert_allclose(step.numpy(),[-.2,-.3,-.4,-np.sqrt(.71)],rtol=0,atol=3e-8)


@pytest.mark.parametrize('contradictory',[False,True])
def test_three_dependent_planes(contradictory):
    G = [[1.,0.,0.],[2.,0.,0.],[-3.,0.,0.]]
    b = [-.6,-1.2,1.5 if contradictory else 1.8]
    step,_ = solve([0.,1.,0.],G,b,1.)
    if contradictory: assert step is None
    else: np.testing.assert_allclose(step.numpy(),[-.6,-.8,0.],rtol=0,atol=1e-12)


def test_more_constraint_rows_than_displacement_dimensions():
    step,_ = solve([-1.,-1.],[[1.,0.],[0.,1.],[1.,1.]],[-.2,-.3,-.5],1.)
    np.testing.assert_allclose(step.numpy(),[-.2,-.3],rtol=0,atol=1e-12)


@pytest.mark.parametrize('planes',[4,8])
def test_all_support_planes_active_with_remaining_tangent_direction(planes):
    G = np.eye(planes+1)[:planes]
    b = np.linspace(-.1,-.2,planes)
    g = np.r_[np.zeros(planes),1.]
    step,info = solve(g,G,b,1.)
    np.testing.assert_allclose(step.numpy(),np.r_[b,-np.sqrt(1-b@b)],rtol=0,atol=1e-12)
    assert info['active']==list(range(planes))


@pytest.mark.parametrize('angle',[1e-4,1e-6,1e-8])
def test_eight_scaled_rows_with_nearly_opposing_intersection(angle):
    G = np.eye(9)[:8]
    G[2] = [-1.,-1.,angle,0.,0.,0.,0.,0.,0.]
    b = np.array([-.2,-.3,.5-.2*angle,-.1,-.1,-.1,-.1,-.1])
    scale = np.logspace(-5,3,8)
    g = np.r_[np.zeros(8),1.]
    step,info = solve(g,G*scale[:,None],b*scale,1.)
    assert step is not None,info
    expected = np.array([-.2,-.3,-.2,-.1,-.1,-.1,-.1,-.1])
    np.testing.assert_allclose(step.numpy(),np.r_[expected,-np.sqrt(1-expected@expected)],rtol=0,atol=4e-8)


@pytest.mark.parametrize('contradictory',[False,True])
def test_eight_dependent_rows_in_small_control_space(contradictory):
    G = np.array([[1.,0.],[2.,0.],[3.,0.],[4.,0.],[-1.,0.],[-2.,0.],[-3.,0.],[-4.,0.]])
    b = np.r_[-.6*np.arange(1,5),(.5 if contradictory else .6)*np.arange(1,5)]
    step,_ = solve([0.,1.],G,b,1.)
    if contradictory: assert step is None
    else: np.testing.assert_allclose(step.numpy(),[-.6,-.8],rtol=0,atol=1e-12)


def test_declared_eight_row_budget_rejects_larger_problem():
    with pytest.raises(ValueError,match='Invalid affine trust-ball'):
        solve([1.,1.],np.ones((9,2)),np.ones(9),1.)


def test_active_metadata_maps_past_zero_rows():
    step,info = solve([0.,1.],[[0.,0.],[1.,0.],[0.,0.]], [0.,-.6,1.],1.)
    np.testing.assert_allclose(step.numpy(),[-.6,-.8],rtol=0,atol=1e-12)
    assert info['active']==[0] and info['active_input_rows']==[1]
