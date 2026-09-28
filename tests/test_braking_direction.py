import torch

from physmorph.pipeline.braking_direction import data_tangent_direction
from physmorph.pipeline.frozen_body_window import project_terminal


def test_terminal_projection_preserves_displacement_at_saturated_node():
    d = torch.tensor([[1.,0.,0.],[.6,0.,0.]])
    original = d.clone()
    b = project_terminal(torch.tensor([[0.,4.,0.],[0.,4.,0.]]),d)
    assert torch.equal(d,original)
    torch.testing.assert_close(b,torch.tensor([[0.,0.,0.],[0.,.8,0.]]))


def test_data_projection_intersection_and_parallel_constraints():
    d = torch.tensor([2.,3.,4.])
    x,y = torch.tensor([1.,0.,0.]),torch.tensor([0.,1.,0.])
    torch.testing.assert_close(data_tangent_direction(d,x,y),torch.tensor([0.,0.,4.]))
    torch.testing.assert_close(data_tangent_direction(d,x,-x),torch.tensor([0.,3.,4.]))
    torch.testing.assert_close(data_tangent_direction(-d,x,y),-d)
