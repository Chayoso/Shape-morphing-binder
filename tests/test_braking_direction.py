import torch
import json
from copy import deepcopy
from pathlib import Path

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


def test_shared_gates_reproduce_frozen_gpu_p306_decisions():
    from scripts.probes.terminal_braking import assess_candidates
    path = Path(__file__).resolve().parents[1]/'docs/evidence/p306/terminal_braking3.json'
    original = json.loads(path.read_text())['rows']
    replay = deepcopy(original)
    assess_candidates(replay,torch.float32)
    for old,new in zip(original[3:],replay[3:]):
        for key in ('checks','nominal_feasible','resolved_braking','feasible'):
            assert old[key]==new[key]
