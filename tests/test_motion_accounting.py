"""Actual position-only control can move geometry while stored momentum remains zero."""
import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory
from physmorph.pipeline.motion_accounting import collect_rollout, summarize


def test_position_control_is_visible_without_inventing_momentum():
    x = np.array([[i, j, k] for i in (-.2, .2) for j in (-.2, .2) for k in (-.2, .2)], np.float32)
    n, steps, dt = len(x), 3, .01
    normal = np.tile(np.array([0., 1., 0.], np.float32), (n, 1))
    pins = np.zeros(n, np.float32); pins[0] = 1
    layer = (np.ones(n, np.float32), normal, np.arange(n, dtype=np.int32)[:, None],
             np.ones((n, 1), np.float32), 0.)
    tr = Trajectory(x, 1., 0., 0., MPMParams(dx=.5, dt=dt, drag=0., nx=16, ny=16, nz=16,
                     grid_min=(-4., -4., -4.)), steps, device='cpu', requires_grad=False,
                    vol0=np.full(n, .01, np.float32), layer=layer,
                    layer_u=wp.array(np.full(n, .06, np.float32), dtype=wp.float32, device='cpu'), pin=pins)
    tr.rollout()
    before = [a.numpy().copy() for a in tr.x]
    accounting = collect_rollout(tr, dt)
    start, end = torch.from_numpy(before[0]), torch.from_numpy(before[-1])
    pic = end.clone(); pic[1:, 1] -= .01
    shifted = pic.clone(); shifted[1:, 0] += .001
    mask = torch.ones(n, dtype=torch.bool)
    report = summarize(accounting, start, end, end, pic, shifted, mask, mask,
                       torch.from_numpy(pins).bool())
    free = report['cohorts']['arrived_free']
    assert free['particles'] == n-1
    assert free['terminal_speed_wu_s']['mpm_mean'] == 0.
    assert free['terminal_speed_wu_s']['geometry_mean'] == pytest.approx(2., abs=1e-5)
    expected_promoted = (shifted-torch.from_numpy(before[-2]))/dt
    assert free['terminal_speed_wu_s']['promoted_geometry_mean'] == pytest.approx(
        float(expected_promoted[1:].norm(dim=-1).mean()), abs=1e-5)
    assert free['terminal_speed_wu_s']['promoted_difference_rms'] == pytest.approx(
        float(expected_promoted[1:].square().sum(-1).mean().sqrt()), abs=1e-5)
    assert free['window_component_rms_wu']['surface_u'] == pytest.approx(.06, abs=1e-7)
    assert free['window_component_rms_wu']['commit_pic'] == pytest.approx(.01, abs=1e-7)
    assert free['window_component_rms_wu']['commit_shift'] == pytest.approx(.001, abs=1e-7)
    assert sum(free['window_signed_fraction'].values()) == pytest.approx(1., abs=1e-5)
    assert report['window_closure_max_wu'] < 1e-6
    assert report['cohorts']['pinned_at_start']['window_net_rms_wu'] == 0.
    assert report['cohorts']['pinned_at_start']['terminal_speed_wu_s']['promoted_geometry_mean'] == 0.
    assert report['cohorts']['transit_free']['particles'] == 0
    for old, current in zip(before, tr.x):
        np.testing.assert_array_equal(current.numpy(), old)
