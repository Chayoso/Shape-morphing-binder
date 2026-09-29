"""Changing material recoupling must retain each forward branch for reverse."""
import numpy as np
import torch
import warp as wp

from physmorph.mpm.function import PersistentAdjoint, RolloutSpec
from physmorph.mpm.state import MPMParams


def changing_activity_case():
    rng = np.random.default_rng(326)
    x = rng.uniform(-.4, .4, (24, 3)).astype(np.float32)
    distance = np.linalg.norm(x[:, None]-x[None], axis=-1)
    nbr = np.argsort(distance, axis=1)[:, 1:5].astype(np.int32)
    rest = np.take_along_axis(distance, nbr, axis=1).astype(np.float32)
    x[0, 0] += 1.8  # Dynamic fragment reconnects during this window.
    horizon = 8
    spec = RolloutSpec(x, 1., 80., 40.,
        MPMParams(dx=.5, dt=.005, nx=16, ny=16, nz=16, grid_min=(-4.,)*3,
                  drag=.1, smoothing=.95), horizon, device='cpu',
        vol0=np.full(24, .015, np.float32), bond_nbr=nbr, bond_rest=rest,
        bond_frag=np.zeros(24, np.float32), body_ctrl=True,
        v0=rng.normal(0, .02, (24, 3)).astype(np.float32))
    dc = torch.tensor(rng.normal(0, .002, (horizon, 24, 3, 3)), dtype=torch.float32)
    body = torch.tensor(rng.normal(0, .001, (24, 3)), dtype=torch.float32, requires_grad=True)
    return spec, dc, body


def activity(tr):
    return torch.stack([wp.to_torch(value).clone() for value in tr.frag_steps])


def observation(out):
    weights = torch.linspace(.3, 1.7, 3, device=out[5].device)
    return (out[5].double()*weights).square().mean() + .003*out[4].double().square().mean()


def body_direction(body):
    value = torch.sin(torch.arange(body.numel(), device=body.device).reshape_as(body).float()*.3)
    return value / value.square().mean().sqrt()


def test_changing_fragment_branches_have_correct_reverse_and_identical_primal():
    spec, dc, body = changing_activity_case()
    fixed = PersistentAdjoint(spec, position_sequence=True)
    out = fixed.apply_with_positions(dc, body_t=body)
    masks = activity(fixed.traj)
    assert torch.equal(masks[:, 0], torch.tensor([1., 1., 1., 0., 0., 0., 0., 0.]))
    assert len({mask.ptr for mask in fixed.traj.frag_steps}) == spec.T
    assert fixed.traj.frag_step is fixed.traj.frag_steps[-1]
    gradient, = torch.autograd.grad(observation(out), body)
    direction = body_direction(body)
    analytical = (gradient.double()*direction.double()).sum().item()
    assert abs(analytical) > 1e-3
    for epsilon in (1e-3, 5e-4):
        losses = []
        for sign in (-1., 1.):
            trial = fixed.apply_with_positions(dc, body_t=body.detach()+sign*epsilon*direction)
            # Recompute each perturbed path's own mask; the stencil stays on the
            # same discrete branch. Changes across time remain essential.
            assert torch.equal(activity(fixed.traj), masks)
            losses.append(observation(trial).item())
        observed = (losses[1]-losses[0])/(2*epsilon)
        assert abs(analytical-observed) <= max(.02*abs(observed), 2e-5)
    broken = PersistentAdjoint(spec, position_sequence=True)
    broken.traj.frag_steps = [broken.traj.frag_steps[0]]*spec.T
    old = broken.apply_with_positions(dc, body_t=body)
    # Reproduce the old scratch lifetime without changing any forward kernel.
    for actual, expected in zip(old, out):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    old_gradient, = torch.autograd.grad(observation(old), body)
    assert (old_gradient.double()*direction.double()).sum().item()*analytical < 0


def test_no_grad_forward_keeps_single_scratch_and_matches_recorded_masks():
    from physmorph.mpm.traj import Trajectory
    spec, dc, body = changing_activity_case()
    recorded = PersistentAdjoint(spec, position_sequence=True)
    out = recorded.apply_with_positions(dc, body_t=body)
    forward = Trajectory(spec.x0, spec.m, spec.lam, spec.mu, spec.prm, spec.T,
        device='cpu', requires_grad=False, persistent=True, vol0=spec.vol0,
        v0=spec.v0, bonds=spec.bonds(), dFc=recorded.dc_wp, body_control=recorded.body_wp)
    assert len({mask.ptr for mask in forward.frag_steps}) == 1
    for step in range(spec.T):
        forward.step(step)
        assert torch.equal(wp.to_torch(forward.frag_step), wp.to_torch(recorded.traj.frag_steps[step]))
    torch.testing.assert_close(wp.to_torch(forward.x[-1]), out[0], rtol=0, atol=0)
