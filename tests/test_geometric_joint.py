import numpy as np
import pytest
import torch

from physmorph.mpm.function import RolloutSpec
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline.geometric import (
    GeometricConfig, forward_geometry, next_window_spec, optimize_geometric_window,
    restrict_render_control, trajectory_health,
    verify_replay,
)


def setup():
    rng = np.random.default_rng(11)
    x = rng.uniform(-.8, .8, (40, 3)).astype(np.float32)
    prm = MPMParams(dx=.75, dt=1/120, nx=16, ny=16, nz=16, grid_min=(-6.,)*3)
    spec = RolloutSpec(x, 1., 800., 400., prm, 3, device="cpu",
                       vol0=compute_rest_volumes(x, 1., prm, "cpu"))
    mask = np.linalg.norm(x, axis=1) > .8
    target = torch.tensor(x)*torch.tensor([1.1, .9, 1.])
    def physics(state, control):
        return ((state[0]-target)**2).mean() + .001*control.square().mean()
    def render(x, fg):
        # A surface observation fixture, not a claim of native raster accuracy.
        return ((x[mask]-target[mask])**2).mean() + .001*(fg[mask, 0]-1.1).square().mean()
    return spec, mask, physics, render


def test_joint_update_and_surface_control_restriction():
    spec, mask, physics, render = setup()
    cfg = GeometricConfig(iterations=2, step_size=.003)
    result = optimize_geometric_window(spec, physics, render, mask, cfg)
    assert result["accepted_steps"] == 2
    for h in result["history"]:
        assert h["g_render_interior"] == 0 and h["raw_render_interior"] > 0
        assert h["true_slope"] < 0 and h["loss"] < h["loss_before"]
        assert h["loss"] == pytest.approx(h["physics"]+h["render"])
        assert h["loss"] <= h["loss_before"] + cfg.armijo*h["step_fraction"]*h["true_slope"]
    with pytest.raises(ValueError, match="boolean"):
        optimize_geometric_window(spec, physics, render, mask.astype(float), cfg)
    promoted = next_window_spec(spec, result["trajectory"])
    assert np.array_equal(promoted.F_geom0, result["trajectory"].F_geom[-1].numpy())
    assert np.array_equal(promoted.vol0, spec.vol0)


@pytest.mark.parametrize("sequence", [False, True])
def test_control_restriction_preserves_surface_and_zeroes_interior(sequence):
    shape = (2, 5, 3, 3) if sequence else (5, 3, 3)
    g = torch.arange(np.prod(shape)).reshape(shape).float()
    mask = torch.tensor([True, False, True, False, True])
    out = restrict_render_control(g, mask)
    if sequence:
        assert torch.equal(out[:, mask], g[:, mask]) and torch.count_nonzero(out[:, ~mask]) == 0
    else:
        assert torch.equal(out[mask], g[mask]) and torch.count_nonzero(out[~mask]) == 0


def test_renderer_cannot_directly_change_interior_control_with_zero_physics_signal():
    spec, mask, _, render = setup()
    def no_physics(state, control):
        return control.square().sum()*0
    result = optimize_geometric_window(spec, no_physics, render, mask,
                                        GeometricConfig(iterations=1, step_size=.0005))
    assert result["accepted_steps"] == 1
    assert torch.count_nonzero(result["control"][:, ~mask]) == 0
    assert torch.count_nonzero(result["control"][:, mask]) > 0
    # Interior still moves through MPM stress coupling, despite zero interior dFc.
    displacement = result["trajectory"].x[-1].numpy()-spec.x0
    assert np.linalg.norm(displacement[~mask]) > 1e-6


def test_guard_rejects_invalid_substep_geometry():
    import warp as wp
    spec, _, _, _ = setup()
    _, tr = forward_geometry(torch.zeros(3, len(spec.x0), 3, 3), spec)
    assert trajectory_health(tr, GeometricConfig())["valid"]
    bad = tr.F_geom[1].numpy()
    bad[0, 0, 0] = -1.
    tr.F_geom[1] = wp.array(bad, dtype=wp.mat33, device="cpu")
    health = trajectory_health(tr, GeometricConfig())
    assert not health["valid"] and health["substep"] == 1


def test_replay_detects_finite_affine_state_change_invisible_to_loss():
    import warp as wp
    spec, _, physics, render = setup()
    control = torch.zeros(3, len(spec.x0), 3, 3)
    state, reference = forward_geometry(control, spec)
    replay_state, replay = forward_geometry(control, spec)
    bad = replay.C[-1].numpy()
    bad[0, 0, 1] += .1
    replay.C[-1] = wp.array(bad, dtype=wp.mat33, device="cpu")
    assert trajectory_health(replay, GeometricConfig())["valid"]
    assert physics(state, control) == physics(replay_state, control)
    assert render(state[0], state[3]) == render(replay_state[0], replay_state[3])
    with pytest.raises(RuntimeError, match="changed C"):
        verify_replay(reference, replay)


def test_surface_restriction_conflict_stops_without_a_physics_only_fallback():
    from physmorph.mpm.function import warp_mpm_geometry
    spec, mask, _, render = setup()
    control = torch.zeros(3, len(spec.x0), 3, 3, requires_grad=True)
    state = warp_mpm_geometry(control, spec)
    gr = torch.autograd.grad(render(state[0], state[3]), control)[0]
    # Construct a local conflict: the restricted direction ascends the true
    # image objective through interior control, with surface gradient cancelled.
    coefficient = -gr.clone()
    coefficient[:, ~mask] *= .5
    def physics(_state, c):
        return (c*coefficient).sum()
    result = optimize_geometric_window(spec, physics, render, mask, GeometricConfig(iterations=2))
    assert result["accepted_steps"] == 0
    assert result["history"][0]["true_slope"] > 0
    assert result["history"][0]["reason"] == "restricted_direction_is_not_joint_descent"
    assert torch.count_nonzero(result["control"]) == 0


def test_rejected_candidate_preserves_control_and_all_rollout_state():
    spec, mask, physics, render = setup()
    control = torch.zeros(3, len(spec.x0), 3, 3)
    _, initial = forward_geometry(control, spec)
    result = optimize_geometric_window(spec, physics, render, mask,
        GeometricConfig(iterations=1, step_size=10., line_search_steps=1))
    assert result["accepted_steps"] == 0
    assert result["history"][0]["reason"] == "joint_line_search_failed"
    assert torch.equal(result["control"], control)
    verify_replay(initial, result["trajectory"])
