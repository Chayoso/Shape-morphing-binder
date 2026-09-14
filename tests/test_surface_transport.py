from dataclasses import replace

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm.function import RolloutSpec, warp_mpm_geometry
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline.geometric import forward_geometry, next_window_spec, verify_replay, trajectory_health, GeometricConfig
from physmorph.pipeline.response_control import surface_support_constraint


def fixture():
    rng = np.random.default_rng(17)
    x = rng.uniform(-.6, .6, (50, 3)).astype(np.float32)
    prm = MPMParams(dx=.75, dt=1/120, nx=12, ny=12, nz=12, grid_min=(-4.5,)*3)
    spec = RolloutSpec(x, 1., 800., 400., prm, 4, device="cpu",
                       vol0=compute_rest_volumes(x, 1., prm, "cpu"), surface0=x[::3].copy())
    c = torch.tensor(rng.normal(0., .01, (4, 50, 3, 3)).astype(np.float32))
    return spec, c


@pytest.mark.parametrize("speed_cap", [0., .005])
def test_passive_surface_matches_coincident_particles_without_changing_physics(speed_cap):
    spec, c = fixture()
    spec = replace(spec, prm=replace(spec.prm, v_max=speed_cap))
    state, tr = forward_geometry(c, spec)
    base, _ = forward_geometry(c, replace(spec, surface0=None))
    for a, b in zip(state[:4], base):
        assert torch.equal(a, b)
    for t in range(spec.T+1):
        np.testing.assert_allclose(tr.surface_x[t].numpy(), tr.x[t].numpy()[::3], atol=1e-7)
        np.testing.assert_allclose(tr.surface_F[t].numpy(), tr.F_geom[t].numpy()[::3], atol=1e-7)
    final_volume = compute_rest_volumes(tr.x[-1].numpy(), spec.m, spec.prm, "cpu")
    np.testing.assert_allclose(tr.surface_density[-1].numpy(), spec.m/final_volume[::3], rtol=2e-6)
    # Restart carries the cumulative surface geometry, not just its positions.
    _, a = forward_geometry(c, next_window_spec(spec, tr))
    _, b = forward_geometry(c, next_window_spec(spec, tr))
    verify_replay(a, b)
    restarted = next_window_spec(next_window_spec(spec, tr), a)
    np.testing.assert_array_equal(restarted.surface_reference0, spec.surface0)
    np.testing.assert_array_equal(restarted.surface_density0, tr.surface_density[0].numpy())


def test_surface_adjoint_matches_finite_response_and_requires_physical_stress():
    spec, c = fixture()
    leaf = c.clone().requires_grad_()
    weights = torch.linspace(-1., 1., len(spec.surface0)*3).reshape(-1, 3)
    def objective(state):
        return (weights*state[4]).sum()+.03*state[5][:, 0, 1].sum()
    ad, = torch.autograd.grad(objective(warp_mpm_geometry(leaf, spec)), leaf)
    direction = ad/ad.norm()
    eps = .005
    plus, _ = forward_geometry(c+eps*direction, spec)
    minus, _ = forward_geometry(c-eps*direction, spec)
    fd = (objective(plus)-objective(minus))/(2*eps)
    assert float(ad.norm()) > 1e-5
    assert float(fd) == pytest.approx(float((ad*direction).sum()), rel=.02, abs=1e-5)
    zero = replace(spec, lam=0., mu=0.)
    state = warp_mpm_geometry(leaf, zero)
    gradient, = torch.autograd.grad(objective(state), leaf)
    assert torch.count_nonzero(gradient) == 0
    assert torch.equal(state[4], torch.tensor(spec.surface0))


@pytest.mark.parametrize("fault,reason", [("endpoint_support", "surface_mass_support"),
    ("original_edges", "surface_triangle_deformation"),
    ("singular_average", "surface_expected_normal_degenerate")])
def test_surface_health_rejects_endpoint_and_reference_failures_without_inverse_exception(fault, reason):
    spec, c = fixture()
    spec.surface_faces = np.array([[0, 1, 2]], np.int64)
    _, tr = forward_geometry(c*0, spec)
    assert trajectory_health(tr, GeometricConfig())["valid"]
    if fault == "endpoint_support":
        tr.surface_density[-1].zero_()
    elif fault == "original_edges":
        # A new window's moderate edges must not replace the original reference.
        tr.surface_reference0 = tr.surface_reference0*.1
    else:
        rotations = np.tile(np.eye(3, dtype=np.float32), (len(spec.surface0), 1, 1))
        for i, angle in enumerate([0., 2*np.pi/3, 4*np.pi/3]):
            ca, sa = np.cos(angle), np.sin(angle)
            rotations[i] = [[ca, -sa, 0.], [sa, ca, 0.], [0., 0., 1.]]
        tr.surface_F = [wp.array(rotations, dtype=wp.mat33, device="cpu") for _ in tr.surface_F]
    health = trajectory_health(tr, GeometricConfig())
    assert not health["valid"] and health["reason"] == reason
    if fault == "endpoint_support":
        assert health["substep"] == spec.T


def test_support_response_has_the_exact_health_boundary_and_persistent_reference():
    spec, c = fixture()
    _, tr = forward_geometry(c*0, spec)
    reference = tr.surface_density[0].numpy().copy()
    tr.surface_density0 = reference.copy()
    endpoint = reference.copy()
    endpoint[:3] *= [.05, .1, .2]
    tr.surface_density[-1] = wp.array(endpoint, dtype=wp.float32, device="cpu")
    row = surface_support_constraint(tr, .1).reshape(spec.T+1, -1)
    assert row[-1, 0] > 1 and float(row[-1, 1]) == pytest.approx(1.) and row[-1, 2] < 1
    # Restart-style first-frame density is not allowed to replace the original.
    tr.surface_density[0] = wp.array(reference*.5, dtype=wp.float32, device="cpu")
    assert torch.allclose(row[-1], surface_support_constraint(tr, .1).reshape_as(row)[-1])
