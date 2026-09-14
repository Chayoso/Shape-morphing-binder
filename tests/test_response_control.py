import numpy as np
import pytest
import torch

from physmorph.mpm.function import RolloutSpec
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline.response_control import (
    ResponseConfig, SurfaceStrainBasis, model_value, optimize_response_window,
    quadratic_model, solve_constrained_response, TemporalSurfaceBasis,
)


@pytest.mark.parametrize("linear", [False, True])
def test_surface_basis_has_physical_strain_units_and_no_interior_support(linear):
    rng = np.random.default_rng(12)
    rest = rng.normal(size=(60, 3)).astype(np.float32)
    mask = np.linalg.norm(rest, axis=1) > 1.2
    basis = SurfaceStrainBasis(rest, mask, linear=linear)
    z = torch.tensor(rng.normal(size=basis.count).astype(np.float32))*.03
    c = basis.expand(z, 4)
    assert torch.count_nonzero(c[:, ~mask]) == 0
    assert torch.allclose(c, c.transpose(-1, -2), atol=1e-7)
    assert torch.equal(c[0], c[-1])
    actual_rms = c[0, mask].square().sum((1, 2)).mean().sqrt()
    assert float(actual_rms) == pytest.approx(float(z.norm()), rel=1e-5)


def models():
    # Image wants (+x,+y); mass prevents +x. The admissible image correction is +y.
    image = quadratic_model(torch.tensor([-.8, -.4]), torch.eye(2))
    mass = quadratic_model(torch.tensor([.2]), torch.tensor([[1.], [0.]]))
    return {"mass": mass, "physics": mass, "image": image}


def test_patch_modes_preserve_uniform_strain_and_resolve_local_actuation():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(240, 3)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    mask = np.arange(len(x)) < 200
    basis = SurfaceStrainBasis(x, mask, patches=8)
    # Partition of unity must preserve a constant strain despite whitening.
    uniform = torch.zeros(len(x), 3, 3)
    uniform[mask, 0, 0] = .1
    z = torch.einsum("knij,nij->k", basis.modes, uniform)/int(mask.sum())
    assert torch.allclose(basis.field(z), uniform, atol=1e-6)
    zlocal = torch.zeros(basis.count); zlocal[-1] = .02
    field = basis.field(zlocal)
    assert torch.count_nonzero(field[~mask]) == 0
    assert field[mask].flatten(1).norm(dim=1).std() > .001
    assert field[mask].square().sum((1, 2)).mean().sqrt() == pytest.approx(.02, rel=1e-5)


def test_constrained_response_uses_admissible_shape_change():
    m = models()
    d, info = solve_constrained_response(m, np.zeros(2), .5, 1.)
    assert info["success"]
    assert d[0] == pytest.approx(0, abs=1e-7)
    assert d[1] == pytest.approx(.4, abs=1e-6)
    assert model_value(m["mass"], d) <= m["mass"]["value"]+1e-9
    assert model_value(m["image"], d) < m["image"]["value"]


def test_positive_loss_rescaling_does_not_change_constrained_solution():
    m = models()
    d, _ = solve_constrained_response(m, np.zeros(2), .3, 1.)
    # Unlike changing relative weights in a sum, changing the units of a single
    # objective or equivalent inequality cannot change the desired solution.
    scaled = {key: {part: value*factor for part, value in m[key].items()}
              for key, factor in (("mass", 300.), ("physics", .005), ("image", 1000.))}
    ds, info = solve_constrained_response(scaled, np.zeros(2), .3, 1.)
    assert info["success"] and np.allclose(d, ds, atol=1e-6)


def test_conflicting_one_dimensional_control_has_no_feasible_image_descent():
    m = quadratic_model(torch.tensor([.2]), torch.tensor([[1.]]))
    image = quadratic_model(torch.tensor([-.8]), torch.tensor([[1.]]))
    d, info = solve_constrained_response({"mass": m, "physics": m, "image": image}, np.zeros(1), .5, 1.)
    assert info["success"]
    assert abs(d[0]) < 1e-7 and info["predicted_decrease"] < 1e-8


def test_active_particle_limit_allows_other_surface_patch_to_move():
    image = quadratic_model(torch.tensor([-.8, -.4]), torch.eye(2))
    constant = {"value": 1., "g": np.zeros(2), "H": np.zeros((2, 2))}
    modes = np.zeros((2, 2, 9)); modes[0, 0, 0] = modes[1, 1, 0] = 1.
    d, info = solve_constrained_response({"mass": constant, "physics": constant, "image": image},
        np.array([.399, 0.]), .3, 1., particle_modes=modes, max_particle_control=.4)
    assert info["success"] and info["particle_constraints"] > 0
    assert .399+d[0] < .4 and d[1] > .29


def test_temporal_basis_has_correct_time_weighted_strain_and_peak():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 3)).astype(np.float32); mask = np.arange(50) < 40
    spatial = SurfaceStrainBasis(x, mask)
    basis = TemporalSurfaceBasis(spatial, 5, 2)
    z = torch.linspace(.001, .012, basis.count)
    control = basis.expand(z, 5)
    assert control.shape == (5, 50, 3, 3)
    assert torch.count_nonzero(control[:, ~mask]) == 0
    assert control[:, mask].square().sum((2, 3)).mean().sqrt() == pytest.approx(float(z.norm()), rel=1e-5)
    assert control.flatten(2).norm(dim=2).max() == pytest.approx(basis.peak(z), rel=1e-5)
    assert not torch.equal(control[0], control[-1])


def test_image_step_preserves_required_feasible_physics_progress():
    m = quadratic_model(torch.tensor([-.6]), torch.tensor([[1.], [0.]]))
    image = quadratic_model(torch.tensor([0., -.5]), torch.eye(2))
    pack = {"mass": m, "physics": m, "image": image}
    dp, _ = solve_constrained_response(pack, np.zeros(2), .3, 1., "physics")
    upper = m["value"]-.8*(m["value"]-model_value(m, dp))
    d, info = solve_constrained_response(pack, np.zeros(2), .3, 1., "image",
        loss_bounds={"mass": upper, "physics": upper}, initial_delta=dp)
    assert info["success"] and model_value(m, d) <= upper+1e-8
    assert d[1] > .1


def test_geometric_limit_deflects_control_into_a_feasible_direction():
    image = quadratic_model(torch.tensor([-.8, -.4]), torch.eye(2))
    constant = {"value": 1., "g": np.zeros(2), "H": np.zeros((2, 2))}
    geometry = {"value": np.array([.99]), "J": np.array([[1., 0.]])}
    d, info = solve_constrained_response({"mass": constant, "physics": constant, "image": image},
        np.zeros(2), .3, 1., state_inequality=geometry)
    assert info["success"] and info["geometric_constraints"] > 0
    assert d[0] <= .0100001 and d[1] > .29


@pytest.mark.parametrize("strain_unit", [1., 1e-4])
def test_trust_coordinates_preserve_solution_with_small_physical_strain(strain_unit):
    # The same feasible problem expressed in two control units, with an active
    # geometry row and nonzero initial control. Both have an analytic solution.
    image = {"value": .4, "g": np.array([-.8, -.4])/strain_unit,
             "H": np.eye(2)/strain_unit**2}
    constant = {"value": 1., "g": np.zeros(2), "H": np.zeros((2, 2))}
    geometry = {"value": np.array([.99]), "J": np.array([[1., 0.]])/strain_unit}
    d, info = solve_constrained_response({"mass": constant, "physics": constant, "image": image},
        np.array([.05, -.02])*strain_unit, .3*strain_unit, strain_unit, state_inequality=geometry)
    assert info["success"] and info["geometric_constraints"] > 0
    assert np.allclose(d/strain_unit, [.01, np.sqrt(.3**2-.01**2)], atol=2e-6)


@pytest.mark.parametrize("bad_image_model,reject_intersection", [(False, False), (True, False), (False, True)])
def test_response_control_updates_real_mpm_and_preserves_interior_controls(bad_image_model, reject_intersection, monkeypatch):
    rng = np.random.default_rng(11)
    source = rng.uniform(-.8, .8, (40, 3)).astype(np.float32)
    mask = np.linalg.norm(source, axis=1) > .8
    prm = MPMParams(dx=.75, dt=1/120, nx=16, ny=16, nz=16, grid_min=(-6.,)*3)
    spec = RolloutSpec(source, 1., 800., 400., prm, 3, device="cpu",
                       vol0=compute_rest_volumes(source, 1., prm, "cpu"))
    basis = SurfaceStrainBasis(source, mask)
    target = torch.tensor(source)*torch.tensor([1.1, .9, 1.])
    def residuals(state, control):
        rm = (state[0]-target).flatten()/len(source)**.5
        rp = torch.cat([rm, .03*state[2].flatten()/len(source)**.5,
                        .01*control.flatten()/control.numel()**.5])
        ri = (state[0][mask, :2]-target[mask, :2]).flatten()/int(mask.sum())**.5
        if bad_image_model:
            # Synthetic observation: its central response is zero at the origin,
            # but a mixed finite step changes it. This must not veto a valid
            # physics-only correction; it is deliberately not a render fixture.
            ri = 1+1e5*control[0, mask, 0, 0].square()
        return {"mass": rm, "physics": rp, "image": ri}
    global_calls = []
    if reject_intersection:
        def global_gate(tr):
            # Inject a detector rejection for every nonzero committed candidate.
            # FD probes bypass this gate; zero-control initial/final states pass.
            peak = float(np.abs(tr._dfc(0).numpy()).max())
            global_calls.append(peak)
            return {"valid": peak == 0., "reason": "surface_self_intersection"}
        monkeypatch.setattr("physmorph.pipeline.surface_validity.no_surface_intersections", global_gate)
    result = optimize_response_window(spec, basis, residuals,
        ResponseConfig(iterations=1, fd_strain=.01, radius=.02,
                       objective="physics" if bad_image_model else "image",
                       global_surface_checks=reject_intersection,
                       geometric_response_constraints=not bad_image_model))
    if reject_intersection:
        assert not result["history"][0]["accepted"]
        assert result["history"][0]["candidate_health"]["reason"] == "surface_self_intersection"
        assert torch.count_nonzero(result["control"]) == 0
        assert result["final"] == result["initial"]
        assert global_calls[:2] == [0., 0.] and global_calls[-1] == 0.
        assert any(p > 0 for p in global_calls)
        return
    assert result["history"][0]["accepted"]
    if bad_image_model:
        assert result["history"][0]["response_validation_error"]["image"] > .2
    else:
        assert result["final"]["image"] < result["initial"]["image"]
    assert result["final"]["mass"] <= result["initial"]["mass"]
    assert result["final"]["physics"] <= result["initial"]["physics"]
    assert torch.count_nonzero(result["control"][:, ~mask]) == 0
    displacement = result["trajectory"].x[-1].numpy()-source
    assert np.linalg.norm(displacement[~mask]) > 1e-6
