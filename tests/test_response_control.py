import numpy as np
import pytest
import torch

from physmorph.mpm.function import RolloutSpec
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import compute_rest_volumes
from physmorph.pipeline.response_control import (
    ResponseConfig, SurfaceStrainBasis, model_value, optimize_response_window,
    quadratic_model, solve_constrained_response,
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


@pytest.mark.parametrize("bad_image_model", [False, True])
def test_response_control_updates_real_mpm_and_preserves_interior_controls(bad_image_model):
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
    result = optimize_response_window(spec, basis, residuals,
        ResponseConfig(iterations=1, fd_strain=.01, radius=.02,
                       objective="physics" if bad_image_model else "image"))
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
