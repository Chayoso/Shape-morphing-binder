"""Coarse control basis (docs/render_controls_physics.md §4): linear, differentiable,
partition of unity, per-particle mode is the identity, time knots interpolate."""
import numpy as np
import pytest
import torch

from physmorph.pipeline.control_basis import ControlBasis


def _cloud(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)


def test_per_particle_mode_is_identity():
    x0 = _cloud()
    b = ControlBasis(x0, T=4, grid=0, tknots=0, device="cpu")
    assert b.per_particle and b.leaf_shape() == (4, len(x0), 3, 3)
    C = torch.randn(*b.leaf_shape())
    assert b.expand(C) is C


def test_grid_partition_of_unity_and_constant_reproduction():
    x0 = _cloud()
    b = ControlBasis(x0, T=5, grid=4, tknots=2, device="cpu")
    assert float(b.w.sum(1).min()) == pytest.approx(1.0, abs=1e-6)
    assert float(b.w.sum(1).max()) == pytest.approx(1.0, abs=1e-6)
    C = torch.zeros(*b.leaf_shape())
    C[:, :, 0, 1] = 0.3                                   # constant node field
    d = b.expand(C)
    assert d.shape == (5, len(x0), 3, 3)
    assert torch.allclose(d[:, :, 0, 1], torch.full((5, len(x0)), 0.3), atol=1e-6)
    assert float(d[:, :, 0, 0].abs().max()) == 0.0


def test_time_knots_interpolate_linearly():
    x0 = _cloud(50)
    b = ControlBasis(x0, T=5, grid=0, tknots=2, device="cpu")
    C = torch.zeros(2, len(x0), 3, 3)
    C[0, :, 0, 0], C[1, :, 0, 0] = 1.0, 3.0
    d = b.expand(C)[:, 0, 0, 0]
    assert torch.allclose(d, torch.tensor([1.0, 1.5, 2.0, 2.5, 3.0]))
    b1 = ControlBasis(x0, T=5, grid=0, tknots=1, device="cpu")
    d1 = b1.expand(torch.full((1, len(x0), 3, 3), 0.7))
    assert torch.allclose(d1, torch.full((5, len(x0), 3, 3), 0.7))


def test_expand_is_linear_and_differentiable_to_nodes():
    x0 = _cloud(200)
    b = ControlBasis(x0, T=3, grid=3, tknots=3, device="cpu")
    C = torch.randn(*b.leaf_shape(), requires_grad=True)
    d = b.expand(C)
    L = (d * torch.randn_like(d)).sum()
    (g,) = torch.autograd.grad(L, C)
    assert torch.isfinite(g).all()
    # nodes that support no particle receive exactly zero gradient
    empty = b.support == 0.0
    if bool(empty.any()):
        assert float(g[:, empty].abs().max()) == 0.0
    # linearity
    C2 = torch.randn(*b.leaf_shape())
    assert torch.allclose(b.expand(C.detach() + C2), b.expand(C.detach()) + b.expand(C2), atol=1e-5)


@pytest.mark.parametrize("grid", [2, 4, 12])
def test_project_recovers_a_basis_field(grid):
    """REFUTE F2 (2026-09-15): the old lumped restriction lost 38-61 % of a basis field
    per application on every non-degenerate grid and the test passed only because
    grid=3 was degenerate (one node). Least-squares CG must reproduce a field IN the
    basis to <2 % and must not contract under repeated application."""
    x0 = _cloud(4000)
    b = ControlBasis(x0, T=4, grid=grid, tknots=2, device="cpu")
    torch.manual_seed(grid)
    C = torch.randn(*b.leaf_shape())
    d = b.expand(C)
    Cr = b.project(d)
    err = float((b.expand(Cr) - d).norm() / d.norm())
    assert err < 0.02, (grid, err)
    d2 = b.expand(b.project(b.expand(Cr)))
    assert float(d2.norm() / d.norm()) > 0.97          # no geometric contraction


def test_small_grids_are_not_degenerate():
    x0 = _cloud(500)
    for g in (2, 3, 4):
        b = ControlBasis(x0, T=3, grid=g, tknots=1, device="cpu")
        assert b.h < 10.0 and int((b.support > 0).sum()) >= g ** 3 // 2
        C = torch.randn(*b.leaf_shape())
        d = b.expand(C)
        assert float(d.std()) > 1e-3                   # spatially varying, not one node


def test_dof_reduction_reported():
    x0 = _cloud(4000)
    b = ControlBasis(x0, T=20, grid=6, tknots=4, device="cpu")
    info = b.describe()
    assert info["n_dof"] == 4 * 216 * 9 and info["n_dof_per_particle"] == 20 * 4000 * 9
    assert info["n_dof"] < info["n_dof_per_particle"] / 50


def test_rejects_bad_requests():
    x0 = _cloud(20)
    with pytest.raises(ValueError):
        ControlBasis(x0, T=3, grid=1, tknots=0, device="cpu")
    with pytest.raises(ValueError):
        ControlBasis(x0, T=3, grid=0, tknots=5, device="cpu")
