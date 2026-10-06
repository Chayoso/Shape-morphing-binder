"""The stress control on the grid (window/basis.py, D109): the particles read the node control with the MPM's own
cubic B-spline (it reproduces a linear field exactly, as the transfer does), the adjoint is the transpose, the
mass-weighted P2G keeps what the grid carries and drops what differs below a cell, and the mass metric gives a smooth
gradient its per-particle norm."""
import numpy as np
import torch

from physmorph.pipeline.window.basis import GridBasis


def _cloud(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.uniform(-1.0, 1.0, (n, 3)), dtype=torch.float32)


DX, GMIN, DIMS = 0.25, (-3.0, -3.0, -3.0), (24, 24, 24)


def test_a_node_field_linear_in_space_is_read_exactly():
    x = _cloud()
    b = GridBasis(x, DX, GMIN, DIMS)
    xn = torch.as_tensor(GMIN) + b.node_ijk.float() * DX                     # the nodes' positions
    C = torch.zeros(2, b.M, 3, 3)
    C[0, :, :, 0] = xn                                                          # a control equal to the node position
    C[1] = 1.0                                                                  # and a constant one
    d = b.to_particles(C)
    assert torch.allclose(d[0, :, :, 0], x, atol=1e-5)                          # linear reproduction: the MPM's kernel
    assert torch.allclose(d[1], torch.ones_like(d[1]), atol=1e-5)               # partition of unity


def test_the_adjoint_is_the_transpose():
    x = _cloud(500)
    b = GridBasis(x, DX, GMIN, DIMS)
    C = torch.randn(3, b.M, 3, 3, requires_grad=True)
    G = torch.randn(3, b.N, 3, 3)
    (b.to_particles(C) * G).sum().backward()
    # <W C, G> = <C, W^T G>: compare against the forward map on the gradient's own direction
    with torch.no_grad():
        lhs = (b.to_particles(C.grad) * G).sum()
        rhs = C.grad.square().sum()
    assert torch.allclose(lhs, rhs, rtol=1e-4)


def test_p2g_keeps_a_smooth_control_and_drops_a_sub_cell_one():
    x = _cloud(20000)
    fine = GridBasis(x, 0.1, GMIN, (60, 60, 60))                              # 20 cells across, 2.5 particles a cell
    smooth = torch.zeros(1, fine.N, 3, 3)
    smooth[0, :, 0, 0] = 0.01 * (2.0 + x[:, 0])
    kept = fine.to_particles(fine.to_nodes(smooth))
    assert float((kept - smooth).norm() / smooth.norm()) < 0.05
    b = GridBasis(x, DX, GMIN, DIMS)
    rough = torch.zeros(1, b.N, 3, 3)
    rough[0, :, 0, 0] = 0.01 * torch.sign(torch.randn(b.N))                   # independent per particle
    left = b.to_particles(b.to_nodes(rough))
    assert float(left.norm() / rough.norm()) < 0.25


def test_the_mass_metric_gives_a_smooth_gradient_its_per_particle_norm():
    x = _cloud(20000)
    b = GridBasis(x, DX, GMIN, DIMS)
    C = torch.zeros(1, b.M, 3, 3, requires_grad=True)
    g = torch.zeros(1, b.N, 3, 3)
    g[0, :, 1, 1] = 1.0 + 0.2 * x[:, 1]
    (b.to_particles(C) * g).sum().backward()
    ratio = float(b.dot(C.grad, C.grad).sqrt() / g.norm())
    assert 0.9 < ratio < 1.1, ratio
