"""The grid's steps and the angular momentum (D99), warp CPU: APIC's angular momentum (x x v plus the affine part,
Jiang et al. 2015) is conserved by the step without drag, and under the drag it decays by the drag's factor every
step, as the momentum does: the drag scales the particle's whole velocity field. On v alone it took the translation
and left the spin, and turned a body at rest (the 40k bunny's whole change of angular momentum by the grid)."""
import numpy as np
import pytest

from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes


def _spinning_ball(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, 3)).astype(np.float32)
    x = x[np.linalg.norm(x, axis=1) < 1.0]
    x -= x.mean(0)
    w = np.array([0.3, -0.2, 2.0], np.float32)
    v = np.cross(w, x).astype(np.float32)                                   # a rigid rotation
    C = np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]], np.float32)
    return x, v, np.broadcast_to(C, (len(x), 3, 3)).copy()


def _angular_momentum(tr, t, dx):
    x, v, C = (tr.x[t].numpy().astype(np.float64), tr.v[t].numpy().astype(np.float64),
               tr.C[t].numpy().astype(np.float64).reshape(-1, 3, 3))
    skew = np.stack((C[:, 2, 1] - C[:, 1, 2], C[:, 0, 2] - C[:, 2, 0], C[:, 1, 0] - C[:, 0, 1]), 1)
    return np.cross(x - x.mean(0), v).sum(0) + dx * dx / 3.0 * skew.sum(0)


@pytest.mark.parametrize("drag", [0.0, 0.9])
def test_the_drag_scales_the_angular_momentum_as_the_momentum(drag):
    x, v, C = _spinning_ball()
    prm = MPMParams(dx=0.5, dt=1.0 / 240.0, drag=drag, smoothing=1.0, grid_min=(-4.0, -4.0, -4.0), nx=16, ny=16, nz=16)
    T = 30
    tr = Trajectory(x, 1.0, 0.0, 0.0, prm, T, v0=v, C0=C, device="cpu", requires_grad=False,
                    vol0=compute_rest_volumes(x, 1.0, prm, "cpu"))                # no elasticity
    tr.rollout()
    L0, LT = _angular_momentum(tr, 0, prm.dx), _angular_momentum(tr, T, prm.dx)
    expected = (1.0 - prm.dt * drag) ** T * L0
    assert np.linalg.norm(LT - expected) < 1e-4 * np.linalg.norm(L0), (L0, LT, expected)
