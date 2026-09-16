"""Material re-coupling of decoupled particles (warp CPU): (1) with every particle coupled
the rollout is bit-identical to the plain one; (2) a decoupled particle is projected back
toward its source neighbours and takes their velocity; (3) the adjoint with re-coupling
matches central finite differences."""
import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree

from physmorph.mpm.function import RolloutSpec, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes

DEV = "cpu"


def _cloud(n=300, seed=5):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (n, 3)).astype(np.float32)


def _prm():
    return MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.0, smoothing=1.0,
                     grid_min=(-6.0, -6.0, -6.0), nx=24, ny=24, nz=24)


def _bonds(x, K=8):
    nbr = cKDTree(x).query(x, k=K + 1, workers=-1)[1][:, 1:].astype(np.int32)
    rest = np.linalg.norm(x[nbr] - x[:, None, :], axis=2).astype(np.float32)
    return nbr, rest


def test_coupled_cloud_is_bit_identical_with_recoupling():
    x0 = _cloud(); prm = _prm(); vol0 = compute_rest_volumes(x0, 1.0, prm, DEV)
    nbr, rest = _bonds(x0)
    C0 = (np.random.default_rng(1).standard_normal((len(x0), 3, 3)) * 0.2).astype(np.float32)
    a = Trajectory(x0, 1.0, 800.0, 400.0, prm, 3, C0=C0, device=DEV, requires_grad=False, vol0=vol0)
    b = Trajectory(x0, 1.0, 800.0, 400.0, prm, 3, C0=C0, device=DEV, requires_grad=False, vol0=vol0, bonds=(nbr, rest))
    a.rollout(); b.rollout()
    assert np.array_equal(a.x[-1].numpy(), b.x[-1].numpy())
    assert np.array_equal(a.v[-1].numpy(), b.v[-1].numpy())
    assert float(b.ncounts[0].numpy().min()) > 1.5              # nobody decoupled


def test_decoupled_particle_is_projected_back_and_takes_the_material_velocity():
    x0 = _cloud(); prm = _prm()
    nbr, rest = _bonds(x0)                                       # bonds from the intact cloud
    x1 = x0.copy(); x1[0] = x0[0] + np.array([3.0, 0.0, 0.0], np.float32)   # 6 dx away, alone
    v0 = np.zeros_like(x0); v0[:, 1] = 0.5                       # the body moves +y; the stray does not
    v0[0] = 0.0
    vol0 = compute_rest_volumes(x0, 1.0, prm, DEV)
    tr = Trajectory(x1, 1.0, 0.0, 0.0, prm, 3, v0=v0, device=DEV, requires_grad=False, vol0=vol0, bonds=(nbr, rest))
    tr.rollout()
    assert float(tr.ncounts[0].numpy()[0]) < 1.5                 # decoupled at step 0
    x_end = tr.x[-1].numpy(); v_end = tr.v[-1].numpy()
    excess0 = np.linalg.norm(x1[nbr[0]] - x1[0], axis=1) - rest[0]
    excess1 = np.linalg.norm(x_end[nbr[0]] - x_end[0], axis=1) - rest[0]
    assert excess1.mean() < 0.999 * excess0.mean()               # moving back (1/T of the excess per step)
    assert abs(v_end[0, 1] - 0.5) < 0.1                          # took the material velocity


@pytest.mark.parametrize("kind", ["x", "vall"])
def test_recoupled_adjoint_matches_finite_difference(kind):
    rng = np.random.default_rng(20260916)
    x0 = rng.uniform(-1.0, 1.0, (40, 3)).astype(np.float32)
    nbr, rest = _bonds(x0)
    x0[0] += np.array([2.5, 0.0, 0.0], np.float32)               # one decoupled particle
    prm = MPMParams(dx=0.75, dt=1.0 / 120.0, drag=0.9, smoothing=0.955,
                    grid_min=(-6.0, -6.0, -6.0), nx=16, ny=16, nz=16)
    vol0 = compute_rest_volumes(x0, 1.0, prm, DEV)
    spec = RolloutSpec(x0=x0, m=1.0, lam=800.0, mu=400.0, prm=prm, T=3, device=DEV, vol0=vol0,
                       bond_nbr=nbr, bond_rest=rest)
    def loss(dfc):
        xT, FT, vT, FgT, V = warp_mpm_ext(dfc, spec)
        return (xT * torch.linspace(0.5, 1.5, 3)).pow(2).sum() * 1e2 if kind == "x" else V.pow(2).sum(2).mean() * 1e2
    torch.manual_seed(2)
    dfc = (torch.randn(3, len(x0), 3, 3) * 2e-2).requires_grad_(True)
    L = loss(dfc)
    (g,) = torch.autograd.grad(L, dfc)
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0
    idx = tuple(int(i) for i in np.unravel_index(int(g.abs().argmax()), g.shape))
    eps = 2e-3
    with torch.no_grad():
        dp, dm = dfc.clone(), dfc.clone(); dp[idx] += eps; dm[idx] -= eps
        fd = float((loss(dp) - loss(dm)) / (2 * eps))
    an = float(g[idx])
    assert abs(fd - an) / max(abs(fd), abs(an), 1e-9) < 0.05, (fd, an)
