"""Extended autograd bridge (warp CPU): geometric F_g and per-step velocities.

Contract (docs/render_controls_physics.md §3): F_g is transported by the velocity
gradient ONLY — no control addition, no smoothing — so a render covariance built on it
cannot change without material motion; and every v_t is exposed so a RUNNING kinetic
term is differentiable. Gradients through both new outputs are checked against
central finite differences on a small deterministic case.
"""
import numpy as np
import pytest
import torch

from physmorph.mpm.function import RolloutSpec, warp_mpm_ext, warp_mpm_full
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes

DEV = "cpu"


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(20260914)
    x0 = rng.uniform(-1.0, 1.0, (40, 3)).astype(np.float32)
    prm = MPMParams(dx=0.75, dt=1.0 / 120.0, drag=0.9, smoothing=0.955,
                    grid_min=(-6.0, -6.0, -6.0), nx=16, ny=16, nz=16)
    vol0 = compute_rest_volumes(x0, 1.0, prm, DEV)
    spec = RolloutSpec(x0=x0, m=1.0, lam=800.0, mu=400.0, prm=prm, T=3, device=DEV,
                       vol0=vol0)
    return x0, spec


def test_ext_matches_full_on_shared_outputs(case):
    x0, spec = case
    torch.manual_seed(0)
    dfc = torch.randn(3, len(x0), 3, 3) * 2e-2
    with torch.no_grad():
        xa, Fa, va = warp_mpm_full(dfc, spec)
        xb, Fb, vb, Fg, V = warp_mpm_ext(dfc, spec)
    assert torch.allclose(xa, xb, atol=1e-7) and torch.allclose(Fa, Fb, atol=1e-7)
    assert torch.allclose(va, vb, atol=1e-7)
    assert V.shape == (3, len(x0), 3) and torch.allclose(V[-1], vb, atol=1e-7)


def test_geometric_F_ignores_direct_control_route(case):
    """At dt=0 nothing moves: the stored F still absorbs (1-s)*dFc, Fg must stay I."""
    x0, spec0 = case
    prm = MPMParams(**{**spec0.prm.__dict__, "dt": 0.0})
    spec = RolloutSpec(x0=x0, m=1.0, lam=800.0, mu=400.0, prm=prm, T=2, device=DEV,
                       vol0=spec0.vol0)
    dfc = torch.zeros(2, len(x0), 3, 3)
    dfc[:, :, 0, 0] = 0.1
    with torch.no_grad():
        xT, FT, _, FgT, _ = warp_mpm_ext(dfc, spec)
    eye = torch.eye(3).reshape(1, 9).expand(len(x0), 9)
    assert torch.allclose(xT, torch.as_tensor(x0), atol=0.0)
    assert float((FT - eye).abs().max()) > 1e-3          # the direct control route
    assert torch.allclose(FgT, eye, atol=1e-7)          # ...is absent from Fg


def test_geometric_F_tracks_motion_without_smoothing(case):
    """Fg_{t+1} = (I + dt C_{t+1}) Fg_t exactly (recomputed from the trajectory C)."""
    x0, spec = case
    torch.manual_seed(1)
    dfc = torch.randn(3, len(x0), 3, 3) * 3e-2
    import warp as wp
    seq = [wp.from_torch(dfc[t].contiguous(), dtype=wp.mat33) for t in range(3)]
    tr = Trajectory(x0, 1.0, 800.0, 400.0, spec.prm, 3, dFc=seq, device=DEV,
                    requires_grad=False, vol0=spec.vol0, track_geom=True)
    tr.rollout()
    Fg = np.tile(np.eye(3, dtype=np.float32), (len(x0), 1, 1))
    for t in range(3):
        C = tr.C[t + 1].numpy()
        Fg = (np.eye(3, dtype=np.float32)[None] + spec.prm.dt * C) @ Fg
    assert np.allclose(tr.Fg[3].numpy(), Fg, atol=1e-6)
    assert not np.allclose(tr.F[3].numpy(), Fg, atol=1e-4)   # smoothed F differs


def _loss(dfc, spec, kind):
    xT, FT, vT, FgT, V = warp_mpm_ext(dfc, spec)
    if kind == "geom":
        return (FgT * torch.linspace(0.5, 1.5, 9)).pow(2).sum() * 1e2
    return V.pow(2).sum(2).mean() * 1e2                 # running kinetic


@pytest.mark.parametrize("kind", ["geom", "vall"])
def test_new_output_gradients_match_finite_difference(case, kind):
    x0, spec = case
    torch.manual_seed(2)
    dfc = (torch.randn(3, len(x0), 3, 3) * 2e-2).requires_grad_(True)
    L = _loss(dfc, spec, kind)
    (g,) = torch.autograd.grad(L, dfc)
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0
    idx = tuple(int(i) for i in np.unravel_index(int(g.abs().argmax()), g.shape))
    eps = 2e-3
    with torch.no_grad():
        dp, dm = dfc.clone(), dfc.clone()
        dp[idx] += eps
        dm[idx] -= eps
        fd = float((_loss(dp, spec, kind) - _loss(dm, spec, kind)) / (2 * eps))
    an = float(g[idx])
    assert abs(fd - an) / max(abs(fd), abs(an), 1e-9) < 0.05, (fd, an)
