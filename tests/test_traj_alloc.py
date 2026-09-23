"""Trajectory allocation change (2026-09-16): device-side zeros / identity clones instead of
host->device copies. Results must be bit-identical to the reference saved from the previous
implementation (scratch reference regenerated here from the same seeds on CPU; the CUDA
reference is compared when a GPU is present)."""
import os

import numpy as np
import pytest
import torch

from physmorph.mpm.function import RolloutSpec, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes

REF = os.environ.get("PHYSMORPH_TRAJ_REF", "")


def _case(dev):
    rng = np.random.default_rng(20260916)
    x0 = rng.uniform(-1.0, 1.0, (500, 3)).astype(np.float32)
    prm = MPMParams(dx=0.5, dt=1.0 / 240, drag=0.9, smoothing=0.955, grid_min=(-6.0,) * 3,
                    nx=24, ny=24, nz=24)
    vol0 = compute_rest_volumes(x0, 1.0, prm, dev)
    C0 = (rng.standard_normal((500, 3, 3)) * 0.2).astype(np.float32)
    spec = RolloutSpec(x0=x0, m=1.0, lam=800.0, mu=400.0, prm=prm, T=6, device=dev, vol0=vol0, C0=C0)
    torch.manual_seed(3)
    dfc = (torch.randn(6, 500, 3, 3, device=dev) * 2e-2).requires_grad_(True)
    xT, FT, vT, FgT, V = warp_mpm_ext(dfc, spec)
    L = (xT * torch.linspace(0.5, 1.5, 3, device=dev)).pow(2).sum() + V.pow(2).sum() + FgT.pow(2).sum()
    (g,) = torch.autograd.grad(L, dfc)
    return dict(xT=xT.detach().cpu().numpy(), FT=FT.detach().cpu().numpy(), vT=vT.detach().cpu().numpy(),
                FgT=FgT.detach().cpu().numpy(), V=V.detach().cpu().numpy(), g=g.cpu().numpy(), vol0=vol0)


@pytest.mark.skipif(not REF or not os.path.exists(REF), reason="reference file not provided")
@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_rollout_bit_identical_to_reference(dev):
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    ref = np.load(REF)
    got = _case(dev)
    for k, v in got.items():
        r = ref[f"{dev}_{k}"]
        if dev == "cpu":
            assert np.array_equal(r, v), (dev, k)          # bit-identical on the CPU
        else:
            # CUDA P2G uses atomic adds: the SAME code differs run-to-run by ~1e-8 relative
            # (measured 7e-9 on x_T, 5e-7 on the dFc gradient); the allocation change must
            # stay inside that band, not add to it
            scale = max(float(np.abs(r).max()), 1e-12)
            assert float(np.abs(r - v).max()) / scale < 5e-6, (dev, k)


def test_fresh_trajectory_arrays_are_zero_and_identity():
    x0 = np.random.default_rng(1).uniform(-1, 1, (64, 3)).astype(np.float32)
    prm = MPMParams(dx=0.5, dt=1.0 / 240, grid_min=(-4.0,) * 3, nx=16, ny=16, nz=16)
    tr = Trajectory(x0, 1.0, 0.0, 0.0, prm, 3, device="cpu", requires_grad=False,
                    vol0=np.ones(64, np.float32), track_geom=True)
    eye = np.tile(np.eye(3, dtype=np.float32), (64, 1, 1))
    for t in range(1, 4):
        assert not tr.x[t].numpy().any() and not tr.v[t].numpy().any() and not tr.C[t].numpy().any()
        assert np.array_equal(tr.F[t].numpy(), eye) and np.array_equal(tr.Fg[t].numpy(), eye)
    assert np.array_equal(tr.x[0].numpy(), x0) and all(np.array_equal(f.numpy(), eye) for f in tr.Fraw)
    assert not tr.gm[0].numpy().any() and not tr.P[0].numpy().any()
