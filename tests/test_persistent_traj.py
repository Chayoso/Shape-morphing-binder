"""Persistent trajectories (speed pass 2026-09-16): a buffer set rolled out many times with
different controls must give the rollout a fresh Trajectory gives (the grid accumulators are
re-zeroed per step), on the CPU bit-identically and on CUDA as a captured graph; the material
re-coupling bonds ride along; the torch assimilation equals the numpy one."""
import numpy as np
import pytest
import torch
import warp as wp
from scipy.spatial import cKDTree

from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes


def _cloud(n=400, seed=3):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (n, 3)).astype(np.float32)


def _prm():
    return MPMParams(dx=0.5, dt=1.0 / 240.0, drag=0.5, smoothing=0.95,
                     grid_min=(-6.0, -6.0, -6.0), nx=24, ny=24, nz=24)


def _controls(N, T, seed):
    return (np.random.default_rng(seed).standard_normal((T, N, 3, 3)) * 0.02).astype(np.float32)


def _fresh(x0, prm, T, dfc, dev, vol0, bonds=None):
    seq = [wp.array(dfc[t], dtype=wp.mat33, device=dev) for t in range(T)]
    tr = Trajectory(x0, 1.0, 800.0, 400.0, prm, T, dFc=seq, device=dev, requires_grad=False,
                    vol0=vol0, bonds=bonds, track_geom=True)
    tr.rollout()
    return tr.x[T].numpy().copy(), tr.v[T].numpy().copy(), tr.F[T].numpy().copy()


def _bonds(x, K=8):
    nbr = cKDTree(x).query(x, k=K + 1, workers=-1)[1][:, 1:].astype(np.int32)
    rest = np.linalg.norm(x[nbr] - x[:, None, :], axis=2).astype(np.float32)
    frag = np.zeros(len(x), np.float32); frag[:3] = 1.0
    return nbr, rest, frag


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_persistent_rollouts_match_fresh_trajectories(dev):
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    x0, prm, T = _cloud(), _prm(), 4
    N = len(x0)
    vol0 = compute_rest_volumes(x0, 1.0, prm, dev)
    bonds = _bonds(x0)
    buf = torch.zeros(T, N, 3, 3, device=dev)
    seq = [wp.from_torch(buf[t], dtype=wp.mat33) for t in range(T)]
    tr = Trajectory(x0, 1.0, 800.0, 400.0, prm, T, dFc=seq, device=dev, requires_grad=False,
                    vol0=vol0, bonds=bonds, track_geom=True, persistent=True)
    captured = tr.capture()
    assert captured == (dev == "cuda")
    for seed in (1, 2, 3):                                  # three controls on ONE buffer set
        dfc = _controls(N, T, seed)
        buf.copy_(torch.as_tensor(dfc, device=dev))
        tr.run()
        if dev == "cuda":
            torch.cuda.synchronize()
        xp, vp, Fp_ = tr.x[T].numpy(), tr.v[T].numpy(), tr.F[T].numpy()
        xf, vf, Ff = _fresh(x0, prm, T, dfc, dev, vol0, bonds)
        if dev == "cpu":
            assert np.array_equal(xp, xf) and np.array_equal(vp, vf) and np.array_equal(Fp_, Ff)
        else:                                               # CUDA atomics: not bit-identical
            assert np.allclose(xp, xf, atol=1e-5) and np.allclose(vp, vf, atol=1e-4)
            assert np.allclose(Fp_, Ff, atol=1e-5)
        assert not np.allclose(xp, x0)                      # the control did move it


def test_torch_assimilation_matches_numpy():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from physmorph.plasticity import assimilation as A
    rng = np.random.default_rng(9)
    N = 25000                                               # above the GPU threshold
    F = (np.eye(3, dtype=np.float32) + 0.3 * rng.standard_normal((N, 3, 3))).astype(np.float32)
    Fp = (np.eye(3, dtype=np.float32) + 0.1 * rng.standard_normal((N, 3, 3))).astype(np.float32)
    # numpy reference: the same call with the GPU branch switched off
    got = A.assimilate_elastic(F, Fp, eta=0.5, smin=0.2, smax=5.0, isochoric=True)
    saved = A._torch_cuda
    try:
        A._torch_cuda = lambda: False
        ref = A.assimilate_elastic(F, Fp, eta=0.5, smin=0.2, smax=5.0, isochoric=True)
    finally:
        A._torch_cuda = saved
    assert np.allclose(got, ref, atol=2e-4), float(np.abs(got - ref).max())
    grow = 1.0 + 0.2 * rng.uniform(0, 1, N).astype(np.float32)
    got_g = A.assimilate_growth(F, Fp, eta=0.5, smin=0.2, smax=5.0, isochoric=True,
                                grow=grow, grow_band=1.5)
    try:
        A._torch_cuda = lambda: False
        ref_g = A.assimilate_growth(F, Fp, eta=0.5, smin=0.2, smax=5.0, isochoric=True,
                                    grow=grow, grow_band=1.5)
    finally:
        A._torch_cuda = saved
    assert np.allclose(got_g, ref_g, atol=2e-4), float(np.abs(got_g - ref_g).max())
