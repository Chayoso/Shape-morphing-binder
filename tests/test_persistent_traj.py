"""Persistent trajectories (speed pass 2026-09-16): a buffer set rolled out many times with
different controls must give the rollout a fresh Trajectory gives (the grid accumulators are
re-zeroed per step), on the CPU bit-identically and on CUDA as a captured graph; the material
re-coupling bonds ride along."""
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


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
@pytest.mark.parametrize("layer_mode", ["none", "position", "deformation"])
def test_forward_scratch_reuse_preserves_all_states_and_replays(dev, layer_mode):
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    x, prm, T = _cloud(120), _prm(), 6
    N = len(x)
    rng = np.random.default_rng(19)
    v0 = rng.normal(0., .02, (N, 3)).astype(np.float32)
    C0 = rng.normal(0., .01, (N, 3, 3)).astype(np.float32)
    bonds = _bonds(x, 4)
    layer = None
    if layer_mode != "none":
        normal = x / np.linalg.norm(x, axis=1, keepdims=True)
        layer = (np.ones(N, np.float32), normal, bonds[0],
                 np.full((N, 4), .25, np.float32), .05)
        if layer_mode == "deformation":
            layer += (rng.normal(0., .1, (N, 4, 3)).astype(np.float32), .2)
    dc = torch.zeros(T, N, 3, 3, device=dev)
    u = torch.zeros(N, device=dev)
    kw = dict(device=dev, requires_grad=False, vol0=np.full(N, .02, np.float32),
              dFc=[wp.from_torch(dc[t], dtype=wp.mat33) for t in range(T)],
              layer=layer, layer_u=wp.from_torch(u, dtype=wp.float32),
              bonds=bonds, bond_history=True, control_steps=3,
              v0=v0, C0=C0, track_geom=True)
    ref = Trajectory(x, 1., 800., 400., prm, T, **kw)
    tr = Trajectory(x, 1., 800., 400., prm, T, persistent=True, **kw)
    assert tr.C[0].ptr != tr.C[-1].ptr
    for name in ["P"] + (["xu", "ld"] if layer else []) + (
            ["Fu"] if layer_mode == "deformation" else []):
        assert len({a.ptr for a in getattr(tr, name)}) == 1
        assert len({a.ptr for a in getattr(ref, name)}) == len(getattr(ref, name))
    assert tr.capture() == (dev == "cuda")
    for repeat in range(3):
        dc.copy_(torch.as_tensor(_controls(N, T, repeat) * .01, device=dev))
        u.copy_(torch.as_tensor(rng.normal(0., .002, N).astype(np.float32), device=dev))
        ref.run()
        tr.run()
        for name in ("x", "v", "F", "Fg"):
            for a, b in zip(getattr(ref, name), getattr(tr, name)):
                np.testing.assert_allclose(a.numpy(), b.numpy(),
                    rtol=2e-5 if dev == "cuda" else 0, atol=2e-6 if dev == "cuda" else 0)
        np.testing.assert_allclose(ref.C[-1].numpy(), tr.C[-1].numpy(),
            rtol=2e-5 if dev == "cuda" else 0, atol=2e-6 if dev == "cuda" else 0)
        np.testing.assert_array_equal(tr.C[0].numpy(), C0)
    # G2P deliberately leaves outputs untouched for invalid positions. They
    # must retain this timestep's previous values, not the prior step's scratch.
    for trajectory in (ref, tr):
        trajectory.step(0)
        invalid = trajectory.x[1].numpy().copy()
        invalid[0] = 1e5
        trajectory.x[1].assign(invalid)
        trajectory.step(1)
    for name in ("C", "F", "Fg"):
        np.testing.assert_allclose(getattr(ref, name)[2].numpy(), getattr(tr, name)[2].numpy(),
            rtol=2e-5 if dev == "cuda" else 0, atol=2e-6 if dev == "cuda" else 0)
    # The adjoint still needs distinct intermediates at every timestep.
    adj = Trajectory(x, 1., 800., 400., prm, T, device=dev, persistent=True,
                     requires_grad=True, vol0=kw['vol0'], layer=layer)
    for name in ("C", "Fraw", "P"):
        arrays = getattr(adj, name)
        assert len({a.ptr for a in arrays}) == len(arrays)



def test_shared_grid_rollout_equals_per_step_grid_rollout():
    """A no-grad trajectory shares one grid across steps; a requires_grad one keeps T. Same
    rollout (CPU: bit-identical)."""
    x0, prm, T = _cloud(), _prm(), 5
    N = len(x0)
    vol0 = compute_rest_volumes(x0, 1.0, prm, "cpu")
    dfc = _controls(N, T, 7)
    seq_a = [wp.array(dfc[t], dtype=wp.mat33, device="cpu") for t in range(T)]
    seq_b = [wp.array(dfc[t], dtype=wp.mat33, device="cpu", requires_grad=True) for t in range(T)]
    a = Trajectory(x0, 1.0, 800.0, 400.0, prm, T, dFc=seq_a, device="cpu", requires_grad=False, vol0=vol0)
    b = Trajectory(x0, 1.0, 800.0, 400.0, prm, T, dFc=seq_b, device="cpu", requires_grad=True, vol0=vol0)
    assert a.share_grid and not b.share_grid and a.gm[0] is a.gm[1] and b.gm[0] is not b.gm[1]
    a.rollout(); b.rollout()
    assert np.array_equal(a.x[T].numpy(), b.x[T].numpy())
    assert np.array_equal(a.v[T].numpy(), b.v[T].numpy())
    assert np.array_equal(a.F[T].numpy(), b.F[T].numpy())


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
@pytest.mark.parametrize("bond_history", [False, True])
def test_persistent_adjoint_matches_the_plain_bridge(dev, bond_history):
    """PersistentAdjoint (graphs on CUDA, plain tape on CPU) gives the _WarpMPMExt outputs and
    control gradient; a second backward with another seed on the same forward is consistent."""
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from physmorph.mpm.function import PersistentAdjoint, RolloutSpec, warp_mpm_ext
    x0, prm, T = _cloud(120), _prm(), 3
    N = len(x0)
    vol0 = compute_rest_volumes(x0, 1.0, prm, dev)
    nbr, rest, frag = _bonds(x0)
    spec = RolloutSpec(x0=x0, m=1.0, lam=800.0, mu=400.0, prm=prm, T=T, device=dev, vol0=vol0,
                       bond_nbr=nbr, bond_rest=rest, bond_frag=frag, bond_history=bond_history)
    adj = PersistentAdjoint(spec)
    torch.manual_seed(0)
    dfc = (torch.randn(T, N, 3, 3, device=dev) * 2e-2)
    w = torch.linspace(0.5, 1.5, 3, device=dev)

    def run(fn, d):
        d = d.clone().requires_grad_(True)
        xT, FT, vT, FgT, V = fn(d)
        L1 = (xT * w).pow(2).sum() * 1e2 + FgT.pow(2).sum() * 1e-1
        L2 = V.pow(2).sum() * 1e2 + (vT * w).sum()
        g1 = torch.autograd.grad(L1, d, retain_graph=True)[0]
        g2 = torch.autograd.grad(L2, d)[0]
        return xT.detach(), FT.detach(), V.detach(), g1, g2
    a = run(lambda d: warp_mpm_ext(d, spec), dfc)
    b = run(lambda d: adj.apply(d), dfc)
    tol = 0.0 if dev == "cpu" else 1e-4
    for ua, ub in zip(a, b):
        assert torch.allclose(ua, ub, rtol=tol, atol=tol * max(1.0, float(ua.abs().max()))), float((ua - ub).abs().max())
    assert float(b[3].abs().max()) > 0 and float(b[4].abs().max()) > 0
    # a second forward on the same buffers with another control
    c = run(lambda d: adj.apply(d), dfc * 0.5)
    e = run(lambda d: warp_mpm_ext(d, spec), dfc * 0.5)
    for ua, ub in zip(e, c):
        assert torch.allclose(ua, ub, rtol=tol, atol=tol * max(1.0, float(ua.abs().max())))


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
@pytest.mark.parametrize("polar_adjoint", [False, True])
def test_control_release_matches_split_physics_and_leaf_finite_differences(dev, polar_adjoint):
    """The tail releases both controls, carries all state, and remains in the adjoint."""
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from physmorph.mpm.function import PersistentAdjoint, RolloutSpec, warp_mpm_ext
    x = np.array([[.25, .2, .2], [.9, .2, .2], [1.1, .25, .2]], np.float32)
    nbr = cKDTree(x).query(x, k=3)[1][:, 1:].astype(np.int32)
    rest = np.linalg.norm(x[nbr] - x[:, None], axis=2).astype(np.float32)
    frag = np.zeros(3, np.float32)
    layer = (np.array([1., 0., 0.], np.float32),
             np.tile(np.array([-1., 0., 0.], np.float32), (3, 1)),
             nbr, np.full((3, 2), .5, np.float32), .25)
    prm = MPMParams(dx=.5, dt=1 / 120, drag=.9, grid_min=(-4.,) * 3,
                    nx=16, ny=16, nz=16)
    v = np.array([[.1, 0., 0.], [.7, 0., 0.], [.3, 0., 0.]], np.float32)
    vol = np.ones(3, np.float32)
    spec = RolloutSpec(x, 1., 800., 400., prm, 8, device=dev, vol0=vol, v0=v,
                       layer=layer, bond_nbr=nbr, bond_rest=rest, bond_frag=frag,
                       bond_history=True, control_steps=4, polar_adjoint=polar_adjoint)
    dc = torch.tensor(_controls(3, 4, 42) * .01, device=dev, requires_grad=True)
    u = torch.tensor([.003, 0., 0.], device=dev, requires_grad=True)
    adj = PersistentAdjoint(spec)

    def expand(z):
        return torch.cat((z, torch.zeros_like(z)))

    def loss(out):
        return out[0].double().square().mean() + out[2].double().square().mean()

    got = adj.apply(expand(dc), u)
    plain = warp_mpm_ext(expand(dc), spec, u_t=u)
    for a, b in zip(got, plain):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    grads = torch.autograd.grad(loss(got), (dc, u))
    ref = torch.autograd.grad(loss(plain), (dc, u))
    for a, b in zip(grads, ref):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)
    for i, g in enumerate(grads):
        direction = -g / g.norm()
        values = []
        for sign in (1., -1.):
            leaves = [dc.detach(), u.detach()]
            leaves[i] = leaves[i] + sign * 1e-3 * direction
            values.append(float(loss(adj.apply(expand(leaves[0]), leaves[1]))))
        assert (values[0] - values[1]) / .002 == pytest.approx(
            float((g * direction).sum()), rel=.03, abs=5e-4)

    kw = dict(device=dev, requires_grad=False, vol0=vol,
              bonds=(nbr, rest, frag), layer=layer, bond_history=True, track_geom=True,
              polar_adjoint=polar_adjoint)
    seq = [wp.from_torch(dc.detach()[i], dtype=wp.mat33) for i in range(4)]
    first = Trajectory(x, 1., 800., 400., prm, 4, dFc=seq, v0=v,
                       layer_u=wp.from_torch(u.detach(), dtype=wp.float32), **kw)
    first.run()
    tail = Trajectory(first.x[4].numpy(), 1., 800., 400., prm, 4,
                      F0=first.F[4].numpy(), v0=first.v[4].numpy(),
                      C0=first.C[4].numpy(), Fg0=first.Fg[4].numpy(), **kw)
    tail.run()
    # Reuse the persistent buffers after the perturbed controls/u above.
    got = adj.apply(expand(dc.detach()), u.detach())
    for actual, expected in zip(got[:4], (tail.x[4], tail.F[4], tail.v[4], tail.Fg[4])):
        torch.testing.assert_close(actual.reshape(-1), wp.to_torch(expected).reshape(-1),
                                   rtol=2e-5, atol=2e-6)
    assert got[-1].shape == (8, 3, 3)
