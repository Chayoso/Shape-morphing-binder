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
    b = Trajectory(x0, 1.0, 800.0, 400.0, prm, 3, C0=C0, device=DEV, requires_grad=False, vol0=vol0, bonds=(nbr, rest, np.zeros(len(x0), np.float32)))
    a.rollout(); b.rollout()
    assert np.array_equal(a.x[-1].numpy(), b.x[-1].numpy())
    assert np.array_equal(a.v[-1].numpy(), b.v[-1].numpy())


def test_decoupled_particle_is_projected_back_and_takes_the_material_velocity():
    x0 = _cloud(); prm = _prm()
    nbr, rest = _bonds(x0)                                       # bonds from the intact cloud
    x1 = x0.copy(); x1[0] = x0[0] + np.array([3.0, 0.0, 0.0], np.float32)   # 6 dx away, alone
    v0 = np.zeros_like(x0); v0[:, 1] = 0.5                       # the body moves +y; the stray does not
    v0[0] = 0.0
    vol0 = compute_rest_volumes(x0, 1.0, prm, DEV)
    frag = np.zeros(len(x0), np.float32); frag[0] = 1.0             # the runner flags it as a fragment
    tr = Trajectory(x1, 1.0, 0.0, 0.0, prm, 3, v0=v0, device=DEV, requires_grad=False, vol0=vol0, bonds=(nbr, rest, frag))
    tr.rollout()
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
    frag = np.zeros(len(x0), np.float32); frag[0] = 1.0
    spec = RolloutSpec(x0=x0, m=1.0, lam=800.0, mu=400.0, prm=prm, T=3, device=DEV, vol0=vol0,
                       bond_nbr=nbr, bond_rest=rest, bond_frag=frag)
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


def test_fragment_mask_flags_only_broken_off_material():
    from physmorph.pipeline.runner import fragment_mask
    prm = _prm()
    body = _cloud(600)                                      # a blob in [-1,1]^3
    tip = body[:40].copy(); tip[:, 0] += 1.4                # a thin feature sticking out (contiguous cells)
    debris = body[:5].copy(); debris[:, 0] += 5.0           # a small cluster far away (> 2 cells past the tip)
    x = np.concatenate([body, tip, debris]).astype(np.float32)
    frag = fragment_mask(x, prm)
    assert not frag[:600].any() and not frag[600:640].any()   # body and its thin feature: connected
    assert frag[640:].all()                                    # the broken-off cluster


def test_fragment_mask_tolerates_a_one_cell_gap_in_a_thin_feature():
    from physmorph.pipeline.runner import fragment_mask
    prm = _prm()                                            # dx 0.5
    body = _cloud(600)
    gap_tip = body[:40].copy(); gap_tip[:, 0] += 1.7        # starts ~0.7 wu past the body: a one-cell gap
    x = np.concatenate([body, gap_tip]).astype(np.float32)
    frag = fragment_mask(x, prm)
    assert not frag.any()                                    # still one body under stencil connectivity


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_position_control_adjoint_preserves_mid_rollout_bond_switch(dev):
    """A late fracture must not retroactively change earlier P2G adjoints."""
    from dataclasses import replace
    from physmorph.mpm.function import PersistentAdjoint

    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = np.array([[0.25, 0.2, 0.2], [0.9, 0.2, 0.2], [1.1, 0.25, 0.2]], np.float32)
    prm = MPMParams(dx=0.5, dt=1 / 120, drag=0., smoothing=1.,
                    grid_min=(-4., -4., -4.), nx=16, ny=16, nz=16)
    nbr, rest = _bonds(x, K=2)
    layer = (np.array([1., 0., 0.], np.float32),
             np.tile(np.array([-1., 0., 0.], np.float32), (3, 1)),
             nbr, np.full((3, 2), 0.5, np.float32), 0.)
    spec = RolloutSpec(x, 1., 0., 0., prm, 4, device=dev,
                       v0=np.array([[0.1, 0., 0.], [0.7, 0., 0.], [0.3, 0., 0.]], np.float32),
                       vol0=np.ones(3, np.float32), layer=layer,
                       bond_nbr=nbr, bond_rest=rest, bond_frag=np.zeros(3, np.float32),
                       bond_history=True)
    adj = PersistentAdjoint(spec)
    dfc = torch.zeros(4, 3, 3, 3, device=dev)

    def loss(u):
        xt, _, vt, _, _ = adj.apply(dfc, u)
        return xt[0, 0].double() + vt.double().square().sum()

    u = torch.tensor([1., 0., 0.], device=dev, requires_grad=True)
    old = warp_mpm_ext(dfc, replace(spec, bond_history=False), u_t=u.detach())
    new = adj.apply(dfc, u.detach())
    if dev == "cpu":
        assert all(torch.equal(a, b) for a, b in zip(old, new))
    else:
        # CUDA P2G atomics need not accumulate in bit-identical order.
        for a, b in zip(old, new):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)
    grad, = torch.autograd.grad(loss(u), u)
    masks = [f.numpy().copy() for f in adj.traj.frag_history]
    assert [int(f[0]) for f in masks] == [0, 0, 1, 1]
    h = 1e-3
    delta = torch.tensor([h, 0., 0.], device=dev)
    lp = loss(u.detach() + delta)
    assert all(np.array_equal(f.numpy(), m) for f, m in zip(adj.traj.frag_history, masks))
    lm = loss(u.detach() - delta)
    assert all(np.array_equal(f.numpy(), m) for f, m in zip(adj.traj.frag_history, masks))
    fd = (lp - lm) / (2 * h)
    assert float(grad[0]) == pytest.approx(float(fd), rel=0.01, abs=1e-4)


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_fragment_p2g_replays_neighbor_velocity_without_changing_forward(dev):
    import warp as wp
    from physmorph.mpm.kernels import k_p2g

    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    def array(data, dtype, grad=False):
        return wp.array(data, dtype=dtype, device=dev, requires_grad=grad)

    x0 = np.array([[.25, .2, .2], [.9, .2, .2]], np.float32)
    v = array(np.array([[.1, 0., 0.], [.7, 0., 0.]], np.float32), wp.vec3)
    zero = array(np.zeros((2, 3, 3), np.float32), wp.mat33)
    eye = array(np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)), wp.mat33)
    one = array(np.ones(2, np.float32), wp.float32)
    nbr = array(np.array([1, 0], np.int32), wp.int32)
    frag = array(np.array([1., 0.], np.float32), wp.float32)
    q = np.zeros((16 ** 3, 3), np.float32)
    q[:, 0] = -4 + .5 * (np.arange(16 ** 3) // 16 ** 2)

    def run(xx, replay, backward=False):
        x = array(xx, wp.vec3, backward)
        gm = wp.zeros(16 ** 3, dtype=wp.float32, device=dev, requires_grad=backward)
        gv = wp.zeros(16 ** 3, dtype=wp.vec3, device=dev, requires_grad=backward)
        with wp.Tape() as tape:
            wp.launch(k_p2g, dim=2, inputs=[
                x, v, zero, eye, zero, zero, one, one, one, nbr, frag, 1,
                gm, gv, wp.vec3(-4., -4., -4.), .5, 2., 1 / 120, 0.,
                16, 16, 16, replay], device=dev)
        mass, momentum = gm.numpy().copy(), gv.numpy().copy()
        value = np.sum(momentum.astype(np.float64) * q)
        if backward:
            tape.backward(grads={gv: array(q, wp.vec3)})
        return value, mass, momentum, x.grad.numpy().copy() if backward else None

    old, fixed = run(x0, 0, True), run(x0, 1, True)
    assert np.array_equal(old[1], fixed[1]) and np.array_equal(old[2], fixed[2])
    xp, xm = x0.copy(), x0.copy()
    xp[0, 0] += 1e-3
    xm[0, 0] -= 1e-3
    fd = (run(xp, 1)[0] - run(xm, 1)[0]) / 2e-3
    # Linear reproduction: momentum first moment = m * neighbor_velocity * x.
    assert fd == pytest.approx(.7, rel=1e-4)
    assert fixed[3][0, 0] == pytest.approx(fd, rel=1e-4)
    assert old[3][0, 0] == 0.0  # Historical baseline remains reproducible.
