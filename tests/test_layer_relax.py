"""Outer-layer relaxation projection (kernels.k_layer_resid / k_layer_project; docs/surface_gradient.md
§6), warp CPU: (1) the projection is zero on a plane-sampled layer (d - dbar = 0), (2) it relaxes a
single out-of-plane particle toward the plane, (3) dL/ddFc through the extended bridge with the
projection on matches central finite differences (the kernels are on the tape)."""
import numpy as np
import pytest
import torch

from physmorph.mpm.function import RolloutSpec, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes
from physmorph.render.surface_recon import layer_relax_data

DEV = "cpu"


def _slab(n_side=8, layers=4, spacing=0.25, seed=0):
    """A slab of particles: layers x n_side x n_side on a jittered grid; the top face is the
    outer layer under test."""
    rng = np.random.default_rng(seed)
    g = np.arange(n_side) * spacing
    X, Y, Z = np.meshgrid(g, np.arange(layers) * spacing, g, indexing="ij")
    x = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1).astype(np.float32)
    x += rng.uniform(-0.05, 0.05, x.shape).astype(np.float32) * spacing
    x -= x.mean(0)
    return x, spacing


def _params():
    return MPMParams(dx=0.75, dt=1.0 / 120.0, drag=0.0, smoothing=0.955,
                     grid_min=(-6.0, -6.0, -6.0), nx=16, ny=16, nz=16)


def test_layer_data_shapes():
    x, sp = _slab()
    mask, nrm, nbr, w = layer_relax_data(x, sp, k=8, h_sp=2.0)
    assert mask.shape == (len(x),) and nrm.shape == (len(x), 3)
    assert nbr.shape == (len(x), 8) and w.shape == (len(x), 8)
    assert 0.2 < mask.mean() < 0.9                      # a slab: faces are the layer
    assert np.all(w[mask < 0.5] == 0)


def test_projection_relaxes_the_rough_residual_over_one_window():
    x, sp = _slab(n_side=12, layers=4)
    prm = _params()
    mask, nrm, nbr, w = layer_relax_data(x, sp, k=8, h_sp=2.0)
    T = 30
    # a bump: push one top-face layer particle (away from the slab's rim) out along its normal
    # by half a spacing
    top = np.where((mask > 0.5) & (nrm[:, 1] > 0.8) & (np.abs(x[:, 0]) < 0.3) & (np.abs(x[:, 2]) < 0.3))[0]
    p = top[len(top) // 2]
    xb = x.copy(); xb[p] += 0.5 * sp * nrm[p]
    vol0 = compute_rest_volumes(xb, 1.0, prm, DEV)
    tr = Trajectory(xb, 1.0, 0.0, 0.0, prm, T, device=DEV, requires_grad=False, vol0=vol0,
                    layer=(mask, nrm, nbr, w, 1.0 / T))   # no elasticity: the projection alone
    tr.rollout()
    d = np.stack([tr.ld[t].numpy() for t in range(1, T + 1)])          # the kernel's own residual
    # the bump's rough residual decays by (1 - 1/T)^T ~ e^-1 over one window (its neighbours
    # absorb a little of it through dbar, so the ratio sits a little above e^-1)
    ratio = abs(d[-1, p]) / abs(d[0, p])
    assert 0.2 < ratio < 0.6, (d[0, p], d[-1, p])
    # the layer as a whole gets smoother: the RMS rough residual (d - dbar) over the top face drops
    wn = w / (w.sum(1, keepdims=True) + 1e-12)
    rough = lambda dd: dd - (wn * dd[nbr]).sum(1)
    top_all = np.where((mask > 0.5) & (nrm[:, 1] > 0.8))[0]
    r0 = np.sqrt((rough(d[0])[top_all] ** 2).mean()); rT = np.sqrt((rough(d[-1])[top_all] ** 2).mean())
    assert rT < 0.6 * r0, (r0, rT)
    # interior particles do not move (no projection off the layer, no elasticity, no gravity)
    xT = tr.x[T].numpy()
    interior = mask < 0.5
    assert np.linalg.norm(xT[interior] - xb[interior], axis=1).max() < 1e-6


def test_position_channel_gradient_matches_finite_differences():
    """The position-mode control leaf u (docs/surface_gradient.md §7): dL/du through the extended
    bridge vs directional central differences, and u = 0 reproduces the plain rollout."""
    x, sp = _slab(n_side=5, layers=3)
    prm = _params()
    mask, nrm, nbr, w = layer_relax_data(x, sp, k=6, h_sp=2.0)
    T = 3
    vol0 = compute_rest_volumes(x, 1.0, prm, DEV)
    spec = RolloutSpec(x0=x, m=1.0, lam=800.0, mu=400.0, prm=prm, T=T, device=DEV, vol0=vol0,
                       layer=(mask, nrm, nbr, w, 0.0))          # channel only, no relaxation
    torch.manual_seed(2)
    dfc = torch.randn(T, len(x), 3, 3) * 2e-2
    wvec = torch.randn(len(x), 3)
    with torch.no_grad():
        x_plain = warp_mpm_ext(dfc, spec)[0]
        x_zero = warp_mpm_ext(dfc, spec, u_t=torch.zeros(len(x)))[0]
    assert torch.allclose(x_plain, x_zero, atol=1e-6)
    u = (torch.randn(len(x)) * 0.05 * sp).requires_grad_(True)

    def L(uu):
        xT, FT, vT, FgT, V = warp_mpm_ext(dfc, spec, u_t=uu)
        return (xT * wvec).sum()

    g, = torch.autograd.grad(L(u), u)
    assert float(g[mask < 0.5].abs().max()) == 0.0          # interior particles: no channel
    for _ in range(3):
        d = torch.randn_like(u); d = d / d.norm()
        eps = 1e-3 * sp
        with torch.no_grad():
            fd = (L(u.detach() + eps * d) - L(u.detach() - eps * d)) / (2 * eps)
        an = (g * d).sum()
        assert abs(float(fd - an)) <= 8e-2 * max(abs(float(fd)), abs(float(an)), 1e-4), (fd, an)


def test_target_surface_normals_on_a_sphere():
    from physmorph.render.surface_recon import target_surface_normals
    rng = np.random.default_rng(0)
    n = 6000
    r = rng.uniform(0, 1, n) ** (1 / 3)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    x = (v * r[:, None]).astype(np.float32)
    sp = float(np.median(np.sort(np.linalg.norm(x[:, None, :400] - x[None, :400], axis=-1), axis=1)[:, 8]))
    nrm, w = target_surface_normals(x, sp)
    outer = r > 1.0 - sp
    cos = (nrm[outer] * v[outer]).sum(1)
    assert np.mean(cos > 0.8) > 0.8, np.mean(cos > 0.8)      # radial normals on the shell
    assert w[outer].mean() > 0.5 and w[r < 0.5].mean() < 0.05  # surface weights: shell ~1, core ~0


def test_adjoint_matches_finite_differences_with_force():
    x, sp = _slab(n_side=5, layers=3)
    prm = _params()
    mask, nrm, nbr, w = layer_relax_data(x, sp, k=6, h_sp=2.0)
    T = 3
    vol0 = compute_rest_volumes(x, 1.0, prm, DEV)
    spec = RolloutSpec(x0=x, m=1.0, lam=800.0, mu=400.0, prm=prm, T=T, device=DEV, vol0=vol0,
                       layer=(mask, nrm, nbr, w, 1.0 / T))
    torch.manual_seed(1)
    dfc = (torch.randn(T, len(x), 3, 3) * 2e-2).requires_grad_(True)
    wvec = torch.randn(len(x), 3)

    def L(d):
        xT, FT, vT, FgT, V = warp_mpm_ext(d, spec)
        return (xT * wvec).sum() + 0.1 * (vT * wvec).sum()

    loss = L(dfc)
    g, = torch.autograd.grad(loss, dfc)
    # directional central differences (the repo's contract for the bridge: single entries are
    # below float32 resolution of the rollout, directions are not — docs/render_controls_physics.md §4)
    torch.manual_seed(3)
    for _ in range(3):
        u = torch.randn_like(dfc); u = u / u.norm()
        eps = 1e-3
        with torch.no_grad():
            fd = (L(dfc.detach() + eps * u) - L(dfc.detach() - eps * u)) / (2 * eps)
        an = (g * u).sum()
        # 8 %: the float32 rollout's replay noise over three elastic steps (the isolated check of
        # the two kernels under wp.Tape matches to four digits — scratch layer_adj2.py)
        assert abs(float(fd - an)) <= 8e-2 * max(abs(float(fd)), abs(float(an)), 1e-4), (fd, an)
