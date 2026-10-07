"""Outer-layer relaxation projection (kernels.k_layer_resid / k_layer_project; docs/surface_gradient.md
§6), warp CPU: (1) the projection is zero on a plane-sampled layer (d - dbar = 0), (2) it relaxes a
single out-of-plane particle toward the plane, (3) dL/ddFc through the extended bridge with the
projection on matches central finite differences (the kernels are on the tape)."""
import numpy as np
import torch

from physmorph.mpm.function import RolloutSpec, warp_mpm_ext
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes
from physmorph.pipeline.window.layer import layer_relax_data as _layer_relax_data

DEV = "cpu"


def layer_relax_data(x, sp, **kw):
    """The pipeline's layer data (torch) as numpy, for the CPU trajectories below."""
    return tuple(t.numpy() for t in _layer_relax_data(torch.as_tensor(x), sp, **kw))


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


def test_a_layer_particle_without_neighbours_is_not_relaxed():
    """A layer particle whose row has no weight (every layer neighbour beyond the Gaussian's reach, or on the
    other side) has no plane to be relaxed onto. Its row is itself, every layer row sums to one, and the
    projection leaves it where it is. Before D39 its row was zero, the kernels read the row's centroid as the
    world's origin and carried the particle to the plane through it: here 20 spacings in one window."""
    x, sp = _slab()
    far = np.array([[0.3, 20 * sp, -0.2]], np.float32)             # one particle 20 spacings above the slab
    x = np.concatenate([x, far])
    p = len(x) - 1
    mask, nrm, nbr, w = layer_relax_data(x, sp, k=8, h_sp=2.0)
    assert mask[p] > 0.5 and nrm[p, 1] > 0.9
    on = mask > 0.5
    assert np.allclose(w[on].sum(1), 1.0, atol=1e-6)
    assert np.all(nbr[p] == p) and w[p, 0] == 1.0 and np.all(w[p, 1:] == 0)
    prm, T = _params(), 30
    vol0 = compute_rest_volumes(x, 1.0, prm, DEV)
    tr = Trajectory(x, 1.0, 0.0, 0.0, prm, T, device=DEV, requires_grad=False, vol0=vol0,
                    layer=(mask, nrm, nbr, w, 1.0 / T))             # no elasticity: the projection alone
    tr.rollout()
    # it moves only with the layer's rigid correction (D98), a small fraction of a spacing
    assert np.linalg.norm(tr.x[T].numpy()[p] - x[p]) < 0.01 * sp


def test_the_relaxation_moves_the_body_neither_along_nor_about_any_axis():
    """D98: over a window the relaxation alone (no elasticity, no velocity) relaxes a rough layer and leaves the
    body's centre of mass where it was and turns it about no axis (its displacements lose their part along the six
    rigid modes of the layer at every step; before, it carried 94 % of the 300k bunny's net drift, D97)."""
    x, sp = _slab(n_side=12, layers=4, seed=3)
    x[:, 1] += np.where(x[:, 1] > x[:, 1].max() - 0.5 * sp, 0.3 * sp * np.sin(9. * x[:, 0]), 0.).astype(np.float32)
    prm, T = _params(), 30
    mask, nrm, nbr, w = layer_relax_data(x, sp, k=8, h_sp=2.0)
    vol0 = compute_rest_volumes(x, 1.0, prm, DEV)
    tr = Trajectory(x, 1.0, 0.0, 0.0, prm, T, device=DEV, requires_grad=False, vol0=vol0,
                    layer=(mask, nrm, nbr, w, 1.0 / T))
    tr.rollout()
    dx = tr.x[T].numpy().astype(np.float64) - x
    gross = np.linalg.norm(dx, axis=1).sum()
    assert gross > 1e-3 * sp * len(x) * 0.01                      # it relaxed something
    r = x - x.mean(0)
    assert np.linalg.norm(dx.sum(0)) < 1e-4 * gross
    assert np.linalg.norm(np.cross(r, dx).sum(0)) < 1e-3 * (np.linalg.norm(r, axis=1) * np.linalg.norm(dx, axis=1)).sum()


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


def test_a_residual_the_reference_has_is_kept():
    """The relaxation's reference (D88): a layer whose rough residual equals the reference everywhere is at rest
    under the projection (the bump stays), and with the reference of a flat layer on a bumped one the bump goes."""
    x, sp = _slab(n_side=12, layers=4)
    prm = _params()
    mask, nrm, nbr, w = layer_relax_data(x, sp, k=8, h_sp=2.0)
    T = 30
    top = np.where((mask > 0.5) & (nrm[:, 1] > 0.8) & (np.abs(x[:, 0]) < 0.3) & (np.abs(x[:, 2]) < 0.3))[0]
    p = top[len(top) // 2]
    xb = x.copy(); xb[p] += 0.5 * sp * nrm[p]
    res = lambda y: ((nrm * (y - (w[..., None] * y[nbr]).sum(1))).sum(1)) * (mask > 0.5)   # noqa: E731
    rough = lambda y: res(y) - (w * res(y)[nbr]).sum(1)                                     # noqa: E731
    vol0 = compute_rest_volumes(xb, 1.0, prm, DEV)

    def end(ref):
        tr = Trajectory(xb, 1.0, 0.0, 0.0, prm, T, device=DEV, requires_grad=False, vol0=vol0,
                        layer=(mask, nrm, nbr, w, 1.0 / T, None, 0.0, None, ref))
        tr.rollout()
        return tr.x[T].numpy()

    kept = end(rough(xb).astype(np.float32))                     # the bumped layer's own residual as the reference
    assert np.abs(kept - xb).max() < 1e-5 * sp                   # nothing to relax: the bump stays
    gone = end(rough(x).astype(np.float32))                      # the unbumped layer's: the bump is the rough part
    assert (xb[p] - gone[p]) @ nrm[p] > 0.25 * sp                # and is taken down, as towards zero


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


def test_the_reference_is_the_residual_of_the_layer_standing_on_the_target():
    """D105: TargetRelief reads the relaxation's own operator on the layer's feet on the target's surface. A flat
    layer under a rippled target surface (wavelength 4 spacings, amplitude 0.1) gets as reference the rough
    residual it would have with every top particle moved onto the ripple; a layer that stands on the target's
    surface points itself gets its own rough residual (nothing to relax)."""
    if not torch.cuda.is_available():
        return
    from physmorph.pipeline.window.layer import TargetRelief
    x, sp = _slab(n_side=16, layers=4)
    xt = torch.as_tensor(x, device="cuda")
    mask, nrm, nbr, w = _layer_relax_data(xt, sp, k=8, h_sp=2.0)
    on = mask > 0.5

    def rough(y):
        res = (nrm * (y - (w[..., None] * y[nbr]).sum(1))).sum(1)
        return res - (w * res[nbr]).sum(1)

    top = on & (nrm[:, 1] > 0.8)
    y0, A, k = float(xt[top, 1].mean()), 0.1 * sp, 2 * np.pi / (4 * sp)
    g = torch.linspace(-3.0, 3.0, 400, device="cuda")
    GX, GZ = torch.meshgrid(g, g, indexing="ij")
    GX, GZ = GX.reshape(-1), GZ.reshape(-1)
    pts = torch.stack((GX, y0 + A * torch.sin(k * GX), GZ), 1)
    nrm_s = torch.nn.functional.normalize(torch.stack((-A * k * torch.cos(k * GX), torch.ones_like(GX),
                                                       torch.zeros_like(GX)), 1), dim=1)
    ref = TargetRelief(pts, nrm_s, sp).at(xt, mask, nrm, nbr, w)
    xb = xt.clone()
    xb[top, 1] = y0 + A * torch.sin(k * xt[top, 0])
    inner = top & (xt[:, 0].abs() < 1.0) & (xt[:, 2].abs() < 1.0)       # 3.6 spacings in from the slab's edges
    want = rough(xb)[inner]
    assert torch.corrcoef(torch.stack((ref[inner], want)))[0, 1] > 0.95
    assert abs(float(ref[inner].norm() / want.norm()) - 1.0) < 0.1
    own = TargetRelief(xt[on], nrm[on], sp).at(xt, mask, nrm, nbr, w)
    assert torch.allclose(own[on], rough(xt)[on], atol=1e-6)


def test_target_surface_normals_on_a_sphere():
    from physmorph.render.surface_recon import target_surface_normals
    rng = np.random.default_rng(0)
    n = 6000
    r = rng.uniform(0, 1, n) ** (1 / 3)
    v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
    x = (v * r[:, None]).astype(np.float32)
    from scipy.spatial import cKDTree
    sp = float(np.median(cKDTree(x).query(x, k=9, workers=-1)[0][:, -1]))
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
        # Keep float32 dynamics; avoid cancellation in the scalar FD reduction.
        return (xT.double() * wvec.double()).sum() + 0.1 * (vT.double() * wvec.double()).sum()

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
