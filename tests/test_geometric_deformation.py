"""Geometry contract gates. CPU only; CUDA probes run separately on hyde06."""
from dataclasses import replace

import numpy as np
import pytest
import torch
import warp as wp

from physmorph.mpm import kernels as K
from physmorph.mpm.function import RolloutSpec, warp_mpm_full, warp_mpm_geometry
from physmorph.mpm.state import MPMParams
from physmorph.mpm.traj import Trajectory, compute_rest_volumes
from physmorph.pipeline.gauss_loss import gaussian_covariance
from physmorph.render.children import expand_children_torch


def case():
    rng = np.random.default_rng(914)
    x = rng.uniform(-.8, .8, (40, 3)).astype(np.float32)
    prm = MPMParams(dx=.75, dt=1/120, grid_min=(-6.,)*3, nx=16, ny=16, nz=16)
    spec = RolloutSpec(x, 1., 800., 400., prm, 3, device="cpu",
                       vol0=compute_rest_volumes(x, 1., prm, "cpu"))
    c = torch.tensor(rng.normal(0, .007, (3, len(x), 3, 3)).astype(np.float32))
    return spec, c


@pytest.mark.parametrize("mode", ["translate", "affine", "cap"])
def test_geom_is_spatial_derivative_of_actual_advection(mode):
    """Compare to analytic affine map and a numeric derivative of k_g2p/k_update.

    Nonuniform F0 catches multiplication order; active speed cap catches the
    Jacobian of the actual advected velocity instead of the unbounded field.
    """
    n, dx, dt = 12, .5, .03
    gmin = wp.vec3(-3., -3., -3.)
    pos = np.array([[.13, -.23, .34]], np.float32)
    eps = .001
    x = np.concatenate([pos, pos + eps*np.eye(3), pos - eps*np.eye(3)]).astype(np.float32)
    F0 = np.array([[1.2, .2, 0], [0, .8, .1], [.1, 0, 1.1]], np.float32)
    A = np.zeros((3, 3), np.float32) if mode == "translate" else np.array(
        [[.2, .31, -.17], [-.08, -.15, .21], [.12, -.19, .1]], np.float32)
    b = np.array([.8, -.2, .3], np.float32)
    nodes = np.stack(np.meshgrid(*([np.arange(n)*dx-3]*3), indexing="ij"), -1).reshape(-1, 3)
    grid = wp.array((nodes @ A.T + b).astype(np.float32), dtype=wp.vec3, device="cpu")
    xx = wp.array(x, dtype=wp.vec3, device="cpu")
    fm = wp.array(np.tile(F0, (len(x), 1, 1)), dtype=wp.mat33, device="cpu")
    fg = wp.zeros(len(x), dtype=wp.mat33, device="cpu")
    z = wp.zeros(len(x), dtype=wp.mat33, device="cpu")
    eta = wp.zeros(len(x), dtype=float, device="cpu")
    v = wp.zeros(len(x), dtype=wp.vec3, device="cpu")
    C = wp.zeros(len(x), dtype=wp.mat33, device="cpu")
    fr = wp.zeros(len(x), dtype=wp.mat33, device="cpu")
    cap = .4 if mode == "cap" else 0.
    wp.launch(K.k_g2p, len(x), inputs=[xx, v, C, fm, z, fr, grid, eta, gmin,
              dx, 1/dx, dt, n, n, n, cap, 1, 1], device="cpu")
    wp.launch(K.k_geom_transport, len(x), inputs=[xx, grid, fm, fg, gmin, dx,
              1/dx, dt, n, n, n, cap], device="cpu")
    advected = x + dt*v.numpy()
    jac_fd = ((advected[1:4]-advected[4:7])/(2*eps)).T
    assert np.allclose(fg.numpy()[0], jac_fd @ F0, rtol=1e-4, atol=3e-5)
    if mode != "cap":
        assert np.allclose(fg.numpy()[0], (np.eye(3)+dt*A) @ F0, atol=2e-6)


@pytest.mark.parametrize("stop", ["dt_zero", "stiffness_zero"])
def test_control_cannot_edit_appearance_without_motion(stop):
    spec, c = case()
    spec = replace(spec, prm=replace(spec.prm, dt=0.)) if stop == "dt_zero" else replace(spec, lam=0., mu=0.)
    c = c.requires_grad_()
    x, fm, v, fg = warp_mpm_geometry(c, spec)
    offsets = torch.full((len(x), 2, 3), .03)
    means, child_f = expand_children_torch(x, fg, offsets)
    _, cov = gaussian_covariance(child_f, .12)
    g = torch.autograd.grad(means.square().sum() + cov.square().sum(), c)[0]
    assert torch.count_nonzero(g) == 0
    assert torch.equal(x, torch.from_numpy(spec.x0))
    assert torch.equal(fg.reshape(-1, 3, 3), torch.eye(3).repeat(len(x), 1, 1))
    assert not torch.allclose(fm.reshape(-1, 3, 3), torch.eye(3).repeat(len(x), 1, 1))


@pytest.mark.parametrize("endpoint, cap", [(0, False), (3, False), (3, True)])
def test_each_geometry_endpoint_pullback_matches_directional_difference(endpoint, cap):
    spec, c = case()
    if cap:
        # External acceleration keeps speeds above the cap, away from its kink.
        spec = replace(spec, prm=replace(spec.prm, v_max=.15, f_ext=(120., 90., -60.)))
    c.requires_grad_()
    out = warp_mpm_geometry(c, spec)
    torch.manual_seed(23)
    q = torch.randn_like(out[endpoint])
    g = torch.autograd.grad((out[endpoint]*q).sum(), c)[0]
    assert g.norm() > 1e-4 and torch.isfinite(g).all()
    direction = g / g.norm()
    analytic = float((g*direction).sum())
    for eps in (.003, .01):
        with torch.no_grad():
            plus = warp_mpm_geometry(c+eps*direction, spec)[endpoint]
            minus = warp_mpm_geometry(c-eps*direction, spec)[endpoint]
            fd = float(((plus-minus)*q).sum()/(2*eps))
        assert abs(fd-analytic)/max(abs(fd), abs(analytic)) < .015


def test_geometry_does_not_change_constitutive_forward_and_restart_is_cumulative():
    spec, c = case()
    old = warp_mpm_full(c, spec)
    new = warp_mpm_geometry(c, spec)
    for a, b in zip(old, new):
        assert torch.equal(a, b)
    def rollout(x, steps, control, **state):
        tr = Trajectory(x, spec.m, spec.lam, spec.mu, spec.prm, steps,
                        dFc=[wp.from_torch(d.contiguous(), dtype=wp.mat33) for d in control],
                        device="cpu", requires_grad=False, track_geometry=True,
                        vol0=spec.vol0, **state)
        tr.rollout()
        return tr
    first = rollout(spec.x0, 1, c[:1])
    second = rollout(first.x[-1].numpy(), 2, c[1:], F0=first.F[-1].numpy(),
                     v0=first.v[-1].numpy(), C0=first.C[-1].numpy(),
                     F_geom0=first.F_geom[-1].numpy())
    assert np.allclose(second.F_geom[-1].numpy().reshape(-1, 9), new[3].numpy(), atol=1e-7)
    assert np.allclose(second.x[-1].numpy(), new[0].numpy(), atol=1e-7)
    with pytest.raises(ValueError, match="explicit F_geom0"):
        rollout(first.x[-1].numpy(), 2, c[1:], F0=first.F[-1].numpy())


def test_bridge_backward_compatibility_and_repeated_geometry_pullback():
    spec, c = case()
    c.requires_grad_()
    lam = torch.full((len(spec.x0),), spec.lam, requires_grad=True)
    mu = torch.full((len(spec.x0),), spec.mu, requires_grad=True)
    old = warp_mpm_full(c, spec, lam, mu)
    new = warp_mpm_geometry(c, spec, lam, mu)
    old_loss = sum(t.square().mean() for t in old)
    new_loss = sum(t.square().mean() for t in new[:3])
    gold = torch.autograd.grad(old_loss, (c, lam, mu))
    gnew = torch.autograd.grad(new_loss, (c, lam, mu), retain_graph=True)
    for a, b in zip(gold, gnew):
        assert torch.allclose(a, b, rtol=2e-5, atol=1e-7)
    geom_loss = new[3].square().mean()
    ga = torch.autograd.grad(geom_loss, c, retain_graph=True)[0]
    gb = torch.autograd.grad(geom_loss, c, retain_graph=True)[0]
    joint = torch.autograd.grad(new_loss+geom_loss, c)[0]
    assert torch.equal(ga, gb)
    assert torch.allclose(joint, ga+gnew[0], rtol=2e-5, atol=1e-7)
