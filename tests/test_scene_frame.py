"""D147: each body's frame (physmorph/pipeline/scene/frame.py) — the Buckens frame's invariance, its exact gradient
(no net force or moment) and the polar backward against finite differences, the sphere's A = s I included; the rigid
velocity of a rigid motion."""
import math

import torch

from physmorph.pipeline.scene.frame import body_frame, deformation_velocity, polar_rotation


def _rot(axis, ang):
    a = torch.tensor(axis, dtype=torch.float64)
    a = a / a.norm()
    K = torch.tensor([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _cloud(n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 3, generator=g, dtype=torch.float64)
    return X - X.mean(0)


def test_frame_is_invariant_under_rigid_motion():
    X = _cloud()
    g = torch.Generator().manual_seed(1)
    x = X @ torch.diag(torch.tensor([1.3, 0.8, 1.0], dtype=torch.float64)) + 0.05 * torch.randn(X.shape, generator=g,
                                                                                               dtype=torch.float64)
    Q, t = _rot([0.3, -1.0, 0.4], 2.1), torch.tensor([3.0, -1.0, 7.5], dtype=torch.float64)
    y1, _, R1 = body_frame(x, X)
    y2, c2, R2 = body_frame(x @ Q.T + t, X)
    assert torch.allclose(y1, y2, atol=1e-10)
    assert torch.allclose(R2, Q @ R1, atol=1e-10)
    assert torch.allclose(c2, x.mean(0) @ Q.T + t, atol=1e-10)


def test_a_spinning_sphere_frame_sees_the_spin():
    X = _cloud()
    Q = _rot([0.0, 0.0, 1.0], 0.7)
    _, _, R = body_frame(X @ Q.T, X)
    assert torch.allclose(R, Q, atol=1e-10)


def test_frame_gradient_has_no_net_force_or_moment():
    X = _cloud()
    g = torch.Generator().manual_seed(2)
    x = (X + 0.3 * torch.randn(X.shape, generator=g, dtype=torch.float64)).requires_grad_(True)
    target = torch.randn(X.shape, generator=g, dtype=torch.float64)
    y, c, _ = body_frame(x, X)
    loss = ((y - target) ** 2).sum() + (y[:, 0] ** 3).sum()
    (gx,) = torch.autograd.grad(loss, x)
    r = x.detach() - x.detach().mean(0)
    assert gx.sum(0).abs().max() < 1e-9 * gx.abs().max() * len(x)
    assert torch.linalg.cross(r, gx, dim=1).sum(0).abs().max() < 1e-9 * (r.abs().max() * gx.abs().max() * len(x))


def _fd_check(A):
    A = A.clone().requires_grad_(True)
    W = torch.randn(3, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    (gA,) = torch.autograd.grad((polar_rotation(A) * W).sum(), A)
    D = torch.randn(3, 3, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    h = 1e-6
    fp = (polar_rotation((A + h * D).detach()) * W).sum()
    fm = (polar_rotation((A - h * D).detach()) * W).sum()
    fd = (fp - fm) / (2 * h)
    an = (gA * D).sum()
    assert abs(float(an - fd)) <= 1e-6 * max(1.0, abs(float(fd))), (float(an), float(fd))


def test_polar_backward_matches_finite_differences():
    _fd_check(torch.randn(3, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64) + 2 * torch.eye(3,
              dtype=torch.float64))


def test_polar_backward_at_the_sphere():
    """A = s Q, every singular value equal: the closed form is finite where the SVD's backward is not."""
    _fd_check(2.5 * _rot([1.0, 2.0, -0.5], 0.4))


def test_rigid_motion_has_no_deformation_velocity():
    X = _cloud()
    w, vc = torch.tensor([0.4, -1.2, 0.7], dtype=torch.float64), torch.tensor([1.0, 0.0, -2.0], dtype=torch.float64)
    x = X + torch.tensor([5.0, 1.0, 0.0], dtype=torch.float64)
    v = vc + torch.linalg.cross(w.expand_as(X), X, dim=1)
    assert deformation_velocity(x, v).abs().max() < 1e-12
