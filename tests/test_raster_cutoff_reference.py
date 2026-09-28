"""Independent closed-form screen Gaussian checks for the P303 dense oracle."""
from types import SimpleNamespace

import torch

from scripts.probes.raster_cutoff_audit import dense_alpha, TAU


def camera():
    view = torch.eye(4, dtype=torch.float64)
    projection = torch.zeros_like(view)
    projection[0, 0] = projection[1, 1] = projection[3, 2] = 1.
    return SimpleNamespace(viewmatrix=view, projmatrix=projection.T,
                           image_height=33, image_width=33, tanfovx=1., tanfovy=1.)


def test_isotropic_center_matches_closed_form_screen_gaussian():
    x = torch.tensor([[0., 0., 3.]], dtype=torch.float64)
    cov = torch.eye(3, dtype=x.dtype)[None]*.01
    opacity = x.new_tensor([.25])
    alpha, raw, _ = dense_alpha(x, cov, opacity, camera())
    variance = (16.5/3)**2*.01 + .3
    expected = .25*torch.exp(x.new_tensor(-.5/variance))
    torch.testing.assert_close(raw[0, 16, 16], opacity[0])
    torch.testing.assert_close(raw[0, 16, 17], expected)
    assert alpha[0, 0, 0] == 0


def test_shifted_alpha_is_continuous_at_support_boundary():
    x = torch.tensor([[0., 0., 3.]], dtype=torch.float64)
    cov = torch.eye(3, dtype=x.dtype)[None]*.01
    values = []
    for delta in (-1e-7, 0., 1e-7):
        opacity = x.new_tensor([TAU+delta], requires_grad=True)
        alpha, _, _ = dense_alpha(x, cov, opacity, camera(), mode='shifted')
        values.append(float(alpha[0, 16, 16].detach()))
    torch.testing.assert_close(x.new_tensor(values), x.new_tensor([0., 0., 1e-7]))


def test_dense_shifted_scale_adjoint_matches_refreshed_support_difference():
    x = torch.tensor([[.01, -.02, 3.]], dtype=torch.float64)
    cov = torch.eye(3, dtype=x.dtype)[None]*.08
    def objective(scale):
        alpha, _, _ = dense_alpha(x, cov*scale**2, x.new_tensor([.25]), camera(), mode='shifted')
        return alpha.sum()
    scale = x.new_tensor(1., requires_grad=True)
    slope, = torch.autograd.grad(objective(scale), scale)
    fd = (objective(1.+1e-5)-objective(1.-1e-5))/2e-5
    torch.testing.assert_close(slope, fd, rtol=1e-7, atol=1e-9)
