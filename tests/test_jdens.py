import numpy as np
import torch

from physmorph.losses.volumetric import d_jdens, density_at


def _grid(n=16, sp=0.1):
    g = np.stack(np.meshgrid(*[np.arange(n) * sp] * 3, indexing="ij"), -1).reshape(-1, 3)
    return torch.as_tensor(g.astype(np.float32))


def test_jdens_zero_at_rest_and_pushes_mass_to_dilated_region():
    x0 = _grid()
    m = torch.ones(len(x0))
    gmin = torch.tensor([-0.5, -0.5, -0.5]); dx = 0.1; dims = (32, 32, 32)
    rho0 = density_at(x0, m, gmin, dx, dims).detach()
    assert float(d_jdens(x0, m, rho0, gmin, dx, dims)) < 1e-9        # J == 1 at rest
    # dilate the top slab (z > 1.0) by 20%: its density drops (J > 1), the rest is
    # compressed relative; the gradient on a top-slab particle must NOT push it further
    # out, and on a bottom (compressed) particle must push toward the dilated region
    x = x0.clone(); top = x[:, 2] > 1.0
    x[top, 2] = 1.0 + (x[top, 2] - 1.0) * 1.2
    x.requires_grad_(True)
    L = d_jdens(x, m, rho0, gmin, dx, dims)
    assert float(L) > 0
    (g,) = torch.autograd.grad(L, x)
    assert torch.isfinite(g).all()
    # mass flows from the compressed bottom toward the dilated top: average -grad_z
    # over particles just below the interface points +z
    band = (x0[:, 2] > 0.7) & (x0[:, 2] <= 1.0)
    assert float((-g[band, 2]).mean()) > 0
