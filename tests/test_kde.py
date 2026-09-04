import numpy as np
import torch

from physmorph.losses.volumetric import d_kde, kde_assign, kde_self_density


def _grid(n=12, sp=0.1):
    g = np.stack(np.meshgrid(*[np.arange(n) * sp] * 3, indexing="ij"), -1).reshape(-1, 3)
    return torch.as_tensor(g.astype(np.float32))


def test_zero_at_coincidence_and_direction():
    tgt = _grid()
    h = 2 * 0.1
    rho = kde_self_density(tgt, h, 32)
    nb = kde_assign(tgt, tgt, 32)
    assert float(d_kde(tgt, tgt, nb[0], nb[1], h, rho)) < 1e-6      # particles == targets
    x = tgt.clone()
    x[0] = x[0] - torch.tensor([0.35, 0.0, 0.0])          # pushed outside along -x
    x.requires_grad_(True)
    nb = kde_assign(x.detach(), tgt, 32)
    L = d_kde(x, tgt, nb[0], nb[1], h, rho)
    (g,) = torch.autograd.grad(L, x)
    assert g[0, 0] < 0                                     # -grad points to +x (toward target)


def test_clump_is_spread():
    tgt = _grid()
    h = 2 * 0.1
    rho = kde_self_density(tgt, h, 32)
    x = tgt.clone()
    i, j = 700, 701
    x[j] = x[i] + 1e-3                                     # clump j onto i
    x.requires_grad_(True)
    nb = kde_assign(x.detach(), tgt, 32)
    L0 = d_kde(x, tgt, nb[0], nb[1], h, rho)
    (g,) = torch.autograd.grad(L0, x)
    with torch.no_grad():
        x2 = x - 0.1 * g / g.norm()
    L1 = d_kde(x2, tgt, nb[0], nb[1], h, rho)
    assert float(L1) < float(L0)
