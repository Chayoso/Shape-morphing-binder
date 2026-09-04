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
    assert float(d_kde(tgt, tgt, nb, h, rho)) < 1e-6           # particles == targets
    x = tgt.clone()
    x[0] = x[0] - torch.tensor([0.35, 0.0, 0.0])              # pushed outside along -x
    x.requires_grad_(True)
    nb = kde_assign(x.detach(), tgt, 32)
    (g,) = torch.autograd.grad(d_kde(x, tgt, nb, h, rho), x)
    assert g[0, 0] < 0                                         # -grad points back (+x)


def test_far_exterior_particle_is_attracted_not_repelled():
    # x3 falsifier: the one-sided form REPELLED far exterior particles. Two-sided:
    # a particle 4 spacings outside must still get a net pull toward the block via the
    # target-side deficit it left behind (its old target point is now empty).
    tgt = _grid()
    h = 2 * 0.1
    rho = kde_self_density(tgt, h, 32)
    x = tgt.clone()
    i = 5                                                      # an edge point
    x[i] = x[i] - torch.tensor([0.4, 0.0, 0.0])                # 4 sp outside along -x
    x.requires_grad_(True)
    nb = kde_assign(x.detach(), tgt, 32)
    (g,) = torch.autograd.grad(d_kde(x, tgt, nb, h, rho), x)
    assert g[i, 0] < 0                                         # -grad -> +x (inward)


def test_clump_is_spread():
    tgt = _grid()
    h = 2 * 0.1
    rho = kde_self_density(tgt, h, 32)
    x = tgt.clone()
    i, j = 700, 701
    x[j] = x[i] + 1e-3
    x.requires_grad_(True)
    nb = kde_assign(x.detach(), tgt, 32)
    L0 = d_kde(x, tgt, nb, h, rho)
    (g,) = torch.autograd.grad(L0, x)
    with torch.no_grad():
        x2 = x - 2e-3 * g / g.norm()
    L1 = d_kde(x2, tgt, nb, h, rho)
    assert float(L1) < float(L0.detach())
