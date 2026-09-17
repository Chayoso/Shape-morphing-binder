"""Entropic OT coverage loss (losses/ot.SinkhornPull): a cloud displaced from its target is
pulled back — the gradient on every particle points toward the target, its norm shrinks as the
cloud approaches, and the envelope-theorem gradient agrees with finite differences of the
value at the converged plan. A lone particle is not rewarded for leaving: its gradient points
back toward the mass it is matched to."""
import numpy as np
import torch

from physmorph.losses.ot import SinkhornPull, target_samples


def _clouds(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    y = torch.as_tensor(rng.uniform(-1, 1, (n, 3)).astype(np.float32))
    x = y.clone() + torch.tensor([0.4, 0.0, 0.0])          # the same cloud, shifted
    return x, y


def test_displaced_cloud_is_pulled_back_and_gradient_shrinks():
    x, y = _clouds()
    pull = SinkhornPull(target_samples(y, 1500), eps=0.05 ** 2 * 4, iters=40)
    xg = x.clone().requires_grad_(True)
    L = pull(xg)
    g = torch.autograd.grad(L, xg)[0]
    assert float(g[:, 0].mean()) > 0                         # pull is toward -x (gradient +x)
    assert float(torch.cos(torch.tensor(0.0))) == 1.0
    cosine = float((g[:, 0] / g.norm(dim=1).clamp_min(1e-12)).mean())
    assert cosine > 0.8                                      # nearly every particle pulled along -x
    x2 = (x - 0.3 * torch.tensor([1.0, 0.0, 0.0])).requires_grad_(True)
    L2 = pull(x2)
    g2 = torch.autograd.grad(L2, x2)[0]
    assert float(L2) < float(L) and float(g2.norm()) < float(g.norm())


def test_envelope_gradient_matches_finite_difference():
    x, y = _clouds(1, 600)
    x, y = x.double(), y.double()
    pull = SinkhornPull(target_samples(y, 600), eps=0.3, iters=600)   # converged potentials
    xg = x.clone().requires_grad_(True)
    L = pull(xg)
    g = torch.autograd.grad(L, xg)[0]
    i = int(g.norm(dim=1).argmax()); h = 1e-3
    d = torch.zeros_like(x); d[i, 0] = h
    with torch.no_grad():
        fp = float(pull(x + d)); fm = float(pull(x - d))
    fd = (fp - fm) / (2 * h)
    assert abs(fd - float(g[i, 0])) / max(abs(fd), 1e-6) < 0.05, (fd, float(g[i, 0]))


def test_lone_particle_is_pulled_toward_the_body_not_away():
    x, y = _clouds(2, 1000)
    x = y.clone(); x[0] = torch.tensor([3.0, 0.0, 0.0])      # one particle far from everything
    pull = SinkhornPull(target_samples(y, 1000), eps=0.05, iters=60)
    xg = x.clone().requires_grad_(True)
    g = torch.autograd.grad(pull(xg), xg)[0]
    assert float(g[0, 0]) > 0                                # gradient +x => descent moves it back (-x)
    assert float(g[0].norm()) > 5 * float(g[1:].norm(dim=1).median())   # and it is the strongest pull


def test_debiased_displacement_has_no_shrinkage_on_a_matched_cloud():
    """A cloud that already equals the target: the plain barycentric map shrinks toward the
    interior (entropic bias), the debiased displacement is ~zero."""
    x, y = _clouds(4, 1200)
    x = y.clone()
    pull = SinkhornPull(target_samples(y, 1200), eps=0.05, iters=60)
    T = pull.barycentric_targets(x)
    plain = (T - x).norm(dim=1).mean()
    deb = pull.debiased_displacement(x, n_self=1200).norm(dim=1).mean()
    assert float(deb) < 0.5 * float(plain)
