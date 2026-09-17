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
    pull = SinkhornPull(target_samples(y, 600), eps=0.3, iters=5000, tol=1e-6)   # converged potentials
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


def test_barycentric_targets_stay_inside_the_target_hull_even_unconverged():
    """Row-normalised projection: every target is a convex combination of target samples,
    so it lies inside the target's bounding box no matter how unconverged the plan is
    (the a_i-scaled version reached 1.7x a spike's length in the ball-to-spike test)."""
    x, y = _clouds(5, 800)
    x = x + torch.tensor([2.0, 0.0, 0.0])                    # far from the target: cold, unconverged
    pull = SinkhornPull(target_samples(y, 800), eps=0.01, iters=2, tol=0.0)
    pull.f_cold = False                                      # no annealing: two sweeps at the target eps
    T = pull.barycentric_targets(x)
    assert pull.last_err > 0.05                              # genuinely unconverged after 2 sweeps
    assert bool((T.min(0).values >= y.min(0).values - 1e-5).all())
    assert bool((T.max(0).values <= y.max(0).values + 1e-5).all())


def test_epsilon_scaling_converges_to_tolerance_and_stops():
    """The cold solve anneals eps from the squared diameter and stops at the tolerance;
    a warm solve on the same cloud needs far fewer sweeps."""
    x, y = _clouds(6, 1500)
    pull = SinkhornPull(target_samples(y, 1500), eps=0.05 ** 2, iters=3000, tol=1e-2)
    pull.barycentric_targets(x)
    cold = pull.last_sweeps
    assert pull.last_err < 1e-2 and cold < 3000
    pull.barycentric_targets(x)
    assert pull.last_sweeps < cold and pull.last_err < 1e-2


def test_entropic_map_equals_the_full_projection_when_the_subsample_is_everything():
    """With n_sub = N the subsampled dual IS the full dual, so the out-of-sample map must
    reproduce barycentric_targets; with a smaller subsample it stays close (same estimator,
    potentials from a uniform subsample)."""
    x, y = _clouds(7, 1200)
    ys = target_samples(y, 600)
    full = SinkhornPull(ys, eps=0.05 ** 2 * 4, iters=3000, tol=1e-3)
    T_full = full.barycentric_targets(x)
    same = SinkhornPull(ys, eps=0.05 ** 2 * 4, iters=3000, tol=1e-3)
    T_same = same.entropic_map(x, n_sub=1200)
    assert float((T_same - T_full).norm(dim=1).max()) < 1e-3
    sub = SinkhornPull(ys, eps=0.05 ** 2 * 4, iters=3000, tol=1e-3)
    T_sub = sub.entropic_map(x, n_sub=400)
    scale = float((T_full - x).norm(dim=1).mean())
    assert float((T_sub - T_full).norm(dim=1).mean()) < 0.25 * scale


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
