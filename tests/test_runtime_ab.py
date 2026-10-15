"""The runtime phase's result-changing A/Bs (docs/experiments.md D142, D143), both behind switches that are off by
default (tests/test_frozen_recipe.py asserts the defaults).

D142 (--ot_warm_trials): a line-search trial's transport solves start from the window's latest solved potentials at the
blur's temperature; a warm solve that does not converge is solved again from zero duals as without a start; the commit
reads the accepted candidate's potentials (no sweep).
D143 (--cleanup_in_phys): the cleanup's gradient joins the physics channel: PCGrad protects, and the render weight
reads, the physics core and the cleanup together; two adjoint sweeps an iteration.
"""
from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from physmorph.losses.grid_ot import GridSinkhornLoss


def _measure(n, seed, dev):
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(n, generator=gen) + 0.05).to(dev)


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_a_warm_solve_converges_from_a_nearby_measures_potentials(dev):
    """(D142) The solve from a nearby measure's potentials ends at the tolerance at the blur, its potentials and value
    within the tolerance's reach of the cold solve's, in fewer blocks (CUDA); without a start the solve is today's."""
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from physmorph.prof import STATE, take
    dims = (9, 8, 7)
    n = int(np.prod(dims))
    tgt = _measure(n, 1, dev)
    loss = GridSinkhornLoss(tgt, torch.zeros(3, device=dev), 0.25, dims, eps=0.04, iters=1600, tol=1e-3,
                            cuda_blocks=dev == "cuda")
    a0 = _measure(n, 2, dev)
    a0 = a0 / a0.sum()
    a1 = a0 * (1.0 + 0.02 * (_measure(n, 3, dev) - 0.5))
    a1 = a1 / a1.sum()
    f0, g0 = loss.solve(a0, loss.b)
    STATE["on"] = True
    take()
    cold = loss.solve(a1, loss.b)
    warm = loss.solve(a1, loss.b, start=(f0, g0))
    counts = take()["n"]
    STATE["on"] = False
    for c, w in zip(cold, warm):
        assert float((c - w).abs().max()) < 0.05 * float(c.abs().max())
    for f, g in (cold, warm):                   # both meet the marginal tolerance at the blur
        fn = loss.transform(g, loss.b.log(), loss.eps)
        assert float((torch.exp(a1.log() + (f - fn) / loss.eps) - a1).abs().sum()) < 2e-3
    if dev == "cuda":
        assert counts.get("ot_warm_blocks", 0) < counts.get("ot_blocks", 0)
    again = loss.solve(a1, loss.b)
    assert all(torch.equal(c, d) for c, d in zip(cold, again))


@pytest.mark.parametrize("dev", ["cpu", "cuda"])
def test_a_warm_solve_that_cannot_converge_falls_back_to_the_cold_one(dev):
    """(D142) GridSinkhornLoss._solve_from: a start the sweeps cannot bring to the tolerance (here NaN potentials) is
    solved again from zero duals, the cold solve's result bit for bit."""
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from physmorph.losses.grid_ot import _GridTransportNotConverged
    dims = (6, 5, 4)
    n = int(np.prod(dims))
    loss = GridSinkhornLoss(_measure(n, 5, dev), torch.zeros(3, device=dev), 0.25, dims, eps=0.04, iters=1600,
                            tol=1e-3, cuda_blocks=dev == "cuda")
    a = _measure(n, 6, dev)
    a = a / a.sum()
    cold = loss.solve(a, loss.b)
    bad = (torch.full_like(a, float("nan")), torch.full_like(a, float("nan")))
    with pytest.raises(_GridTransportNotConverged):
        loss.solve(a, loss.b, start=bad)
    got = loss._solve_from(a, loss.b, bad)
    assert all(torch.equal(c, d) for c, d in zip(cold, got))


def test_the_warm_flag_starts_the_next_call_from_the_latest_solved_one():
    """(D142) The flag is the caller's for one call: the call solves from the latest solved potentials, then the flag
    is down; off, every call solves from zero duals."""
    dims = (6, 5, 4)
    n = int(np.prod(dims))
    loss = GridSinkhornLoss(_measure(n, 7, "cpu"), torch.zeros(3), 0.25, dims, eps=0.04, iters=1600, tol=1e-3)
    starts = []
    solve = loss.solve

    def spy(a, b, start=None):
        starts.append(start is not None)
        return solve(a, b, start)

    loss.solve = spy
    m0, m1 = _measure(n, 8, "cpu"), _measure(n, 9, "cpu")
    loss(m0)
    loss.warm = True
    loss(m1)
    assert loss.warm is False
    loss(m0)
    assert starts == [False, False, True, True, False, False]


def _stub_optimizer(cleanup_in_phys, scale=1.0):
    from physmorph.pipeline import PipelineConfig
    from physmorph.pipeline.render_loss import LambdaBalancer
    from physmorph.pipeline.window.solve import WindowOptimizer
    cfg0 = PipelineConfig()
    opt = object.__new__(WindowOptimizer)
    opt.cfg = types.SimpleNamespace(cleanup_in_phys=cleanup_in_phys, work_telemetry=False, grad_dump="", iters=8,
                                    render_weight_scale=scale)
    opt.on_iter = None
    opt._zd = None          # B13's one-off force-units measurement: none in a dFc run
    opt.tele = {}
    opt.balancer = LambdaBalancer(cfg0.lambda_auto, cfg0.lambda_ema, None, cap_rel=20.0)
    gen = torch.Generator().manual_seed(4)
    opt.dFc = torch.randn(2, 5, 3, 3, generator=gen, dtype=torch.float64).requires_grad_(True)
    opt.u = torch.randn(5, generator=gen, dtype=torch.float64).requires_grad_(True)
    opt.leaves = [opt.dFc, opt.u]
    W = [torch.randn(2, 5, 3, 3, generator=gen, dtype=torch.float64) for _ in range(3)]
    V = [torch.randn(5, generator=gen, dtype=torch.float64) for _ in range(3)]

    def losses():
        d, u = opt.dFc, opt.u
        Lp = ((W[0] * d).sum() - 0.3) ** 2 + (V[0] * u).sum() ** 2
        Lc = ((W[1] * d).square().sum()) * 0.1 + (V[1] * u.sin()).sum()
        Lr = ((W[2] * d).sum() + (V[2] * u).sum()) ** 2
        return Lp, Lc, Lr

    opt.obj = types.SimpleNamespace(phys_core=lambda e: e.Lp, cleanup=lambda xT: xT)
    return opt, losses


@pytest.mark.parametrize("on", [False, True])
def test_the_gradient_is_composed_by_its_definition(on):
    """(D143) WindowOptimizer.gradient at the window's first iteration (the weight calibrated): off, ∇Lp + λ pc(∇Lp,
    ∇Lr) + ∇Lc with λ from |∇Lp|; on, ∇(Lp + Lc) + λ pc(∇(Lp + Lc), ∇Lr) with λ from |∇(Lp + Lc)|."""
    from physmorph.pipeline.render_loss import LambdaBalancer
    from physmorph.pipeline.window.solve import pcgrad, _norm
    opt, losses = _stub_optimizer(on)
    Lp, Lc, Lr = losses()
    e = types.SimpleNamespace(Lp=Lp, xT=Lc, lr=Lr)
    got, _ = opt.gradient(e, 0)
    Lp, Lc, Lr = losses()
    gp = torch.autograd.grad(Lp, opt.leaves, retain_graph=True)
    gc = torch.autograd.grad(Lc, opt.leaves, retain_graph=True)
    gr = torch.autograd.grad(Lr, opt.leaves)
    keep = [a + b for a, b in zip(gp, gc)] if on else list(gp)
    pc = pcgrad(keep, [r.clone() for r in gr])
    bal = LambdaBalancer(opt.balancer.alpha_lam, opt.balancer.ema, None, cap_rel=20.0)
    lam = bal.update(_norm(keep), _norm(pc))
    want = [k + lam * p + (0.0 if on else c) for k, p, c in zip(keep, pc, gc)]
    assert abs(opt.lam_r - lam) <= 1e-12 * abs(lam)
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)


def test_the_twin_takes_the_same_step_either_way():
    """(D143) With the render weight at zero (the physics-only twin) the step is ∇Lp + ∇Lc either way, to the rounding
    of one sum."""
    steps = []
    for on in (False, True):
        opt, losses = _stub_optimizer(on, scale=0.0)
        Lp, Lc, Lr = losses()
        got, _ = opt.gradient(types.SimpleNamespace(Lp=Lp, xT=Lc, lr=Lr), 0)
        steps.append(got)
    for a, b in zip(*steps):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)


def test_the_commit_reads_the_accepted_candidates_potentials(monkeypatch):
    """(D142) In a short run with the switch, a window's commit after an accepted candidate solves no transport and
    its energy is computed on the candidate's potentials; trials solve warm."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    import physmorph.pipeline.window.solve as S
    from physmorph.mpm.state import MPMParams
    from physmorph.pipeline import PipelineConfig, run_pipeline
    rng = np.random.default_rng(11)
    src = rng.uniform(-1.5, 1.5, (300, 3)).astype(np.float32)
    tgt = (rng.uniform(-1.5, 1.5, (300, 3)) * np.array([1.3, 0.8, 1.0])).astype(np.float32)
    prm = MPMParams(dx=1.0, nx=32, ny=32, nz=32)
    cfg = PipelineConfig(T=4, iters=2, animations=3, loss_res=12, render_views=2, render_elevs=(0.0, 0.5),
                         render_res=24, dt_res=32, patience=3, render_exterior=True, ot_warm_trials=True)
    n = {"solves": 0, "warm": 0}
    rows = []
    solve, commit_rollout = GridSinkhornLoss.solve, S.commit_rollout

    def counting(self, a, b, start=None):
        n["solves"] += 1
        n["warm"] += int(start is not None)
        return solve(self, a, b, start)

    def commit(win, obj, leaf, u, lam_r):
        ot = obj.tgt.grid_ot
        reuse, before = bool(ot.repeat), n["solves"]
        out = commit_rollout(win, obj, leaf, u, lam_r)
        rows.append((reuse, n["solves"] - before, out.E_final))
        return out

    monkeypatch.setattr(GridSinkhornLoss, "solve", counting)
    monkeypatch.setattr(S, "commit_rollout", commit)
    run_pipeline(src, tgt, prm, cfg, log=lambda *_: None)
    assert any(r for r, _, _ in rows), rows
    for reuse, solved, E in rows:
        assert np.isfinite(E)
        if reuse:
            assert solved == 0
    assert n["warm"] > 0
