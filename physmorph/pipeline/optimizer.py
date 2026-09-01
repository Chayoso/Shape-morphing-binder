"""optimize_window — one window of multi-leaf trajectory optimisation (docs/pipeline_v2.md §3).

Evolves trajectory_opt.optimize_sequence (the C++ CompGraph parity port) with:
  * leaves = [dFc sequence] + optional [s material field]  (render→material channel),
  * terminal kinetic loss mean|v_T|^2                      (arrive at rest),
  * asymmetric multi-elevation D_render                    (holes/spray see gradient),
  * a box leash relu(|x|−extent)^2                         (gradient exists BEYOND the render
    viewport and the D_vol grid — the far-field term the adversarial round showed D_render
    cannot provide once a particle leaves every view),
  * λ_R estimated ONCE per window from gradient norms (EMA across windows) — within a window
    the line search therefore decreases a single fixed objective; a per-iteration λ made
    "monotone acceptance" meaningless (adversarial finding),
  * acceptance requires a FINITE ROLLOUT STATE (x,F,v), not just finite scalars — the kernels'
    valid_pos guard silently drops NaN particles from the splats, so a NaN rollout can LOWER
    D_vol by deleting mass and would otherwise be accepted (adversarial finding),
  * history reuses the accepted line-search evaluation     (no wasted extra rollout).

Step control is the C++ recipe: persistent Adam moments, backtracking line search that
rejects any non-improving step (restoring leaves AND moments), adaptive initial step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from ..losses.volumetric import d_vol
from ..mpm.constitutive import lame
from ..mpm.function import RolloutSpec, warp_mpm_full
from ..mpm.state import MPMParams
from ..mpm.traj import Trajectory
from .config import PipelineConfig
from .render_loss import LambdaBalancer, d_render


@dataclass
class TargetPack:
    """Precomputed target quantities shared by every window (built once in runner)."""
    grid: torch.Tensor          # D_vol target mass grid
    lgmin: torch.Tensor
    ldx: float
    ldims: tuple
    m: torch.Tensor             # unit masses (N,)
    views: list                 # [(theta, phi)]
    sils: list | None           # target alpha images (None when render channel off)
    extent: float


def _norm(gs) -> float:
    """Joint L2 norm over a list of per-leaf gradients."""
    return float(torch.sqrt(sum(g.pow(2).sum() for g in gs)).item())


def _finite(tensors) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors)


def optimize_window(x0, prm: MPMParams, cfg: PipelineConfig, tgt: TargetPack,
                    balancer: LambdaBalancer, F0=None, Fp=None, v0=None, C0=None,
                    s_init=None, log=print):
    """Optimise dFc[0..T-1] (+ material s) over one horizon. Returns
    (frames, F_seq, end_state, s_out, hist, stats)."""
    dev = cfg.device
    x0 = np.ascontiguousarray(x0, np.float32)
    N, T = x0.shape[0], cfg.T
    lam0, mu0 = lame(cfg.young, cfg.poisson)
    spec = RolloutSpec(x0=x0, m=1.0, lam=lam0, mu=mu0, prm=prm, T=T,
                       F0=F0, Fp=Fp, v0=v0, C0=C0, device=dev)

    dFc = torch.zeros(T, N, 3, 3, device=dev, requires_grad=True)
    leaves = [dFc]
    s = None
    if cfg.opt_material:
        s0 = np.zeros((2, N), np.float32) if s_init is None else np.asarray(s_init, np.float32)
        s = torch.tensor(s0, device=dev, requires_grad=True)
        leaves.append(s)
    mom = [torch.zeros_like(p) for p in leaves]
    vel = [torch.zeros_like(p) for p in leaves]
    lr_scale = [1.0] + ([cfg.mat_lr_scale] if s is not None else [])
    adam_t = 0

    def material():
        if s is None:
            return None, None
        return lam0 * torch.exp(s[0]), mu0 * torch.exp(s[1])

    def losses_of(xT, vT):
        lv = d_vol(xT, tgt.m, tgt.grid, tgt.lgmin, tgt.ldx, tgt.ldims)
        lk = vT.pow(2).sum(1).mean()
        lr = (d_render(xT, tgt.sils, tgt.views, cfg.render_res, tgt.extent,
                       cfg.sil_k, cfg.w_hole, cfg.w_spray) if balancer.active else None)
        return lv, lk, lr

    def terms(dfc):
        """Differentiable path: torch Function + wp.Tape (gradient phase only)."""
        lam_t, mu_t = material()
        xT, FT, vT = warp_mpm_full(dfc, spec, lam_t, mu_t)
        lv, lk, lr = losses_of(xT, vT)
        return (xT, FT, vT), lv, lk, lr

    def eval_terms(dfc):
        """No-grad path for line-search candidates: plain rollout, NO tape, NO adjoint
        buffers — the graph path allocates ~2x memory and tape bookkeeping that a
        candidate evaluation (up to max_ls_iters per iteration) never uses."""
        with torch.no_grad():
            lam_t, mu_t = material()
            lam_e = lam_t.detach().cpu().numpy() if lam_t is not None else lam0
            mu_e = mu_t.detach().cpu().numpy() if mu_t is not None else mu0
            dc = dfc.detach().contiguous()
            seq = [wp.from_torch(dc[t].view(N, 3, 3), dtype=wp.mat33) for t in range(T)]
            tr = Trajectory(x0, 1.0, lam_e, mu_e, prm, T, F0=F0, Fp=Fp, v0=v0, C0=C0,
                            dFc=seq, device=dev, requires_grad=False)
            tr.rollout()
            xT = wp.to_torch(tr.x[T])
            FT = wp.to_torch(tr.F[T]).reshape(N, 9)
            vT = wp.to_torch(tr.v[T])
            lv, lk, lr = losses_of(xT, vT)
        return (xT, FT, vT), lv, lk, lr

    def phys_total(lv, lk, dfc, xT):
        L = lv + cfg.w_kin * lk + cfg.w_ctrl * dfc.pow(2).sum() / (T * N)
        if cfg.w_box > 0:      # far-field leash: differentiable everywhere, zero inside box
            L = L + cfg.w_box * torch.clamp(xT.abs() - tgt.extent, min=0).pow(2).sum(1).mean()
        if s is not None:
            L = L + cfg.w_mat * s.pow(2).mean()
        return L

    def scalars(lv, lk, lr, lam_r, dfc, xT):
        L = float(phys_total(lv, lk, dfc, xT).detach())
        return L if lr is None else L + lam_r * float(lr)

    hist, accepted, rejected = [], 0, 0
    alpha, g0_norm = cfg.alpha, None
    lam_r = (balancer.lam or 0.0) if balancer.active else 0.0
    grad_converged = False

    for it in range(cfg.iters):
        # ---- gradients. λ_R is fixed for the WHOLE window (estimated from the first
        # iteration's per-term norms), so every accepted step decreases one objective. ----
        state, lv, lk, lr = terms(dFc)
        Lp = phys_total(lv, lk, dFc, state[0])
        if balancer.active and it == 0:
            gp = torch.autograd.grad(Lp, leaves, retain_graph=True)
            gr = torch.autograd.grad(lr, leaves)
            lam_r = balancer.update(_norm(gp), _norm(gr))
            g = [a + lam_r * b for a, b in zip(gp, gr)]
        elif balancer.active:
            g = list(torch.autograd.grad(Lp + lam_r * lr, leaves))
        else:
            g = list(torch.autograd.grad(Lp, leaves))
        cur = scalars(lv, lk, lr, lam_r, dFc, state[0])
        if not np.isfinite(cur):
            log(f"[win] iter {it}: non-finite loss, aborting window")
            break
        gn = _norm(g)
        if g0_norm is None:
            g0_norm = max(gn, 1e-12)
        if gn < cfg.gd_tol * g0_norm:
            grad_converged = True
            log(f"[win] converged at iter {it} (||g||={gn:.4g})")
            break

        # ---- adaptive alpha (C++) ----
        a_try = alpha
        if cfg.adaptive_alpha:
            a_try *= max(cfg.min_alpha_scale, min(1.0, cfg.target_norm / max(gn, 1e-6)))

        # ---- backtracking line search over the Adam step ----
        bak = [p.detach().clone() for p in leaves]
        bak_m = [m_.clone() for m_ in mom]
        bak_v = [v_.clone() for v_ in vel]
        step_ok = False
        new = cur
        lv_n = lk_n = lr_n = None
        for _ls in range(cfg.max_ls_iters):
            with torch.no_grad():
                t_ = adam_t + 1
                for p, gi, m_, v_, sc in zip(leaves, g, mom, vel, lr_scale):
                    m_.mul_(cfg.beta1).add_(gi, alpha=1 - cfg.beta1)
                    v_.mul_(cfg.beta2).addcmul_(gi, gi, value=1 - cfg.beta2)
                    mh = m_ / (1 - cfg.beta1 ** t_)
                    vh = v_ / (1 - cfg.beta2 ** t_)
                    p -= (a_try * sc) * mh / (vh.sqrt() + cfg.eps)
                if cfg.dfc_clip > 0:
                    n = dFc.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1)
                    dFc *= (cfg.dfc_clip / n.clamp_min(1e-8)).clamp(max=1.0)
                if s is not None:
                    s.clamp_(-cfg.mat_clamp, cfg.mat_clamp)
            state_n, lv_n, lk_n, lr_n = eval_terms(dFc)
            with torch.no_grad():
                new = scalars(lv_n, lk_n, lr_n, lam_r, dFc, state_n[0])
            # acceptance requires a FINITE STATE: NaN particles vanish from the splats
            # (valid_pos), so a poisoned rollout can show a finite, LOWER loss.
            if np.isfinite(new) and new < cur and _finite(state_n):
                adam_t = t_
                alpha = min(a_try * 1.1, cfg.alpha)          # C++ grows alpha on acceptance
                step_ok = True
                accepted += 1
                break
            with torch.no_grad():                            # reject: restore and halve
                for p, b in zip(leaves, bak):
                    p.copy_(b)
                for m_, b in zip(mom, bak_m):
                    m_.copy_(b)
                for v_, b in zip(vel, bak_v):
                    v_.copy_(b)
            a_try *= 0.5
        if not step_ok:
            rejected += 1
            alpha *= 0.5
            if alpha < 1e-8:
                log("[win] alpha underflow, stopping window")
                break
            continue

        # history from the ACCEPTED evaluation (v1 burned a full rollout re-logging here)
        hist.append({"iter": it, "loss": new, "d_vol": float(lv_n), "kin": float(lk_n),
                     "d_render": float(lr_n) if lr_n is not None else None,
                     "lambda": lam_r if balancer.active else None,
                     "grad_norm": gn, "alpha": a_try,
                     "dfc_absmax": float(dFc.detach().abs().max()),
                     "s_absmax": float(s.detach().abs().max()) if s is not None else None})

    # ---- final rollout: every intermediate state + FULL end state ----
    with torch.no_grad():
        lam_t, mu_t = material()
        lam_np = lam_t.detach().cpu().numpy() if lam_t is not None else lam0
        mu_np = mu_t.detach().cpu().numpy() if mu_t is not None else mu0
        dc = dFc.detach().contiguous()
        seq = [wp.from_torch(dc[t].view(N, 3, 3), dtype=wp.mat33) for t in range(T)]
        tr = Trajectory(x0, 1.0, lam_np, mu_np, prm, T, F0=F0, Fp=Fp, v0=v0, C0=C0,
                        dFc=seq, device=dev, requires_grad=False)
        tr.rollout()
        frames = [tr.x[t].numpy().copy() for t in range(T + 1)]
        F_seq = [tr.F[t].numpy().copy() for t in range(T + 1)]
        end = {"F": tr.F[T].numpy().copy(), "v": tr.v[T].numpy().copy(),
               "C": tr.C[T].numpy().copy()}
    s_out = s.detach().cpu().numpy() if s is not None else None
    stats = {"accepted": accepted, "rejected": rejected, "grad_converged": grad_converged}
    return frames, F_seq, end, s_out, hist, stats
