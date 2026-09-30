"""optimize_window — one window of settled-transport optimisation.

Leaves: the driven controls dFc (T, N, 3, 3) and the outer-layer offset u (N,). Each
iteration differentiates the physics and render objectives separately through the same
2T-step adjoint; the render gradient loses any component that opposes the physics
gradient (one-sided PCGrad) and is weighted by lambda, calibrated once so that
lambda |g_render| = lambda_auto |g_physics| and then held; the cleanup gradient joins
after. Step control: persistent Adam moments and a backtracking line search that accepts a
step only when the full objective decreases by the Armijo amount (or the replay-noise
floor) and the whole trajectory stays valid; rejected trials restore leaves and moments.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import warp as wp

from ..config import PipelineConfig
from ..render_loss import LambdaBalancer
from ..target import TargetPack
from .objective import Objective
from .rollout import Commit, Eval, commit_rollout, eval_terms, graph_terms, state_ok
from .setup import StartState, Window
from .telemetry import collect_grad_dump, support_record, work_record, write_grad_dump

_TELE_KEYS = ("render_work", "render_work_x", "render_work_F", "phys_work", "phys_work_x",
              "phys_work_F", "phys_work_v", "step_norm", "render_cos", "phys_cos")
_STAT_KEYS = ("g_cos", "g_raw_cos", "g_share", "g_phys_norm", "g_rend_norm",
              "predicted_decrease") + _TELE_KEYS
# line-search diagnostics: trials, failure reasons, and (cfg.ls_probe) each failed trial split by channel
_LS_KEYS = ("ls_trials", "ls_fail_merit", "ls_fail_state", "ls_probe")


@dataclass
class WindowResult:
    commit: Commit | None             # None: no accepted step (a null window)
    hist: list                        # one record per accepted iteration
    stats: dict = field(default_factory=dict)


def _norm(gs) -> float:
    """Joint L2 norm over a list of per-leaf gradients."""
    return float(torch.sqrt(sum(g.pow(2).sum() for g in gs)))


def _dot(a, b):
    return sum((x * y).sum() for x, y in zip(a, b))


def pcgrad(g_keep, g_strip):
    """Strip from g_strip its component along g_keep when they conflict (joint over leaves)."""
    dot = _dot(g_keep, g_strip)
    if float(dot) >= 0:
        return list(g_strip)
    k2 = sum(x.pow(2).sum() for x in g_keep).clamp_min(1e-30)
    return [b - (dot / k2) * a for a, b in zip(g_keep, g_strip)]


class WindowOptimizer:
    def __init__(self, start: StartState, prm, cfg: PipelineConfig, tgt: TargetPack,
                 balancer: LambdaBalancer, vol0, bonds, alpha_scale=1.0, on_iter=None, log=print):
        self.cfg, self.tgt, self.balancer, self.log, self.on_iter = cfg, tgt, balancer, log, on_iter
        self.win = Window(start, prm, cfg, tgt, vol0, bonds)
        self.obj = Objective(self.win)
        N = self.win.N
        self.dFc = torch.zeros(cfg.T, N, 3, 3, device=cfg.device, requires_grad=True)
        self.u = torch.zeros(N, device=cfg.device, requires_grad=True)
        self.leaves = [self.dFc, self.u]
        self.mom = [torch.zeros_like(p) for p in self.leaves]
        self.vel = [torch.zeros_like(p) for p in self.leaves]
        self.adam_t = 0
        self.alpha_scale = alpha_scale
        self.alpha = cfg.alpha * alpha_scale
        if tgt.settled_step is not None:
            # warm-start the step search, not its acceptance: every trial runs the checks
            self.alpha = min(self.alpha, 1.1 * (tgt.settled_step * alpha_scale))
        if tgt.settled_scale is not None:
            # the calibration is not candidate state: restore it before any comparison
            balancer.lam, balancer.capped = tgt.settled_scale
        # render_weight_scale multiplies lambda wherever it is set: the render-off twin (0)
        # keeps every other term, the checks and the render telemetry
        self.lam_r = (balancer.lam or 0.0) * float(cfg.render_weight_scale)
        self.lam_capped = None
        self.tele, self.dump = {"ls_trials": 0, "ls_fail_merit": 0, "ls_fail_state": 0}, {}
        self.accepted = self.rejected = 0

    def scalar(self, e: Eval) -> float:
        return self.obj.scalar(e.lv, e.lk, e.lr, self.lam_r, e.dfc, e.xT, e.FT, e.lk_var)

    def eval(self) -> Eval:
        return eval_terms(self.win, self.obj, self.dFc, self.u)

    # ---- window start ----
    def warm_start(self, dfc_init):
        """Decayed previous controls, kept only if they give a valid state that beats the
        zero start (dFc is an absolute control: verbatim reuse double-applies it)."""
        e0 = self.eval()
        E0 = self.scalar(e0) if state_ok(e0) else np.inf
        with torch.no_grad():
            self.dFc.copy_(dfc_init * self.cfg.warm_decay)
        ew = self.eval()
        Ew = self.scalar(ew)
        if not (state_ok(ew) and np.isfinite(Ew) and Ew < E0):
            with torch.no_grad():
                self.dFc.zero_()
            for b in self.mom + self.vel:
                b.zero_()
            self.adam_t = 0

    def replay_noise(self) -> float:
        """CUDA atomics make two rollouts of one control differ: the relative difference at
        the start control (10x it floors the commit-rollout tolerance)."""
        ea, eb = self.eval(), self.eval()
        EA, EB = self.scalar(ea), self.scalar(eb)
        if np.isfinite(EA) and np.isfinite(EB):
            return abs(EA - EB) / max(abs(EA), self.win.loss_floor_eff)
        return 0.0

    # ---- one iteration ----
    def gradient(self, e: Eval, it: int):
        """(g, diag): the composite control gradient and, on the first and last iteration,
        the endpoint position-space gradients of both channels (telemetry only)."""
        cfg, obj, leaves = self.cfg, self.obj, self.leaves
        Lp_core = obj.phys_core(e.lv, e.lk, e.dfc, e.xT, e.FT, e.lk_var)
        Ldt = obj.cleanup(e.xT)
        diag = None
        if (self.on_iter is not None or cfg.work_telemetry) and it in (0, cfg.iters - 1):
            gp_x = torch.autograd.grad(Lp_core + Ldt, e.state(), retain_graph=True, allow_unused=True)
            gr_x = torch.autograd.grad(e.lr, (e.xT, e.FT), retain_graph=True, allow_unused=True)
            diag = (gp_x, gr_x)
        gp = torch.autograd.grad(Lp_core, leaves, retain_graph=True)
        gdt = torch.autograd.grad(Ldt, leaves, retain_graph=True)
        if cfg.grad_dump and it == 0:
            self.dump.update(collect_grad_dump(e, Lp_core, gp, leaves, self.u, cfg.w_pbr))
        gr_raw = [r.detach().clone() for r in torch.autograd.grad(e.lr, leaves)]
        gr = pcgrad(gp, gr_raw)
        if it == 0:
            self._calibrate_lambda(gp, gr, gr_raw)
        g = [a + self.lam_r * b for a, b in zip(gp, gr)]
        return [gi + di for gi, di in zip(g, gdt)], diag

    def _calibrate_lambda(self, gp, gr, gr_raw):
        """lambda from the PROJECTED render gradient, once per target resolution; the
        render-influence telemetry of this window."""
        tgt, bal = self.tgt, self.balancer
        if tgt.settled_scale is not None:
            bal.lam, bal.capped = tgt.settled_scale
            lam = bal.lam
        else:
            lam = bal.update(_norm(gp), _norm(gr))
            if np.isfinite(lam) and lam > 0:
                tgt.settled_scale = (float(lam), bool(bal.capped))
        self.lam_r = lam * float(self.cfg.render_weight_scale)
        self.lam_capped = int(bool(bal.capped))
        np_, nr_ = _norm(gp), _norm(gr)
        np_raw, nr_raw = np_, _norm(gr_raw)
        self.tele.update(g_cos=float(_dot(gp, gr)) / max(np_ * nr_, 1e-30),
                         g_raw_cos=float(_dot(gp, gr_raw)) / max(np_raw * nr_raw, 1e-30),
                         g_share=self.lam_r * nr_ / max(np_ + self.lam_r * nr_, 1e-30),
                         g_phys_norm=np_, g_rend_norm=nr_)
        if self.cfg.grad_dump:
            self.dump.update(lam_r=float(self.lam_r), g_share=float(self.tele["g_share"]))

    def line_search(self, g, gn: float, cur: float, e: Eval):
        """Backtracking over the Adam step. Returns the accepted Eval and its step, or None."""
        cfg, win = self.cfg, self.win
        a_try = self.alpha
        if cfg.adaptive_alpha:
            a_try *= max(cfg.min_alpha_scale, min(1.0, win.target_norm_eff / max(gn, 1e-30)))
        bak = [p.detach().clone() for p in self.leaves]
        bak_m, bak_v = [m.clone() for m in self.mom], [v.clone() for v in self.vel]
        new, e_n, required = cur, None, 0.0
        for _ in range(cfg.max_ls_iters):
            t_ = self.adam_t + 1
            with torch.no_grad():
                for p, gi, m_, v_ in zip(self.leaves, g, self.mom, self.vel):
                    m_.mul_(cfg.beta1).add_(gi, alpha=1 - cfg.beta1)
                    v_.mul_(cfg.beta2).addcmul_(gi, gi, value=1 - cfg.beta2)
                    mh = m_ / (1 - cfg.beta1 ** t_)
                    vh = v_ / (1 - cfg.beta2 ** t_)
                    p -= a_try * (mh / (vh.sqrt() + win.eps_eff))
                if cfg.dfc_clip > 0:
                    n = self.dFc.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1)
                    self.dFc *= (cfg.dfc_clip / n.clamp_min(1e-8)).clamp(max=1.0)
                self.u.clamp_(-win.sp0, win.sp0)           # one spacing per window
            e_n = self.eval()
            with torch.no_grad():
                new = self.scalar(e_n)
                pred = -float(sum((gi.detach() * (p - b)).sum() for gi, p, b in zip(g, self.leaves, bak)))
                self.tele["predicted_decrease"] = pred
                # the Armijo slope counts only when the model predicts descent; else the
                # noise floor (a stale moment can point against a fresh gradient)
                noise_floor = cfg.ls_noise_rel * max(abs(cur), 1.0 / self.tgt.unit_ratio)
                required = max(cfg.armijo_c1 * pred, noise_floor) if pred > 0.0 else noise_floor
            merit_ok, st_ok = bool(np.isfinite(new) and new <= cur - required), state_ok(e_n)
            self.tele["ls_trials"] += 1
            if merit_ok and st_ok:
                self.adam_t = t_
                self.alpha = min(a_try * 1.1, cfg.alpha * self.alpha_scale)
                self.accepted += 1
                return e_n, a_try, new, bak
            self.tele["ls_fail_merit"] += int(not merit_ok)
            self.tele["ls_fail_state"] += int(not st_ok)
            if cfg.ls_probe:
                self._probe(bak, cur, e, e_n, a_try, st_ok)
            with torch.no_grad():                             # reject: restore and shrink
                for p, b in zip(self.leaves, bak):
                    p.copy_(b)
                for m_, b in zip(self.mom, bak_m):
                    m_.copy_(b)
                for v_, b in zip(self.vel, bak_v):
                    v_.copy_(b)
            a_try *= 0.5
        self.rejected += 1
        self.log(f"[win] line search exhausted (cur={cur:.6g} last_new={new:.6g} required={required:.3g} "
                 f"||g||={gn:.3g} a_try={a_try:.3g} state_ok={state_ok(e_n)}; last-attempt deltas "
                 f"d_vol={float(e_n.lv - e.lv.detach()):.3g} kin={float(e_n.lk - e.lk.detach()):.3g} "
                 f"render={float(e_n.lr - e.lr.detach()):.3g} lam={self.lam_r:.3g})")
        return None

    def _probe(self, bak, cur, e: Eval, e_n: Eval, a_try, st_ok):
        """Diagnostic (cfg.ls_probe): a failed trial split by channel. The same step is
        evaluated on dFc alone and on u alone; per variant, the objective change relative to
        the current value, the changes of the transport, kinetic and render terms, and the
        state check. The caller restores the backup afterwards."""
        parts_n = self._parts(e_n)                          # before the next rollout rewrites the flags
        with torch.no_grad():
            d_t, u_t = self.dFc.detach().clone(), self.u.detach().clone()
            self.u.copy_(bak[1])
        e_a = self.eval()
        parts_a = self._parts(e_a)
        with torch.no_grad():
            self.dFc.copy_(bak[0])
            self.u.copy_(u_t)
        e_b = self.eval()
        parts_b = self._parts(e_b)
        with torch.no_grad():
            self.dFc.copy_(d_t)
        scale = max(abs(cur), 1e-30)

        def row(ev, ok):
            return [(self.scalar(ev) - cur) / scale, float(ev.lv - e.lv.detach()), float(ev.lk - e.lk.detach()),
                    float(ev.lr - e.lr.detach()), int(ok)]
        self.tele.setdefault("ls_probe", []).append(
            [float(a_try), float((u_t - bak[1]).abs().max())] + row(e_n, st_ok) + row(e_a, state_ok(e_a))
            + row(e_b, state_ok(e_b)) + parts_n + parts_a + parts_b + self._parts(e)[:3])

    def _parts(self, ev: Eval) -> list:
        """The transport term taken apart (diagnostic): the transport without the support bound,
        the support penalty B, its largest per-particle value, and the particles flagged decoupled
        at the last step of the latest eval rollout."""
        tgt, sup = self.tgt, self.tgt.support
        with torch.no_grad():
            xT, vT = ev.xT.detach(), ev.vT.detach()
            saved, tgt.grid_ot.support = tgt.grid_ot.support, None
            try:
                base = float(tgt.grid_ot.state_energy(xT, tgt.m, vT, self.obj.horizon))
            finally:
                tgt.grid_ot.support = saved
            b = sup.penalty_per_point(xT) if sup is not None else xT.new_zeros(1)
            tr = self.win.tr
            nfrag = float(wp.to_torch(tr.frag_step).sum()) if getattr(tr, "bonds", None) else 0.0
        return [base, float(b.mean()), float(b.max()), nfrag]

    def record(self, it, e_n: Eval, new, a_try, gn) -> dict:
        lpbr = float(e_n.lpbr)
        return {"iter": it, "loss": new, "d_vol": float(e_n.lv), "kin": float(e_n.lk),
                "kin_run": float(e_n.lk_run), "kin_var": float(e_n.lk_var),
                "d_sil": float(e_n.d_sil), "d_render": float(e_n.lr) - self.cfg.w_pbr * lpbr,
                "d_pbr": lpbr, "lambda": self.lam_r, "grad_norm": gn, "alpha": a_try,
                "predicted_decrease": self.tele.get("predicted_decrease"),
                **{k: self.tele.get(k) for k in _TELE_KEYS},
                "dfc_absmax": float(e_n.dfc.abs().max())}

    # ---- the window ----
    def run(self, dfc_init=None) -> WindowResult:
        cfg, log = self.cfg, self.log
        if dfc_init is not None and cfg.warm_decay > 0:
            self.warm_start(dfc_init)
        replay_rel = self.replay_noise() if cfg.replay_calibrate else 0.0
        leaf0 = self.dFc.detach().clone() if cfg.grad_dump else None
        hist, grad_converged, ls_exhausted = [], False, False
        g0_norm = L_start = None
        self.tele["null_reason"] = None
        for it in range(cfg.iters):
            e = graph_terms(self.win, self.obj, self.dFc, self.u)
            g, diag = self.gradient(e, it)
            cur = self.scalar(e)
            if not np.isfinite(cur):
                log(f"[win] iter {it}: non-finite loss, aborting window")
                self.tele["null_reason"] = "nonfinite_loss"
                break
            L_start = cur if L_start is None else L_start
            gn = _norm(g)
            g0_norm = max(gn, 1e-12) if g0_norm is None else g0_norm
            if gn < cfg.gd_tol * g0_norm:
                grad_converged = True
                log(f"[win] converged at iter {it} (||g||={gn:.4g})")
                break
            found = self.line_search(g, gn, cur, e)
            if found is None:
                self.tele["null_reason"] = "ls_exhausted" if not hist else None
                # an exhausted search leaves point, moments and gradient unchanged: the next
                # iteration would re-test rejected steps, so the window ends here
                self.alpha *= 0.5
                ls_exhausted = self.alpha >= 1e-8
                break
            e_n, a_try, new, bak = found
            if diag is not None:
                self.tele.update(work_record(diag[0], diag[1], e.state(), e_n.state()))
                self.tele["step_norm"] = float((self.dFc.detach() - bak[0]).norm())
            rec = self.record(it, e_n, new, a_try, gn)
            if self.on_iter is not None:
                self._stream(it, e_n, rec, diag)
            hist.append(rec)
        return self.commit(hist, replay_rel, grad_converged, ls_exhausted, L_start, leaf0)

    def _stream(self, it, e_n, rec, diag):
        host = lambda t: None if t is None else t.detach().cpu().numpy().astype(np.float32)  # noqa: E731
        self.on_iter(it, host(e_n.xT), host(e_n.FT.reshape(-1, 3, 3)),
                     {**rec, **{k: self.tele.get(k) for k in _STAT_KEYS},
                      "_grad_phys": host(diag[0][0]) if diag else None,
                      "_grad_render": host(diag[1][0]) if diag else None})

    def commit(self, hist, replay_rel, grad_converged, ls_exhausted, L_start, leaf0) -> WindowResult:
        cfg, win = self.cfg, self.win
        win._adj = None                     # the tape is done: free it before the runner goes on
        commit = commit_rollout(win, self.obj, self.dFc, self.u, self.lam_r)
        E_accept = hist[-1]["loss"] if hist else None
        replay_tol = (max(cfg.ls_noise_rel, 10.0 * replay_rel)
                      * max(abs(E_accept or 0.0), win.loss_floor_eff))
        replay_bad = E_accept is not None and commit.E_final > E_accept + replay_tol
        if not commit.valid:
            replay_bad, grad_converged = True, False
            self.tele["null_reason"] = "commit_invalid"
            hist, self.accepted = [], 0
        if (not np.isfinite(commit.jt_final) or commit.jt_final <= 1e-4 or replay_bad) and self.accepted > 0:
            self.log(f"[win] commit rollout failed trajectory check (jt={commit.jt_final:.3g}) - "
                     "discarding window (replay/accepted-candidate mismatch)")
            self.tele["null_reason"] = "commit_replay" if replay_bad else "commit_jt"
            hist, self.accepted = [], 0
        selection_merit = None
        if self.accepted > 0:
            # windows are compared with current geometry, not window-frozen cleanup assignments
            with torch.no_grad():
                xf = commit.x[-1]
                selection_merit = (commit.E_final - float(self.obj.cleanup(xf))
                                   + float(self.obj.cleanup(xf, common_geometry=True)))
            if not np.isfinite(selection_merit):
                self.tele["null_reason"] = "merit_nonfinite"
                hist, self.accepted, grad_converged = [], 0, False
        if cfg.grad_dump and self.dump.get("gx_phys") is not None:
            commit.x = [t.clone() for t in commit.x]        # the dump's rollouts reuse the buffers
            commit.F = [t.clone() for t in commit.F]
            write_grad_dump(cfg.grad_dump, self.dump, win, leaf0, self.dFc.detach().clone(),
                            self.u.detach().clone(), win.expand(self.dFc.detach()),
                            commit.x[-1].detach().cpu().numpy())
        stats = {"replay_rel": replay_rel, "accepted": self.accepted, "rejected": self.rejected,
                 "grad_converged": grad_converged, "ls_exhausted": ls_exhausted, "L_start": L_start,
                 "u_gate": self.obj.u_gate_frac, "lambda_capped": self.lam_capped,
                 "dfc": self.dFc.detach()[:cfg.T].clone(),
                 **{k: self.tele.get(k) for k in _STAT_KEYS + _LS_KEYS}}
        stats.update(null_reason=self.tele.get("null_reason") if self.accepted == 0 else None, E_accept=E_accept,
                     commit_E_final=float(commit.E_final), commit_jt=float(commit.jt_final))
        if self.accepted > 0:
            stats["selection_merit"] = selection_merit
            stats.update(support_record(self.tgt, self.obj.horizon, commit.x[-1], commit.end_v))
        elif selection_merit is not None and not np.isfinite(selection_merit):
            stats["invalid_selection"] = True
        self.tgt.settled_step = (hist[-1]["alpha"] / self.alpha_scale
                                 if self.accepted > 0 and self.alpha_scale > 0 else None)
        return WindowResult(commit=commit if self.accepted > 0 else None, hist=hist, stats=stats)


def optimize_window(start: StartState, prm, cfg: PipelineConfig, tgt: TargetPack,
                    balancer: LambdaBalancer, vol0, bonds, dfc_init=None, alpha_scale=1.0,
                    on_iter=None, log=print) -> WindowResult:
    opt = WindowOptimizer(start, prm, cfg, tgt, balancer, vol0, bonds, alpha_scale, on_iter, log)
    return opt.run(dfc_init)
