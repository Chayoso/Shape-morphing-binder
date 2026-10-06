"""Xu et al. as published, on this simulator (D118): one episode and the optimisation of its control layer.

M. Xu, C. Song, D. Levin, D. Hyde, "A Differentiable Material Point Method Framework for Shape Morphing", IEEE TVCG 2025
(DOI 10.1109/TVCG.2025.3591729, the accepted author's version; earlier: SCA 2024 poster, arXiv 2409.15746 v1). The
numbers of their method are in XuPaper, each with where it comes from: the TVCG text where it gives one, else the C++
working copy (bffb9e2:legacy/DiffMPMLib3D, CompGraph.cpp OptimizeDefGradControlSequence, PointCloud.cpp,
legacy/utils/physics_utils.py). That copy is PhysMorph-GS's: its [FIX]-marked additions (render-gradient injection,
PCGrad, the physics weight, the adaptive initial alpha, Adam moments kept across passes), the out-of-target x5 of its
EndLayerMassLoss and the AMSGrad, step clipping and EMA of PointCloud::Descend_Adam are not Xu et al.'s and are not used
here; the loss is the published one (losses/volumetric.d_vol_xu with PipelineConfig.xu_kw, xu_form "tvcg": Eqs. 7-9).

The simulator is ours (window/setup.Window: the same MLS-MPM kernels, material, dt, grid, sampling and N). It is Xu et
al.'s modified MLS-MPM (arXiv v1 appendix A, cited as Appendix A11 in the TVCG text: the control added to F in the
stress and in the F update, so that it accumulates into F; the F interpolation F <- (1 - gamma) F_new + gamma F of
Section VI in the forward pass and in back-propagation (Section VII); the damped P2G momentum), at our dt of 1/240 s
against their 1/120 s; the paper's durations are kept in time.

An episode (their "simulation network", V-D): `steps` driven steps from the end state of the previous episode, no
released half; one control layer (V-B: control layers every 10 timesteps, networks of 10 timesteps): a per-particle
control deformation gradient F~ (N, 3, 3) at the episode's first paper timestep, held over the `layer` simulator
steps that span it, zero after; the loss read at the episode's end (TVCG Algorithm 1, line 8: "the loss after
simulation from n to N"). Optimisation (Algorithm 1 for that layer, V-C passes): `passes` passes of `iters`
iterations; each pass a fresh Adam (m, v, t = 0, Algorithm 1 lines 2-4; the working copy keeps them across passes
only in a [FIX]) at the original step size, from the control the previous pass left ("using the computed F~ as
initial/improved guesses", V-C); each iteration one gradient and a
line search that halves the step until the loss decreases (Algorithm 2), at most `ls_iters` trials; when none does,
the control stays the last accepted one and the pass ends (the working copy: "Line search failed. Moving to next
control step."). The episode is then kept, whatever its line search did.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import PipelineConfig
from ..target import TargetPack
from .objective import Objective
from .rollout import Commit, commit_rollout, eval_terms, graph_terms
from .setup import StartState, Window


@dataclass(frozen=True)
class XuPaper:
    """The numbers of Xu et al.'s method: quoted from the TVCG text (section, table, figure) or, where it gives none, the
    C++ working copy's (bffb9e2)."""
    dt: float = 1.0 / 120.0      # "Our simulation uses a timestep size of dt = 1/120 seconds" (V-B; Table II caption)
    network: int = 10            # "networks of 10 timesteps each, i.e., 0.083-second segments at 120 FPS" (V-D)
    interval: int = 10           # "we apply control gradients every 10 timesteps" (V-B): one control layer per network
    iters: int = 4               # Table II caption: "performed 4 gradient descent iterations (3 optimization passes)"
    passes: int = 3
    alpha: float = 1e-3          # Fig. 8 (D to Dragon): "beta1 = 0.90; learning rates: Lion/Adam = 1e-3" (the working
    beta1: float = 0.9           #   copy's configs use 0.01); beta2 and epsilon are not in the paper: CompGraph.cpp
    beta2: float = 0.999
    eps: float = 1e-3
    ls_iters: int = 10           # Algorithm 2 halves while L > L0, unbounded; the working copy stops at max_ls_iters 10
    gd_tol: float = 1e-4         # the tolerance kappa is not in the paper: the working copy's, relative to the first norm
    density: float = 75.0        # Table II, Sphere to Bunny, density 75 (kg/m^3 in arXiv v1) on their 32^3 grid of unit
                                 #   cells (the working copy: m_p = point_dx^3 density, grid_dx 1): a full cell holds 75

    def layout(self, dt: float) -> tuple[int, int]:
        """(simulator steps of an episode, simulator steps of its control layer) at the simulator's dt: an episode is
        `network` paper timesteps (10/120 s), the control layer one of them (1/120 s)."""
        if self.interval != self.network:
            raise ValueError("one control layer per network (V-B and V-D: both 10 timesteps)")
        steps = max(1, int(round(self.network * self.dt / dt)))
        return steps, min(steps, max(1, int(round(self.dt / dt))))

    def episodes(self, timesteps: int) -> int:
        """The episodes of a morph of `timesteps` paper timesteps (Table II's "# of Timesteps": Sphere to Bunny 420;
        Table IV: D to Dragon 950)."""
        return max(1, int(round(timesteps / self.network)))


XU = XuPaper()


class EpisodeWindow(Window):
    """A window of Xu et al.'s protocol: `steps` driven steps and no released half; the control leaf (1, N, 3, 3) is the
    one control layer, held over the first `layer` steps; no material bonds (ours, not theirs)."""

    def __init__(self, start: StartState, prm, cfg: PipelineConfig, tgt: TargetPack, vol0, steps: int, layer: int):
        self._steps, self._layer = int(steps), int(layer)
        super().__init__(start, prm, cfg, tgt, vol0, None)

    def layout(self, cfg: PipelineConfig) -> tuple[int, int, int]:
        return 1, self._steps, self._layer

    def expand(self, leaf: torch.Tensor) -> torch.Tensor:
        """(1, N, 3, 3) -> (steps, N, 3, 3): the layer's control on its steps, zero after."""
        k = self._layer
        rest = leaf.new_zeros((self.T - k,) + tuple(leaf.shape[1:]))
        return torch.cat((leaf.expand(k, -1, -1, -1), rest), dim=0)


@dataclass
class AdamState:
    m: torch.Tensor
    v: torch.Tensor
    t: int = 0                   # accepted steps of this pass


def adam_direction(g: torch.Tensor, st: AdamState, P: XuPaper = XU):
    """(direction, m, v) of the next Adam step, not yet committed: the non-stochastic Adam of Algorithm 1 (lines 10-15,
    all particles every iteration) as the working copy feeds it, the control layer's gradient divided by its norm
    (PointCloud::Descend_Adam_Stochastic and Descend_Adam), the bias correction at the step being taken."""
    gh = g / g.norm().clamp_min(1e-12)
    m = P.beta1 * st.m + (1.0 - P.beta1) * gh
    v = P.beta2 * st.v + (1.0 - P.beta2) * gh * gh
    t = st.t + 1
    d = (m / (1.0 - P.beta1 ** t)) / ((v / (1.0 - P.beta2 ** t)).sqrt() + P.eps)
    return d, m, v


def bisection(loss, x: torch.Tensor, d: torch.Tensor, L0: float, alpha: float, ls_iters: int):
    """Algorithm 2 with the working copy's bound: the step x - a d for a = alpha, alpha / 2, ... until the loss is below
    L0 (strictly, as the working copy accepts). Returns (a, L, trials); a is None when no trial decreased the loss."""
    a, L = float(alpha), float("nan")
    for k in range(int(ls_iters)):
        L = float(loss(x - a * d))
        if np.isfinite(L) and L < L0:
            return a, L, k + 1
        a *= 0.5
    return None, L, int(ls_iters)


@dataclass
class EpisodeResult:
    commit: Commit                    # the episode's rollout with its final control (kept whatever happened)
    leaf: torch.Tensor                # that control layer (1, N, 3, 3)
    stats: dict = field(default_factory=dict)


def _finite(e) -> bool:
    """Our loss scatters a NaN particle nowhere (its stencil index is invalid), so a non-finite state would not show
    in the value as it does in the working copy's: the state itself is checked."""
    return all(bool(torch.isfinite(t).all()) for t in (e.xT, e.FT, e.vT))


def optimize_episode(start: StartState, prm, cfg: PipelineConfig, tgt: TargetPack, vol0, P: XuPaper | None = None,
                     log=print) -> EpisodeResult:
    """One episode: Algorithm 1 for its control layer over the passes, then the commit rollout."""
    P = P or XU
    steps, layer = P.layout(prm.dt)
    win = EpisodeWindow(start, prm, cfg, tgt, vol0, steps, layer)
    obj = Objective(win)
    leaf = torch.zeros(1, win.N, 3, 3, device=cfg.device)
    stats = {"steps": steps, "layer": layer, "L0": None, "passes": []}

    def value(c):                                   # the loss at the episode's end for a candidate control (no tape)
        e = eval_terms(win, obj, c, None)
        return obj.scalar(e, 0.0) if _finite(e) else float("inf")

    for _ in range(P.passes):
        st = AdamState(torch.zeros_like(leaf), torch.zeros_like(leaf))
        alpha, g0 = P.alpha, None
        rec = {"accepted": 0, "ls_failed": 0, "trials": 0, "converged": 0, "nonfinite": 0, "grad_norm": [], "alpha": []}
        for _ in range(P.iters):
            x = leaf.detach().clone().requires_grad_(True)
            e = graph_terms(win, obj, x, None)
            L = obj.phys_core(e) + obj.cleanup(e.xT)
            cur = float(L.detach())
            if stats["L0"] is None:
                stats["L0"] = cur                   # the zero control's loss: the episode without control
            if not (np.isfinite(cur) and _finite(e)):
                rec["nonfinite"] = 1
                break
            g, = torch.autograd.grad(L, x)
            gn = float(g.norm())
            g0 = gn if g0 is None else g0
            rec["grad_norm"].append(gn)
            if gn <= 0.0 or gn < P.gd_tol * g0:
                rec["converged"] = 1
                break
            d, m, v = adam_direction(g.detach(), st, P)
            a, new, trials = bisection(value, leaf, d, cur, alpha, P.ls_iters)
            rec["trials"] += trials
            if a is None:                           # the control stays the last accepted one; the pass ends
                rec["ls_failed"] = 1
                break
            leaf = (leaf - a * d).detach()
            st = AdamState(m, v, st.t + 1)
            alpha = a                               # Algorithm 1 line 14: the step size carries, halved only
            rec["accepted"] += 1
            rec["alpha"].append(a)
            rec["loss"] = new
        stats["passes"].append(rec)
    win._adj = None                                 # the tape is done: free it before the commit
    commit = commit_rollout(win, obj, leaf, None, 0.0)
    stats.update(ctrl_absmax=float(leaf.abs().max()), ctrl_rms=float(leaf.square().mean().sqrt()),
                 accepted=sum(r["accepted"] for r in stats["passes"]),
                 ls_failed=sum(r["ls_failed"] for r in stats["passes"]),
                 ls_trials=sum(r["trials"] for r in stats["passes"]))
    return EpisodeResult(commit=commit, leaf=leaf, stats=stats)
