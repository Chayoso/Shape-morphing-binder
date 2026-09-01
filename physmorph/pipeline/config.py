"""PipelineConfig — every knob of the v2 blessed path in one dataclass.

docs/pipeline_v2.md §3 defines the loss; §5 the gates the defaults are tuned to pass.
The physics-only baseline arm is lambda_auto=0, opt_material=False — the SAME code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # ---- horizon / outer loop ----
    T: int = 20                     # rollout length = control layers per window (C++ num_timesteps)
    iters: int = 8                  # optimiser iterations per window
    animations: int = 30            # outer commits (C++ num_animations)
    hold_after_converge: bool = True  # pad frames once frozen (smooth gif tails)

    # ---- optimiser (line-searched Adam; C++ step control) ----
    alpha: float = 0.02             # initial step size
    max_ls_iters: int = 10          # backtracking line-search attempts
    adaptive_alpha: bool = True
    target_norm: float = 2500.0
    min_alpha_scale: float = 0.1
    gd_tol: float = 1e-3            # stop window when ||g|| < gd_tol * ||g_0||
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    dfc_clip: float = 0.0           # optional per-particle |dFc| cap (0 = off; C++ has none)

    # ---- loss terms (docs/pipeline_v2.md §3.3) ----
    loss_res: int = 32              # D_vol grid resolution
    w_kin: float = 0.5              # terminal kinetic mean|v_T|^2 (arrive at rest)
    w_ctrl: float = 1e-3            # running control cost sum_t|dFc|^2/(T N) (anti-slam)
    w_box: float = 10.0             # far-field leash relu(|x|-extent)^2: the render views and
                                    # the D_vol grid have NO gradient for far ejecta; this does

    # ---- render channel (§3.4, §3.6) ----
    lambda_auto: float = 0.0        # 0 = physics-only arm; >0 = norm-balanced render weight
    lambda_ema: float = 0.3         # EMA rate for the balanced lambda (anti-oscillation)
    render_views: int = 6           # azimuths per elevation ring
    render_elevs: tuple = (0.0, 0.5, -0.5)   # elevation angles (v1 was equator-only)
    render_res: int = 64
    sil_k: float = 1.5              # alpha saturation 1-exp(-k w)
    w_hole: float = 2.0             # deficit inside target (holes/missing extremities)
    w_spray: float = 1.0            # excess outside target (ejecta pulled back by the objective)

    # ---- v3: optimisation accelerators (docs/method.md §6) ----
    warm_start: bool = False        # init window's dFc from the previous window's solution
    render_gs_iters: int = 0        # >0: Sobolev/grid-GS smoothing of the render pull before
    render_gs_kappa: float = 4.0    #     the adjoint pullback (screened diffusion strength)

    # ---- v3: VBD-MPM quasi-static arm (docs/method.md §7; used by runner_vbd) ----
    vbd_sweeps: int = 60            # max colored sweeps per commit
    vbd_tol: float = 5e-3           # stop when |grad E| < tol * |grad E_0|
    vbd_step: float = 0.9           # damped block step
    vbd_ls: int = 4                 # per-color backtracking halvings

    # ---- material channel (§3.2 ch.2) ----
    opt_material: bool = False      # optimise per-particle log-Lame multipliers s
    mat_lr_scale: float = 0.25      # material step = alpha * this (slower than dFc)
    mat_clamp: float = 1.0          # |s| <= mat_clamp  (e^{±1} = 0.37..2.7x)
    w_mat: float = 1e-2             # ridge toward base material

    # ---- plastic assimilation channel (§3.5; exact elastic-stretch version) ----
    assim: float = 0.5              # eta: F_e -> R_e S_e^{1-eta} per commit; 0 disables
    assim_smin: float = 0.2         # cumulative Fp band; wide, because a saturated Fp stops
    assim_smax: float = 5.0         # tracking the motion and re-arms spring-back

    # ---- convergence freeze ----
    patience: int = 5               # commits without tol improvement before freeze
    tol: float = 0.003              # relative improvement threshold on the tracked loss

    # ---- material base / misc ----
    young: float = 1.4e5
    poisson: float = 0.2
    device: str = "cuda"

    history: list = field(default_factory=list)
