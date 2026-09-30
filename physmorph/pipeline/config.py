"""PipelineConfig — the constants of the settled-transport pipeline in one dataclass.

The formulation is fixed (README.md): each window runs T driven and T released steps and is
scored at the released end; the physics objective is the debiased grid Sinkhorn divergence
to the fixed target plus the residual drift and the transport-bounded local support; the
render objective is the multi-view silhouette plus the matched shading term, weighted by a
lambda calibrated once and held; the controls are a per-particle stress increment dFc and
the normal offset u of the outer layer. The fields below are its numbers. Weights are in
the legacy cell-sum unit and converted by the unit ratio measured at the source.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    # ---- windows and stopping ----
    T: int = 20                     # driven steps per window (and as many released steps)
    iters: int = 8                  # optimiser iterations per window
    animations: int = 300           # window budget
    patience: int = 5               # windows without a merit improvement before stopping
    reject_stop: int = 3            # consecutive rejected windows that stop the run
    tol: float = 0.003              # relative merit improvement that counts as progress
    anneal_stale: float = 0.7       # step scale x this after a non-improving window
    best_truncate: bool = True      # deliver the trajectory up to its best-merit window
    hold_after_converge: bool = True  # append one held frame when the run stops

    # ---- line-searched Adam ----
    alpha: float = 0.02
    max_ls_iters: int = 10
    adaptive_alpha: bool = True
    target_norm: float = 2500.0     # legacy-unit gradient norm of the adaptive step
    min_alpha_scale: float = 0.1
    gd_tol: float = 1e-3            # the window stops once ||g|| < gd_tol ||g_0||
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-3               # legacy-unit Adam epsilon
    armijo_c1: float = 1e-4
    ls_noise_rel: float = 1e-7      # improvements below this relative size are noise
    replay_calibrate: bool = True   # measure the rollout replay noise at every window start
    dfc_clip: float = 0.02          # per-particle, per-step |dFc| cap
    warm_decay: float = 0.5         # warm start = previous window's control x this

    # ---- material and discretisation ----
    young: float = 1.4e5
    poisson: float = 0.2
    mass_ref_n: int = 40000         # dynamics mass per particle = mass_ref_n / N
    loss_res: int = 64              # loss grid resolution (the prepare stage sets it from dx)
    unit_ref_res: int = 64          # the legacy grid on which the unit ratio is measured
    assim: float = 0.5              # fraction of the elastic stretch made plastic per window
    assim_smin: float = 0.2         # singular-value band of the plastic deformation
    assim_smax: float = 5.0

    # ---- physics objective (legacy units) ----
    w_kin: float = 5.0              # end kinetic energy
    w_kin_var: float = 200.0        # velocity variance (driven) and all motion (released)
    w_ctrl: float = 1e-3            # control magnitude
    w_creg: float = 100.0           # control smoothness over the source kNN graph
    creg_k: int = 8
    w_box: float = 10.0             # far-field leash beyond the target extent
    w_jvol: float = 50.0            # volume prior (J - 1) log J on the stored F
    ot_iters: int = 1600            # Sinkhorn sweep budget per solve
    ot_tol: float = 0.01            # marginal error of a converged solve
    support_weight: float = 8.0     # local support bound, E + E wB / (E + wB)
    support_target_ref: bool = False  # support floor: half the target density at the nearest target point
                                      #   (False: half the target median density, one global floor)
    support_form: str = "log"       # per-particle deficit penalty: "log" relu(log f - log s)^2 or "ratio"
                                    #   relu(1 - s/f)^2 (the missing fraction of the local mass, at most 1)
    loss_follows_n: bool = False    # loss cell = MPM cell x min(1, (mass_ref_n / N)^(1/3)): the transport grid
                                    #   and blur follow the particle spacing above the reference N

    # ---- cleanup (fixed weights, outside the render balance) ----
    w_dt: float = 0.2               # W1 pull of isolated particles down the target DT
    dt_iso_lo: float = 1.2          # isolation gate ramp, in median kNN distances
    dt_iso_hi: float = 1.8
    dt_res: int = 160               # the target DT's own fine grid
    dt_clamp_frac: float = 2.0      # DT clamp, in target extents
    w_nn: float = 0.2               # near-band pull to the nearest target point
    nn_berth_k: float = 1.0         # berth in target spacings (the prepare stage resolves it)
    nn_far_k: float = 1000.0

    # ---- render objective ----
    lambda_auto: float = 0.5        # lambda |g_render| = lambda_auto |g_physics| at calibration
    lambda_ema: float = 0.3
    render_weight_scale: float = 1.0  # x lambda wherever it is set; 0 = the render-off twin
    render_views: int = 6           # azimuths per elevation ring
    render_elevs: tuple = (0.0, 0.5, -0.5)
    render_res: int = 64
    render_res_hi: int = 96         # coarse-to-fine: targets rebuilt at this resolution ...
    c2f_event: bool = True          # ... when the run at the coarse resolution would stop (the plateau, the
                                    #   patience or the rejection streak); it then goes on at the fine resolution
                                    #   to its own stop (before 2026-09-30: at half the window budget, a schedule
                                    #   that C_R's 40k runs never reached and the 300k dragon reached by run length)
    sil_k: float = 1.5              # alpha = 1 - exp(-k w)
    w_hole: float = 2.0             # silhouette deficit inside the target
    w_spray: float = 1.0            # silhouette excess outside it
    w_pbr: float = 1.0              # shading term
    pbr_ambient: float = 0.25

    # ---- outer layer (relaxation and the u control) ----
    layer_k: int = 24               # same-side layer neighbours
    layer_h_sp: float = 2.0         # neighbour weight width, in spacings
    layer_gate_ot_cells: float = 1.0  # u acts within this many MPM cells of its transport image
    coh_k: int = 8                  # material bond neighbours (source kNN)

    # ---- outer acceptance ----
    outer_merit_tol: float = 1e-4   # a repeated rejected merit within this is a replay
    outer_reversal_cos: float = -0.2  # once latched, a low-gain reversal is rejected
    outer_reversal_gain: float = 5e-3

    # ---- output ----
    grad_dump: str = ""             # directory of per-window gradient dumps (visualisation)
    ls_probe: bool = False          # diagnostic: every failed line-search trial re-run on dFc alone and u alone
    work_telemetry: bool = True     # first/last-iteration steering telemetry
    device: str = "cuda"

    def __post_init__(self):
        import math
        if self.support_form not in ("log", "ratio"):
            raise ValueError("support_form must be \"log\" or \"ratio\"")
        for name in ("support_weight", "render_weight_scale"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lambda_auto <= 0:
            raise ValueError("settled transport calibrates the render weight: lambda_auto > 0 "
                             "(use render_weight_scale 0 for the render-off twin)")
        if self.T < 1 or self.iters < 1 or self.animations < 1:
            raise ValueError("T, iters and animations must be positive")
