"""PipelineConfig — the constants of the settled-transport pipeline in one dataclass.

The formulation is fixed (README.md): each window runs T driven and T released steps and is
scored at the released end; the physics objective is the debiased grid Sinkhorn divergence
to the fixed target plus the residual drift and the transport-bounded local support; the
render objective is the multi-view silhouette plus the matched shading term, weighted by a
lambda calibrated at every window; the controls are a per-particle stress increment dFc and
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
    assim_volume: bool = False      # A/B (D135): the assimilation takes the elastic stretch's volume too (isochoric=False),
                                    #   so a volume the end state keeps becomes the body's rest volume (det Fp) instead of
                                    #   being held by the control; the band [assim_smin, assim_smax] then bounds every
                                    #   principal stretch of Fp and so det Fp too; defined with volume_exact off, carried or
                                    #   smoothed (where F's volume is the one the stress reads); False = isochoric as before
    volume_exact: str = "off"       # A/B: the stress reads a tracked volume J per particle (carried across windows like F)
                                    #   on the smoothed F's shape, F_eff = (J / det F)^(1/3) (F + dFc) (mpm/kernels.k_stress_vx).
                                    #   The smoothing kept 4.5 % of each step's increment, so the stress read J ~ 1.00 where
                                    #   the transit stream was at J ~ 4 (D128). "history" (D129; True): J of the unsmoothed
                                    #   history, J_t det(F_new) / det(F_t), the control's volume included at full size;
                                    #   "motion" (D130): the motion's own volume, J_t det(I + dt C) = det Fg, the control's
                                    #   volume acting within its own step only; "carried" (D131): the same volume carried in the smoothed F
                                    #   itself (k_volume_carry: det F = J after every step; the stress reads F + dFc as on the old path, so
                                    #   F's scale cannot drift and reweight the control); "smoothed" (D134): carried the same way, J the
                                    #   motion's volume at the smoothing's rate (the old path's volume without the control's
                                    #   accumulated part); "off" (False) = the old path bit for bit

    # ---- physics objective ----
    ot_iters: int = 1600            # Sinkhorn sweep budget per solve
    ot_tol: float = 0.01            # marginal error of a converged solve
    support_weight: float = 8.0     # local support bound, E + E wB / (E + wB)
    support_target_ref: bool = False  # support floor: half the target density at the nearest target point
                                      #   (False: half the target median density, one global floor)
    support_form: str = "log"       # per-particle deficit penalty: "log" relu(log f - log s)^2 or "ratio"
                                    #   relu(1 - s/f)^2 (the missing fraction of the local mass, at most 1); or
                                    #   "proximity": the target-surface proximity in place of the support (the
                                    #   body's nearest particle's kernel at every outer target point against
                                    #   half the kernel at one sampling pitch; radius^2 mean relu(1 - K/floor)^2,
                                    #   no bound, no weight)
    loss_follows_n: bool = False    # loss cell = MPM cell x min(1, (mass_ref_n / N)^(1/3)): the transport grid
                                    #   and blur follow the particle spacing above the reference N
    cell_shape: float = 0.0         # D137: the shape's MPM cell (prepare's Prepared.cell_shape) when the MPM cell follows
                                    #   N above mass_ref_n (prepare's cell_ref_n); the u gate's grid and radius keep it;
                                    #   0 = the MPM cell is the shape's (prm.dx), the code as it was

    # ---- cleanup (fixed weights, outside the render balance) ----
    w_dt: float = 0.2               # W1 pull of isolated particles down the target DT
    spray_gate: str = "knn"         # D126 A/B: which particles the spray cleanup acts on: "knn", the kNN-ratio ramp below
                                    #   (the code as it was); "grid", the MPM's own decoupling test (mpm/kernels.k_frag_step,
                                    #   the one the material bonds use: no other particle in the 3^3 cells around its own, or
                                    #   the runner's commit-time fragment mask), binary, no constant (dt_iso_lo/hi unused)
    dt_iso_lo: float = 1.2          # isolation gate ramp, in median kNN distances
    dt_iso_hi: float = 1.8
    dt_res: int = 160               # the target DT's own fine grid
    dt_clamp_frac: float = 2.0      # DT clamp, in target extents
    w_nn: float = 0.2               # near-band pull to the nearest target point, between the berth and one loss cell
    nn_berth_k: float = 1.0         # berth in target spacings (the prepare stage resolves it)

    min_spacing: float = 0.0        # the position update keeps particles this far apart, in pitches of the rest
                                    #   volume (0: off; 0.9 is the spacing of a Poisson-disk sample of that density, D70)

    # ---- the comparison baseline (D112) ----
    baseline: str = ""              # "xu": the same simulator, windows and step control with Xu et al.'s objective
                                    #   alone (their EndLayerMassLoss as the C++ oracle computes it, losses/volumetric.
                                    #   d_vol_xu); nothing of ours: no transport, surface proximity, drift or released
                                    #   motion, cleanup, relaxation, u, minimum spacing, relief reference or render;
                                    #   "xu_spray": the same with our spray cleanup (the ejection guard) as well, in
                                    #   Xu's units (divided by the scale that makes Xu's gradient norm D_vol's at
                                    #   the source, as our transport's), so it weighs against the loss as ours does
    xu_mass: float = 1.0            # that loss's per-particle mass: 1 / ppc, so that a full cell holds 1 as in the oracle
    xu_form: str = "oracle"         # which form of that loss: "oracle", the C++ code's (a min-mass penalty, the out-of-
                                    #   target nodes' gradient x5, so not the gradient of its value); "paper", the
                                    #   published one (Xu et al., arXiv 2409.15746): 1/2 sum (ln(m+1) - ln(m*+1))^2 and
                                    #   its own gradient

    # ---- render objective ----
    lambda_auto: float = 0.5        # lambda |g_render| = lambda_auto |g_physics| at calibration
    lambda_ema: float = 0.3
    render_weight_scale: float = 1.0  # x lambda wherever it is set; 0 = the render-off twin
    render_exterior: bool = False   # the render terms read on the exterior (surface discs on the particles' zero set,
                                    #   render/exterior.py) in place of the particle cloud (D62)
    exterior_radius: float = 3.0    # D123 A/B: the exterior field's kernel radius in pitches (D59's 3 = the old path bit
                                    #   for bit); its offset follows in proportion, 0.8 x radius / 3, the rule that keeps the
                                    #   surface's mean offset from the mesh where the radius 3 has it (D123)
    u_off: bool = False             # D124 A/B (ablation): u's gate is zero on every particle, so u never acts; the leaf
                                    #   stays (zero gradient) and everything else is as it is
    render_body_only: bool = False  # D127 A/B: the render terms read only the discs of the body's largest connected set
                                    #   (the display's rule, render/exterior.connected_sets: discs linked within 2.2 lattice
                                    #   pitches), decided once per search of the window's discs; a flake of discs apart from
                                    #   the body earns the render nothing. The target's discs are unchanged
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
    work_telemetry: bool = False    # diagnostic records: the first/last-iteration steering telemetry, and per window the
                                    # support split, the active sets, the scale and control records, the OT divergence, the
                                    # det F quantiles and the thin metrics (about 2 s a window at 300k); off in production
    profile: bool = False           # diagnostic: wall-clock split of a window (synchronises the GPU around each part)
    term_dump: str = ""             # diagnostic: directory of each term's per-particle position gradient per window
    device: str = "cuda"

    def xu_kw(self) -> dict:
        """The keyword arguments of losses/volumetric.d_vol_xu for the baseline's form."""
        return dict(out_of_target=1.0, penalty_weight=0.0, eps=0.0) if self.xu_form == "paper" else {}

    def __post_init__(self):
        import math
        if self.support_form not in ("log", "ratio", "proximity"):
            raise ValueError("support_form must be \"log\", \"ratio\" or \"proximity\"")
        if self.xu_form not in ("oracle", "paper"):
            raise ValueError("xu_form must be \"oracle\" or \"paper\"")
        if self.spray_gate not in ("knn", "grid"):
            raise ValueError("spray_gate must be \"knn\" or \"grid\"")
        self.volume_exact = {False: "off", True: "history"}.get(self.volume_exact, self.volume_exact)
        if self.volume_exact not in ("off", "history", "motion", "carried", "smoothed"):
            raise ValueError("volume_exact must be \"off\", \"history\", \"motion\", \"carried\" or \"smoothed\"")
        if self.assim_volume and self.volume_exact in ("history", "motion"):
            # those modes' stress reads (J / det F)^(1/3) F: F's own volume is not the one the stress reads, so
            # assimilating it would not move the body's rest volume
            raise ValueError("assim_volume is defined with volume_exact off, carried or smoothed")
        for name in ("support_weight", "render_weight_scale"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lambda_auto <= 0:
            raise ValueError("settled transport calibrates the render weight: lambda_auto > 0 "
                             "(use render_weight_scale 0 for the render-off twin)")
        if self.T < 1 or self.iters < 1 or self.animations < 1:
            raise ValueError("T, iters and animations must be positive")
