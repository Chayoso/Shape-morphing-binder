"""PipelineConfig — the constants of the settled-transport pipeline in one dataclass.

The formulation is fixed (README.md): each window runs T driven and T released steps and is
scored at the released end; the physics objective is the debiased grid Sinkhorn divergence
to the fixed target plus the residual drift and the target-surface proximity; the
render objective is the multi-view silhouette plus the matched shading term, weighted by a
lambda calibrated at every window; the controls are a per-particle stress increment dFc and
the normal offset u of the outer layer. The fields below are its numbers. Weights are in
the legacy cell-sum unit and converted by the unit ratio measured at the source.

The defaults here are the code as it was before each A/B switch (the reference the tests compare against); the frozen
recipe (tag freeze-2026-10-09b, D139) is scripts/pipeline_run.py's defaults: render_exterior, min_spacing 0.9, the
layer relief, lambda_ema 1, volume_exact "carried", assim_volume and the source's density match on the MPM cell from the
shape (D105, D120, D131, D132, D135); D137's cell following N above 40k is an option (--cell_follows_n; cell_shape is
then set by the run script from prepare).
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
    replay_noise: str = "window"    # D164: the commit check reads the window's own replay pairs (floored at E's float32
                                    # resolution) and the commit reads the accepted candidate's transport potentials and
                                    # proximity lists (amendment 1); "start": the tag's check and commit (comparison only)
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
                                    #   principal stretch of Fp and so det Fp too; False = isochoric as before
    volume_exact: str = "off"       # A/B (D131): "carried", the motion's own volume J_t det(I + dt C) tracked per particle and
                                    #   carried in the smoothed F itself (mpm/kernels.k_volume_carry: det F = J after every step;
                                    #   the stress reads F + dFc as on the old path); the smoothing kept 4.5 % of each step's
                                    #   increment, so the stress read J ~ 1.00 where the transit stream was at J ~ 4 (D128);
                                    #   "off" (False) = the old path bit for bit

    # ---- physics objective ----
    ot_iters: int = 1600            # Sinkhorn sweep budget per solve
    ot_tol: float = 0.01            # marginal error of a converged solve
    ot_warm_trials: bool = False    # A/B (D142): a line-search trial's transport solves start from the window's latest
                                    #   solved potentials at the blur, the commit reads the accepted candidate's
    cleanup_in_phys: bool = False   # A/B (D143): the cleanup's gradient in the physics channel (PCGrad, the render weight)
    # the fine part of the geometry energy is the target-surface proximity (losses/support.SurfaceProximity)
    loss_follows_n: bool = False    # loss cell = MPM cell x min(1, (mass_ref_n / N)^(1/3)): the transport grid
                                    #   and blur follow the particle spacing above the reference N
    cell_shape: float = 0.0         # D137: the shape's MPM cell (prepare's Prepared.cell_shape) when the MPM cell follows
                                    #   N above mass_ref_n (prepare's cell_ref_n); the u gate's grid and radius keep it;
                                    #   0 = the MPM cell is the shape's (prm.dx), the code as it was

    # ---- cleanup (fixed weights, outside the render balance) ----
    w_dt: float = 0.2               # W1 pull of isolated particles down the target DT
    dt_iso_lo: float = 1.2          # isolation gate ramp, in median kNN distances
    dt_iso_hi: float = 1.8
    dt_res: int = 160               # the target DT's own fine grid
    dt_clamp_frac: float = 2.0      # DT clamp, in target extents
    w_nn: float = 0.2               # near-band pull to the nearest target point, between the berth and one loss cell
    nn_berth_k: float = 1.0         # berth in target spacings (the prepare stage resolves it)
    proximity_berth: bool = False   # D161b A/B: the surface proximity counts a target point covered within the berth (its floor
                                    #   K(berth) in place of K(sp) / 2: the near band's free radius); off = the frozen path
    placement_at_cell: bool = False  # D162 (2) A/B: the placement terms read the body at the transport's resolution, the loss
                                    #   cell: the surface proximity covers a target point from one loss cell (floor K(ldx)), the
                                    #   near band's berth is the loss cell (its band, berth to one loss cell, is then empty) and the
                                    #   spray cleanup's isolation gate reads the 8th-neighbour distance in loss cells; off = the
                                    #   frozen path

    min_spacing: float = 0.0        # the position update keeps particles this far apart, in pitches of the rest
                                    #   volume (0: off; 0.9 is the spacing of a Poisson-disk sample of that density, D70)
    spacing_symmetric: bool = False  # D149 A/B: the minimum spacing's neighbours are the symmetric closure of each
                                    #   particle's 16 nearest at the window start (q is p's iff p lists q or q lists p),
                                    #   so every pair's push is equal and opposite and the push keeps the centre of mass;
                                    #   off = each particle's own 16 nearest (the frozen path), where a pair listed by one
                                    #   side only moves one particle

    # ---- the comparison baseline (D112) ----
    baseline: str = ""              # "xu": the same simulator, windows and step control with Xu et al.'s objective
                                    #   alone (their EndLayerMassLoss as the C++ oracle computes it, losses/volumetric.
                                    #   d_vol_xu); nothing of ours: no transport, surface proximity, drift or released
                                    #   motion, cleanup, relaxation, u, minimum spacing, relief reference or render;
                                    #   "xu_spray": the same with our spray cleanup (the ejection guard) as well, in
                                    #   Xu's units (divided by the scale that makes Xu's gradient norm D_vol's at
                                    #   the source, as our transport's), so it weighs against the loss as ours does;
                                    #   "force" (B13): everything ours, but the driven control is an external force per
                                    #   particle and step in place of dFc, held at zero (mpm/force.py); "force_mf": that
                                    #   force with no net force and no net torque at every step
    force_scale: float = 1.0        # B13: the force leaf's unit, x (lam + 2 mu) V / (m dx) (window/setup.py)
    xu_mass: float = 1.0            # that loss's per-particle mass: 1 / ppc, so that a full cell holds 1 as in the oracle
    xu_form: str = "oracle"         # which form of that loss: "oracle", the C++ code's (a min-mass penalty, the out-of-
                                    #   target nodes' gradient x5, so not the gradient of its value); "paper", the
                                    #   published one (Xu et al., arXiv 2409.15746): 1/2 sum (ln(m+1) - ln(m*+1))^2 and
                                    #   its own gradient

    # ---- render objective ----
    lambda_auto: float = 0.5        # lambda |g_render| = lambda_auto |g_physics| at calibration
    lambda_ema: float = 0.3
    render_weight_ref: str = "residual"  # D162 (1) A/B: "residual" = the rule lambda |g_r| = lambda_auto |g_p| at every window
                                    #   (the frozen path); "floor" = the render weight referenced to the render's own excess over
                                    #   its floor: lambda_t = lambda_0 max(0, 1 - L_floor / L_r(t)), lambda_0 the rule's value at
                                    #   the first window, L_r(t) the render term at the window's first evaluation, L_floor the
                                    #   render term of the pictures' own first sample read as the body is read (amendment 2);
                                    #   "shape" (D162b) = the same weight with the excess read on the body's shape, 1 - b_floor / b(t):
                                    #   b the render term of uniform resamples of the body's solid (pipeline/shape_read.py)
                                    #   "floor_pitch" (D162d) = F's weight with L_r and L_floor read on pictures low-passed at one
                                    #   exterior pitch (target.render_term_lowpass): the sub-pitch arrangement below, the strokes above
    render_weight_scale: float = 1.0  # x lambda wherever it is set; 0 = the render-off twin
    settle_u: str = "zero"          # D166 A/B: u in the settle windows: zero (D162h, the default) or last = the last
                                    #   committed window's u, held (rigid-free and clamped in each settle window, as u is)
    settle_relax: bool = True       # D166 A/B: the outer-layer relaxation in the settle windows (on = D162h, the default)
    colour_kernel: str = "kernel"   # D173 A/B: the render's disc colours, "kernel" (D160's 3-pitch weights, the frozen path) or
                                    #   "nearest" (render/exterior.nearest_weights: the nearest particle's colour, softmax at half a pitch)
    render_split: str = ""          # D172 A/B: the render gradient's weight by control block; "" = one weight on both (the
                                    #   frozen path); "u" = the run's weight on u, the rule's per-window value on dFc;
                                    #   "dFc" = the run's weight on dFc, the rule's on u (the merit keeps the run's weight)
                                    #   "u_rule" = the run on the rule (merit, stops) with u at the first calibration's value
    render_every: int = 1           # D174 A/B (runtime): the render gradient (its adjoint sweep) at every k-th iteration of a
                                    #   window (0, k, 2k, ...); between, the window's last one, projected against the fresh
                                    #   physics gradient and weighted as always; the render term's value is read at every
                                    #   evaluation as before; 1 = every iteration (the frozen path)
    keyframe_time: tuple = ()       # D175 (docs/design/keyframe_render_control.md; () = off, the frozen path): the times (s) of
                                    #   the picture keyframes, increasing. A window whose end frame lies in (frame(t_k-1),
                                    #   frame(t_k)], frame(t) = round(t / (2T dt)) 2T, reads keyframe k's pictures and none of
                                    #   the final target's geometry or cleanup (only the released motion and the end drift); the
                                    #   render weight there is the held weight on both blocks; a segment window with no accepted
                                    #   step commits its zero-control rollout (the settle window's); the stops read the final
                                    #   target's epoch only. Needs render_exterior and lambda_ema 0 (lambda held from a pre-pass
                                    #   at the source on the final target)
    keyframe_compare: str = "silhouette"  # D175b: the keyframe comparison; "silhouette" = the exterior silhouette + shading
                                    #   (D175); "column" = the particles' projected count picture against the keyframe's
                                    #   (render_loss.d_count); the final target's render term is the exterior either way
    keyframe_soundness: str = "every"  # D175c: "every" = the released motion and end drift in every window (D175); "final" =
                                    #   only with the final target: a keyframe segment's merit is its comparison alone
    keyframe_geometry: str = "own"  # D175 ablation (arm N): "final" keeps the final target's geometry and cleanup in a segment
                                    #   (only the pictures are swapped); "own" = the storyboard's ownership (the method)
    layer_relax: bool = True        # D170 A/B: the outer-layer relaxation in every window (on = the frozen path); off = its
                                    #   fraction 0 throughout, as the xu baseline has it (window/setup.py), u and all else as they are
    settle_windows: int = 0         # D162h / D162i: after the run stops, this many windows of released physics with zero controls
                                    #   (the same material and assimilation), delivered and recorded (a rest test / a settled end)
    render_exterior: bool = False   # the render terms read on the exterior (surface discs on the particles' zero set,
                                    #   render/exterior.py) in place of the particle cloud (D62)
    exterior_radius: float = 3.0    # D123 A/B: the exterior field's kernel radius in pitches (D59's 3 = the old path bit
                                    #   for bit); its offset follows in proportion, 0.8 x radius / 3, the rule that keeps the
                                    #   surface's mean offset from the mesh where the radius 3 has it (D123)
    exterior_field: str = "zb"      # D140 / D144 A/B: the exterior's field, "zb" Zhu and Bridson's (the code as it was),
                                    #   "aniso" the anisotropic density field (render/exterior.Anisotropic, D140) or "zbs" Zhu
                                    #   and Bridson's with Solenthaler et al.'s correction (ZhuBridsonSolenthaler, D144),
                                    #   read by the target's pictures and the render terms alike
    render_target: str = "samples"  # D159 A/B: the render's target pictures; "samples" = the target sample and its draws
                                    #   through the exterior (the frozen path), "mesh_shell" = the target mesh grown by the
                                    #   run's own measured exterior offset (render/mesh_shell.py), needs the mesh and the exterior
    u_off: bool = False             # D124 A/B (ablation): u's gate is zero on every particle, so u never acts; the leaf
                                    #   stays (zero gradient) and everything else is as it is
    render_views: int = 6           # azimuths per elevation ring
    render_elevs: tuple = (0.0, 0.5, -0.5)
    render_res: int = 96            # the pictures' resolution (the run script scales it with N above mass_ref_n, D74)
    sil_k: float = 1.5              # alpha = 1 - exp(-k w)
    w_hole: float = 2.0             # silhouette deficit inside the target
    w_spray: float = 1.0            # silhouette excess outside it
    w_pbr: float = 1.0              # shading term
    pbr_ambient: float = 0.25

    # ---- outer layer (relaxation and the u control) ----
    layer_k: int = 24               # same-side layer neighbours
    layer_h_sp: float = 2.0         # neighbour weight width, in spacings
    layer_gate_ot_cells: float = 1.0  # u acts within this many MPM cells of its transport image
    relax_driven: bool = False      # D146 A/B: the layer relaxation acts in the T driven steps only, the steps u acts in,
                                    #   so the released half has no position edit; off = every step of the 2T-step window
                                    #   (the code since the settled transport, 835af64: the frozen path)
    gap_split: str = "off"          # D148 A/B: the two sides of a target gap on their own grid fields (mpm/gap_step.py):
                                    #   "on" (variant A) at every gap node, "interface" (variant B) where the node's particle
                                    #   volume is below one cell, "soft" (D148c) A with soft sides (each particle's share of
                                    #   side 1 the plan's mass beyond the plane under its blur), "fine" (D148b) A with the side
                                    #   points from the loss grid's plan; "walls" (D148c) one wall per particle and gap
                                    #   region, persistent, with B's gate (mpm/gap_walls.py); "off" = the single field
    gap_vp: float = 0.0             # variant B: the volume per particle (the sample's fill volume / N), set by the run script
    u_adam_carry: bool = False      # D146b A/B: u's Adam moments carry from one committed window to the next (u itself still
                                    #   starts at zero), each coordinate's bias correction counting its own age; reset on an
                                    #   outer reject; dFc's stay per window; off = fresh moments every window (the frozen path)
    coh_k: int = 8                  # material bond neighbours (source kNN)

    # ---- outer acceptance ----
    outer_merit_tol: float = 1e-4   # a repeated rejected merit within this is a replay
    outer_reversal_cos: float = -0.2  # once latched, a low-gain reversal is rejected
    outer_reversal_gain: float = 5e-3

    # ---- output ----
    grad_dump: str = ""             # directory of per-window gradient dumps (visualisation)
    work_telemetry: bool = False    # diagnostic records: the first/last-iteration steering telemetry, and per window the
                                    # proximity split, the active sets, the scale and control records, the OT divergence, the
                                    # det F quantiles and the thin metrics (about 2 s a window at 300k); off in production
    profile: bool = False           # diagnostic: wall-clock split of a window (synchronises the GPU around each part)
    term_dump: str = ""             # diagnostic: directory of each term's per-particle position gradient per window
    device: str = "cuda"

    def xu_kw(self) -> dict:
        """The keyword arguments of losses/volumetric.d_vol_xu for the baseline's form."""
        return dict(out_of_target=1.0, penalty_weight=0.0, eps=0.0) if self.xu_form == "paper" else {}

    def __post_init__(self):
        import math
        if self.xu_form not in ("oracle", "paper"):
            raise ValueError("xu_form must be \"oracle\" or \"paper\"")
        if self.baseline not in ("", "xu", "xu_spray", "force", "force_mf"):
            raise ValueError("baseline must be \"\", \"xu\", \"xu_spray\", \"force\" or \"force_mf\"")
        if self.baseline.startswith("force"):
            if not (math.isfinite(self.force_scale) and self.force_scale > 0):
                raise ValueError("force_scale must be finite and positive")
            if self.gap_split != "off" or self.grad_dump:
                raise ValueError("the force baseline (B13) runs without gap_split and grad_dump")
        self.volume_exact = {False: "off"}.get(self.volume_exact, self.volume_exact)
        if self.volume_exact not in ("off", "carried"):
            raise ValueError("volume_exact must be \"off\" or \"carried\"")
        if self.gap_split not in ("off", "on", "interface", "soft", "fine", "walls"):
            raise ValueError("gap_split must be \"off\", \"on\", \"interface\", \"soft\", \"fine\" or \"walls\"")
        if self.gap_split in ("interface", "walls") and not self.gap_vp > 0:
            raise ValueError("gap_split interface reads the node volume: gap_vp must be the volume per particle")
        if self.settle_u not in ("zero", "last"):
            raise ValueError("settle_u must be \"zero\" or \"last\"")
        if self.render_weight_ref not in ("residual", "floor", "shape", "floor_pitch"):
            raise ValueError("render_weight_ref must be \"residual\", \"floor\", \"shape\" or \"floor_pitch\"")
        if self.render_weight_ref in ("floor", "shape", "floor_pitch") and not self.render_exterior:
            raise ValueError("render_weight_ref floor / shape reads the render term on the exterior: it needs render_exterior")
        if self.exterior_field not in ("zb", "aniso", "zbs"):
            raise ValueError("exterior_field must be \"zb\", \"aniso\" or \"zbs\"")
        if self.exterior_field == "aniso" and self.exterior_radius != 3.0:
            raise ValueError("the anisotropic field (D140) has its own reach: exterior_radius must stay 3")
        if self.render_target not in ("samples", "mesh_shell"):
            raise ValueError("render_target must be \"samples\" or \"mesh_shell\"")
        if self.render_target == "mesh_shell" and not self.render_exterior:
            raise ValueError("the mesh shell (D159) is drawn on the exterior: render_exterior must be on")
        if not math.isfinite(self.render_weight_scale) or self.render_weight_scale < 0:
            raise ValueError("render_weight_scale must be finite and nonnegative")
        if self.lambda_auto <= 0:
            raise ValueError("settled transport calibrates the render weight: lambda_auto > 0 "
                             "(use render_weight_scale 0 for the render-off twin)")
        if self.T < 1 or self.iters < 1 or self.animations < 1:
            raise ValueError("T, iters and animations must be positive")
        if int(self.render_every) < 1 or int(self.max_ls_iters) < 1:
            raise ValueError("render_every and max_ls_iters must be at least 1")
        self.keyframe_time = tuple(float(t) for t in (self.keyframe_time or ()))
        if self.keyframe_geometry not in ("own", "final"):
            raise ValueError("keyframe_geometry must be \"own\" or \"final\"")
        if self.keyframe_soundness not in ("every", "final"):
            raise ValueError("keyframe_soundness must be \"every\" or \"final\"")
        if self.keyframe_compare not in ("silhouette", "column"):
            raise ValueError("keyframe_compare must be \"silhouette\" or \"column\"")
        if self.keyframe_time:                        # D175: the storyboard's preconditions (the design, section 5)
            if not self.render_exterior or self.lambda_ema != 0.0:
                raise ValueError("keyframes read their pictures on the exterior with the held weight: they need "
                                 "render_exterior and lambda_ema 0")
            if (self.render_split not in ("", "u") or self.render_weight_ref != "residual" or self.baseline
                    or self.render_target != "samples"):
                raise ValueError("keyframes run on the held weight's paths only: render_split \"\" or \"u\", "
                                 "render_weight_ref residual, no baseline, render_target samples")
            if any(not (t > 0 and math.isfinite(t)) for t in self.keyframe_time) or list(self.keyframe_time) != sorted(
                    set(self.keyframe_time)):
                raise ValueError("keyframe_time: positive, finite and strictly increasing")
