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
    eps: float = 1e-3              # C++ oracle scale; avoids sign-like steps near stationarity
    armijo_c1: float = 1e-4        # sufficient decrease along the preconditioned step
    ls_noise_rel: float = 1e-7     # reject improvements smaller than rollout/atomic noise
    replay_calibrate: bool = True  # measure the replay noise of the start control each
                                   # window (two rollouts) and floor the commit-rollout
                                   # tolerance at 10x it (probe: 1e-7 discarded ~10% of
                                   # accepted windows)
    dfc_clip: float = 0.0           # optional per-particle |dFc| cap (0 = off; C++ has none)
    pace_budget: float = 0.0        # >0: derive the per-window pace cap FROM THE
                                    # ANIMATION BUDGET as 1 - pace_budget^(1/animations)
                                    # (an exponential glidepath whose residual at the
                                    # last anim is pace_budget). A fixed 12%/window cap
                                    # finishes the descent geometrically in ~20 commits
                                    # and leaves the remaining budget to hold/oscillate
                                    # (user directive 2026-09-02: the morph must
                                    # progress across ALL frames, no snap); overrides
                                    # `pace` when set.
    pace_quasistatic: bool = True   # a paced window may exit early on its shape budget
                                    # only if it did not increase the kinetic term
                                    # (b8: momentum-carrying commits -> every next
                                    # candidate regresses -> brake pins the run)
    pace: float = 0.0               # TRAJECTORY pacing: each window may reduce its starting
                                    # loss by at most this fraction (0 = off). Stops the
                                    # "morph done in 2 commits" snap — the deliverable is the
                                    # trajectory, not just the endpoint.

    # ---- loss terms (docs/pipeline_v2.md §3.3) ----
    loss_res: int = 32              # D_vol grid resolution
    w_kin: float = 0.5              # terminal kinetic mean|v_T|^2 (arrive at rest)
    w_ctrl: float = 1e-3            # running control cost sum_t|dFc|^2/(T N) (anti-slam)
    w_tctrl: float = 0.0            # temporal dFc first-difference cost (anti bang-bang)
    w_box: float = 10.0             # far-field leash relu(|x|-extent)^2: the render views and
                                    # the D_vol grid have NO gradient for far ejecta; this does

    # ---- render channel (§3.4, §3.6) ----
    lambda_auto: float = 0.0        # 0 = physics-only arm; >0 = norm-balanced render weight
    lambda_ema: float = 0.3         # EMA rate for the balanced lambda (anti-oscillation)
    lambda_cap: float = 5e3         # hard cap: the raw ratio diverges once D_render
                                    # saturates (live-observed 1.77e5); converged render fades
    render_views: int = 6           # azimuths per elevation ring
    render_elevs: tuple = (0.0, 0.5, -0.5)   # elevation angles (v1 was equator-only)
    render_res: int = 64
    sil_k: float = 1.5              # alpha saturation 1-exp(-k w)
    w_hole: float = 2.0             # deficit inside target (holes/missing extremities)
    w_spray: float = 1.0            # excess outside target (ejecta pulled back by the objective)
    w_pbr: float = 0.0              # PBR-lite: headlight-Lambertian shaded-image L2 inside the
    pbr_ambient: float = 0.25       #   balanced render scalar (curvature-sensitive feedback)
    grad_project: bool = False      # PCGrad one-sided: project the render grad off the physics
                                    #   grad when they conflict (cos<0 — measured late-run -0.74)
    c2f_at: float = 0.0             # >0: coarse-to-fine — rebuild render targets at this
    render_res_hi: int = 96         #   fraction of the run at render_res_hi
    lg_sweeps: int = 0              # LOCAL-GLOBAL: >0 runs a surface-band GS pass per
    lg_young: float = 2e3           #   commit (global MPM owns bulk transport; the band,
                                    #   interior-pinned, owns the rim render residual —
                                    #   the alternative to PCGrad under strong conflict)
    w_creg: float = 0.0             # spatial regularisation of the CONTROL field: kNN-
    creg_k: int = 8                 #   Laplacian penalty on dFc so no single boundary
                                    #   particle can be actuated alone (the thin-feature
                                    #   fringe's creation mechanism — forensics 2026-09-01)
    w_dt: float = 0.0               # one-sided-W1 cleanup: SUM_p m_p x gate_p x 3D DT.
    dt_gate: str = "knn"            #   Gate A/B settled on the honest metric (§7.6):
    dt_iso_lo: float = 1.2          #   kNN selectivity (full pull on singletons all run
    dt_iso_hi: float = 1.8          #   long, dense rim untouched) beat the budget
    dt_budget: float = 0.01         #   scalar (mid-run partial rub on every out
    dt_res: int = 160               #   particle: no cleanup + rim damage). The kNN
                                    #   gate's inversion side effect is owned by the
                                    #   trajectory-det acceptance guard; its clump
                                    #   blind spot is UNOWNED (stack-review f7/F10 —
                                    #   fill v1 was measured ineffective and the
                                    #   flagship runs without fill). SUM form: per-
                                    #   stray pull = w_dt exactly, N-invariant (Opus
                                    #   finding 1: mean form was a 300-3000x no-op).
                                    #   dt_budget also caps the FILL budget (F15).
    fill_cap_frac: float = 0.02     # fill v4: pair cap (<= this fraction of N matched)
    w_fill: float = 0.0             # fill v4: FIXED pull weight on TARGET-HARD
                                    #   assigned pairs (deficit cell -> nearest donor
                                    #   particles, capacity ~ cell shortfall). v3's
                                    #   norm-balanced weight died with the physics
                                    #   gradient (0.062->0.009) before thin features
                                    #   filled; growth v1 proved particle-side demand
                                    #   is a universal surface signal (99% of high
                                    #   demand outside the ears). Anchoring on the
                                    #   deficit CELLS is the partial-OT target-hard
                                    #   form. (history below: the v3 note)
                                    # HOLE-side W1 v3: ALPHA of the fill norm-
                                    #   balancer (fill_lam = w_fill*||g_phys||/
                                    #   ||g_fill||, EMA, cap) — v2's constant
                                    #   weight dominated late 30:1 (measured);
                                    #   balanced form bounds fill <= w_fill x
                                    #   the physics gradient by construction.
    fill_thresh: float = 0.6        #   nearby mass toward under-covered TRUE-support
    fill_sigma: float = 2.0         #   cells. Support-AND (Opus F1: v1's blurred-only
    fill_range_frac: float = 0.2    #   mask was 95-100% OUTSIDE support = an outward
                                    #   fringe factory). Budget on deficit MASS (F2:
                                    #   in-range particle count attenuated the useful
                                    #   mode 22-78x). Range 0.2*extent (F3: ear-tip-
                                    #   to-body is 0.171*extent; v1's 0.1 stranded 29%
                                    #   of the ear). thresh 0.6 (F13: 0.3's fixed point
                                    #   was a 30%-filled feature). Smooth locality ramp
                                    #   (F12). Residual: no hysteresis (bang-bang
                                    #   bounded by the budget; Schmitt mask queued).
    dt_clamp_frac: float = 2.0      # W1 DT clamp: own FINE target-fitted grid at
                                    #   dt_res (Opus finding 2: loss-grid cells left a
                                    #   ~1-unit dead radius over the fringe band). 3D,
                                    #   not per-view (visual hull hides interior
                                    #   concavities — forensics). Clamp 2*extent: no
                                    #   force-free interior gap (Codex f4); beyond the
                                    #   box the quadratic leash dominates the linear
                                    #   tail. NOT lambda-scaled: lambda->cap x constant
                                    #   gradient = mass-ejection mode (arXiv:2409.15746).

    w_nn: float = 0.0               # GRID-FREE near-band W1: constant pull toward the
    nn_berth_k: float = 1.0         #   assigned nearest target particle, active only in
                                    #   (1.5 -> 1.0, x2 2026-09-03: no dead band - 40k
                                    #   chamfer 0.0706, out_nn 11.0 -> 9.5%, far -9%,
                                    #   max_dt 19.4 -> 18.0 sp, guards 0)
    w_jdens: float = 0.0            # DENSITY-measured volume prior (REVISION 3): sKL on
                                    # J = rho0/rho from a CIC mass grid at jdens_res,
                                    # calibrated once to the D_vol gradient norm
                                    # (w_jdens=1 -> parity). The stored F lags the true
                                    # deformation, so w_jvol (on det F) cannot see the
                                    # 1.32x ear dilation / 0.94x body compression this
                                    # term measures directly.
    jdens_res: int = 128            #   its own raster (64^3 is biased vs the kNN J)
    w_h1: float = 0.0               # NON-LOCAL mass balance (REVISION 3 amendment):
                                    # H^-1 norm of the density residual on the loss
                                    # grid (FFT Poisson). Same minimiser as D_vol; its
                                    # gradient is the Coulomb pull of every deficit
                                    # cell on every surplus particle, so body mass is
                                    # told to move into the ears. D_vol/d_dt/d_sil are
                                    # all local: v3 40k converged to a wrong fixed
                                    # point (ear frac 0.069 of 0.100, d_vol floor ~54
                                    # vs 30 at 20k). 1 = equal gradient norm with
                                    # D_vol at the source (calibrated once).
    w_kde: float = 0.0              # PARTICLE-SCALE density matching (SPH D_vol):
                                    # 0 = off; >0 multiplies a ONE-SHOT equal-norm
                                    # calibration against the D_vol gradient at the
                                    # first window (w_kde=1 -> same gradient magnitude
                                    # as D_vol at calibration; a rule, not a tuned
                                    # scalar). Census 2026-09-03: particles cluster
                                    # 2-2.7x at sub-cell scale; the fringe halo and the
                                    # frayed ears live below the CIC/silhouette
                                    # resolution, where only this term has direction.
    kde_h_k: float = 2.0            #   kernel width in target-NN spacings
    kde_k: int = 32                 #   frozen neighbour count (particles and targets)
    nn_far_k: float = 1000.0        #   the berth..far band (x target-NN spacing). Owns
                                    #   EVERYTHING beyond the berth since E4 (2026-09-02):
                                    #   the observability probe showed the unowned residual
                                    #   floater class was clumps beyond the old 4.5x cap
                                    #   (DT gate leaves clumps unowned); E4 at 300 paced
                                    #   commits: chamfer tie, out_nn 22.3->20.5%, far
                                    #   floaters 1743->1511, max_dt 26.4->23.6sp, guards 0.
    nn_tail_frac: float = 0.0       #   optional bounded far-cluster cleanup: among points
                                    #   beyond nn_far_k, pull only the worst fraction per
                                    #   frozen window (avoids a full Chamfer servo).
                                    #   the 0.05-0.10 wu fork-halo that lives inside the
                                    #   fine-DT grid's dilation dead band (forensic
                                    #   2026-09-02: 90% gate-closed AND zero force
                                    #   there). Frozen assignment per window; rim inside
                                    #   the berth feels nothing.
    w_grow: float = 0.0             # GROWTH channel (morphoelastic F=Fe·G): commit-
    grow_band: float = 1.5          #   time rest-volume command grow=1+w_grow*demand,
                                    #   demand = blurred coverage shortfall on TRUE
                                    #   support gathered at the particle (zero when
                                    #   covered -> stops by construction). NOT the
                                    #   falsified ratchet (uncontrolled absorption):
                                    #   demand-driven + per-commit cap + cumulative
                                    #   det(Fp) governor [1/band, band] (Stomakhin-snow
                                    #   lesson). Lives OUTSIDE the optimizer's gradient
                                    #   balance - the fill-v3 verdict: loss-side pulls
                                    #   die with the physics gradient before finishing
                                    #   thin features (fill_lam 0.062->0.009 measured).
    w_jvol: float = 0.0             # volume prior on terminal F: mean (J-1)·log J —
                                    #   the sKL/log-unbiased registration regularizer
                                    #   (Yanovsky/Leow CAM 07-49): zero iff J=1,
                                    #   symmetric in log measure, and a SOFT inversion
                                    #   barrier as J->0+ (F5 literature verdict: the
                                    #   permanent volumetric spring of isochoric
                                    #   assimilation arms inversions; bare eta_vol is
                                    #   the ratchet on a slower clock). Weight kept an
                                    #   order below the render gradient (adjoint
                                    #   conditioning). Tampubolon-style log-volume
                                    #   ledger pre-registered as the follow-up.

    use_gauss_loss: bool = False    # REAL 3DGS render loss (diff_gauss) replaces the
    gauss_covariance: bool = True   # supervise viewer-visible Sigma=sigma0^2 F F^T
    gauss_mix: float = 0.0          # >0: HYBRID render channel — silhouette + this
                                    # times the gauss L1, equal-magnitude calibrated
                                    # once per window at the first evaluation. g1
                                    # (2026-09-02) falsified pure replacement:
                                    # image-L1 saturates once the silhouette
                                    # roughly matches (chamfer +29%, fork 622 vs
                                    # 520) yet puts gradients exactly where the
                                    # viewer sees residue (out_nn better, detFmin
                                    # 0.62 vs 0.43) — so mix, don't replace.
    gauss_res: int = 96             #   CIC soft-silhouette in the lambda channel: the
    gauss_sigma_scale: float = 1.0  # target-surface NN multiplier; keep splats at sample scale
    gauss_children: int = 1         # 1=legacy parent splat; 2..4 massless tangent children
    gauss_child_sigma_scale: float = 0.55  # detail objective; display coverage is a separate knob
    gauss_child_offset_scale: float = 0.35 # tangent offset / calibrated parent sigma
    gauss_child_k: int = 16         # active-surface neighbours for frozen tangent PCA
    w_cov: float = 0.0              # penalise Gaussian-visible F singular values outside
    cov_smin: float = 0.5           #   this band; det(F)>0 alone does not bound anisotropy
    cov_smax: float = 2.0
                                    #   viewer's own forward model as the objective —
                                    #   sub-cell arrangement and viewer-visible
                                    #   floaters finally generate gradients (the CIC/
                                    #   saturation blindness that spawned the whole
                                    #   fringe tranche). Positions-only gradients v1.

    # ---- v3: optimisation accelerators (docs/method.md §6) ----
    warm_start: bool = False        # init window's dFc from the previous window's solution
    warm_decay: float = 0.5         # dFc is an ABSOLUTE control added into F: verbatim reuse
                                    # double-applies it (measured: Jmin 0.71->0.37->1e-4).
                                    # Decayed init + the window-start safeguard make it safe.
    render_gs_iters: int = 0        # >0: Sobolev/grid-GS smoothing of the render pull before
    render_gs_kappa: float = 4.0    #     the adjoint pullback (screened diffusion strength)
    surface_grad_frac: float = 0.0  # >0: persistent soft surface mask on render covectors
    surface_grad_floor: float = 0.05 # interior still receives a small, nonzero render signal
    surface_grad_k: int = 24        # source-material kNN surface estimator
    render_surface_only: bool = False  # render only a frozen material-skin subset;
                                       # simulation particles/mass are unchanged
    grad_h1: bool = False           # Sobolev (H1) descent direction on the material kNN graph:
                                    # the TOTAL control gradient is replaced by the solution of
                                    # (I + kappa (I - A)) u = g, solved to convergence — the
                                    # search direction cannot differ between material neighbours
                                    # at sub-stencil scale (mass ejection: lone surface particles
                                    # pushed at the clip every window)
    control_h1_iters: int = 0       # screened kNN solve on adjoint dFc render gradient
    control_h1_kappa: float = 2.0   # surface signal propagation without editing physical state

    # (VBD quasi-static arm retired 2026-09-01 -> deprecated/; its vbd_* fields removed)

    # ---- render-controls-physics contract (docs/render_controls_physics.md, 2026-09-14) ----
    control_grid: int = 0           # >0: dFc lives on a G^3 trilinear node grid over the
                                    # window's start positions (control_basis.py) instead
                                    # of per particle — the low-dimensional physical field
                                    # of PhysDreamer/NeuMA, and the basis-realised
                                    # Gauss-Seidel propagation (§4, §6). 0 = per particle.
    control_tknots: int = 0         # >0: piecewise-linear in time over this many knots
                                    # (1 = one control shared by every step, the C++
                                    # control_stride=T form); 0 = one per step.
    render_F_geom: bool = False     # render covariance rides the GEOMETRIC F_g (velocity-
                                    # gradient transport only) instead of the controlled,
                                    # smoothed F: the image can then change only through
                                    # motion (audit Finding 1, §3).
    w_kin_running: float = 0.0      # RUNNING kinetic mean_t mean_p |v_t|^2 over the whole
                                    # window (stop-and-go driver C, oscillation_triage.md);
                                    # 0 keeps the terminal-only term.
    w_coh: float = 0.0              # MATERIAL-COHERENCE prior on the window displacement:
                                    # mean_i |u_i - mean_{j in N_src(i)} u_j|^2 / sp^2 with
                                    # u = x_T - x_0 and N_src the frozen source-material
                                    # kNN (coh_k). Zero for locally affine motion (coherent
                                    # stretch/rotation), positive for a particle leaving
                                    # its neighbours — the thin-feature vanguard measured
                                    # 2026-09-15 (55-60 % of particles arriving on thin
                                    # targets locally sparse; not caused by the nn/W1 pulls,
                                    # scatter_probe ablation). State-side twin of w_creg.
    coh_k: int = 8
    w_bond: float = 0.0             # ONE-SIDED BOND-STRETCH bound (Luiten et al. 2023
                                    # rigidity/isometry weights on frozen source kNN,
                                    # bond-based-PD stretch criterion): per window,
                                    # mean_i sum_j w_ij relu(|x_T,j - x_T,i| - (1+bond_s0)
                                    # |x_0,j - x_0,i|)^2 / sp^2, w_ij = exp(-|x_src,j -
                                    # x_src,i|^2 / (2 (2 sp)^2)). Rest lengths are THIS
                                    # window's start (a rate limit on separation, never a
                                    # spring back), compression is free (ears may thin).
    bond_s0: float = 0.3            #   allowed per-window stretch of a material bond
    vol_frontier: bool = False      # evaluate D_vol only on target cells within one loss
                                    # cell of the window's current occupancy: removes the
                                    # far-field pull that rewards throwing a few particles
                                    # into distant thin-feature cells (Xu 2025: "mass
                                    # ejections lead to the quickest decrease"); the ear
                                    # then fills by frontier growth from the head.
    w_kin_var: float = 0.0          # VELOCITY-VARIANCE term mean_p [mean_t |v_t|^2 -
                                    # |mean_t v_t|^2]: zero iff every particle moves at
                                    # constant velocity inside the window, positive for
                                    # any reversal - targets the measured window-locked
                                    # limit cycle (hyde06 2026-09-15: speed 0.47 -> 0.10
                                    # -> 0.45 inside every window, 95 % of the speed
                                    # power at period T) WITHOUT penalising net progress.
    grad_project_mode: str = "render"  # which side PCGrad strips when cos<0 (grad_project):
                                    # "render" (legacy: physics descent preserved),
                                    # "phys" (render-first: physics component removed),
                                    # "cagrad" (Liu et al. 2021 two-task CAGrad, c=cagrad_c).
    cagrad_c: float = 0.5
    render_gs_cheb: bool = False    # Chebyshev-accelerated grid sweeps (Wang 2015) for
                                    # render_gs_iters; both x AND F covectors are smoothed.
    gauss_cov_sat: float = 0.0      # >0: STATELESS stretch saturation in the render forward
                                    # model, Sigma = s0^2 M (I + M/r^2)^-1 with M = F F^T
                                    # (eigenvalue lam -> lam/(1+lam/r^2): identity for small
                                    # stretch, -> r^2 for large). Replaces the commit-time
                                    # F_g relaxation, which changed the image without motion
                                    # (REFUTE-2 F11) and saturated the covariance at ~1.5 %
                                    # anisotropy. Smooth, no SVD, same map inside and across
                                    # windows. 0 = off (legacy s0^2 F F^T).
    unit_ref_res: int = 64          # loss_units="density": the legacy loss grid the weight
                                    # conversion refers to (REFUTE-2 F6: measured on the
                                    # run's own grid, the converted weights inherited the
                                    # cell-sum's resolution dependence — 3.2x weaker at
                                    # --ppc 8 / 149^3 than at 64^3 for the same flag)
    gauss_robust_eps: float = 0.0   # >0: Charbonnier sqrt(r^2+eps^2) instead of |r| in the
                                    # Gaussian image loss (audit: L1 sign crossings made
                                    # the finite-difference check fail at 12k)
    phys_loss: str = "density"      # "auto": "ot" when more than half of the source particles sit in
                                    # cells the target leaves empty (the sphere inside the C's hole:
                                    # the cell sum has only an outward push there), else "density"
                                    # "density": the cell-sum D_vol; "ot": entropic optimal transport
                                    # (losses/ot.SinkhornPull) rescaled once to D_vol's gradient norm;
                                    # "ot_leash": D_vol + a hinge on the distance of each particle to
                                    # its transport-map image beyond the plan's blur radius (the cell
                                    # sum refines inside the radius, nothing may leave the body by more)
                                    # "ot_pace": D_vol against the DISPLACEMENT-INTERPOLATED target: the
                                    # cloud advected along the plan by at most one blur radius per
                                    # particle, rasterised; local moves only, no far-cell reward
                                    # "ot_resid": as ot_pace but the plan moves only the cell sum's own
                                    # RESIDUAL (excess mass -> deficit mass); satisfied particles stay
    ot_eps_cells: float = 0.0       # Sinkhorn epsilon = (ot_eps_cells * loss cell)^2; 0: the
                                    # plan resolves to the particle spacing, eps = (nn spacing)^2
                                    # (= (dx / ppc^(1/3))^2, half a cell at ppc 8: the resolution
                                    # of the cloud itself, derived, not tuned)
    ot_samples: int = 8192          # target samples for the plan
    ot_iters: int = 400             # Sinkhorn sweep CAP per plan; the solve stops at ot_tol
    ot_tol: float = 1e-2            # row-marginal error at which a plan counts as converged
                                    # (the cold plan anneals eps geometrically from the squared
                                    # target diameter; warm plans re-solve at the target eps)
    ot_debias: bool = False         # debiased divergence: subtract the self-transport map (no shrinkage)
    ot_handoff: bool = False        # ot_pace: once no target cell with mass lies beyond one cell of
                                    # the body, the window target is the FIXED target (fill at full
                                    # strength); first tried 2026-09-17 while the merit read the
                                    # paced loss (a bug) — re-tested on the fixed merit
    render_until: int = 0           # >0: the render channel is switched OFF from this window on
                                    # (intervention experiment: does the render gradient change
                                    # the physics trajectory? 2026-09-17)
    dvol_form: str = "log"          # density-unit residual: "log" (log(1+m/m_ref)) or "linear"
                                    # ((m - m_t)/m_ref: gradient proportional to the deficit, no
                                    # empty-cell amplification)
    loss_units: str = "legacy"      # "legacy": D_vol = 1/2 sum_cells log-mass residual^2
                                    # (Xu et al., grid-count units); "density": the same
                                    # residual on m/m_ref averaged over target-support
                                    # cells — dimensionless and loss_res-invariant (§2),
                                    # so lambda_R is O(1) and lambda_cap never binds.

    # ---- material channel (§3.2 ch.2) ----
    opt_material: bool = False      # optimise per-particle log-Lame multipliers s
    mat_lr_scale: float = 0.25      # material step = alpha * this (slower than dFc)
    mat_clamp: float = 1.0          # |s| <= mat_clamp  (e^{±1} = 0.37..2.7x)
    w_mat: float = 1e-2             # ridge toward base material

    # ---- plastic assimilation channel (§3.5; exact elastic-stretch version) ----
    assim: float = 0.5              # eta: F_e -> R_e S_e^{1-eta} per commit; 0 disables
    assim_smin: float = 0.2         # cumulative Fp band; wide, because a saturated Fp stops
    assim_smax: float = 5.0         # tracking the motion and re-arms spring-back
    assim_consensus: bool = False   # plastic increment from the stencil-neighbourhood
                                    # consensus F_e (self excluded): a particle cannot flow
                                    # plastically away from its material (mass ejection fix)
    assim_iso: bool = False         # isochoric plasticity: assimilate only the deviatoric
                                    # stretch (det Fp = 1) so lambda resists ALL volume
                                    # change forever — the unnormalised ratchet measured
                                    # |J-1|>0.3 on 34% of particles, detF->0 at 120c,
                                    # and squeeze-ejecta floaters (forensics 2026-09-01)

    # ---- convergence freeze ----
    mom_carry: float = 0.0          # >0: carry Adam moments across windows (first
                                    # moment decayed by this factor, second moment
                                    # and step count verbatim; reset with a rejected
                                    # warm start). g2_anneal (2026-09-02): annealing
                                    # cut late-run reversal -0.523 -> -0.345 but only
                                    # acts after improvement stops; the zigzag during
                                    # descent comes from each window's zero-moment
                                    # Adam restart - the control is ONE parameter
                                    # conceptually, so its optimizer state should
                                    # persist like its value (warm start) does.
    anneal_stale: float = 0.0       # >0: multiply the NEXT window's initial alpha by
                                    # this on every no-improvement commit (slow 1.15x
                                    # recovery on improvement, floor 0.05). The late-
                                    # run zigzag forensic (2026-09-02): per-window
                                    # Adam restarts overshoot-return across windows
                                    # near the optimum (37.5% of particles reverse
                                    # direction each commit, ears just make it
                                    # visible); a plateau-scheduled step is the
                                    # optimizer-side cure, not state damping.
    gauss_in_objective: bool = True # False: the gauss loss is built (dressing + d_gauss
                                    # telemetry) but the WINDOW render channel stays the
                                    # silhouette - the global solver is then byte-identical
                                    # to the flagship (t1 forensic: with gauss in the
                                    # objective and gates on d_sil, every candidate from
                                    # ~anim 20 regressed the fixed merit by 11.6% and was
                                    # brake-rejected for the rest of the budget).
    surface_mask_objective: bool = True  # False: surface_w is used for gauss parents /
                                    # dressing only, NOT to mask the window's render
                                    # covector (keeps the flagship window untouched).
    local_dress_iters: int = 0      # Tier D (design v2 §4): >0 = post-gate dressing
                                    # solve iterations on accepted commits. 0 = off.
                                    # The stage-0b benchmark SELECTS this from
                                    # {5,10,20} (largest whose p50 overhead <= 50%
                                    # of the global commit p50) — never hand-tuned.
    dress_cap_frac: float = 0.5     # world-space cap ||F.off|| <= this x h_src
                                    # (frozen source-surface median NN spacing);
                                    # A/B {0.5, 1.0} at stage 1b, pick-by-rule.
    work_telemetry: bool = True     # record render/phys linearized work even headless
    layer_relax: bool = False       # outer-layer relaxation PROJECTION in the forward model (kernels
                                    #   k_layer_resid/k_layer_project; docs/surface_gradient.md §6):
                                    #   the rough plane residual removed over one window (1/T per step)
    layer_frac: float = 0.0         #   projection fraction per step: 0 = 1/T (the rough residual
                                    #   relaxed over one window, e^-1 per window, as the bonds re-join);
                                    #   1 = a hard per-step constraint on the rough residual
    layer_ctrl: bool = False        # POSITION-MODE CONTROL CHANNEL (docs/surface_gradient.md §7): a
                                    #   per-window normal displacement leaf u on the outer layer, applied
                                    #   1/T per step in k_layer_project; the render covector reaches it
                                    #   without the grid's low-pass. |u| <= one spacing per window.
    sil_kernel: str = "cic"         # splat kernel of the silhouette / shading rasterisers: 'cic' (bilinear,
                                    #   2x2, derivative discontinuous at pixel edges — neighbouring particles
                                    #   get different, even opposite, pulls) or 'quad' (quadratic B-spline,
                                    #   3x3, the lowest order with a continuous derivative; docs/
                                    #   surface_gradient.md §7 covector rough share). losses.silhouette.set_kernel
    layer_ctrl_smooth: bool = False # the u channel's Adam STEP projected onto the layer's smooth subspace
                                    #   with the relaxation's own W (h = 2 spacings, no new constant): the
                                    #   rough part of the step is what the relaxation undoes one step later
                                    #   (docs/surface_gradient.md §7); a search-direction transform
    pbr_denoised: bool = False      # G1: the shading target rendered from the TARGET's reconstructed
                                    #   surface (Poisson mesh normals, surface weights) instead of the
                                    #   noisy density gradient; the morph's normals on a render-pixel
                                    #   grid blurred by 1.5 spacings (the renderer's density)
    layer_k: int = 24               #   same-side layer neighbours of the plane fit
    layer_h_sp: float = 2.0         #   Gaussian weight width in spacings (the layer's thickness)
    layer_F: bool = False           # P3 (docs/final_plan.md 2; kernels.k_layer_F): the u channel THROUGH the
                                    #   deformation gradient: F <- (I + grad delta) F on the layer each step, so a
                                    #   rough u is a strain the stress resists and the adjoint reaches u through F
    layer_F_depth: float = 1.0      #   depth of the normal extension in spacings (the layer below did not move);
                                    #   0 = tangential gradient only (no normal term)
    layer_gate: bool = False        # P2 (docs/surface_gradient.md 12; surface_recon.layer_u_gate): u may act only
                                    #   where the particle-scale density residual at the window start exceeds
                                    #   nsig x the Poisson shot-noise floor at the surface scale (sigma_sp)
    layer_gate_sigma_sp: float = 1.5  #   the renderer's / level set's blur (spacings)
    layer_gate_nsig: float = 2.0      #   standard deviations above the floor
    layer_gate_geom: bool = False   # the geometric gate (docs/surface_gradient.md 15; surface_recon.layer_u_gate_geom):
                                    #   u may act only where the target's outer layer is within layer_gate_geom_cells
                                    #   MPM cells of the particle at the window start (the sub-grid residual)
    layer_gate_geom_cells: float = 1.0  #   the radius in MPM cells (dx = the grid's resolution, not a tuned length)
    layer_u_render_only: bool = False  # P1 (docs/surface_gradient.md 13): the u leaf receives the render channel's
                                    #   gradient only (lambda x g_render); the physics gradient through u is dropped
    grad_dump: str = ""             # directory: per window, dump the terminal covectors of each
                                    #   channel (physics / silhouette / shading), their control
                                    #   (leaf) gradients and the linear-response rollouts of each
                                    #   channel alone (docs/surface_gradient.md; probes/grad_stage.py)
                                    # (REFUTE M18: the P-render metric was None in
                                    # every run without --live_port)
    patience: int = 5               # commits without tol improvement before freeze
    reject_stop: int = 3            # consecutive rejected candidates (any kind) that end the run at
                                    # its best commit — an early stop. v7 forensics: every run's
                                    # rejections are a terminal streak of 2–5 (the plateau); the C
                                    # replayed one rejected step 12 times. An isolated rejection is
                                    # followed by descent (A 177 -> 217, beast 264 -> 274), so a
                                    # streak, not a count, is the stop signal.
    tol: float = 0.003              # relative improvement threshold on the tracked loss
    persistent_rest_volume: bool = True  # compute Vp0 once at the sampled source state
    best_truncate: bool = True      # deliver the trajectory up to its best shape-merit
                                    # commit (continuous, no snap); the flat-valley
                                    # wandering tail is dropped from frames, kept in
                                    # history (b4: best d_vol 62 at a435, final 158)
    reattach: bool = False          # conservative particle resampling at the commit: a particle
                                    # whose grid cell is not connected to the body (fragment
                                    # mask on the dilated occupancy) is not a continuum element
                                    # and is merged back onto the nearest body particle (its
                                    # state copied); mass conserved, 0 fragments by construction
    bonds: bool = False             # MATERIAL BONDS for decoupled particles (forward model,
                                    # kernels.k_bond_force, method.md eq 24): one-sided tension
                                    # bonds to the frozen source neighbours (coh_k), rest length
                                    # re-based at the window start, stiffness (6/K)(λ+2μ) r, acting
                                    # only on particles with no other particle in their 3^3 cells.
    continuity: bool = False        # DISCRETE-CONTINUITY line-search feasibility (2026-09-16,
                                    # the mass-ejection mechanism): a step is accepted only if
                                    # every particle's window-end velocity relative to its frozen
                                    # material neighbours is <= one local spacing per window
                                    # (sp_i / (T*dt)) or no worse than the reference rollout.
                                    # Scale from the discretisation; no tuned constant.
    eject_veto: bool = False        # MASS-EJECTION VETO (2026-09-16): reject a candidate
                                    # window that increases the number of ISOLATED particles
                                    # (nearest neighbour > eject_iso_k target spacings);
                                    # shrinks the step like a brake reject. Needs outer_merit.
    eject_iso_k: float = 6.0        # 3.0 rejected EVERY early window (the sphere's surface
                                    # dilutes legitimately to ~3 spacings; a detached particle
                                    # sits at > 6): dragon/armadillo froze at anim 14, 2026-09-16
    w_esc: float = 0.0              # ESCAPE-VELOCITY hinge on the window-end velocity relative
                                    # to the frozen source neighbours (coh_k):
                                    # mean relu(|v_i - mean_j v_j| - esc_k*median|v|)^2 / thr^2
    esc_k: float = 3.0
    archive_stride: int = 1         # keep every k-th per-step state in frames/F_frames
    outer_merit: bool = False       # fixed-scale trust gate for production runs
    outer_merit_tol: float = 1e-4   # relative sufficient decrease required for a commit
    outer_gate_move_frac: float = 6e-3 # RETIRED as latch evidence (s1: reachable at 10% of descent; s3: pacing makes every move small) — kept for provenance
    outer_gate_merit_max: float = 0.55 # normalized fixed merit required before latching
    anneal_on_reversal: float = 0.0  # >0: multiply the next window's alpha by this
                                    # when the accepted commit REVERSES the previous
                                    # commit's displacement (reversal_cos < -0.2) -
                                    # sign-change step control, no rejection, no
                                    # patience impact (v4: unlatched reversal REJECT
                                    # froze the run at a56 via patience)
    outer_reversal_always: bool = False  # apply the low-gain reversal reject WITHOUT
                                    # the plateau latch (v4 test, 2026-09-04: the solid-
                                    # target tail still zigzags, rev-cos -0.44 / 61%)
    outer_reversal_cos: float = -0.2  # block low-gain cross-window reversals
    outer_reversal_gain: float = 5e-3 # reversal is allowed above this relative merit gain

    # ---- material base / misc ----
    young: float = 1.4e5
    poisson: float = 0.2
    device: str = "cuda"

    history: list = field(default_factory=list)
