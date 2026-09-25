# PhysMorph method contract (v3)

This file is what the code cites as `docs/SPEC.md` (historical name). **Equation numbers
(1)–(14) and section anchors §3.3 / §4.2 are load-bearing** — docstrings in
`physmorph/mpm/*` and `physmorph/losses/*` reference them; keep them stable.

---

## §1 Continuum + discretisation

Elastoplastic MLS-MPM, fixed corotated elasticity, multiplicative plasticity. Particle
state: x, v, APIC affine C, total deformation F, plastic rest state Fp, per-particle Lamé.
Domain grid: `MPMParams` (default dx=0.5, 64³, dt=1/240, F-smoothing 0.955 — the legacy
morphing line). Oracle: `legacy/DiffMPMLib3D` (Xu et al.), ported kernel-for-kernel.

## §2 Constitutive (code: mpm/constitutive.py)

```
(1)  λ = Eν/((1+ν)(1−2ν)),  μ = E/(2(1+ν))
(2)  ψ(F) = μ Σ(σ_i−1)² + λ/2 (J−1)²                     (fixed corotated energy)
(3)  P(F) = 2μ(F−R) + λ(J−1)J F^{−T},  R from proper SVD (reflection-repaired)
```

## §3.3 One MPM step (code: mpm/kernels.py, mpm/step.py)

```
(3') F_e = (F + dFc)·Fp^{−1};  stress uses P(F_e)         ← the CONTROL enters here
(4)  P2G mass:      m_g = Σ_p w_gp m_p                     (cubic B-spline 4³ stencil)
(5)  P2G momentum:  mom_g = Σ_p w_gp [m_p v_p(1−dt·drag) + G_p(x_g−x_p)],
                    G_p = −C0·dt·V_p·P(F_e)(F+dFc)^T + m_p C_p,  C0 = 3/dx²
(6)  grid:          v_g = mom_g/m_g + dt·f_ext             (+ floor contact, optional)
(7)  G2P:           v_p = Σ w v_g;  C_p = C0 Σ w v_g(x_g−x_p)^T  (+ η damping: eta_sym
                    damps only sym(C) — objective; eta_mode 1 = exp(−dt·η), dt-consistent)
(8)  v_max clamp    OFF in every blessed path (prm.v_max = 0)
(9)  update:        F ← blend[(I + dt·C)(F + dFc), F; smoothing];  x ← x + dt·v
(10) volumes:       V_p = m_p/ρ_p from a one-time mass P2G at rest
```

## §4.2 Differentiable rollout (code: mpm/traj.py, mpm/function.py)

Per-step arrays on a `wp.Tape`; a torch `autograd.Function` bridges leaves
`{dFc[t] (T,N,3,3), λ_i, μ_i (N,)}` → `(x_T, F_T, v_T)`. One backward yields every
per-layer control gradient at once (the reason autodiff beats the layer-by-layer C++).
Grads are read per-leaf only if that input required them. Constant-sequence ≡ shared
control is gate G1a; a finite-difference check of dL/ds is gate G1b.

## §5 Dynamic formulation (v2 blessed path; code: pipeline/optimizer.py, runner.py)

Window loss over horizon T (terminal):

```
(13) D_vol   = ½ Σ_cells [log(m+1) − log(m_tgt+1)]²        (mass matching, Xu et al.)
(14) D_render= mean_views mean_px [w_hole·relu(α_t−α)² + w_spray·relu(α−α_t)²],
               α = 1−exp(−k·CIC(x));  views = azimuth ring × elevations {0,±φ}
L = D_vol + λ_R·D_render + w_kin·mean|v_T|² + w_ctrl·Σ|dFc|²/(TN)
    + w_box·mean relu(|x_T|−r_box)² + w_mat·mean|s|²
```

- **λ_R** = α_λ·‖∇phys‖/‖∇render‖, estimated ONCE per window (single objective for the
  line search), EMA across windows; α_λ=0 ⇒ the physics-only arm, same code path.
- **Four render→physics channels**: dFc (adjoint), material s=(s_λ,s_μ) with
  Lamé = base·eˢ (same adjoint), Fp (assimilation of the optimised motion, §8), v_T
  (arrive-at-rest term).
- **Step control**: hand-rolled Adam over leaves + backtracking line search (reject +
  restore leaves AND moments) + adaptive α; acceptance requires a FINITE rollout state
  (NaN particles vanish from splats and can fake a lower loss). Candidate evaluations use
  a plain no-tape rollout.
- **Outer loop**: FULL state promotion (x, F repaired-and-counted, v, C); plastic
  assimilation; plateau freeze on RAW components (λ-free); guard counters must read zero.
- **The box leash** exists because pixels are local: a particle outside every viewport and
  the loss grid has exactly zero data gradient — escape must remain visible to the
  objective.

### §5.1 Surface-Gaussian render channel

The production render representation is distinct from the volumetric MPM discretisation.
A frozen material-coordinate surface score selects the source Gaussians once; the target
has its own independently selected surface set.  The actual Gaussian loss uses the same
rest radius as the viewer and

```
Sigma_p = sigma0^2 F_p F_p^T.
```

Thus rendering supplies both `dL/dx_T` and `dL/dF_T`.  The surface mask is applied once to
that terminal covector, then the complete MPM adjoint maps it to every `dFc[t]`.  It does
not mask physics particles, MPM grid transfers, or the control gradient after the
pullback.  Production uses about 50% of the 20k material points as render primitives,
`sigma0 = 1.0 * median target-surface NN = 0.04032`, and a hybrid silhouette/Gaussian
render objective.

Viewer/export opacity has one additional target-free validity check based on frozen
source-material neighbours.  It can suppress a Gaussian that no longer has continuum
support, but is deliberately outside the differentiable objective; simulation state and
all raw metrics remain unchanged.

## §6 Sobolev / grid-GS render direction (v3; code: pipeline/grid_smooth.py)

The raw ∂D_render/∂x_T is a Jacobi-style signal (all particles react to the same pixels at
once; zero in coverage pockets; high-frequency dominated). Before the adjoint pullback we
optionally replace it with a Sobolev-metric gradient: scatter to the loss grid (CIC),
red-black screened-diffusion sweeps `(I+κL)u = ĝ`, gather, rescale to the raw norm. The
smoothed field seeds `∂x_T`-backward, so the pullback to dFc still goes through the exact
MPM adjoint — this is a **search-direction transform, not a physics change**; nothing here
needs to be differentiable. Lineage: Sobolev preconditioning (Repulsive Curves) and
Preconditioned Deformation Grids (PG 2025). Arm: `render_gs` (+ warm-started dFc).

This is an experimental arm, not the production configuration.  The current grid smoother
drops the covariance (`F`) component of the exact Gaussian covector, and the kNN/H1 control
variant worsened all tested shape/deformation criteria.  Both are therefore disabled for
the final run.

## §7 VBD-MPM quasi-static arm (v3; code: vbd/solver.py, pipeline/runner_vbd.py)

Morphing is quasi-static at heart (the dynamic path spends w_kin forcing rest). The VBD
arm makes that structural: each commit solves, on the ACTIVE grid nodes (CIC stencil
frozen at commit start; rebinned between commits),

```
u* = argmin_u  Σ_p V_p ψ_SNH((I+∇u(x_p))·F·Fp^{−1}) + D_vol(x+u) + λ_R·D_render(x+u)
               + w_box·leash
```

ψ_SNH = stable Neo-Hookean (Smith 2018; SVD-free, defined for all J — the energy VBD
itself uses). Solver = VBD transplanted from mesh vertices to **grid nodes**: 2-color
parity blocks, per-node diagonal elastic preconditioning, per-color backtracking on the
total energy, a trust-region-like global step that grows 1.5× on clean sweeps (the data
terms' curvature is unknown a priori), stop at ‖∇E‖ ≤ tol·‖∇E₀‖. Then x ← x+u,
F ← (I+∇u)F, assimilation as in §8. Rest is by construction — no velocities exist.

Render information reaches the physics as a genuine **energy of the equilibrium**; λ_R is
balanced per commit from gradient norms on u. Differentiability (for a future material/
system-ID channel): validated in `scripts/probe_gs_differentiability.py` — unrolled and
IFT-adjoint gradients both exact (<0.1%) at convergence, biased when unconverged ⇒ solve
tolerance is a *correctness* gate.

This family remains a research comparison.  A surface-only VBD/GS post-correction is not
composable with the dynamic production path because changing `x/F` without the matching
inertial solve and `v/C` update breaks the state contract.  A future integration must be a
complete implicit MPM integrator, or use an SPD VBD/PD solve only as a control-space
preconditioner followed by the unchanged rollout and line search.

## §8 Plastic assimilation (both families; code: plasticity/assimilation.py)

```
F_e = F·Fp^{−1} = R_e·S_e  (per-particle polar) ⇒ Fp ← clamp_sv(S_e^η·Fp)
⇒ F_e_new = R_e·S_e^{1−η}   EXACTLY
```

Rigid motion is a strict no-op; dilation is assimilated in full; fixed-corotated energy
decreases monotonically (tested). Rejected precursors (measured, in git history): OT
`update_fp` (symmetrised Jacobian fabricates strain from rotation; isochoric ⇒ blind to
dilation) and displacement-field polar assimilation (mismatches the dFc-inflated F —
commit-boundary stress spikes, kin 66→509).

## §9 Conditioning & guards (code: mpm/conditioning.py)

`condition_F(clamp=False)`: repairs ONLY non-finite rows and SVD reflections, both
counted; healthy F returns bit-exact. No silent singular-value projection anywhere in a
blessed path. Guard counters (domain clamp, NaN x/v/C, F resets/flips, any-step
inversions) are containment + telemetry: **a fired counter invalidates the run** (gate G2).


## H⁻¹ mass balance (2026-09-04, REVISION 3 amendment)

Every shape term above is local (D_vol reaches one cell beyond a mismatch, the DT term
vanishes inside the target, the silhouette sees outlines), so a surplus region far from a
deficit receives no signal: the solid hero converged to a wrong fixed point with the ears
30% under-filled. The same residual r = ρ − ρ_t is therefore also measured in the H⁻¹ norm,

    D_h1 = ½ ‖r‖²_{H⁻¹} = ½ Σ_k |r̂_k|² / |k|²  = ½ ∫ |∇φ|²,   ∇²φ = r,

computed by an FFT Poisson solve on the (×2 zero-padded) loss grid; the DC mode is dropped
(mass is matched by construction). ∂D_h1/∂x_i = −m_i ∇φ(x_i) through the CIC kernel (φ is
negative in deficits: the pull of every deficit cell on every surplus particle). The code
works in grid units (mass per cell, k in cycles per cell), so raw d_h1 values are comparable
only at a fixed loss_res. Same minimiser as D_vol (r ≡ 0); note D_vol is the log-mass form,
so the two are not the same functional away from the minimiser.
Weight: `w_h1` × a one-shot scale equalising its POSITION-gradient norm with D_vol at the
source (`w_h1 = 1` → parity; the control-space norms after the MPM Jacobian differ, a known
caveat). The scale is preserved across the c2f target rebuild (REFUTE 2026-09-04 F1). The
term is added OUTSIDE the physics core, like the W1 term, so the λ-balancer numerator and
the PCGrad reference are unchanged by it (REFUTE F3, the W1 precedent); it is tracked per
commit (`d_h1`) and counted in the plateau/merit/delivery tracks when active.

Self-energy correction (REFUTE Opus 2026-09-04 F1). The CIC self-energy of a particle,
½ m_i² Σ_ab w_a w_b G(c_a−c_b), depends only on its sub-cell position (minimum at the cell
centre) and not on the ambient density: an uncorrected H⁻¹ term is a density-blind lattice
attractor, one per loss cell (measured self-force constant from 1 to 54 particles per cell,
44× D_vol's at mid-run) — precisely the sub-cell clustering scale of docs/floaters.md. D_vol's
log form suppresses its own self-term ~750× at flagship occupancy; the linear residual does
not. `d_h1` therefore subtracts the self-energy analytically (the P3M self-force correction,
Hockney & Eastwood), using the 8×8 stencil Green's matrix from the same FFT kernel; the
self-force is exactly zero at every sub-cell position (tests/test_h1.py). The term sits in
the physics core (Opus F3 over Codex F3: control before attribution; attribution by the v7c
arm and the paired-at-equal-d_vol census); its per-window ratio s·|g_h1|/|g_vol| is logged
(`h1_ratio`) because the source-calibrated parity drifts along a morph (Opus F2).


## §10 Render-controls-physics contract (2026-09-14; code: pipeline/control_basis.py, mpm/kernels.py, pipeline/grad_combine.py)

Full rationale, literature and the pre-registered ladder: `docs/render_controls_physics.md`.
Every item is opt-in; with all flags off the §5 path is unchanged.

```
(15) F_g,{t+1} = (I + dt·C_{t+1}) F_g,t                (geometric F: velocity-gradient
                                                        transport only — no dFc, no
                                                        smoothing; render_F_geom renders
                                                        Sigma = sigma0^2 F_g F_g^T from it)
(16) dFc[t,p] = sum_k w_k(x0_p) · sum_j B_tj · C[j,k]    (control basis: trilinear nodes on a
                                                        G^3 grid over the window start,
                                                        piecewise-linear in time over K
                                                        knots; grid=0 -> identity)
(17) L += w_kin_running · mean_t mean_p |v_t|^2         (running kinetic; every v_t is an
                                                        output of the extended bridge)
(18) D_vol^density = 1/2 · (1/n_support) sum_cells [log(1+m/m_ref) - log(1+m_t/m_ref)]^2
                                                       (loss_units="density"; same
                                                        minimiser as (13); fixed weights
                                                        divided by the MEASURED source
                                                        ratio D13/D18, gradient constants
                                                        by |grad D13|/|grad D18|; REFUTE
                                                        2026-09-15 F1)
(19) Chebyshev sweep (grid_smooth): u_{k+1} = w_{k+1}(g(S(u_k)-u_k) + u_k - u_{k-1}) + u_{k-1},
     w_1 = 1, w_2 = 2/(2-r^2), w_{k+1} = 4/(4 - r^2 w_k), r = (kappa/(1+kappa))^2;
     applied to the (N, 3+9) field [dL/dx_T, dL/dF_T] before the adjoint pullback
(20) composite direction: grad_project_mode in {render (one-sided PCGrad, legacy),
     phys (mirror), cagrad (Liu 2021), blend (physics-anchored magnitude)}
(21) L += w_kin_var · mean_p [ mean_t |v_t|^2 − |mean_t v_t|^2 ]   (velocity variance over
                                                        the window: zero for constant-
                                                        velocity motion, positive for a
                                                        reversal — the measured window-
                                                        locked limit cycle; 2026-09-15)
```

Contract statements (tested on warp-CPU, `tests/test_ext_bridge.py`,
`test_control_basis.py`, `test_chebyshev_smooth.py`, `test_render_controls_physics.py`):
(15) at dt=0 the stored F absorbs (1−s)·dFc while F_g stays exactly I; (16) is linear with
partition of unity to 1e-6 and zero gradient on unsupported nodes; (17)/(15) gradients
match central differences to <5 %; (19) reaches a 25–200× lower error to the converged
solution than plain sweeps at 12–40 iterations (κ 4–50) and the same limit; the archive keeps the PHYSICS F in `F_frames` (metrics,
assimilation) and F_g at accepted commits in `Fg_commits`.

### 10.5 Support-gated APIC (forward model, opt-in; 2026-09-15; code: mpm/kernels.py `k_cell_count`, `k_support_gate`; mpm/step.py `gate_omega`, `nominal_support`)

```
(22) ω_p = S((n_p/n₀ − r_lo)/(r_hi − r_lo)),   S(s) = c²(3 − 2c), c = clamp(s, 0, 1)
     P2G momentum (5) with  G_p = −C₀ Δt V_p P_p F_eff,pᵀ + ω_p m_p C_p
```

`n_p` = number of particles in the 3³ grid cells around p's cell at the current step; `n₀` =
median of `n_p` over the source cloud, fixed ONCE per run (`MPMParams.gate_n0`, set by the
runner) so every window gates alike; `r_hi ≤ r_lo` turns the gate off and (22) reduces to (5)
exactly (ω ≡ 1). A particle in a depleted neighbourhood hands the grid PIC momentum only: the
affine term `m C (x_g − x_p)` of a front particle (steep velocity gradient at a fringe) is what
gives the empty-side nodes an outward velocity, and with no other particle on those nodes the
grid cannot pull it back (numerical fracture, Yue 2015). Source: Yao & Zhao 2026 (arXiv
2603.03860, "support-gated APIC"); ASFLIP (Fei 2021) for the fringe diagnosis.

Adjoint: ω is piecewise constant in x (∂ω/∂x = 0 a.e.), so it is computed outside the tape
per step and the P2G adjoint reads it as a constant; ∂(mom_g)/∂C_p = ω_p m_p (x_g − x_p) w_gp.
Contract (`tests/test_support_gate.py`, warp CPU): gate-off and a saturated gate are
bit-identical to plain APIC; an isolated particle carrying C keeps it under APIC (APIC
reproduces an affine field exactly on a lone particle: `Σ w d dᵀ = dx²/3 I`) and loses it
under the gate while its translation is unchanged (a lone particle cannot accelerate itself
either way: `Σ w d = 0`); the gated adjoint matches central FD within 5 %. Only physical
variables move — the gate changes how the material transfers momentum, not the state.

### 10.6 Discrete continuity (line-search feasibility; the mass-ejection mechanism, 2026-09-16; code: pipeline/optimizer.py `cont_check`)

```
(23)  for every particle i, with N(i) its frozen source-material neighbours (coh_k) and
      sp_i = mean_{j∈N(i)} |x_j − x_i| at the window start:
          | v_i(T) − mean_{j∈N(i)} v_j(T) | · (T·Δt)  ≤  sp_i
      A line-search candidate (control α·d) is ACCEPTED only if (23) holds wherever it held
      for the iteration's reference rollout (the state before the step); otherwise the step
      is rejected and α halved, exactly like a non-finite or orientation-reversing state.
```

Meaning: no particle may outrun its own material neighbourhood by more than one local
spacing per window — the discrete statement that the deformation increment is continuous at
the particle scale. The scale is the discretisation (local spacing, window length T·Δt); there
is no tuned constant. Measured on the 40k archives: ejected particles run at 5–6 wu/s
relative to a body p95 of 0.25–0.28 wu/s, i.e. 4–5× above the limit sp/(T·Δt) = 0.108/0.083
= 1.3 wu/s at 40k (0.83 wu/s at 150k); coherent motion including thin-feature stretching
stays an order of magnitude below it. Because acceptance is decided inside the line search,
no accepted commit can launch a particle, and a launch already present in the reference
(from an earlier window) is not made worse — the loss then pulls it back. Only physical
variables move; the rule constrains which control increments are admissible, not the state.

Rejected parameter routes (docs/experiments.md 2026-09-16 ladder): a window-level isolation
veto (freezes on legitimate stretching: k = 3 and k = 6 both replayed the same candidate to
a freeze), an escape-velocity hinge (a weight), and a G2P speed cap `v_max` (a cap; froze
the dragon at 2 min with 17 % far particles). They remain opt-in for the record.

### 10.7 Material re-coupling of decoupled particles (forward model; the mass-ejection mechanism, 2026-09-16; code: mpm/kernels.py `k_p2g` / `k_update`, mpm/traj.py `_bond_args`)

```
(24)  decoupled(p) :⇔ no other particle in the 3^3 grid cells around p's cell
      (then Σ_g w_gp (x_g − x_p) = 0: the grid cannot act on p)
      N(p) = the K frozen source-material neighbours (coh_k), r_pj = |x_j − x_p| at the
      window START (re-based every window: plastic)
      P2G (5) for a decoupled p uses the MATERIAL velocity  v̄_p = (1/K) Σ_j v_j  in place of v_p
      advection (9) for a decoupled p adds the bond projection
          x_p ← x_p + Δt v_p + (1/K) Σ_j (|x_j − x_p| − r_pj)_+ (x_j − x_p)/|x_j − x_p|
```

Why this and not a penalty, a cap or a spring: the measured ejecta (docs/experiments.md
2026-09-16) are not launched — their velocity relative to their material neighbours is
0.5–1.1 wu/s (below the continuity limit) while the absolute speed is 2–5 wu/s; they drift
away over many windows after leaving every other particle's stencil, which is numerical
fracture (Yue 2015 §2): once alone, nothing in the continuum reaches them. (24) restores
the transfer the grid would have provided, through the material: the velocity a decoupled
particle contributes is the one it would have received from shared nodes (material PIC),
and its position is projected toward its bonds' rest lengths (position-based dynamics, the
PB-MPM / Lagrangian-bond lineage of Jiang 2017, Han 2019, Lewin 2024). The decoupling test
is binary and comes from the discretisation; the projection is complete (no stiffness, no
weight, no threshold). Coupled particles are untouched (the rollout is bit-identical to
(5)+(9)); both operations are gathers, so the Warp adjoint is exact
(`tests/test_material_bonds.py`: FD within 5 %). The decoupling test (24) is a property
of the CURRENT state and is evaluated every step (2026-09-18, mpm/traj.py `_bond_args`,
kernels `k_frag_step`: the 3^3-cell count of x_t, outside the tape, OR-ed with the runner's
commit-time fragment mask); until then the mask was fixed at the window start, and the
single-particle leaders of the expansion phase — which clear the gap inside one window —
were bonded a window too late and left in empty space (150k bob without the net: 54
particles, 1.3–3.7 wu out, static). Momentum bookkeeping: the decoupled
particle's momentum change is not returned to the neighbours (one particle against the
body; recorded, not hidden). An explicit bond SPRING with stiffness (6/K)(λ+2μ) r was
implemented first and rejected: integrated explicitly with a multi-wu extension it is
unstable (dragon: 9 % of particles far, frozen at anim 23).

### 10.8 Transport-paced target for the cell sum (the mass-ejection mechanism, 2026-09-17; code: pipeline/optimizer.py `phys_loss == "ot_pace"`, losses/ot.py)

Cause (H3, confirmed on 19 meshes): the cell sum (13) rewards a lone particle in an empty
target cell with the largest marginal gain, so a surface particle facing a distant unfilled
feature is pulled away alone, decouples numerically (§10.6-10.7) and drifts. A transport
loss has no such reward (mass moves as a flow) but cannot fill at the particle scale: the
entropic map image sits ~0.9 spacings inside the target regardless of the sample count, and
the fixed cell-sum merit/tracker then stops the run (`ot40b`, `ot40g`).

Formulation. Per window, from the start positions x0:
  1. Sinkhorn dual between a fixed uniform subsample of n = ot_samples particles and n
     target samples, cost |x - y|^2, eps = (h_s)^2 with h_s = particle spacing x (N/n)^(1/3)
     (the spacing of the sample sets the plan is computed on); geometric eps-scaling from the
     squared target diameter, stopped at the L1 marginal error ot_tol (1e-2); the potentials
     warm-start the next window.
  2. Out-of-sample entropic map T(x_i) = softmax_j((g_j - |x_i - y_j|^2)/eps) . y for all N
     particles (row-normalised barycentric projection), debiased by the same map onto the
     subsample itself (T - T_self), and averaged over the k material neighbours inside one
     blur radius (k from the blur volume; the continuum map is smooth, the sampled one is not).
  3. Paced target: x_int,i = x0,i + min(1, h_s / |d_i|) d_i with d_i = T(x0,i) - x0,i, i.e. the
     cloud advected along the plan by at most one blur radius per particle (McCann
     displacement interpolation, one plan per window), rasterised with the loss CIC splat
     into the window target grid m_tgt^(k).
  4. The window objective is (13) in density units against m_tgt^(k); the outer merit, the
     brake and the convergence tracker keep reading the FIXED target grid.

Properties. Every particle is asked to move at most one blur radius toward where the plan
puts its mass, so no cell far from a particle can reward it for leaving the body; when all
|d_i| <= h_s the paced target is the image cloud (the transport end state) and the term is
the ordinary fill. No new constant: the pace is the plan resolution h_s, the neighbour count
follows from it, the tolerance is the standard Sinkhorn stopping rule. Cost: O(n^2) per
sweep independent of N (~1 s per window at 40k with two runs per GPU).

Evidence (40k, no re-attachment, docs/experiments.md 2026-09-17): end fragments bob 85 -> 1,
dragon 41 -> 2, armadilo 12 -> 0 with silIoU +3..+13 points and chamfer within 0.008;
falsified alternatives on the way: OT as the loss (holes, tracker stop), a transport leash
on the cell sum (weak: oscillation; strong: tears the bulk), denoised leash anchors (same).

### 10.9 Particles per cell and the numerical-fracture gap (2026-09-17; the mesh-size lever)

A particle leaves the continuum when it shares no grid node with any neighbour. With the
quadratic B-spline stencil that needs a gap of about one cell between the particle and the
body. At 8 particles per cell (cell = 2 spacings) a one-spacing lead, which the density
gradient gives a surface leader in a few windows (dfc clip 0.02 per window), is enough; at
27 particles per cell (cell = 3 spacings, the 3 x 3 x 3 arrangement that is the standard
high-quality MPM sampling) the same lead has to be three spacings, which the elastic
neighbourhood does not allow. Measured at 40k, no re-attachment, log density loss
unchanged: end fragments dragon 41 -> 0, bob 85 -> 2, armadilo 12 -> 0, with silIoU
+10 / +11 / +2 points and chamfer within 0.006; the MPM grid is 41^3 instead of 59^3, so the
runs are faster. This is a discretisation choice (dx = (V ppc / N)^(1/3) with ppc = 27), not
a per-shape constant, and it supersedes the transport pacing of 10.8 as the ejection
remedy: the pacing removed the far-cell reward but paid with a paced surface; the coarser
cell removes the fracture the reward exploits. The lever the user named ("mesh size").
Ladder record (same day, 40k, dragon / bob / armadilo): loss_res 64 -> 32 halves dragon's
fragments only (23 / 87 / 5); a linear density residual is not a drop-in (the merit,
tracker and unit calibration are log form; runs freeze).

Mesh size vs ppc — the definition (user question, same day). The mesh size is not an
independent knob: with the particle spacing h = (V/N)^(1/3) the MPM cell is
dx = (V ppc / N)^(1/3) = h ppc^(1/3), so ppc 8 / 27 / 64 mean a cell of 2 / 3 / 4 spacings,
and at fixed ppc the cell shrinks as N^(-1/3). The decoupling gap is one cell (no shared
node under the quadratic stencil), i.e. ppc^(1/3) spacings; the lead a surface leader can
build per window is bounded by the control clip, so the boundary between "fractures in a
few windows" (2 spacings) and "the elastic neighbourhood holds" (3 spacings) is a measured
fact, not a derivation. The loss cell ldx = bbox / loss_res is a separate discretisation
and is not the ejection variable (loss_res 32 vs 64 above). Rule: ppc is the discretisation
constant (27 = 3^3), dx follows N; the rule holds while dx <= t_min / 2 for the thinnest
target feature t_min (bunny ear ~0.3-0.4 wu against dx 0.205 at 150k). ppc 64 (4 spacings)
is being measured to close the curve.

### 10.10 Transport-paced cell sum with the cell-wise hand-off; the loss regime (2026-09-17 evening; code: pipeline/optimizer.py `phys_loss == "ot_pace"` + `cfg.ot_handoff`, pipeline/runner.py `phys_loss == "auto"`)

The residual ejection at the cell of 10.9 is the EXPANSION phase: while the sphere spreads
toward the target, the cell sum rewards the outermost particles most (empty cells ahead),
the surface layer accelerates ahead of the interior and sheds isolated particles and, on
thin features, cell-sized chunks (2–5 % of the cloud isolated at the peak, dragon / bob /
beast at 150k). Remedy, no new constant:
  1. Paced target (10.8) with the pace = max(plan blur radius, loss cell): the window
     target is the cloud advected along the transport plan by at most one loss cell per
     particle, so the expansion is a coherent flow with no far-cell reward (a pace below
     the loss cell stalls — the cell sum is blind to sub-cell shifts, C forensic).
  2. Cell-wise hand-off: every DEFICIT target cell within one cell of the occupied set
     (the reach of the CIC gradient) carries the fixed target mass — ordinary fill at full
     strength; excess cells and far cells keep the paced mass (evacuation and transport
     at the pace). The global hand-off (fixed target only once no deficit cell is far) never
     fires on shapes with unreachable tips; cells of the fixed target around a body that
     sits in a target hole re-create the runaway (C), which is why only deficit cells hand
     off.
  3. Merit: the physics component of the fixed-scale outer merit is the Sinkhorn
     divergence to the fixed target (losses/ot `SinkhornPull.divergence`) — what the recipe
     descends, defined on the fixed target, monotone along a transport path where the cell
     sum plateaus; the record's d_vol stays the fixed-target cell sum (the paced loss value
     in the record froze every early ot_pace run through the brake).
Evidence (40k, cell 0.31, no re-attachment; docs/experiments.md 2026-09-17 evening): end
fragments bunny / dragon / bob 0 / 0 / 0 (density recipe 0 / 0 / 2), silIoU 0.961 / 0.965 /
0.974 (0.960 / 0.955 / 0.958), whole-run isolated-particle peak 0.09 / 0.14 / 0.29 % (0.11 /
2.3 / 1.1 %).

Loss regime (`--phys_loss auto`): measured once at the start, the fraction of source
particles whose target cell carries no mass. Above one half (the sphere inside the C's
hole, 68 %) the cell sum has only an outward push and its inertia overshoots the arms
(v2–v6 froze C at 20 windows under every material: poisson 0.0 and 0.45 alike — the
failure is not compressibility); the per-particle transport loss (`ot`) morphs it (silIoU
0.96 vs 0.72). Below one half the paced cell sum of this section applies. The regime is a
property of the discretised problem, not a per-shape setting.

Hole-regime target (`ot`; pipeline/optimizer.py, efc7fa0): the per-particle loss is the
squared distance to a window target, and the target is NOT the raw map image. The raw
image pulls each particle in proportion to its remaining distance, so the particles
farthest behind (the sphere material bound for the C's arm tips, 2–3 wu away) are pulled
hardest, lead the body and fracture — the leader mechanism of 10.6 in per-particle form
(150k: 1900–2700 re-attachments in bursts of 100–440 particles while |v|max sat at 2.5
wu/s; 40k: ~110). The window target is instead the material-smoothed map displacement
(averaged over the k particles inside one plan blur radius at the source, the material
graph of 10.8) bounded to one pace = max(plan blur radius, loss cell) per particle per
window: every particle is asked for at most the move the grid resolves in a window, the
pull is uniform and bounded, and the target still walks the whole map one pace per
window. No new constant; the same pace and neighbourhood as the cell-sum regime.

### 10.10a The limit case C: why detached material is irreversible, and why C keeps the net (2026-09-18)

The control of this model is a stress (the deformation-gradient control dFc enters through
k_stress). The internal stress of an isolated body integrates to zero net force, so a chunk
that has separated from the body cannot be translated by any control — the same reason the
whole body keeps its centre of mass. Ejection is therefore irreversible by construction: the
only external force in the model is the re-attachment step (conservative resampling of
grid-disconnected particles into the body, 10.7), and "no safety net" can only mean "no
fracture". Prevention is complete for single particles (the per-step decoupling test, 10.7)
and for the box band (10.11); it is not complete for the one target whose source lies inside
the target's hole (C): its material must cross empty space to the arms, the per-particle
transport pull leads the arm fronts by more than a cell, and cell-sized chunks (85–105
particles at 150k) come off the fronts and stop where the pull can no longer move them
(the transport map still points inward there — probe 2026-09-18 — the stress simply cannot
act). Everything that slows the front stalls the morph instead: paced target (0.88 / 0.55),
grid-resolved displacement (0.59), material-kNN smoothing (0.79), elasticity assim 0.1 / 0.25
(0.87 / 0.87 at 23 windows), discrete continuity (0.84 at 20 windows), ot_pace + hand-off
(0.56 at 150k). C is reported with the net and its re-attachment count (530 at 150k) as the
model's documented limit.

### 10.11 Domain walls (forward model; the 150k shedding mechanism, 2026-09-17 night; code: mpm/kernels.py `k_grid_op`, `WALL_NODES`)

The MPM grid had no boundary treatment on the six faces of the domain box (only the
optional floor). A particle within the stencil half-support of the box edge (two cells,
cubic B-spline) deposits on and gathers from a truncated stencil — it loses momentum
every step and freezes in that band, and the commit-time clip keeps it there. The band
was a TRAP, and the "chunks" the safety net re-attached at 150k were material stranded in
it: on the 150k C the probe (`output/chunk_origin.py`) found every chunk static at the box
corners, 1.5–3.3 wu from any target point, with 600–900 particles beyond the box leash in
every window of the arm phase; the count of particles clipped at the box (`GUARD clamp`)
tracks the re-attachments across runs — v6 dragon 10706 band hits / 755 re-attachments,
v6 bob 10934 / 486, 150k C 27–59 k / 885–1914, against 0 / 0 for every run that never
reached the band (40k C, teapot, heart, A, nn150 bunny/dragon). The 40k runs never reach
the band; the 150k transients do (the auto domain is the far-field leash plus the two
stencil cells, so the free region ends one cell beyond the target's outer surface).

Walls: on the outermost `WALL_NODES` = 2 node layers of every face the outward normal
grid velocity is zeroed; tangential and inward motion stay free (a separating wall, the
same treatment as the floor without friction). A body that overshoots slides along the
wall and is pulled back by the loss instead of freezing. The constant is the stencil
half-support, not a tuning. Runs whose particles never touch the band are bit-identical
(the wall acts on nodes that carry no mass). Tests: adjoint vs finite differences,
smoke, bonds (26 passed).

Deliverable rule (scripts/render_photoreal.py `--min_cells 1`, `--iso auto`, `--bridge`): an
isosurface component is drawn iff it is a continuum element of the grid — it holds at
least one cell of MASS, ppc = N dx³ / V particles (10.9), counted by voxel-label membership,
and encloses at least dx³. The mass condition is the one that matters: the blurred surface
of a compressed chunk of a few dozen particles encloses more than dx³ at the filament level
and would be drawn as a marble (150k C, 53 frames) while it holds a third of a cell of
material. Closed components whose signed volume has the sign opposite to the body's are
interior cavities (a hollow inside the bunny's ear, 90 frames), removed and counted apart,
never as pieces. The per-frame sidecar records raw, dropped, cavity, bridged and drawn
component counts and the isolated-particle count, so the record is complete. The
isosurface level is not a free constant either: at half the bulk density a neck two
particles across (a teat, a whisker) falls below the level and its bulb renders as a
detached ball although the material is connected at the particle scale (cow at 150k, 62
frames: connected at every cell scale down to 0.07 wu). The level is therefore the one at
which a filament two particles across still renders — a 2×2 bundle of particles at spacing
s blurred by σ has the peak density 4 / (2π σ² s) against the bulk 1 / s³, so iso = 2 s² /
(π σ²) of the bulk, capped at 0.5 (0.28 at the renderer's σ = 1.5 s); a single particle
peaks at s³ / ((2π)^{3/2} σ³) = 0.02 of the bulk and stays invisible, and a cluster that
does reach the level but is smaller than a cell is dropped by the volume rule. Below even
that — a feature one particle across — the isosurface cannot follow, and the physics does
produce such features where the target is thinner than the cell (the cow's teat at 150k:
a 72-particle bulb packed on the target's teat, tied to the udder by a single-particle
thread; one body at every cell scale down to 0.07 wu). For these the renderer draws the
PARTICLE connectivity (`--bridge`): particles the isosurface does not enclose but which
connect the body to another drawn component (a path of particles within 2.5 spacings of
each other, found by union-find with each enclosed component as one node) are drawn as a
filament one particle spacing thick. The rendered topology then follows the particles,
not the threshold; the sidecar records the bridged components per frame, and
`scripts/probes/grid_fragments.py` records, without any renderer, the physical fragments
(clusters sharing no dilated cell with the body, ≥ one cell of material) so that "the
isosurface drew two pieces" and "the physics has two bodies" are separate columns of the
report. v7 (2026-09-18): physical fragments ≥ 1 cell in 0 frames for 18 of 19 targets,
C 6 of 202 frames.

### 10.12 The rendered surface: the outer particle layer, not the density level (2026-09-19; code: render/surface_recon.py, scripts/render_photoreal.py `--surface poisson`, probe scripts/probes/surface_gt.py)

The deliverable surface up to v8 was the marching-cubes level set of the blurred particle
density. Two things were wrong with it, both measured against the true asset mesh mapped
into the cloud frame (the probe reproduces the target cloud bit for bit and carries the
mesh along).

*The bulk.* The "fraction of the bulk density" in 10.11 was taken over the occupied
VOXELS. The blur's halo — three σ of sub-bulk voxels around the whole body — pulls that
median to about half of the density a particle actually sees, so the level was ~0.14 of the
interior density and every surface sat 1.6–1.9 spacings outside the true one (bunny 150k
+1.56, dragon +1.91, cow +1.76 spacings). The bulk is now the median over the PARTICLES
(what a typical particle sees; robust whenever more than half the particles are interior).
The marching-cubes surface then sits +0.5 … +1.2 spacings out, which is the level's own
offset: a level below half the bulk is outside the continuum boundary by construction
(Φ⁻¹(0.28) σ = 0.86 spacing) — the price of drawing a two-particle filament with a level.

*The bumps.* The texture of the level set is the sampling: the target cloud is a voxel
fill of pitch 0.8–1.3 spacings with 0.6–2 particles per voxel drawn with replacement, i.e.
a Poisson-random subsample; blurred at 1.5 spacings its shot noise is ~14 %, which moves
the level set by ~0.6 spacing with a 3-spacing correlation length. No kernel shape
changes this (anisotropic PCA kernels: the same 13° roughness; a smaller kernel resolves
the gaps and is worse), and no mesh filter reaches a 3-spacing bump.

The surface that is right is the one through the OUTER PARTICLE LAYER, denoised as a
surface: (1) the layer = particles whose relative density gradient |∇ρ|/ρ exceeds the
half-space value one spacing deep, φ(1/σ)/(σ Φ(1/σ)) = 0.285 per spacing at σ = 1.5 —
a rule invariant to the local density, so a stretched region of a morph frame (0.6 × bulk
throughout) is not all "layer" (the density rule takes it whole, several particles thick,
and the surface breaks); (2) each layer particle gets a plane by weighted PCA over its 24
layer neighbours (Gaussian weight of two spacings, the layer's thickness), the normal
oriented by the density gradient, and is PULLED onto that plane — the plane-pulling
constraint of 3D Gaussian Triangulation, one MLS projection — which removes most of the
half-pitch jitter and the ~20° noise of the gradient normals; (3) screened Poisson
reconstruction (Kazhdan & Hoppe 2013) of these oriented surfels with the finest octree
cell of ONE spacing — one surfel per cell, the quadratic B-spline averaging three — and
vertices farther than 3σ of the blur from every surfel removed (no particle's kernel
reaches there). A finer cell follows the noise (dragon frame 400: 12.9° → 19.5°, 69
components); a coarser one leaves the surface too far from the surfels. The density keeps
every role it had: the level (for the mass rule's voxel labels), the mass rule, the
cavities and the bridges; only the drawn surface changes.

Measured on the target clouds of bunny, dragon and cow (`surface_gt.py`; spacings and
degrees): mean distance to the true surface 1.16–1.19 → 0.20–0.29 with the bias +1.2 → 0;
completeness 1.2–1.3 → 0.26–0.37; roughness at two spacings 12.3–13.2° → 8.4–9.7° (the true
dragon is 15.4° — its scales are below what any 1.5-spacing kernel resolves); on the morph
frames 12.5–13.2° → 8.5–12.9° with the horns, whiskers and legs present wherever the
particles are. Costs: a frame carries 10–13 raw components instead of one (small flaps of
the open trim and dimples counted as cavities, dropped by the mass rule and recorded in
the sidecar), a filament thinner than the B-spline's reach relies on the bridge rule as
before, and the reconstruction is CPU-bound (~4× the render time). The other candidates
of the pre-registration (IMLS, surfel triangulation, bilateral filtering, PCA kernels) and
their numbers are in docs/experiments.md 2026-09-19.

The deliverable rules of 10.10 read the surface that is drawn, not the level set. With the
Poisson surface: a component's mass is the number of particles it encloses (ray-casting
occupancy of the closed component), the body is the heaviest component, a component below
one cell of mass whose centroid the body encloses is a cavity; "enclosed" for the bridge
rule is occupancy by the drawn mesh with a tolerance of one spacing (the surface passes
through the outer particle layer it was fitted to), and the filament drawn is the shortest
particle chain (Dijkstra) from the body to the linked piece — the literal rule, not every
free particle the chain touches, which mattered once the surface stopped hiding the
expansion-phase spray inside a fat level set. The reconstruction runs in a separate
interpreter per frame (Open3D 0.19 segfaults now and then across repeated calls); a frame
whose reconstruction fails twice is a level-set frame and the sidecar says so. The
marching-cubes path keeps the voxel-label rules of 10.10 so that the v8 numbers remain
comparable.

### 10.13 The volume fill of a non-watertight mesh, and the exterior test of the outer layer (2026-09-22; code: sampling/mesh.py `_fill_ortho_reliable`, `_fill_pockets`, `FILL_MODE`; render/surface_recon.py `exterior_surfels`; probe scripts/probes/fill_check.py)

**The fill.** A target cloud is one jittered particle per filled voxel (10.12, §4 of
docs/surface_gradient.md). The fill of a mesh with holes (bunny, dragon, beast, armadillo,
maxplanck: boundary loops, no inside/outside) was the orthographic rule — a voxel is filled if,
along each of the three axes, a surface voxel lies before and after it. Above a hole in the base
the axis through the hole finds no surface below, so the whole column stays empty; several holes
side by side make a comb of empty and filled columns, a region at 40 % of the bulk density
(bunny: 4.5 % of the interior deeper than two spacings, in a column from the base to the back;
maxplanck 10 %). The density loss then carries the comb into the morph, and the outer-layer rule
of 10.12 draws surfaces on its density steps. Fix: the mesh's boundary loops are projected along
each axis and rasterised into a 2-D footprint (holes filled, dilated by one voxel); a column inside
the footprint is not asked about that axis — the voxel is filled if enclosed along every RELIABLE
axis, provided at least two axes are reliable (a single axis draws one-voxel streaks between
unrelated surfaces, as the old 'base' fill did); otherwise the plain intersection stands. Then
the streak strip of 2026-09-16 and a majority pocket fill (an empty voxel with four or more of
its six face-neighbours filled is interior; iterated), which closes sub-voxel pockets and
tunnels only. A watertight mesh has no footprints and is filled exactly as before (a torus hole
is enclosed along two axes only, and stays open). Acceptance (`fill_check.py`, the share of the
interior deeper than two spacings whose 1.5-spacing count is below 60 % of the bulk): bunny 4.79 →
0.42 %, maxplanck 9.95 → 0.33 %, beast 0.00, armadillo 0.00, the fourteen watertight targets 0.00;
dragon 0.54 % (unchanged, far from any loop — a property of the mesh, recorded). `FILL_MODE =
"legacy"` reproduces the fill of every archive before this date; `surface_gt` retries the
target-cloud reproduction with it, so old runs stay measurable.

**The exterior test.** The outer-layer rule |∇ρ|/ρ ≥ 0.285 per spacing fires at any density
step, inside the material as well as at the surface; the surfels it produces at an interior
step (a target pocket, a compressed region of a morph) make Poisson draw an interior sheet,
which the metric counts (bunny d_95 5.4 spacings with the sheet, 1.4 without). A surface has
empty space on its outward side; an interior step has material there. Per surfel, the
particles in the outward cap of the two-spacing ball beyond 0.75 spacing (2.5 × the layer's own
plane residual, so a rough true surface puts 0.07 particles there on average) are counted; the
cap holds cap_vol / p_vol³ = 7.8 spacings³ × 1.9 = 14.8 particles at the bulk density and 5.9 at
the faintest pocket seen (40 %). A surfel with two or more (a tenth of the bulk cap) is interior
and dropped before the plane pulling and the Poisson solve; a true surfel is lost with
probability 0.2 %, a 40 %-pocket surfel kept with 2 %. The bunny target's reconstruction drops
2 907 of its surfels this way. Discretisation numbers at 40k: spacing 0.135 wu (8-NN median),
p_vol = spacing / 1.24, cap radius 0.27 wu, plane 0.10 wu.

### 10.14 The outer-layer relaxation, the position-mode control channel, the denoised shading reference and stratified sampling (2026-09-19 … 22; code: mpm/kernels.py `k_layer_resid` / `k_layer_project`, mpm/traj.py, pipeline/optimizer.py, render/surface_recon.py `layer_relax_data` / `target_surface_normals`, sampling/mesh.py `sample_volume_stratified`; evidence docs/surface_gradient.md §4, §6–§10, §14)

These four are in the recipe (`scripts/ops/hyde06_env.sh`, frozen 2026-09-22) and were
documented in docs/surface_gradient.md; this section is their formulation contract.

**The outer layer.** At every window start the layer is the set of particles whose
offset from the centroid of their 32 nearest neighbours exceeds half a spacing (the SPH
surface rule; spacing = the median 8-NN distance of the cloud, 0.135 wu at 40k), with the
outward normal along that offset; each layer particle keeps its K = 24 nearest LAYER
neighbours with Gaussian weights of width h = 2 spacings times the normal agreement (same
side only: a sheet two particles thick has both faces in the K nearest), rows normalised to
one. The layer, normals, neighbours and weights are frozen for the window.

**The relaxation projection** (`--layer_relax`). After the MPM update of each step, the
plane residual d_p = n_p · (x_p − Σ_q w_pq x_q) is computed on the layer and the ROUGH part
of it, d_p − Σ_q w_pq d_q (what the neighbourhood mean does not explain), is removed along
the normal at the fraction 1/T per step: over one window of T = 20 steps a rough residual
decays by (1 − 1/T)^T ≈ e⁻¹, the rate at which the material bonds re-join. A hard per-step
projection (fraction 1) is not a contraction and diverges (§6). The projection is a
constraint on positions; F is not updated by it (contact-like). A particle FORCE for the
same purpose was falsified: P2G/G2P average a one-spacing pattern away (2.4 % of a bump
per window). Effect at 40k: the morph's layer plane-residual RMS 0.44 → 0.29 spacings, below
the target cloud's own 0.29–0.34, at −0.5 … −1.2 silIoU points (the silhouette's per-particle
pull no longer buys IoU with noise).

**The position-mode control channel u** (`--layer_ctrl`). A second optimiser leaf, one
scalar per layer particle per window, applied as the displacement (u_p / T) n_p per step in
the same projection kernel, clipped to one spacing per window; its adjoint is the identity
times the physics response of the remaining steps, so the render covector reaches the
layer without the grid's low-pass (through the stress control the covector's neighbour
correlation at two spacings is 0.87–0.89 — the grid's; through u 0.46). Both loss channels
drive it. The factorial of §10 (docs/surface_gradient.md) and the candidate round (§11–§13)
establish: the outline gain of the render channel does not need u (the stress control at
the cell scale carries it); u adds silhouette +0.4 … +1.9 points on every target and the
better bunny surface, and costs 1.1–1.4° of normal error on bob / dragon; coupling it to F
(sub-cell strain the 3.6-spacing grid cannot relax: det F collapses), gating it by the
density residual, or driving it by the render channel alone are all worse. It stays
kinematic and ungated (the decision of 2026-09-22, docs/final_plan.md §6).

**The denoised shading reference G1** (`--pbr_denoised`). The shading target is rendered
from the target's RECONSTRUCTED surface: per particle the normal of the nearest triangle of
the Poisson mesh of 10.12 and the weight exp(−(dist / spacing)²); the morph's normals on the
render-pixel grid (pixel = 2 · extent / render_res) blurred by 1.5 spacings. The reference no
longer carries the sampled cloud's shot noise (rough share of the shading covector at two
spacings 0.67 → 0.55).

**Stratified sampling** (`--sampler stratified`, G5). One jittered particle per fill voxel,
the fill resolution found by bisection so that the fill holds at least n voxels, the surplus
dropped without replacement (bunny 40k: 44³ voxels, 6.5 % dropped). Proof and measurement in
docs/surface_gradient.md §9: drawing with replacement is a Poisson process with relative
density fluctuation (p/σ)^{3/2}/√(8π^{3/2}) = 5.9 % at σ = 1.5 spacings; the jittered lattice
leaves a dipole field with (p/σ)^{5/2}/√(64π^{3/2}) = 1.1 % (measured cube 5.5 → 1.1 %,
bunny 8.1 → 2.9 %, dragon 5.2 → 2.2 %; p = the volumetric spacing = spacing / 1.24).

**Addendum 2026-09-22 (direct contact of drawn pieces; g40 cow frames 267 / 270 / 438 / 480).**
The bridge rule walked from the body to another drawn piece through FREE particles only (the
ones the drawn surface does not enclose). When the surface breaks across a thin feature while
the particles continue — a leg two particles thick whose hoof the Poisson surface caps off —
the connecting particles lie within the one-spacing tolerance of both caps and are all
"enclosed": there is no free particle to walk through, and the piece was reported unbridged
although the cloud is a single connected component at the link radius and the grid probe
counts no fragment. The graph now also carries an edge between two drawn pieces whose
enclosed particles come within the link radius (the nearest such pair), and the filament
drawn for that edge is the segment between those two particles. Nothing else changes: the
radius is still max(2.5 spacings, one cell), the path is still the shortest chain, a piece
farther than a cell from everything is still a piece.

### 10.15 Surface tracking for the morph video (2026-09-22; code: scripts/render_photoreal.py `--track`, `advect_vertices`, `closest_on`)

The deliverable surface of 10.12 is reconstructed independently for every video frame. Two
things follow that the physics does not contain: the frame-to-frame RE-FIT JITTER — a few
particles cross the outer-layer rule, the surfel set changes, the Poisson solve moves the
whole surface a fraction of a spacing — and false BREAKS at thin necks, where a feature one
or two particles thick falls out of the reconstruction for a few frames and its far end is
drawn as a piece (bridged by 10.10's filament, still visibly a tube with a bulb). Both are
properties of the reconstruction, not of the material, which is connected (grid fragments 0)
and moves smoothly.

The tracked surface carries the mesh with the material: each vertex is bound every frame to
its k = 8 nearest particles of the previous frame with Gaussian weights of one spacing and
moves by their weighted mean displacement (advection); then it is pulled toward the fresh
reconstruction of the current frame by the fraction α = 0.3 of its distance to the closest
point on it (an exponential filter with a ~3-frame memory: drift is corrected, the re-fit
jitter is not followed). The mesh is re-solved only when the drawn topology changes (the
number of drawn pieces, bridges or cavities of the fresh reconstruction) or when the mean
drift before the pull exceeds one spacing; the last frame is always a fresh reconstruction,
so the end-frame metrics of 10.12–10.13 are unchanged. The sidecar records per frame the
re-fit jitter of the independent reconstruction (the previous fresh mesh advected against
the current one, in spacings), the tracked drift and the re-mesh events; the QA columns
(pieces, bridges, cavities) keep coming from the fresh reconstruction, so a tracked video is
judged by the same per-frame counts. Metrics never read the tracked mesh. Discretisation:
spacing 0.135 wu at 40k, video frame = 3 archived frames = 24 MPM steps.

**Addendum 2026-09-22 (evening; what the per-frame reconstruction actually does, and the re-mesh
policy).** Measured on the g40 cow: the re-fit jitter of the independent reconstruction is 0.024
spacings (p90 0.035) and the rendered-image change between consecutive frames 0.08 % of full
scale — there is no geometric flicker to remove. What the eye reads as "the surface re-adjusting"
is the TOPOLOGY EVENTS: at a thin neck the reconstruction breaks for a few frames, the far end is
drawn as a bridged piece and re-joins (cow: 24 frames in four episodes). The tracked mesh is
therefore re-solved only when the PARTICLES confirm a topology change (a drawn piece the filament
rule cannot tie to the body, or the cavity count), when the median drift exceeds one spacing, or
every 60 video frames to bound the stretching of the advected tessellation; the pull toward the
fresh surface is applied only to vertices within one spacing of it, so a neck the reconstruction
lost stays a tube. Cow: 10 re-meshes (the periodic ones and the sphere-phase cavities), none at
the neck episodes; the legs are continuous through frames 207–270 and 432–483. Tracking is the
default of `photoreal_batch.sh` (`TRACK=0` restores the per-frame surface); the QA columns are
still those of the fresh reconstruction.


**Addendum (2026-09-23; code: scripts/render_photoreal.py `--track_keep`, `--track_stretch`; probe
scripts/probes/video_jumps.py).** The user's reading of the first tracked galleries: a connecting
part wiped in one frame, and connections drawn with messy triangles. Measured on all 38 tracked
videos (per video frame: the image change, the object's pixel area and its change, the re-mesh
flag from the QA sidecar, the particles' displacement): at re-mesh frames the image change is
2–10× the median and the area change 3–9× that of ordinary frames while the particles move as
usual — the re-mesh replaced a tracked mesh that still carried a tube (a neck the fresh
reconstruction lacked) or a web (triangles stretched between two growing ears, bunny frame
357) with the fresh mesh. Two rules fix it without a new constant: (1) at a re-mesh that the
particles did not ask for (drift or stretch), the tracked triangles with no fresh counterpart —
all three vertices farther than one spacing from the fresh surface — are kept, so a lost neck
stays a tube; a particle-confirmed topology change still re-meshes fully; (2) the tracked mesh
re-meshes when its 99th-percentile edge exceeds twice the fresh mesh's own 99th-percentile edge
(the Nyquist factor: a stretched triangle cannot carry the fresh detail), and a stretched
triangle is never kept — the web never forms. The periodic re-mesh is off (it only wiped).
Readings (g41 bunny / cow): re-mesh frames' mean area change 14.4 % / 7.1 % with the old
policy, 2.4 % / 4.5 % with the new (ordinary frames 3.9 % / 1.8 %); the per-frame reconstruction
has a median frame-to-frame image change 1.5–1.8× the tracked one (the re-fit and the
particle-scale texture). `photoreal_batch.sh TRACK=1` uses the new policy; the default stays
per-frame until the user chooses.

### 10.16 The transport gate of the u channel (2026-09-22 evening; code: pipeline/optimizer.py, the `ot_pace` block; config `layer_gate_ot`, `layer_gate_ot_cells`; evidence docs/surface_gradient.md §15)

The mid-morph surface of the g40 gallery was lumpy (outer-layer plane-residual RMS 1.8–2.4× its
end value at frames 15–135, cratered patches in the stills). The cause is the position-mode
channel u of §10.14: its per-particle step, from a covector that is half noise at two spacings
(§7 of surface_gradient.md), is re-applied every step on material that is still in transit, and
the relaxation (W at two spacings) does not see the residual it leaves at the cell scale. The
render × u factorial's traces put it beyond doubt: u on adds +30–50 % to the morph-mean
roughness on bunny / bob / dragon with or without the render channel; the render channel without
u adds nothing beyond the twin spread. A coarser control basis (§15a: node spacing one and two
cells) does not touch the lumps — u is a separate leaf — and costs end detail.

The gate: at every window start the pace block has, for every particle, the remaining transport
to its entropic-map image, d_p = |x_p − T(x_p)| (material-kNN averaged like the pace itself).
u may act only on outer-layer particles with d_p ≤ dx (one MPM cell): the residual the grid
cannot resolve is u's regime; a particle in transit gets none. The gate multiplies the existing
per-particle gate buffer of the trajectory (`layer_ug`) and is recomputed per window with the plan.
Nothing is tuned: the radius is the grid's cell, the residual the transport loss already computes.

Readings (§15c; first 300 frames, mean roughness at two spacings, the peak at the cell in
brackets; end silhouette IoU against the g40 recipe): bunny 0.288 (0.442) → 0.237 (0.304), −0.2
points; bob 0.320 (0.501) → 0.270 (0.433), −0.04; dragon 0.369 (0.506) → 0.291 (0.400), −0.04.
u off (`u0`) reaches 0.203 / 0.250 / 0.285 at −0.6 / −0.3 / −0.3 points; the end frame
against the true mesh is the same for every lever and for the recipe when re-measured like for like today (bunny surface_gt mean 0.27–0.28, hp_res 0.15–0.16; the same as the g40 report's end-frame row; the 0.22 quoted at first was the target-floor row). The gate's active share starts
at 0–13 % of the layer and reaches 80–100 % once the material has arrived (the dragon's thin
features never do: 16–22 % of its layer stays gated). Rendering influence unchanged in kind:
g_share of the control update 0.36–0.38 with and without the gate; the gate acts on u only.
Refuted alternatives, all pre-registered (surface_gradient.md §15): the coarse control basis
(one and two cells), the u-step projection (§7's `--layer_ctrl_smooth`, −5 … −13 %), the
geometric gate on the distance to the target's nearest surface (−1 … −3 %: a surface point is
within a cell of the target long before the material under it has arrived — the wrong residual).

**Addendum (2026-09-22 22:00; code: pipeline/runner.py, the outer-merit block; evidence
surface_gradient.md §15d–15f).** The gate's first gallery (g41) ran 18 of 19 targets as predicted
(mid-morph roughness −6 … −29 %, end silIoU within ±0.3 points) and stopped nefertiti at anim 16
(0.935): its arriving front spills outside the outline for a few windows (the stray-cleanup term
d_dt +65 % in one window, 455 → 954 over five) while the transport divergence keeps falling, and
the outer merit's catastrophe brake — a full-merit regression beyond one pace budget, written for
runaways of the descended objective — rejected the candidate three times; the annealed replays
meet the same state, because the trajectory has to pass through it. Under the old recipe the
in-transit u pulled the front back every window and hid the spill; under u off the spill is
accepted at −1 % a window and the run recovers. The brake now reads the primary objective alone
(the scaled transport divergence, or the cell sum for density recipes); the latched low-gain rule
and the reversal rule keep the full merit. Nefertiti under the gate: 0.9676 (old recipe 0.9683),
90 windows, roughness −25 %. Refuted on the way, pre-registered: the brake on every physics
component (d_dt still trips it) and a gate on the normal component of the remaining transport
(admits half the layer in the expansion phase — no gate). The gate does not act in the `ot`
(hole-topology) regime, where the pace's kNN-averaged residual is not formed; C runs ungated.

### 10.17 The dynamics mass of the discretisation (2026-09-23; code: pipeline/config.py `mass_ref_n`, pipeline/optimizer.py; evidence docs/experiments.md 2026-09-23)

The particle mass of the forward model was unit at every N. The control force per cell is a sum of particle rest volumes (∝ 1/N each) and does not grow with N, so the acceleration a unit control produces scaled as 1/N: the 300k body was 7.5× more sluggish than the 40k one (first-window displacement 0.004 against 0.027 wu, kinetic energy 0.014 against 0.54) and the transport took 212 windows against 59. The dynamics mass is now mass_ref_n / N per particle with mass_ref_n = 40000, the discretisation the constants were calibrated at: the body’s mass, density, wave speed and control response are the same at every N, the 40k runs are bit-identical, and the loss-side unit masses (whose normalisations — m_ref, the density units — are built on them) are untouched. With it the 300k first windows equal the 40k ones (0.024 wu, 0.48) and the end silhouette is 0.9677 (40k 0.961). The 300k run still runs to 176 windows: the finer cloud keeps improving below the loss cell, where the 40k cloud has reached its noise floor — a window budget of 70–90 delivers the 40k-level silhouette in 15–20 min.

### 10.17a The reference discretisation of the spacing-derived constants (2026-09-24; code: pipeline/config.py `disc_ref`, `disc_ref_factor`; scripts/render_photoreal.py `--ref_n`; evidence docs/experiments.md 2026-09-23 night)

10.17 makes the DYNAMICS of a cloud at N > `mass_ref_n` those of the reference cloud sampled
finer: the same grid (10.9), the same mass, the same response to a control. Every other
constant the pipeline derives from the particle spacing was still taken at the native
spacing, so at 300k on the 40k grid (spacing 0.069 against 0.135 wu) the outer layer was half
as deep, its relaxation half as wide, the u channel clipped to half the length per window,
the splats of the render loss half the size, the shading target's normal grid twice as fine,
the cleanup band and the ejection radius half as long, and every neighbour COUNT (the layer
and asymmetry kNN, the isolation gate's k, the bond neighbours, the decoupling count of one
particle) covered a seventh of the mass. The measured consequence at the bunny's ear tip
(thinnest extent 0.35 wu = 1.13 cells; the tip cell holds 0.60 of a bulk cell's mass): the
300k body puts 5.9 reference particles' mass within 0.25 wu of the tip against 13 at 40k, and
the material that does arrive is stretched to 1.7–2.4 native spacings (0.11–0.16 wu, within
half a cell — connected for the grid, a separate bead for a renderer whose kernel is the
native spacing), 18 particles of it off the body at 1.5 spacings at the end. None of the
three mechanisms for detached material sees it: the per-step decoupling test counts single
particles (10.7), the fragment mask is grid connectivity (10.7), the kNN gate's ratio is 1
inside a compact clump (10.5) and the DT pull is zero on the target anyway.

```
(38)  f = (N / mass_ref_n)^(1/3)   (1 for N <= mass_ref_n)
      every spacing-derived LENGTH   <- length x f
      every neighbour COUNT          <- count x f^3
      the decoupling count (24)      <- N_p(3^3 cells) <= f^3   (one reference particle's mass)
```

Under `disc_ref` the pipeline applies (38) at its sources: the target NN spacing
(`runner.build_target`, hence the OT plan blur radius of 10.8/10.10, the cleanup band, the
ejection radius, the re-attachment jitter, the KDE width), the layer spacing `sp0` of the
relaxation and the u channel (the layer depth, the relaxation width, the u clip), the splat
size of the render loss and the shading target's spacing, and the counts (layer 24, asymmetry
32, isolation 8, bonds `coh_k`, the decoupling count). The reference cloud and every run
without the flag are bit-identical (f = 1). No new constant: f is the mass contract of 10.17.

The deliverable renderer applies the same rule (`--ref_n`, default `mass_ref_n`): its kernel
width, outer-layer threshold, level, Poisson octree cell, trim, bridging radius and tracking
tolerances are taken at the reference spacing. At the native spacing the 300k octree was one
level finer than the 40k one (cell 0.034 against 0.069 wu), the sub-cell beads of the ear
tips were separate components and the bump angle of the end surface was 1.9° against 1.2°; at
the reference spacing the beads are gone (one component, nothing dropped), the fork of the
left ear is one tip, the bump is 1.5° — the knobs that remain at both tips are the stretched
tip material itself, which the render cannot and should not hide. Whether (38) in the physics
closes the tip is the pre-registered twin `d300_bunny` (docs/experiments.md 2026-09-23 night).

Addendum (2026-09-23 night, the twin `d300_bunny`; docs/experiments.md): (38) in the physics
brings the tip's mass to 18.3 reference particles (5.9 / 13 at 40k), its 8-NN to 0.58 native
spacings (1.67), the end silIoU to 0.9699 (0.9677) in 52 windows / 16 minutes (176 / 44) with
det F min 0.745 (0.478) — and leaves the sub-cell arrangement to itself: the native-scale bump
2.3° (1.9°), the mid-morph roughness 2×, 95 particles in small pieces (37). The rule is kept
opt-in (`disc_ref`, not in the recipe); the deliverable renderer keeps `--ref_n`. What the pair
c300 / d300 establishes: the method has no term that orders the quadrature below the cell, and
the two runs are the two ways of not having one (chasing sub-cell residuals for 120 windows with
a window-to-window reversal, or ignoring them). The next mechanism is a sub-cell ordering step
that is loss-neutral and mass-preserving (particle shifting / redistribution), designed after
the literature (docs/related_work.md, pending).

### 10.18 Fickian shifting of the sub-cell arrangement (2026-09-23 night; code: mpm/shifting.py `fickian_shift`, pipeline/runner.py at the commit; config `shift_sub`, `shift_h_sp`; evidence docs/experiments.md 2026-09-23 night, docs/related_work.md)

The quadrature below the cell is a null space of every term of the objective (10.17a: the pair
c300 / d300). Shifting is the SPH remedy (Lind, Xu, Stansby, Rogers 2012): the particle
concentration C_i = Σ_j (m_j/ρ_j) W_ij is 1 in a uniform arrangement and each particle
diffuses down its gradient, one explicit step at the stability limit, at every window commit:

```
(39)  Δx_i = −½ h² ∇C_i,   ∇C_i = Σ_j (m_j/ρ_j) [1 + R (W_ij / W(Δp))^n] ∇_i W_ij
      W Gaussian of width h = Δp (native spacing), R = 0.2, n = 4 (Monaghan 2000), k = 40 neighbours
      outer layer (one-sided neighbourhood): the tangential part only, weight → 1 at ½ Δp of offset
      |Δx_i| ≤ ½ Δp; positions only — m, v, C, F untouched; applied to the COMMIT state
```

Constants and their origin: ½ is the explicit-diffusion stability limit (the 2020 review's
10.3–10.4); the anti-pairing factor is Monaghan's tensile correction that Lind 2012 added to the
shifting gradient (R = 0.2, n = 4 as in the δ⁺-SPH form; Lind's own values unverified,
docs/related_work.md) — without it the kernel gradient vanishes for close pairs and the step
dis-orders a cloud (measured, experiments.md); h = Δp for the Gaussian is the width of the cubic
spline at the standard SPH ratio h = 1.3 Δp (σ 0.71 against 0.78 Δp), the one that orders;
k = 40 is where the Gaussian is below 2 % (a ball of radius 2 Δp holds ≈ 34 particles); the
free-surface rule is Lind's (shifting restricted at the free surface), with the outer layer's
own asymmetry measure (surface_recon.layer_by_asymmetry) as the weight; the ½-spacing cap is
the explicit step's own bound (no particle passes a neighbour). The shift is mass-preserving (positions only) and acts on an arrangement the objective leaves
UNDERCONSTRAINED — the CIC weights of the cell sum vary continuously inside a cell and the images
depend on every position, so "loss-neutral by construction" overstates it (the independent audit,
docs/diagnosis_300k_20260923.md); the measured change of the objective across a commit's shift is
to be recorded per run. The 40k gallery is untouched (opt-in). Whether it orders the
quadrature of the morph without harming the fit is the pre-registered run `f300_bunny`.

### 10.19 Sign-history damping of the u channel (2026-09-23 night; code: pipeline/runner.py at the accepted commit, pipeline/optimizer.py `u_scale_init` / `stats["u_final"]`; config `u_rprop`; evidence docs/oscillation.md Addendum 9)

The tail of a 300k run breathes: the outer layer's u step (10.15/10.16, a normal displacement per
layer particle bounded by one spacing a window) goes out on the render gradient and comes back
the next window — sign flips in 60–80 % of the layer per window, 2.5 spacings of motion summed
for 0.04 of net drift over thirty windows. The window loop re-linearises from u = 0 every window,
so nothing in it remembers that a particle's last step was undone. Rprop (Riedmiller & Braun
1993) is the step rule built for exactly this signal — a per-parameter step size adapted by the
sign of successive updates:

```
(40)  after an accepted window w, per particle p with u_w(p) u_{w−1}(p) ≠ 0:
        s_p ← η⁻ s_p  if sign u_w(p) ≠ sign u_{w−1}(p)      (η⁻ = 0.5)
        s_p ← min(1, η⁺ s_p)  if the sign is kept          (η⁺ = 1.2)
        s_p ∈ [0.05, 1];   the next window's bound on u(p) is s_p × one spacing
```

The constants are Rprop's own (η⁻ = 0.5, η⁺ = 1.2) and the anneal floor (0.05); a particle whose
u keeps its sign descends at the full bound, one that oscillates has its bound halved each flip,
and a particle that leaves the layer (u = 0) keeps its scale. Positions, the stress control and
the relaxation are untouched; the rule acts only on the channel whose sign history is measured
to alternate. It is the source-side counterpart of the delivered-trajectory rule of Addendum 9
(`stop_on_cycle`, which stops the tail; this removes the breathing that makes the tail). The
rendering influence of the channel is reported by g_share and the u gate as before; the damping
reduces the render's push only where the push was being undone.

### 10.20 The null-space projection of the window's displacement (2026-09-24; code: mpm/gridfilter.py `grid_project`, pipeline/runner.py at the commit; config `commit_pic`; evidence docs/experiments.md 2026-09-24, docs/related_work.md "MPM particles ↔ grid")

The grid sees a particle field only through the transfer: P2G with the cubic B-spline weights,
G2P back. P = G2P ∘ P2G maps particle fields onto what the grid can represent; I − P is the grid's
null space — the sub-cell modes the dynamics cannot act on and the cell sum cannot see, where the
measured sub-cell disorder and the tail's breathing live (10.17a, oscillation.md Addendum 9).
Gritton and Berzins (2017) remove that null space per cell by an SVD of the P2G operator;
XPIC(m) (Hammerquist and Nairn, 2017) removes it by alternating transfers, exactly as m → ∞. Here
it is applied ONCE per window to the window's displacement, at the commit and before the shift:

```
(41)  d = x_end − x_start,   d_f = d − (I − P)^m d,   x_end ← x_start + d_f      (m = 5)
      P(f)_p = Σ_g w_gp (Σ_q w_gq m_q f_q) / (Σ_q w_gq m_q),  w = the simulation's cubic stencil at x_start
```

On a random cloud P is not an exact projection (a linear field returns with a first-order
sampling error, 6 % at m = 1), so the order matters: at m = 5 a linear field is reproduced within
2 % and uncorrelated sub-cell noise is removed to 7 % of its RMS (`tests/test_gridfilter.py`); a
two-cell sinusoid is removed as well — the cubic stencil cannot carry it — which is the price the
pre-registration puts on the ear tip (P51). No constant beyond the order; positions only; v, C,
F untouched (the removed part is a fraction of a spacing). The share of the window's displacement
in the null space is logged (`pic_null_share`; 28 % at the first window of a 20k smoke).

### 10.21 Windows from rest (2026-09-24; code: pipeline/runner.py at the accepted commit; config `rest_commit`; evidence docs/experiments.md 2026-09-24, the rebound diagnostic)

The morph is delivered as a sequence of windows, each of which ends at rest as far as the
terminal kinetic term (10.2) can ask; the next window has started from the commit's velocity and
APIC affine state. The rebound diagnostic (`rebound_probe`: a zero-control rollout from every
accepted commit) reads what that carried state does: the body travels on, along the committed
displacement, by 0.9 of a window's motion in the expansion and 0.3–0.4 in the tail — an
overshoot the next window's control must cancel, whose reversed momentum the following commit
then carries: the coherent two-window alternation at the resolved scale that the spectral
criterion finds in every run (correlation −0.5 … −0.98, the 40k reference included). Nothing
springs back elastically (the same rollout from rest moves far less).

```
(42)  at an accepted commit:   v_p <- 0,   C_p <- 0        (x, F, F_p unchanged)
```

A sequence of equilibria carries no momentum between its members; (42) makes the terminal
rest the kinetic term asks for exact, at no constant, and removes the carried overshoot at its
source. What it may cost is the free travel that helped the expansion phase (0.9 of a window's
motion at windows 2–8); the pre-registration of `m300_bunny` (experiments.md) puts a window
budget on that.

Addendum (2026-09-24 01:00): the unconditional form of (42) is refuted — at 20k it left the morph at
silIoU 0.919 against 0.967 after 30 windows, and at 300k the transport gate stood at 22 % at window
10 against 65 % (run m300, stopped): the expansion rides the carried momentum (0.9 of a window's
travel is free in windows 2–8), and a body restarted from rest every window must re-accelerate.
The rule therefore applies only once the transport has ARRIVED, read from the u transport gate of
10.16 (the fraction of the layer within one cell of its OT image): from the first accepted commit
at which it reads 100 % (`rest_commit_gate`, latched), windows start from rest; before that the
momentum is the transport. No new constant: the gate is the recipe's own arrival measure.

Addendum 2 (2026-09-24 01:40): m300b's gate latched at window 62 of 66 — it crept from 93 % to
100 % over 28 windows, the last stragglers defining arrival — so the rule was engaged for four
windows before the merit gate stopped the run: no reading of the mechanism. Arrival is read
better by the runner's own reversal cosine (the cosine between consecutive accepted windows'
displacements): +0.9 through the transport, a zero crossing, then negative at every window to the
end of the run (m300b from window 44, l300 from 49, the 40k reference from 20). Second form
(`rest_commit_reversal`): the latch fires at the second accepted commit in a row whose reversal
cosine is negative — one full period of the two-window alternation, the carried momentum an
overshoot by definition. No constant beyond the sign; the gate of the first form still applies.
Pre-registered as o300 (experiments.md P62–P65).

### 10.22 The support-preserving paced target (2026-09-24; code: losses/projection.py, pipeline/optimizer.py in the paced-target block; config `pace_project`; evidence docs/experiments.md 2026-09-24 01:45)

The paced target of the transport regimes (`phys_loss` ot_pace / ot_shape) advects the cloud
along its plan by one loss cell per window, x_int = x0 + min(1, h/|d|) d with d the
material-denoised plan displacement, and rasterises it with the loss's own CIC splat. That is the
straight-ray (displacement) interpolant of the plan, and for a volume-preserving but anisotropic
map it is not volume preserving in transit:

```
(43)  rho_t = rho_0 / det(I + t (J - I)),   det(I + t (J - I)) = prod_i (1 + t (lambda_i - 1))  !=  1   (0 < t < 1, J != I)
```

Where the map stretches (a patch of the sphere's crown into a 1.1-cell ear, lambda = (4, 1/2, 1/2))
the paced density falls to rho_0 / 1.4 at t = 1/2, and it rises above rho_0 where the rays converge
at the feature's root. The cell sum then asks for a dense root and a sparse stream, and a cloud
whose spacing is a third of a cell can supply exactly that — a filament of bulk density in part of
the cell, breaking into pieces at the native spacing (experiments.md 2026-09-24 01:45: the 300k
ear, a base slab at 1.5–1.7× and a stream at 0.5–0.7 of the target thickness; the 40k tongue,
which cannot form a sub-cell filament, at 0.8–1.0).

```
(44)  d' = d - grad p,    -Lap p = -div d  on Omega,   p = 0 on and outside the boundary of Omega,
      Omega = { lattice nodes whose CIC mass >= 1/2 of the bulk node mass }
```

(44) removes the divergent part of the paced step on the body Omega (Chorin's projection; the
admissible-velocity projection of the density-constrained crowd model of Maury, Roudneff-Chupin
and Santambrogio 2010; the Hele-Shaw limit of Perthame, Quirós and Vázquez 2014, where the pressure
lives on the saturated set and vanishes on the free surface). The advected cloud keeps its density
and a thin feature grows as a tongue extruded from the body, fed through its base at bulk density,
instead of being assembled from a stream. The plan is re-solved every window from the advected
cloud, so the endpoint is unchanged; only the path through transit is.

Discretisation: the loss grid's node lattice; MAC face velocities from the mass-weighted CIC
deposit of the step at the lattice shifted by half a cell along each axis; the seven-point
Laplacian with p = 0 outside Omega (the Dirichlet free surface), conjugate gradients to a 1e-4
relative residual; the correction -grad p on the faces gathered back at the particles with the
same shifted CIC, FLIP-style — the particle keeps its own denoised step minus the grid's divergent
part. No constant beyond the discretisation: the half-bulk occupancy is the CIC value of a node
lying on the surface, the tolerance a solver residual. Logged per window: the step's divergence
rms before and after (a relative volume change per window) and the correction in cells
(`pace_proj_div0 / div1 / corr` in the record). Tests: tests/test_projection.py — a translation
and a rotation of a ball pass unchanged (< 3 % / 5 %), a radial expansion is removed inside the
ball (< 20 % of the step remains), the divergence falls by more than 85 %.

Addendum 3 (2026-09-24 03:10) — the alternation as the plateau. o300 (the second form, latched at
window 43) answers the question 10.21 asked: the windows started from rest reverse their
predecessors just the same (reversal cosine −0.63 … −0.31 with the kinetic energy down seven-fold),
so the carried momentum is not the alternation's carrier; nor is the step size (l300 and the 40k
reference alternate at the annealed floor) nor the balancer (λ and the gradient share do not
alternate). What alternates is the shape merit itself — at 40k the outer gain flips sign at every
tail window (±1–2 %), the silhouette term with it, the cell sum flat: the commits alternate between
a better and a worse state. The outer merit gate has, since 2026-09-04, a rule for exactly this
(reject a reversing candidate whose gain is below 0.5 %), but its latch reads the λ-free plateau
tracks, and the alternation's up-swings set new bests by a fraction of a per-mille each cycle and
disarm it — l300 committed 35 alternating windows, the 40k reference 40, before three rejections
ended them. `outer_latch_reversal` arms the gate at the alternation's onset — the second accepted
commit in a row whose displacement reverses the previous one, the same reading as the second
form's latch — and keeps it armed: reversing low-gain candidates are rejected, three in a row end
the run at the best commit, and the deliverable carries no alternation. No new constant: the sign,
one period, and the gate's existing gain threshold. What it does not do is say WHY consecutive
optimisations from a stationary state alternate; the intervention twins (p300: the render channel
off from the arrival; then the fixed target) are pre-registered for that in experiments.md.

### 10.23 The particle scale at a thin feature (2026-09-24; code: losses/volumetric.py d_kde / kde_assign, config `w_kde`, `plan_native`; evidence docs/experiments.md 2026-09-24 07:55, y300)

The reference discretisation (10.17a) delivers the mass to the ear region, and the cell sum
receives it as the straight-ray target says: a converging fan piles at the base, a stretching one
thins into the ear. At 300k the front realises the sub-bulk stream as a filament of bulk density
three particles across — a sparse cell and a dense filament in part of the cell have the same CIC
mass, so the cell sum cannot tell them apart, and the supply through the base piles up behind a
front that takes it up slowly (experiments.md: the base slab at 1.5–1.7×, the mid-ear at 0.5–0.7
of the target thickness, the 40k tongue at 0.8–1.0 because its spacing forbids a filament thinner
than a cell). The projected target (10.22), the linear cell sum and the plan's blur (below) do
not change this; the term that does sees the particle scale:

```
(45)  D_kde = mean_p [ rho_h(x_p; particles) - rho_h(x_p; target points) ]^2 / rho_ref^2,
      rho_h(y; S) = sum_{s in S, k nearest} exp(-|y - s|^2 / h^2),   h = k_h * (target NN spacing at the reference),
      neighbour lists frozen per window; weighted by w_kde * s_kde with s_kde = |grad D_vol| / |grad D_kde| at the source (once)
```

(45) compares the kernel density of the particles with the kernel density of the target's own
points at every particle (the SPH form of the cell sum, 2026-09-03), and its gradient runs from
crowded to deficient regions: a filament surrounded by target volume it leaves empty is a deficit
at that scale, a pile an excess. With w_kde = 1 the term enters at the cell sum's gradient norm —
a ratio, not a constant. y300 (l300 + the term): the base slab's maximum 1.11× (1.49), the ear
grown as a tongue from the base at 0.90–1.05 of the target thickness where it has mass, the tip
at 16.1 reference particles with 4 strays, silIoU 0.9782; the price is a later ear (0.31 filled at
t = 0.20 against 0.83) and det F 0.59, and the tongue's leading edge still fragments at the
native spacing (the thin-feature carrier of related_work.md — split particles or a spine where the
destination is thinner than two cells — is the open item for that).

The plan's blur under 10.17a (correction). The plan's blur is sample-derived — the target NN
spacing × (N / ot_samples)^(1/3), the spacing of the ot_samples-point sample of the shape, hence
N-independent (0.119 wu at 40k, 0.116 at 300k native) — and 10.17a's scaling of the target NN
spacing doubled it to 0.227 wu. `plan_native` keeps it at the sample's spacing; u300 read no
change in the ear and none in the transport, so it is a correctness fix of rule (38), not a
mechanism.

### 10.24 The per-particle Rprop on the control step (2026-09-24; code: pipeline/optimizer.py at the Adam step, pipeline/runner.py at the accepted commit; config `ctrl_rprop`, `u_rprop_floor`; evidence docs/experiments.md 2026-09-24 07:30 and 10:30)

Every run alternates once its bulk has arrived, and every intervention that changed a carrier —
the momentum (o300), the render channel (p300), the relaxation (w300), the moving target (v300,
in part), the balancer, the u bound (j300) — left the alternation's amplitude where it was: the
layer's per-window normal step sits at 0.002–0.003 wu at 40k and at 300k alike, and l300 and the
40k reference alternate with the step at its annealed floor (0.05 of α). A first-order controller
with a fixed minimum step against a residual that changes sign when crossed is a limit cycle of
that step's amplitude; the cure is a step that keeps shrinking on reversal (the Robbins–Monro
condition), and the reason the recipe has a floor at all is the transport: a global step decayed
to zero would freeze a part that is still travelling (z300's gate cut the ear short). The two are
separated per particle:

```
(46)  at an accepted commit, for every particle p with d_p = x_p - x_p,start and d'_p its previous accepted displacement:
      r_p <- r_p / 2                if d_p · d'_p < 0            (the step reversed)
      r_p <- min(1, 1.2 r_p)        if d_p · d'_p >= 0           (the step kept its direction)
      the control's Adam step for p is scaled by r_p (no floor); particles with |d| below 1e-4 cell keep r_p
```

This is Rprop (Riedmiller, Braun 1993: η⁻ = ½, η⁺ = 1.2, the standard values; here without Δ_min),
applied per particle to the control leaf, with the same rule the u channel has had since 10.19
(`u_rprop`) whose floor of 0.05 spacings — the breathing's own amplitude — is now a value
(`u_rprop_floor`, 0 in this recipe). A particle still in transport moves the same way window
after window and keeps r_p = 1; a particle of the arrived body reverses at every window and its
step halves each time — the alternation decays geometrically instead of persisting at the floor,
and the run ends on the plateau rule when nothing moves. No constant beyond Rprop's two, which
are the literature's; the floor is removed, not set.

What it does not do: it does not change the window's objective, the paced target or the render
channel, so the merit reversal's cause (the moving target, 10.21 addendum 3) is untouched — the
step decays under it. Read together with the reconstruction question (the kind of the tail
motion, experiments.md 09:40): if the tail's visible change is a tangential re-sampling of the
Poisson fit (hypothesis c), the decayed step removes that too, since a particle that no longer
moves cannot rearrange.

Addendum (2026-09-24 15:50) — the neighbourhood-smoothed form (`ctrl_rprop_smooth`, `ctrl_rprop_k`).
The per-particle rule read on the bunny at 300k: the layer's flip fraction 0.74 → 0.41, its
per-window normal step 0.0017 → 0.00095 wu, the low-band consecutive-window correlation −0.67 →
+0.04, the delivered video's tail 0.0016 → 0.0013 per frame (the 40k reference's value) — and det
F min 0.664 → 0.388: neighbouring particles' control updates differing by orders of magnitude are
a sub-cell control noise the regulariser (creg, a kNN Laplacian on the control) exists to forbid,
and the material pays in local compression. The second form reads the reversal on the window
displacement averaged over a material neighbourhood (frozen at the source) and applies the
neighbourhood mean of the per-particle scales:

```
(47)  d̄_p = mean_{j ∈ N(p) ∪ p} d_j  (same for d'),   r_p updated by (46) on d̄_p · d̄'_p,   the step scaled by  r̄_p = mean_{j ∈ N(p) ∪ p} r_j
```

N(p) = the coherence kNN (≈ 60 at 300k, half a cell; ad300: det F 0.64 kept, the decay diluted to
a correlation of −0.31 and a step of 0.0012 wu) or the control regulariser's own kNN (creg_k = 8;
ag300, running): the neighbourhood is the regulariser's constant, not a new one. What the rule
costs the transport: a particle whose arrival wiggle reverses once keeps half its step for good —
the ear tip in ac300 held 10.7 reference particles against l300's 18.5; the reading of the form
on the tongue is af300 (with the KDE term).

### 10.26 The settled body's viscosity and the settle at commit (2026-09-24 evening; config `settle_eta`, `settle_commit`; DIAGNOSTIC forms)

The freeze twin (g41f) showed that a settled body with its control zeroed, its velocity zeroed
at commits and its elastic stretch assimilated still moves 0.0016 wu a window: the residual
per-window motion is the rollout's own — the grid carries the last arrivals into settled
material, the layer relaxation follows — and not the optimiser's. Two forms read whether that
motion decays if the settled material is quasi-static:

```
(48)  eta_p = 1 / (T dt)  for settled particles (arrived, twice reversed), 0 otherwise    [settle_eta: inside the windows]
(49)  x, F <- rollout(zero control, eta = 1 / (T dt) everywhere, T steps) from the accepted state; v, C <- 0   [settle_commit]
```

(48) is the forward model's own per-particle viscosity (traj.Trajectory eta, the Kelvin–Voigt
damping of the MPM) with the time constant of one window — the constant is the discretisation's,
not a material's; (49) is the quasi-static commit: the delivered commit is an equilibrium and the
next window is linearised at rest. **Both change the material during the morph**: a viscous
settled body responds to an external force during the morph with that damping, and the
committed state of (49) is a relaxed one. They are therefore diagnostics of where the residual
motion lives, and are not in the deliverable recipe unless the user accepts a damped settling
phase; the deliverable object itself (the archived x, F, Fp with λ, μ) carries no viscosity —
after the morph it responds to external forces through its elastic parameters alone, and with
the plastic assimilation its rest state is the morphed shape.

### 10.27 The pinned settled body (2026-09-24 night; config `settle_pin`; the user's "once optimised, lock it")

The user's requirement is that the settled body's oscillation be exactly zero, not small. The
freeze (10.25) and the viscous forms (10.26) act on the optimiser's step and on the momentum
respectively, and the settled body still moved 0.0016 wu a window (g41f), the rollout's floor:
the grid carries the last arrivals into settled material and the layer relaxation follows. The
pin closes that floor at the kinematics, inside the forward model:

```
(50)  P = { p : arrived_p ∧ rev_p ≥ 2 }    (the settle_eta / freeze reading, monotone: once in P, always in P)
      for p ∈ P, every step t of every later window:
          v_p^{t+1} = 0,  C_p^{t+1} = 0,  F_p^{t+1} = F_p^t,  x_p^{t+1} = x_p^t          [k_g2p, k_update]
          the layer relaxation and the u channel skip p                                 [k_layer_project]
      the control of p is zeroed and its Rprop scale set to 0 (no step), u_p = 0, v_p = C_p = 0 at the commit
```

The pinned particle still carries its mass and its (zero) momentum to the grid, so the
transporting material sees the settled body as a fixed obstacle: the constraint is the same
kinematic one a boundary condition imposes, the grid solve is unchanged, and the adjoint through
a pinned particle is the identity on x and F (its tape branch is a copy). Nothing is added to
the material: λ, μ, F_e, F_p of a pinned particle are the ones it arrived with, and the
delivered object (the archived x, F, F_p) responds to an external force after the morph through
its elastic parameters alone — the pin exists only while the morph runs, like the OT pace or
the window loop itself. Its cost is the one the freeze paid: a pinned particle cannot correct
its arrival error any more, so the fit can only lose from the pin's onset (P183 bounds it), and
the arrivals that come after it must flow around the pinned body instead of through it (P184
reads the compression that costs). Read by the frames alone: a pinned particle's frame-to-frame
step is 0 in float32, an unpinned one's never is (`scratch/pin_probe.py`).

Addendum (2026-09-25 morning) — the clear-neighbourhood pin (`settle_pin_clear`). The pin's
two costs on the gallery and at 300k are one mechanism: a pinned body is a fixed obstacle, and
material that still has to pass through or settle against it is compressed (g41z: det F
−0.05…−0.11 on six targets, the minimum in the UNPINNED set on cow / armadilo — the last
arrivals against the pinned boundary) or cut off (ap300: the ear's tip 3.9 reference particles
against ai300's 13.6, the last two slabs 0.66 / 0.18 against 0.89 / 1.13 — the base pinned
while the tip was still fed through it). The rule that keeps the channel open and the boundary
clear reads the paced target's own arrival scale:

```
(50b)  p ∈ P  only if  arrived_p ∧ rev_p ≥ 2  ∧  min_{q unarrived} |x_p − x_q| > r_pace,    r_pace = max(leash, Δx_loss)
```

r_pace is the radius within which the paced target declares a particle arrived (optimizer
`pace_r`); no constant is added. A particle next to material in transit stays free until that
material has arrived, so the pinned body's boundary is always one arrival radius inside the
settled region, and a channel through which material still flows is never pinned shut.

Addendum 2 (2026-09-25) — the pinned body is stress-free (`settle_pin_assim`). The per-window
det F series split by pin state (`scratch/detf_time.py`) shows what the pin's det F cost is: on
the bunny a particle pinned at window 30 with det F 0.762 keeps it for good (the un-pinned
material's own minimum recovers 0.75 → 0.84 by elastic expansion, as in g41), on the cow the
material still in transit compresses progressively (0.857 → 0.770 over windows 20–42 while the
pinned fraction rises 18 → 61 %) and relaxes to 0.80 once it settles. The pinned particle's F is
the TOTAL deformation (k_stress reads F_e = F F_p^{-1}), so a transient compression at the pin
is locked as elastic strain — and a pinned body with locked elastic strain is not at
equilibrium: once the morph ends and the object answers to λ, μ alone, that strain releases.
The settled body must therefore be pinned stress-free:

```
(50c)  at the pin of p:  F_p ← S_e F_p  with  R_e S_e = polar(F F_p^{-1})   (η = 1: F_e → R_e; the freeze's assimilation, 10.25)
```

F is kept (the transient compression becomes the rest volume there — a density excess of a few
per cent in a few particles, invisible), the pinned particle's stress is zero, the arrivals
settle against a wall that neither pushes nor pulls, and the delivered object is at rest. The
det F read on the archive stays the total deformation's; the elastic det F_e of the pinned set
is 1 by construction (P192).

Addendum 3 (2026-09-25) — the transit rays (`settle_pin_ray`). The pinned body is a wall, and
material whose plan image lies beyond it has to pass through it: on nefertiti the crown's
stream through the settled bust made every candidate 5 % worse at a reversal cosine of +0.94
(a transport window, no oscillation) and the run stopped at 42 of 90 windows under every pin
variant that pinned the bust (g41q, g41n, g41pz); at 300k the bunny's ear was fed through its
pinned base and the tip starved. The clear rule (50b) protects the boundary at pin time only;
the traffic is known from the plan itself — the paced target moves each particle along the
straight McCann ray from its position to its plan image — so the rule that keeps every stream
open is:

```
(50d)  p ∈ P  only if  arrived_p ∧ rev_p ≥ 2  ∧  min_{q unarrived, s ∈ [0,1]} |x_p − (x_q + s (y_q − x_q))| > r_pace
```

y_q the plan image of q; the rays sampled at r_pace and put in one kd-tree; a candidate within
r_pace of any sample stays free. Nothing is added: r_pace is the arrival radius, the rays are the
plan's. A settled region a stream still has to cross stays a yielding body (the no-pin
behaviour there) until the stream has arrived, and pins afterwards.

Correction to addendum 3 (2026-09-25, g41pr): the clearance radius is the grid kernel's
support, 2 Δx (the cubic B-spline), not r_pace. With r_pace (≈ 0.3 Δx at 40k) nefertiti stopped
at 38 of 90 windows exactly as before: a pinned particle carries its mass to every node of its
stencil, and a moving particle that shares a node with it receives the mass-weighted, near-zero
momentum — the pinned body is a no-slip wall with a boundary layer one support wide, and a
stream passing along the settled bust within that layer is dragged to a stop whether or not
the ray itself is clear. (50b) and (50d) therefore read `max(r_pace, 2 Δx)`: the discretisation's
own length. A settled region within one support of a transit ray stays a yielding body until the
stream has arrived; far from every stream it pins as before.

Addendum 4 (2026-09-25) — the settled body yields to a passing stream (`settle_pin_yield`).
With the stencil clearance nefertiti's crown stream is no longer stopped (g41ps: 114 windows,
silIoU 0.9727, the best fit of any run on it) but it is squeezed: between pinned walls two
supports apart the free set's det F falls 0.78 → 0.46 over windows 45–85 and the arrivals pin at
0.55. A body that is fed by a stream has to yield where the stream passes — in the un-pinned
run that yielding is a large part of what is seen as breathing. The rule keeps the yielding
and only the yielding:

```
(50e)  a settled particle within 2 Δx of a transit ray is RELEASED for the window:  pin_p = 0, control_p = 0, u_p = 0
       (passive material: it moves by the physics alone); it is pinned again when no ray is within 2 Δx, with F_e → R_e
```

The optimiser drives nothing in the released set (no step, no relaxation move), so it cannot
alternate; the motion there is the elastic response to the stream, which ends when the stream
has arrived. Far from every stream the body is exactly still, as before. The delivered object
is stress-free at every pinned particle (the re-pin assimilates).

Addendum 5 (2026-09-25) — the pinned body as a separating collider (`settle_pin_slip`). Every
pin variant so far kept the pinned particles' mass in the grid's momentum average: a node they
cover carries their mass with zero momentum, and the free material that shares the node is
dragged toward rest — the wall was no-slip by construction, which is what the course notes
call the Dirichlet (sticky) condition (Jiang et al. 2016 §12.1) and what every engine avoids
for a collider by applying the collision to the grid velocity after the forces, relative to
the collider's velocity, and only when approaching (taichi_elements / warp-mpm `separate`,
Houdini's collider projection). The pinned body is such a collider:

```
(50f)  P2G:  a pinned particle deposits nothing (no mass, no momentum, no stress force)
       once per window:  m_pin(node) = Σ_{p ∈ P} w_ip m_p                         [k_pin_mass]
       grid, after forces, at nodes with m_pin > 0:  n = ∇m_pin / |∇m_pin| (into the body),
           v ← v − n max(n·v, 0)                                                   [k_grid_op]
```

The free material's velocity at a shared node is its own; the approaching normal component is
removed, the tangential and any separating motion stay — a slip wall. The pinned particles
themselves keep (50): x, v = 0, C = 0, F fixed, and with addendum 2 they are stress-free, so
dropping their stress force from P2G changes nothing. The clearance and yield rules (50b–50e)
are geometric substitutes for this and are not needed with it; the boundary wedging of the
last arrivals (the over-fill at the arrival snap) is a separate item (dossier §14.0).

### 10.28 The arrival's capacity (2026-09-25 night; config `arrive_cap`; dossier §14.0)

The paced target declares a particle arrived within one pace radius of its plan image and
then replaces the image by the NEAREST target point (the entropic image lies inside the blur;
without the snap the end state was fuzzy — chamfer 0.098 against 0.076 at 150k). The snap has
no exclusivity: near a thin feature every arrival snaps to the same few target points and the
cell-sum objective packs the material into them. The dragon at 300k compresses at its spikes
under every discretisation (50–270 particles below det F 0.3; worse at the finer grid), and
under the pin the last arrivals wedge against pinned neighbours (nefertiti: 3–15 particles at
det F < 0.7, all touching pinned ones) — the same over-fill. The snap with capacity:

```
(51)  cap = N / |target points|   (the mass ratio; no constant)
      for each target point q: the cap closest arrivals whose nearest point is q snap to q; the rest keep their plan image
```

The surplus is still driven toward the feature by its plan image, but no longer to a point
already occupied at the target's density; a spike or an ear tip fills to the target's own
count and not beyond. Read by the end-state det F quantiles (dragon p1, the count below 0.3)
and by the wedged count on the pin runs (P212).
