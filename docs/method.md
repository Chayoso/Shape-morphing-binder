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
(`tests/test_material_bonds.py`: FD within 5 %). Momentum bookkeeping: the decoupled
particle's momentum change is not returned to the neighbours (one particle against the
body; recorded, not hidden). An explicit bond SPRING with stiffness (6/K)(λ+2μ) r was
implemented first and rejected: integrated explicitly with a multi-wu extension it is
unstable (dragon: 9 % of particles far, frozen at anim 23).
