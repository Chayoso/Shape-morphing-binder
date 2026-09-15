# Rendering controls physics — the contract, the literature, the changes (2026-09-14)

*Pre-registered design. Everything numerical in this file is a CPU-test measurement or a
literature number; no hyde06 run has been made under this contract yet (server access was
unavailable on 2026-09-14 — see §9).*

**Premise (fixed, user directive):** the render gradient must control the physics. The
only variables the image is allowed to move are physical — the control stress `dFc`, the
material field `s`, the plastic rest state `Fp`, the terminal velocity. No displacement
injection, no geometric post-op, and — new here — **no route by which the image can change
without material moving** (§3).

This document answers the ten questions posed on 2026-09-14 in order: gradient scale (§2),
holes (§7), grid dependence (§7), gradient conflict (§5), viewer (§8 → `docs/viewer.md`),
VBD/Chebyshev propagation (§6), PhysGaussian/PhysDreamer gradient flow (§1, §4), C++
parity (§1.2), oscillation (§9 → `docs/oscillation_triage.md`), and the literature
(§1). §10 is the pre-registered A/B ladder.

---

## §1 What the 2022–2026 literature does with a render gradient

Survey scope: 40+ papers checked against arXiv/publisher pages on 2026-09-14
(PhysMorph-GS excluded by instruction). The full table with arXiv ids is in
`docs/related_work.md` (updated); the mechanism facts that decide our design:

| pattern | who | consequence for us |
|---|---|---|
| **The image gradient is accumulated into a LOW-DIMENSIONAL physical field**, never a per-particle-per-step control: a triplane Young's-modulus field (PhysDreamer 2404.13026, 8³), a KAN material field (DreamPhysics 2406.01476), per-part log parameters (Resonance4D 2604.01994), a LoRA residual on a constitutive law (NeuMA 2410.08257), 2,048 spring anchors (Spring-Gaus 2403.09434), a global velocity (PAC-NeRF 2303.05512, GIC 2406.14927) | every GS+MPM inverse paper | §4: the control lives on a coarse node grid × time knots |
| **The horizon of the render→MPM gradient is short**: gradient only to the previous frame (PhysDreamer), M≪N frame mini-batches (OmniPhysGS 2501.18982), 10-step chained graphs (Xu et al. 2409.15746), 8-step batches (Bolliger 2512.13214), h=16–32 with a critic (SHAC 2204.07137: ‖g‖>1e6 beyond) | all | our T-step windows are already this; the running kinetic term (§9) is the in-window regulariser these papers use (kinetic-energy cost in Bolliger) |
| **Velocity/geometry first, material after; progressive unfreezing** (PAC-NeRF, GIC, NeuMA, MASIV 2508.01112, ProJo4D 2506.05317 — up to 10× accuracy) | all | the material channel `s` stays a second-stage leaf (`mat_lr_scale`); the control-grid arm runs geometry first |
| **The observation loss is moved off raw pixels** when it must scale: silhouettes/masks (GIC, MASIV), optical flow (PhysFlow 2411.14423 states the render loss "is restricted to subtle motions due to instability and noise"), Sinkhorn silhouettes (MonoPhysics 2605.30320), spectra (Resonance4D), Eulerian mass (Xu) | most | the hybrid silhouette + Gaussian channel stays; the Gaussian L1 gets a Charbonnier option (§2.3) |
| **Gaussians ride a kinematic F: Σ = F Σ₀ Fᵀ** (PhysGaussian 2311.12198, NeuMA's `A = Ā A₀ Āᵀ`, VR-GS 2401.16663) with F the MPM deformation gradient **transported by the velocity gradient** | all | §3: our render covariance must not see the control's direct addition into F |
| **No GS+MPM paper uses PCGrad-style surgery**; fixed additive weights + staging. The physics-domain evidence on surgery is cautionary: GradBlend (2607.25060) finds PCGrad/GradNorm inflate the primary metric 2–100× and anchors the step to the primary gradient; Adam's moments re-introduce conflict after any projection (2609.01558) | — | §5: modes implemented, default off, A/B pre-registered |
| **No VBD-for-MPM exists.** Closest: Position-Based MPM (Lewin, SIGGRAPH Talks 2024) — per-particle Jacobi constraint projection, unconditionally stable, artificially soft; APS-MPM (Visual Computer 2025) XPBD-style affine projection, 10× dt. VBD (2403.06321) uses Chebyshev with ρ=0.95 and *skips acceleration on colliding vertices*; the 2026 residual-driven Chebyshev (Visual Computer) estimates ρ online | — | §6 |
| **Holes** = numerical fracture at low particles-per-cell: fixes are particle resampling (Yue 2015 Continuum Foam), split/merge (CMAME 2023), C² kernels (CK-MPM 2412.10399), or decoupling quadrature from particles (MPM Lite 2602.07853); GaussianFluent (2601.09265) documents hollow surfaces/needle floaters when interior particles have no render attributes | — | §7 |

### §1.1 PhysGaussian vs PhysDreamer: how the gradient actually flows
PhysGaussian is **forward-only**: Gaussian = MPM particle, `Σ_p(t) = F_p(t) Σ_p(0) F_p(t)ᵀ`,
SH rotated by the polar rotation of F. Its solver (`warp-mpm`) is Warp-differentiable, and
that is what PhysDreamer uses: image loss (λ L1 + (1−λ) D-SSIM against video-diffusion
frames) → rendered Gaussians → Σ and centres → MPM adjoint (768 substeps/frame, gradient
truncated to the previous frame, checkpoint+recompute) → **a triplane E(x) and a triplane
v₀(x)**, optimised in two stages (v₀ on 3 frames, then E). The gradient never reaches a
per-particle quantity; ν is fixed. The scale problem we hit (a pixel-mean loss against a
grid-count loss) does not exist there because there is ONE data term and the parameter is
low-dimensional (Adam is scale-free per parameter, and the pixel signal is summed over
thousands of particles per triplane cell).

### §1.2 The C++ oracle, re-read against the Python bridge
`legacy/DiffMPMLib3D` injects a render covector once per backward pass at the terminal
layer (`accumulate_render_grads`, `render_gain_`) and balances by norm ratio
(`get_control_layer_grad_norm`). Its hand adjoint differs from the written forward in two
places (gradient_flow_audit, 2026-09-14, verified again here): the stored-F derivative
`(1-s)AᵀG + sG` is used as the `dFc` gradient (the direct control derivative has no `sG`),
and the F→C contribution omits `(1-s)` (a 22× factor at s=0.955). The Warp tape
differentiates the forward it was given, so the Python gradient is the exact one for the
Python forward; the finite-difference tests (`tests/test_render_control_gradient.py`,
`tests/test_ext_bridge.py`) pass at <3–5 %. **Do not "correct" Python toward the C++
adjoint.** What the C++ has that Python lacked has been ported: line search (v2), adaptive
α (v2), norm balancing (v2), and now the same terminal-covector injection through the
geometric F (§3).

---

## §2 The 10³–10⁴ gradient gap (question 1)

Measured (audit, 12k sphere→bunny initial state): ‖∂D_vol/∂dFc‖ = 26.9, ‖∂D_gauss/∂dFc‖ =
0.0025 (ratio 1.1e4); the transfer-function probe showed the MPM adjoint changes the ratio
by ≤1.65×. **It is loss units**: `D_vol = ½ Σ_cells (log-mass residual)²` sums over
262k grid cells with masses of tens of particles; the image loss is a mean over pixels in
[0,1]. Three consequences and their fixes:

1. **Direction is unaffected** — the λ balancer (`λ = α‖g_p‖/‖g_r‖`) already makes the
   composite direction unit-free. What was NOT unit-free: the cap `lambda_cap = 5e3`
   bound at the measured ratios (6.5e3–1.1e4), silently under-weighting the render
   channel by up to 2×. The balancer now reports `lambda_capped` in every commit record.
2. **Per-entry signal is noise-level.** The audit's single-entry finite-difference check
   of the image gradient FAILED at 12k (34 %/19 %/8 % error) while directional checks
   passed — the render gradient is right as a *field* and wrong per entry. This is the
   real reason a per-particle-per-step control cannot be driven by an image: §4.
3. **Reported numbers are incommensurable.** `loss_units="density"` (opt-in) replaces the
   cell sum by the per-support-cell mean of the residual on `m/m_ref`:
   `D = ½ (1/n_support) Σ_cells [log(1+m/m_ref) − log(1+m_t/m_ref)]²` —
   same minimiser, same log form, dimensionless. CPU measurement
   (`tests/test_loss_units.py`, 300k-particle ball, loss_res 24→48): legacy ×4.3, density
   ×1.5 (the residual band is under-resolved at 24; a Riemann-sum effect, not a unit
   effect). The optimizer's absolute constants (Adam `eps`, `target_norm`, the
   line-search noise floor) and every FIXED weight added to D_vol (`w_kin`,
   `w_kin_running`, `w_ctrl`, `w_tctrl`, `w_box`, `w_creg`, `w_jvol`, `w_cov`, `w_mat`,
   `w_dt`, `w_nn`, `w_fill`) are legacy-unit numbers; in density mode they are converted
   by TWO ratios MEASURED at the source state (`runner.calibrate_units`):
   `D_vol(legacy)/D_vol(density)` for the weights and `‖∇D_vol(legacy)‖/‖∇D_vol(density)‖`
   for the gradient constants; the calibrated terms (h1/jdens/kde) follow `dvol()`
   automatically, and the λ cap becomes RELATIVE (20× the first raw ratio) instead of
   the legacy-unit absolute 5e3. REFUTE 2026-09-15 F1: the first version used one
   analytic per-cell constant `n_support·2m_ref/(1+m_ref)`, which the reviewer measured
   at 4–45× below the true loss ratio and 5–44× below the gradient ratio (loss_res
   32/48/64: true 7.7e3/5.5e4/1.2e4 vs 609/1661/2879), and the two true ratios differ
   by 1.2–1.3×, so a single scalar could not serve both. Measured on the CPU smoke
   without any weight conversion: the kinetic penalty out-weighed the rescaled D_vol
   50:1 and no line-search step was accepted. The measured ratios make the relative
   weighting equal to legacy AT THE SOURCE only; they drift along the morph like the
   h1 calibration and are logged, not assumed constant. Density mode stays an A/B arm.

### §2.3 The image loss itself
The audit's render+loss segment disagreed with finite differences by 7–34 % where the
MPM pullback agreed to <0.04 %. `|r|` per pixel has a sign crossing wherever a
perturbation flips a residual; `gauss_robust_eps` switches to Charbonnier
`√(r²+ε²) − ε` (smooth, same minimiser, same units), ε = 0.02 in the new arms.

---

## §3 The image may change only through motion (audit Finding 1 → fixed)

Forward step (both Python and C++): `F_{t+1} = (1−s)(I + Δt C_{t+1})(F_t + c_t) + s F_t`.
The control `c_t` is added straight into the stored F. Rendering `Σ = σ₀² F Fᵀ` from that F
means the image can improve with **zero particle motion** (measured: at dt=0 or at zero
stiffness the image loss still moved, 0.0715→0.0715 with max centre change 0). Under the
premise this is illegal — the render gradient would be "controlling" a bookkeeping
variable.

**Fix (`render_F_geom=True`):** the engine now also carries a *geometric* deformation
`F_g`, `F_g,{t+1} = (I + Δt C_{t+1}) F_g,t` — velocity-gradient transport only, no control
addition, no temporal smoothing (the PhysGaussian/NeuMA kinematics). The render covariance
and the viewer use `F_g`; the constitutive law keeps the controlled, smoothed F. Tests:
at dt=0 with `c_xx=0.1` the stored F moves by `(1−s)·0.1` and `F_g` stays exactly I;
`F_g` equals the recomputed product of `(I+ΔtC)` to 1e-6; gradients through `F_g` match
central differences (<5 %). Runner: `F_frames` keeps the physics F (metrics, assimilation);
`Fg_commits` archives `F_g` at every accepted commit; the live viewer receives `F_g`
(`rec["F_kind"]="geom"`).

---

## §4 The control lives on a coarse basis (questions 1, 2, 4, 6, 7)

`pipeline/control_basis.py`: `dFc(t,p) = Σ_k w_k(x0_p) · B(t) · C[knot, node_k]` —
trilinear nodes on a `G³` grid over the window's start positions (one cell margin),
piecewise-linear in time over `K` knots (`K=1` is the C++ `control_stride=T` form). The
map is linear, frozen per window, exactly differentiable in torch before the Warp bridge;
partition of unity holds to 1e-6; nodes without support receive exactly zero gradient;
the per-particle mode (`control_grid=0`) is the identity (legacy path bit-for-bit).

Why this is the literature's answer to four of the questions:
- **SNR (q1):** one node accumulates the image gradient of ~ppc·(cell/dx)³ particles over
  `T/K` steps; at 40k particles, G=12, K=4 the DOF count drops from 7.2e6 to 6.2e4
  (`describe()` reports both), i.e. ~100 particle-steps vote on each coefficient.
- **Lone-particle actuation = hole factory (q2):** a node moves every particle in its
  support coherently; `w_creg` (the kNN Laplacian that existed to stop single-particle
  actuation) is redundant and is set to 0 in the new arms.
- **Conflict (q4):** most of the measured render/physics conflict lived in the
  high-frequency per-particle components that the basis cannot represent.
- **Propagation (q6):** "update one particle and let it propagate to the neighbours" is
  what a node update does by construction, without a Gauss–Seidel sweep on the state.
- **Grid dependence (q3):** the control resolution is decoupled from the MPM grid.

Warm start (`warm_start`) projects the previous window's *expanded* field onto the new
window's basis (`ControlBasis.project`). REFUTE 2026-09-15 F2: the first version was a
lumped `D⁻¹Wᵀ` restriction — a node-space smoother, `W D⁻¹ Wᵀ ≠ I` — that lost 38–61 %
of a basis field per application on every real grid, and its test passed only because
`G=3` was degenerate (a one-cell margin made `h ≈ 4e6`). Now: the node box is the bbox
expanded by 5 % per side (every `G ≥ 2` valid), and `project` is the least-squares
solution `(WᵀW)C = Wᵀd` by Jacobi-preconditioned CG (30 iterations, matrix-free);
measured reproduction error <2 % at `G ∈ {2,4,12}` and no contraction under repeated
application (`tests/test_control_basis.py`). Also noted (F11): `dfc_clip` on the node
leaf bounds every particle's control (convex combination) but conservatively — at
`G=12, K=4` the admissible set is strictly smaller than the per-particle one at the
same `dfc_clip`, so basis-vs-flagship comparisons at a fixed clip confound the basis
with the control budget; the ladder reports both clipped and unclipped runs. The kNN
control preconditioner (`control_h1_iters`) indexes particles and is EXCLUSIVE with a
node leaf (it raised an IndexError mid-window when combined; the optimizer now fails
fast at window start — the basis already does its job).

---

## §5 Combining the two gradients (question 4)

`pipeline/grad_combine.py`, selected by `grad_project` + `grad_project_mode`:
`"render"` = legacy one-sided PCGrad (render's conflicting component removed; physics
descent preserved), `"phys"` = the mirror image (render-first inside the conflict cone),
`"cagrad"` = two-task CAGrad (Liu 2021, exact 1-D simplex solve), `"blend"` = physics-
anchored magnitude with render steering the direction by a FIXED β = α_λ (the
GradBlend rule from 2607.25060). REFUTE 2026-09-15 F3: with β derived from the
balanced λ the blend was direction-identical to the plain sum (cos 1.000); with the
fixed β it is the norm-balanced composite without the balancer's EMA and cap, with the
step magnitude pinned to ‖g_p‖ — a distinct but modest variant, stated as such. All are search-direction rules; the fixed-λ window objective is still line-
searched, which is what guarantees descent. Measured history: standalone PCGrad was
falsified (v4), and removing it from the flagship was ≥ tie (h13); with the near-band
term the late-run conflict vanished (h15, cos −0.86 → 0.00). **Default stays off**;
the modes exist for the pre-registered ladder (§10), and every commit still logs
`g_raw_cos`, `g_cos`, `render_cos`, `phys_cos` so the *effective* conflict of the accepted
step (after Adam) is measured, per 2609.01558.

---

## §6 VBD / Chebyshev / Gauss–Seidel propagation (question 6)

What was already tried: a full VBD transplant to MPM grid nodes (quasi-static energy
minimisation; `deprecated/vbd`) — it converged to equilibria that crawl (per-term exit
gradients cancel; 8–16× slower; retired 2026-09-01, `docs/rationale.md` §4). A
Gauss–Seidel sweep that edits `x/F` after the rollout leaves `v`/`C` inconsistent and was
retired for that reason (`docs/oscillation.md`). No published VBD-for-MPM exists (§1).

What VBD/PD *do* contribute that is legitimate here — and is now implemented:
1. **Chebyshev semi-iterative acceleration** (Wang 2015; the acceleration VBD inherits)
   of the grid Gauss–Seidel that preconditions the render covector (`grid_smooth.py`,
   `render_gs_cheb`): `u_{k+1} = ω_{k+1}(γ(S(u_k)−u_k) + u_k − u_{k−1}) + u_{k−1}`,
   `ω₁=1, ω₂=2/(2−ρ²), ω_{k+1}=4/(4−ρ²ω_k)`, ρ = (κ/(1+κ))² for the red-black sweep of the
   screened diffusion, 2 warm-up sweeps, the Wang recursion restarted at ω₁=1 on the
   first accelerated step; below 8 sweeps the function runs plain sweeps because the
   accelerated iterate is worse there (REFUTE F6: κ=20, 4 sweeps 0.87 vs 0.48). The ρ
   estimate is the periodic bound; the boundary-truncated `avg6` makes the true ρ
   slightly smaller. CPU measurement (`tests/test_chebyshev_smooth.py`,
   6k particles, 16³ grid, error to the converged solution): κ=20 at 20 sweeps 1.5e-2 →
   4.9e-4 (30×), κ=4 at 12 sweeps 2.0e-4 → 7.9e-6 (25×), κ=50 at 40 sweeps 2.1e-2 →
   1.0e-4 (200×); the first ~4 sweeps are worse (warm-up transient, as in Wang 2015);
   both iterations reach the same limit (same linear system).
2. **Both covectors are propagated.** The smoothing branch dropped the F covector (audit
   Finding 2: 42–57 % of the pullback norm missing); the field is now (N, 3+9), smoothed
   with one operator, each channel's norm preserved, and both seeds go through the MPM
   adjoint.
3. The basis (§4) is the particle-level "propagate to the neighbours" mechanism.

What is NOT done, on purpose: a state-level VBD/GS solve. The engineering review's two
admissible integrations stand (a complete implicit MPM integrator, or an SPD control-space
preconditioner followed by the unchanged rollout + line search); the Chebyshev-GS above is
the second.

---

## §7 Holes and grid dependence (questions 2, 3)

Both are particles-per-cell effects. `mpm/discretisation.py` makes `dx` a function of
`(N, sampled volume, ppc)`: `dx = (V·ppc/N)^{1/3}`, tiled to the domain cube. The loss
grid follows (`loss_res = grid_n`) ONLY in density units — REFUTE 2026-09-15 F4: the
legacy `D_vol` is a cell sum, so letting `loss_res` follow dx (32 → 109 at 20k, ppc 8)
multiplied it ~535× (gradient ×97) against every fixed weight; in legacy units
`loss_res` is left as given. The Gaussian rest size is reported (`sigma0`), not
enforced (the loss keeps `sigma0_from_nn`). The stability numbers are reported with
every run (from the arms' actual material, not literals): `ρ = N m / V`, `c = √((λ+2μ)/ρ)`,
`CFL = c Δt/dx`, elastic period `2L/c`, kernel support / spacing. `pipeline_run.py --ppc 8`
enables it. `measure_ppc` reports the occupied-cell occupancy and the fraction of cells
below 4 ppc (the hole-risk telemetry). Default MPM numbers (dx 0.5, 20k, bunny volume ≈
65 wu³, unit mass): ρ≈300, c≈23, CFL≈0.19, ppc≈38 — stable; at 5k the same dx gives
ppc≈10; below ~4 ppc the cubic stencil loses partition of unity locally and tears
(Steffen 2008). The render-side answer to holes (the premise) is unchanged: the hole
side of the silhouette (`w_hole`) and the Gaussian children see holes at the render scale
and pull mass to close them; the basis (§4) stops the optimizer from *creating* them.
Particle splitting (Yue 2015) is deferred: variable `N` breaks the archive/metric
contracts (`frames` is (M,N,3)), and at ppc ≥ 8 it has not been needed.

---

## §8 Real-time viewer (question 5) → `docs/viewer.md`

Decoupled: runs publish atomically to a directory (`--live_dir`), a standalone
`scripts/viewer_serve.py` serves every run under a root on one port (list, live follow,
replay scrub of accepted commits, 2/4-run comparison), and a local `scripts/viewer_tunnel.py`
keeps the SSH tunnel alive and reconnects. The in-process `--live_port` mode is unchanged.

---

## §9 Oscillation (question 9) → `docs/oscillation_triage.md`

The dossier closed the *optimizer-side* zigzag (sub-spacing, invisible). What the user
sees during the simulation has three candidate drivers, and the triage probe
(`scripts/probes/oscillation_triage.py`) classifies a run archive by pre-registered rules:
(A) volume (J breathing), (B) stiffness (elastic ringing at `2L/c`, CFL), (C) window
stop-and-go (per-window terminal-rest objective + zero-restart controls). The mechanism
fixes that exist for each: A — `w_jvol`, isochoric assimilation (adopted); B — the
discretisation contract (§7) reports CFL and the elastic period against the window
length; C — the running kinetic term `w_kin_running · mean_t mean_p |v_t|²`, time
knots (`control_tknots`), and — added after the first hyde06 archives — the
**velocity-variance term** `w_kin_var · mean_p [mean_t |v_t|² − |mean_t v_t|²]`, which is
zero for constant-velocity motion and positive for any reversal inside the window.

**Measured 2026-09-15 (hyde06, 20k, T=20, dt=1/240, dx=0.5; `docs/experiments.md`):**
the in-simulation vibration is a **window-locked control limit cycle**. Inside every
window the mean speed goes 0.47 → 0.10 → 0.45 with the turning point mid-window and
continuity across the boundary; 95 % of the speed power sits at period T; the elastic
period is 132 steps, J peak-to-peak 0.008, CFL 0.24 — neither stiffness nor volume. The
probe's original sprint-then-brake rule missed it (sag 0, jump 0.93); driver C now also
fires on intra-window speed modulation > 2 (measured 2.7–4.1 on every arm). Remedies
measured against the pre-registered falsifier (window-locked power < 0.5, visible
fraction < 1 %, no chamfer regression > 2 %):

| remedy (baseline arm) | s̄ | kin_var | power @T | visible | chamfer / silIoU |
|---|---|---|---|---|---|
| none (w_kin 0.5) | 0.308 | 0.128 | 0.95 | 9.6 % | 0.1599 / 0.9655 |
| w_kin 5 | 0.206 | 0.042 | 0.81 | 5.9 % | 0.1593 / 0.9663 |
| w_kin_running 10 | 0.247 | 0.064 | 0.85 | 7.3 % | 0.1586 / 0.9668 |
| warm start + w_kin 5 (continuity across windows) | 0.222 | 0.048 | 0.90 | 10.5 % | 0.1588 / 0.9655 |
| w_kin_var 10 | 0.248 | 0.057 | 0.84 | 10.5 % | 0.1589 / 0.9644 |
| w_kin_var 50 | 0.134 | 0.0083 | 0.58 | 2.5 % | 0.1596 / 0.9655 |
| w_kin 5 + w_kin_var 50 | 0.113 | 0.0053 | 0.50 | 2.1 % | 0.1591 / 0.9651 |

Reading: the cycle is NOT caused by the cold start of each window's control (warm start
leaves the power at 0.90) and is only attenuated by kinetic magnitude penalties; it is
the optimizer's own per-window solution under a terminal-only objective — a push-and-
return trajectory costs nothing there. The variance term prices exactly that reversal
and leaves progress free: at 50 it halves the window-locked power and cuts the visible
fraction 4× with chamfer/silIoU/holes unchanged and G3 passing with 2.5–5× margin. The
falsifier is not yet met (power 0.50–0.58, visible 2.1–2.5 %); the dose-response
continues at w_kin_var 200 (batch f). The coarse control basis makes the same cycle
spatially coherent (excursion p99 1.0–1.3 sp vs 0.6–0.8), a second reason the
per-particle flagship stays.

---

## §10 Pre-registered ladder (to run on hyde06)

All arms N=20k/40k sphere→bunny (real-volume sampler), T=20, dt=1/240, dx=0.5 (or `--ppc 8`),
300 commits, pace 0, the adopted recipe; every number reported with its discretisation.

| arm | what changes | falsifier |
|---|---|---|
| `render_full_dt_iso_nn` | baseline (flagship) | — |
| `render_ctrl` | + basis 12³×4, F_g render, running kinetic 1.0, Chebyshev GS, w_creg 0 | chamfer > flagship +2 % or holes ↑ |
| `render_ctrl_gauss` | + hybrid 3DGS image loss on surface parents, Charbonnier 0.02 | d_gauss not below `render_ctrl`'s at equal d_vol |
| `render_ctrl_first` | + `grad_project_mode="phys"` | any G2 guard, or chamfer +2 % |
| `render_ctrl` + `--loss_units density` | density units | λ trace O(1); loss_res 32 vs 64 changes d_vol by <1.6× |
| `render_ctrl` + `--ppc 8 --loss_units density` | dx from N, loss grid follows | ppc telemetry; hole_frac vs fixed dx at 5k |
| grid sweep 6/12/24 | basis resolution | non-monotone → basis is not the lever |
| triage probe on every arm | oscillation drivers | driver C must vanish with `w_kin_running`; else B/A per the rules |

Acceptance is the existing gate set (G1–G6) plus the triage verdict; adversarial REFUTE
(Codex gpt-5.6-sol xhigh + Opus) before adoption, per AGENTS.md rule 5.

**Status 2026-09-15:** everything above is implemented, REFUTE-reviewed (`docs/reviews/refute_rcp_opus_20260915.md`, 12 findings fixed/answered) and covered by 58 new CPU tests
(`tests/test_ext_bridge.py`, `test_control_basis.py`, `test_chebyshev_smooth.py`,
`test_grad_combine.py`, `test_discretisation.py`, `test_loss_units.py`,
`test_render_controls_physics.py`, plus the viewer and triage suites). The ladder has NOT
run: `ssh -J chayo@hyde01.dabh.io` rejected the local key (`Permission denied
(publickey,password)` at the jump host) all day; the JumpCloud key sync must be re-done
by the user before the commands in `docs/experiments.md` can be launched.
