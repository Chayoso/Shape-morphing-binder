# Local(render) + Global(physics) split solver — design (lg2)

Status: **DESIGN — not implemented**. Date 2026-09-02, branch `v3-grid-gs`.
Inputs: `docs/root_analysis.md` (the diagnosis), `docs/oscillation.md`, `docs/floaters.md`
(forensic dossiers), `docs/method.md` (formulation contract),
`docs/engineering_review_20260902.md` (architecture constraints), the retired
`physmorph/pipeline/surface_local.py` and its `lg_sweeps` guard in `runner.py`.

**Mechanism in one paragraph.** The global solver is untouched: MPM windows, volumetric +
hybrid render losses, full adjoint to `dFc`, line search, outer fixed-merit gate. A new
LOCAL channel runs once per accepted commit and owns the frequencies the global channel
measurably cannot reach (root_analysis: render-attributable work 1e-6..1e-3 vs physics
0.03..33). It has two tiers, neither of which touches `x/v/C/F`. **Tier D (dressing)** is
a pure observation change: bounded, tangent-plane, sub-Nyquist degrees of freedom on the
existing massless render children, optimized directly against the Gaussian image loss —
no MPM adjoint in the loop, so the high-frequency render signal is used at full strength
where it lives. **Tier R (rest-state demand)** revives the retired surface-band solver but
inverts its one fatal move: the band displacement `u` it computes is **never applied to
the state**. Instead its strain content is absorbed as a *virtual* deformation
`F_virt = exp(clamp(sym ∇u))·F` through the **existing** plastic-assimilation contract
(isochoric, SV-band, per-particle exact), so the next global window's *real*
elastodynamics — mass, momentum, grid transfer all untouched — physically transports the
surface toward the render-preferred shape. High-frequency corrections thus route *around*
the low-pass MPM adjoint (they never need to survive it as a gradient), while every
realized motion still comes out of the unchanged forward physics and is arbitrated by the
unchanged outer gate.

---

## §1 Problem restated as requirements

`docs/root_analysis.md` (all measured): the visually important modes — surface-local,
fine-scale, thin features — are

- **R1 unobservable**: CIC/silhouette are low-pass; floaters live in the loss null space
  (nn-band forensic: making one null-space mode observable killed its floater class);
- **R2 uncontrollable**: the render covector's unique content is high-frequency; the
  cubic-B-spline P2G/G2P adjoint plus F-smoothing 0.955 attenuate exactly that.
  Render work share in accepted steps ~0.02%, unchanged after the sub-pixel splat fix.
  Scale gap: grid dx=0.5 vs splat σ≈0.04 — 12×, crossed twice by the gradient;
- **R3 flat**: near-optimum curvature in those modes is ~0; the iterate wanders the
  valley (oscillators ON the surface at 1.2× spacing). Drivers #1–#5 removed the noise
  sources; the flatness is the objective's property.

What a fix must therefore do (root_analysis "what the frame dictates"):

1. observe at the visual scale (Gaussian forward model — partially done: gauss hybrid,
   Nyquist res floor);
2. **route fine corrections around the low-pass** — they must not ride `dFc` through the
   dynamics;
3. respect the state contract: no partial state edits after an explicit rollout
   (engineering review: VBD-style partial updates leave `v`, APIC `C`, interior state
   inconsistent); anything permanent outside the rollout goes through the
   plasticity/assimilation contract or is a pure observation/parametrization change;
4. no new tuned scalars without a pre-registered calibration rule.

Why the retired `surface_local.py` failed, precisely (its own docstrings + the
`lg_sweeps` guard + the h15 verdict):

- **F1 state overwrite**: it wrote `x' , F'` after the window, leaving `v`/`C`/interior
  inconsistent — the exact inadmissible move the review names;
- **F2 objective mismatch**: its energy excluded the one-signed W1/fill terms, so it could
  undo an accepted W1 step and assimilation then ratcheted the regression (the
  `ValueError` guard in `runner.py:run_pipeline`);
- **F3 blind objective**: its render term was the *same* CIC soft-silhouette the global
  window already exhausts — "the band had nothing left to descend" (h15 lg isolation:
  tie at 3.3× wall-clock, λ_loc pinned at cap). It shared the global loss's null space,
  so it inherited the global loss's blindness.

lg2 keeps F1 impossible by construction (nothing applies `u`), dissolves F2 (the demand
competes *inside* the next global solve where all terms live, and the outer gate can
reject the outcome), and fixes F3 (the local objective is the Gaussian image loss at the
Nyquist-floored resolution — the channel whose unique content is precisely what the
silhouette cannot see).

---

## §2 Literature grounding (read/verified 2026-09-02)

| # | paper (venue, link) | what it contributes to THIS design |
|---|---|---|
| 1 | **Projective Dynamics: Fusing Constraint Projections for Fast Simulation** — Bouaziz, Martin, Liu, Kavan, Pauly, ACM TOG 33(4), SIGGRAPH 2014. https://users.cs.utah.edu/~ladislav/bouaziz14projective/bouaziz14projective.pdf | The canonical local/global alternation: cheap per-constraint nonlinear *projections* onto admissible manifolds, coupled to a global solve through auxiliary variables. lg2's local step is a projection of the render residual onto *admissible rest-state increments* (bounded stretch, isochoric); the coupling variable is `Fp`, and the "global solve" is the full MPM window. |
| 2 | **DiffPD: Differentiable Projective Dynamics** — Du, Wu, Ma, Wah, Spielberg, Rus, Matusik, ACM TOG 41(2), 2021. https://arxiv.org/abs/2101.05917 | Differentiating an implicit local/global solve is only unbiased at *convergence* (and reuses the forward factorization). lg2 rule: the local band solve's convergence gate is a **correctness** gate — an unconverged solve donates no demand. Matches our own probe (`scripts/probe_gs_differentiability.py`: exact at ‖∇E‖ tol, 17–54% error at 10 blind sweeps). |
| 3 | **Vertex Block Descent** — A. H. Chen, Liu, Yang, Yuksel, ACM TOG 43(4), SIGGRAPH 2024. https://graphics.cs.utah.edu/research/projects/vbd/ | What IS usable after our review rejected naive partial updates: the *solver machinery* — per-block energy-monotone Gauss-Seidel with coloring and diagonal preconditioning — applied to a **complete variational subproblem in its own DOFs**. The band solve for the *virtual* field `u` (Dirichlet-anchored interior) is such a problem; applying block updates to post-rollout dynamic state is not, and stays banned. |
| 4 | **As-Rigid-As-Possible Surface Modeling** — Sorkine, Alexa, SGP 2007. https://dl.acm.org/doi/10.5555/1281991.1282006 | The grandfather split (closed-form local rotation fit / global Poisson solve) and its warning: the guarantees come from both phases descending ONE energy. lg2 deliberately gives that up (two objectives at two scales) and substitutes the outer fixed-scale merit gate + rollback as the arbiter — stated, not hidden (§11.1). |
| 5 | **Sobolev Active Contours** — Sundaramoorthi, Yezzi, Mennucci, IJCV 73(3):345–366, 2007. https://link.springer.com/article/10.1007/s11263-006-0635-2 | L2 shape gradients are pathologically high-frequency; Sobolev metrics yield coherent, stable shape flows that favor coarse motions and dodge local minima. Grounds solving `u` in a screened/H1 metric on the band (and explains the raw render pull's jitter). Note our measured caveat: the H1 metric belongs on the *shape field u*, not on the control `dFc` — the kNN/H1 control preconditioner was tested and worsened everything (method.md §6). |
| 6 | **Repulsive Curves** — Yu, Schumacher, Crane, ACM TOG 40(2), 2021. https://www.cs.cmu.edu/~kmcrane/Projects/RepulsiveCurves/ | Resolution-independent large steps for geometric energies via (fractional) Sobolev preconditioning. Lineage of our grid-GS smoother (method.md §6) and of the band solve's diagonal preconditioner + line search. |
| 7 | **2D Gaussian Splatting for Geometrically Accurate Radiance Fields** — Huang, Yu, Chen, Geiger, Gao, SIGGRAPH 2024. https://surfsplatting.github.io/ | Collapse 3D Gaussians to oriented planar disks for surface-accurate splatting. Grounds Tier D's *tangent-plane-only* dressing DOFs (a surfel's freedom is exactly in-plane) and the upgrade path if isotropic children still bleb. |
| 8 | **High-quality Surface Reconstruction using Gaussian Surfels** — Dai, Xu, Xie, Liu, Wang, Xu, SIGGRAPH 2024. https://dl.acm.org/doi/10.1145/3641519.3657441 | z-scale→0 flattening improves optimization *stability* and surface alignment — evidence that removing the normal DOF (as Tier D does) helps rather than hurts the fit. |
| 9 | **SuGaR: Surface-Aligned Gaussian Splatting…** — Guédon, Lepetit, CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Guedon_SuGaR_Surface-Aligned_Gaussian_Splatting_for_Efficient_3D_Mesh_Reconstruction_and_CVPR_2024_paper.html | Binding Gaussians to a surface (regularization/attachment) makes them editable and non-floating. Tier D binds every dressed primitive to a material parent with bounded offsets — an unsupported *dressed* splat is impossible by construction. |
| 10 | **Mip-Splatting: Alias-free 3D Gaussian Splatting** — Yu, Chen, Huang, Sattler, Geiger, CVPR 2024 (best student paper). https://niujinshuchong.github.io/mip-splatting/ | Primitive size must be constrained by the sampling frequency (3D smoothing filter). Grounds the existing Nyquist res floor in `gauss_loss.py` and Tier D's amplitude/scale band: dressing capacity is deliberately limited to sub-Nyquist content so it *cannot* repaint macroscopic geometry. |
| 11 | **PhysGaussian: Physics-Integrated 3D Gaussians for Generative Dynamics** — Xie et al., CVPR 2024. https://xpandora.github.io/PhysGaussian/ | Same kernels for simulation and rendering; splat kinematics driven by the MPM deformation gradient. Grounds `Σ = σ0² F Fᵀ`, child advection by `F`, and the general "what is seen is what is simulated" contract Tier D must not break. |
| 12 | **Stress-dependent finite growth in soft elastic tissues** — Rodriguez, Hoger, McCulloch, J. Biomech. 27(4):455–467, 1994. https://pubmed.ncbi.nlm.nih.gov/8188726/ | The multiplicative decomposition of deformation into a *commanded rest-state change* plus elastic accommodation — the physical license for Tier R: editing the zero-stress reference (`Fp`) and letting real elasticity realize it is textbook morphoelasticity, not state hacking. Our `assimilate_growth` already cites this lineage; Tier R is its deviatoric sibling. |
| 13 | **Variational multirate integrators** — Ober-Blöbaum et al., arXiv:2406.12991 (and Stability of AVIs, JCP 2008). https://arxiv.org/abs/2406.12991 | Different parts of one mechanical system may legitimately run on different clocks inside a variational framework — supports the per-commit local / per-window global cadence — with the standing caveat that resonance between the two clocks is a real failure mode (→ §11.5 telemetry). |

---

## §3 Design overview — division of labor by frequency

```
wavelength:   body scale ──────── dx=0.5 ──────── NN spacing≈0.04 ──────── 0
              │  GLOBAL (unchanged)  │   TIER R (demand)     │  TIER D (dressing)
carrier:      │  dFc via MPM adjoint │  Fp via assimilation  │  render parametrization
signal path:  │  loss → adjoint      │  local solve → rest   │  local solve → observation
              │  → control → rollout │  state → NEXT rollout │  (no physics at all)
```

- **Global (unchanged)**: coarse transport. `optimize_window` and its losses, adjoint,
  λ-balancing, line search, outer merit gate, plateau freeze — byte-for-byte the current
  blessed path. The surface restriction stays where the review put it: on the terminal
  observation covector.
- **Tier R** owns render corrections at wavelengths the grid can transport (≳dx) but the
  adjoint attenuates. It converts them from a *gradient that must survive the adjoint*
  into a *rest-state demand the forward physics realizes*. One crossing of the low-pass
  (forward), not two.
- **Tier D** owns wavelengths below what grid forces can shape at all (sub-dx surface
  texture, pinholes, splat-scale arrangement — where the late-run oscillators live).
  These are *unphysical for this discretization by definition* (no grid force can create
  them), so the only honest place for them is the observation layer, with capacity
  band-limited so it can never impersonate transport.

Both tiers preserve the state contract trivially: **neither writes `x`, `v`, `C`, or
`F`** — the two failure classes that killed `surface_local.py` and the displacement
assimilation (commit-boundary kin 66→509) cannot occur.

---

## §4 Tier D — surface dressing (observation layer)

### 4.1 DOFs

The render children already exist (`render/children.py`): per surface parent, up to 4
massless splats at frozen tangent-PCA offsets, advected by the parent's `F`, sharing the
parent covariance. Tier D makes the *coefficients* live:

- per child `c` of parent `p`: tangent coefficients `(a_pc, b_pc)` in the **frozen**
  source-material basis `(t1_p, t2_p)` (the PCA basis `tangent_child_offsets` already
  computes), plus one log-scale `s_pc`;
- effective material offset: `off_pc = baseline_pc + a_pc·t1_p + b_pc·t2_p` — no normal
  component (2DGS/Gaussian-Surfels: the surfel's freedom is in-plane; removing the
  normal DOF is what makes it stable and non-repainting);
- rendered as now: `center = x_p + F_p·off_pc`, `Σ_pc = (σ_child·e^{s_pc})² F_p F_pᵀ`.

Hard caps (structural, §7): `|off_pc| ≤ 0.5·h_src` (`h_src` = frozen source-surface
median NN spacing), `s_pc ∈ [log cov_smin, log cov_smax]` (the band `w_cov` already
declares), zero-centroid per parent re-projected after every step (the children.py
contract: parent COM exact). **No opacity DOF** — the floaters dossier's contract that
"tearing cannot be rewarded by a disappearing render loss" stays absolute; opacity
remains the target-free export-only support check.

### 4.2 Solve

Per accepted commit, on the promoted terminal state with everything else frozen:
minimize `L_gauss(x_T, F_T; dressing)` (the existing `GaussViews.loss`, all views,
Nyquist-floored res) over the dressing coefficients by the line-searched Adam recipe of
`optimizer.py` (fixed max iters, backtracking, monotone acceptance, convergence gate on
‖g‖). Deterministic given the state (modulo CUDA rasterizer atomics — same caveat and
the same noise-floor tolerance as the existing replay check).

During the *next* window the dressing is **frozen**, so the window's objective is fixed
(the property the line search and the "single objective per window" λ rule require). By
the envelope argument, evaluating the window's gauss term at the locally-optimal frozen
dressing means the global gradient carries only what dressing *cannot* explain — the
coarse residual. That is the routing statement made precise: sub-Nyquist content is
explained locally and stops leaking into `dFc` as flat-valley gradient noise.

### 4.3 Invariants

Touches nothing physical: `x/v/C/F/Fp`, particle count, mass, momentum all bit-identical.
Children stay massless observation-only (children.py contract). Finiteness: line search
rejects non-finite energies; coefficients live in compact boxes. All gate metrics are
computed on raw particle state and never consume the renderer (`metrics.py` contract), so
**dressing cannot move a single gate number** — it can only move the G6 visual QA and the
gauss-residual telemetry. That is deliberate anti-gauge armor (§11.3).

---

## §5 Tier R — rest-state demand through the existing assimilation contract

### 5.1 The virtual band solve

Revive the `SurfaceLocal` machinery (band = occupied cells with an empty 6-neighbor;
DOFs = CIC corner nodes with weight ≥ w_min; interior pinned as Dirichlet anchors;
8-color energy-monotone block descent with diagonal preconditioning, line search,
convergence gate — the VBD-lesson solver, kept verbatim) with two changes:

1. **objective**: `E(u) = Σ_p V_p ψ_SNH((I+∇u)F_e) + w_box·leash + λ_loc·L_gauss(x+u, (I+∇u)F; dressing frozen)`
   — the Gaussian image loss replaces the silhouette (fix for F3: this objective sees
   sub-cell arrangement, lone splats linearly via L1, and thin-feature holes at the
   Nyquist-floored resolution);
2. **output**: `u` is returned as a *virtual* field. `x`, `F` are NOT updated. Ever.

λ_loc comes from the dedicated local `LambdaBalancer` in u-space (diag-curvature ×
trust-radius vs ‖∂L_gauss/∂u‖ — the retired pass's own calibrated recipe, kept).

### 5.2 Demand extraction and absorption

Per band particle (interior: demand = identity):

```
e_p     = clamp_norm( sym(∇u(x_p)), r_p )          # strain-only; rotation demand discarded
A_p     = exp(e_p)                                  # SPD stretch, eigendecomposition
F_virt  = A_p · F_p                                 # virtual total deformation
Fp_new  = assimilate_elastic(F_virt, Fp, eta=cfg.assim, smin, smax, isochoric=cfg.assim_iso)
```

with `r_p = min( 1/3, τ )` and the throttle `τ` defined in §7. The **one existing
assimilation call absorbs global + local in one ratchet** (`_assimilate` unchanged):
with `F_e' = A_p F_e = R'S'`, the contract gives `Fp ← clamp_sv(S'^η Fp)`, hence the
*actual* new elastic state is `F_e,new = F_e S'^{-η}` — when the demand asks for
expansion, the material is left relatively compressed against its new rest state and
**real elastic stress pushes the surface outward during the next window**, through
untouched P2G/G2P mass and momentum transfer. When `u = 0`: `A = I`, `S' = S_e`,
`F_e,new = R_e S_e^{1-η}` — **bit-exactly the current path**. The mechanism is
morphoelastic remodeling (Rodriguez–Hoger–McCulloch), the pattern `assimilate_growth`
already uses for the volumetric channel; Tier R is its deviatoric sibling driven by the
render residual instead of the coverage-shortfall field.

### 5.3 Invariants, step by step

| step | finite | det F > 1e-4 | SV band on Fp | det Fp = 1 (iso) | x/v/C/F/mass/momentum |
|---|---|---|---|---|---|
| band solve (virtual u) | line search rejects non-finite E | state untouched | untouched | untouched | untouched |
| `e_p` clamp, `A_p = exp(e_p)` | eig of sym 3×3, bounded box | `A_p` SPD, `det A_p ∈ [e^{-0.58}, e^{0.58}]` at r=1/3 — but irrelevant: `F` itself is untouched | — | — | untouched |
| assimilation of `F_virt` | rows with `det F_e' ≤ 1e-6` skipped (contract) | `F` untouched ⇒ unchanged; next window's `_state_ok` still enforces whole-trajectory det | clamp LAST (contract) | alternating log-space projection (contract, f16 fix) | only `Fp` changes |
| next window | standard guards | standard `_state_ok` + trajectory det | — | — | standard promotion |

What is deliberately *not* preserved: the monotone elastic-energy-decrease property of
plain assimilation. Tier R **injects** elastic energy — that is its fuel. It is bounded
(‖e_p‖ ≤ r_p, band-limited support), logged (`lg2_demand_energy = Σ V_p[ψ(F_e,new) −
ψ(R_e S_e^{1−η})]`), and gated (§10 kill K-R2: window-start kinetic kick ≤ 2× the
matched baseline's median — the displacement-assimilation spike, kin 66→509, is the
pre-registered failure signature to watch).

Volume demand is structurally impossible: the isochoric projection keeps `det Fp = 1`
exactly, so Tier R is deviatoric-only by inheritance; commanded volume remains the
separately-governed `w_grow` channel's monopoly.

### 5.4 Why this addresses R2 (the dead channel)

The render covector no longer has to survive `(cubic B-spline adjoint)² × smoothing^T`
to become motion. The local solve reads it at full strength in state space, converts it
to a demand, and the demand becomes motion through *forward* physics — which is a
low-pass on each step but integrates to arbitrary above-dx displacement over a window.
The telemetry that proves/kills it: `lg2_realized_cos` = cos(u_demand, Δx of the next
committed window on the band) and `lg2_realized_frac` = ‖projection‖/‖u‖ (§8 P-render).

---

## §6 What the global solver keeps (and the envelope semantics)

- `optimize_window`: unchanged losses, λ_R once-per-window, PCGrad-free flagship stack,
  line search, `_state_ok`, replay validation. The gauss term evaluates with the frozen
  dressing (one added argument through `TargetPack`).
- `runner.py` outer loop: promotion, guards, outer fixed-scale merit gate, plateau
  freeze on raw λ-free tracks, rollback + cold restart — all unchanged. The freeze/gate
  components stay **dressing-free** (D_vol, silhouette d_render, dt, fill on raw
  particles): the gate cannot be gamed by the observation layer (§11.3).
- Differentiated vs frozen: `dFc`, `s` differentiated through the MPM tape (unchanged);
  dressing differentiated only inside Tier D's local solve; `u` differentiated only
  inside Tier R's local solve; `Fp` never differentiated (as now); nothing new enters
  the wp.Tape.

---

## §7 Every constant, with its pre-registered rule (no free weights)

| constant | value | rule (evidence-bearing, pre-registered) |
|---|---|---|
| λ_loc | balanced | the local `LambdaBalancer` in u-space (diag-curvature × trust radius vs ‖∂L_gauss/∂u‖, own EMA) — the retired pass's calibrated recipe, reused unchanged. Never shared with the global balancer (documented poisoning). |
| dressing amplitude cap | `0.5·h_src` | Nyquist: a tangent offset ≤ half the sample spacing can re-distribute coverage between neighboring parents but cannot represent (hence cannot repaint) any feature the particle sampling itself resolves (Mip-Splatting's sampling-bound argument in material space). A/B `{0.5, 1.0}` in the ladder; pick by raw-chamfer-tie + max gauss-residual drop. |
| dressing scale band | `[cov_smin, cov_smax]` | the band the existing `w_cov` term already declares as viewer-legal anisotropy/scale. No new constant. |
| ∇u strain cap `1/3` | fixed | provable, not tunable: ‖sym∇u‖ ≤ 1/3 ⇒ all eigenvalues of `e_p` in [−1/3,1/3] ⇒ `A_p` SPD with κ(A) ≤ e^{2/3}; and `A` never touches `F` anyway. Chosen as the largest cap that keeps the small-strain reading of `I+∇u ≈ exp(e)` honest to <6% error. |
| demand throttle `τ` | `median_p‖log S_e,p‖` of the just-committed window (band particles) | *the local channel may not demand more strain than the physics is currently carrying*: self-calibrating (no constant), self-terminating (at convergence accepted strain → 0 ⇒ τ → 0 ⇒ demand → 0 — the ratchet cannot run after the freeze), and scale-free. Falsifier: the held-phase probe (§10 K-R4). |
| band solve sweeps / tol | `lg_sweeps=10`, `tol=5e-2` | existing defaults of the retired solver; the convergence flag is a **correctness gate** (DiffPD): `lg_converged=False` ⇒ **no demand this commit** (dressing keeps its last monotone-accepted value — safe by construction). |
| dressing iters | 20, all views | fixed budget; convergence gate may stop early; determinism requires the full view set (no stochastic view subsampling). |
| when the pass runs | accepted commits only | null/rejected commits donate no demand and no dressing change — retrying an identical stale state must not accumulate anything (the s4 absorbing-state lesson). |

---

## §8 The three symptoms → three falsifiable predictions

All at matched seed/budget/discretisation vs the flagship baseline
(`render_full_dt_iso_nn` + gauss hybrid, N=20k stage / 40k hero, T=20, dt=1/240, dx=0.5,
smoothing 0.955), telemetry already in `runner.py` history records unless marked new.

**P-float (unobservable modes).** The local gauss objective sees lone splats linearly
(L1, no α-saturation) and thin-feature holes at Nyquist-floored res; hole-side residual
at ears becomes outward demand where fill-v3's loss-side pull dies with the physics
gradient. Predict: `out_nn_frac` (>2·sp) improves ≥20% and `out_nn_far_frac` (>4.5·sp)
does not regress; bunny fork lo-band count ≤ 45 (h18 baseline 64, −30%); ear cov<0.3
reaches the <15% target h18 missed (18.7%). **Kill**: `out_nn_frac` regression >5%, or
far-count regression, at the stage-3 budget.

**P-osc (flat valley).** Once dressing explains sub-Nyquist residual, the gauss term's
flat-valley gradient noise leaves the global problem (envelope, §4.2); the surviving
render gradient is coarse and coherent. Predict: over the last 40 pre-freeze commits,
`reversal_cos` min > −0.2 **without a single outer-gate rejection needed** (the gate
stays armed but idle), and tail mean `move` ≤ 0.5× baseline (h15: 0.003–0.004 → ≤0.002);
freeze commit index ≤ baseline's. **Kill**: rev-cos min below the matched baseline's, or
freeze fails inside the baseline's freeze budget +10%.

**P-render (dead channel).** New telemetry: `lg2_demand_work = −⟨g_render,x, u⟩` at
commit, `lg2_realized_cos/frac` = alignment of the next committed window's band Δx with
u. Predict: median `lg2_realized_cos ≥ 0.3` after commit 20, and total
render-attributable work share — `(render_work + realized demand work) / (phys_work +
…)` — rises ≥10× from the measured ~0.02% floor (i.e., ≥0.2%, target O(5%)). **Kill**:
median `lg2_realized_cos < 0.1` by commit 40 — physics is ignoring the demand and the
channel is dead again; Tier R is falsified regardless of endpoint metrics.

---

## §9 Integration plan (file:function, cadence, cost)

### 9.1 Code changes

| where | change |
|---|---|
| `physmorph/render/children.py` | add `DressingState` (per-child `(a,b,s)` in the frozen `(t1,t2)` basis + caps + zero-centroid projection); `expand_children_torch` accepts optional live coefficients; numpy twin for export parity. |
| `physmorph/pipeline/gauss_loss.py:GaussViews._render/loss` | accept a dressing override for `source_offsets` and a per-child log-scale (multiplies `render_sigma` per splat / scales `cov6` rows). Target baking unchanged (targets are truth; never dressed). |
| `physmorph/pipeline/surface_local.py` → `local_correction_pass` rewrite | (a) Tier D dressing sub-solve (line-searched Adam, fixed budget, convergence gate); (b) Tier R band solve with `L_gauss` in the energy and the **virtual** contract: returns `(A_field, dressing_new, tele)`; **never** returns `x'/F'`. `SurfaceLocal.__init__/solve` kept verbatim (band, coloring, diag, node_cap, gate). |
| `physmorph/pipeline/runner.py:run_pipeline` | replace the LOCAL-phase block (currently guarded by `cfg.lg_sweeps > 0`): no `x/Fc` overwrite, no post-pass metric recompute (state is unchanged, so the archived state *is* the window state again — the old adversarial patch dissolves). Compute `F_virt = A·Fc` and route it into the **existing** assimilation call site (`assimilate_elastic(F_virt, …)` / `assimilate_growth(F_virt, …)`); when the pass is off, `F_virt = Fc` bit-exactly. Rollback dict gains `Fp_predemand` and `dressing` (§11.2); the `lg_sweeps`+`w_dt/w_fill` `ValueError` guard is retired **with a comment stating why its precondition (state overwrite) no longer exists**. |
| `physmorph/pipeline/optimizer.py:losses_of` | gauss loss reads the frozen dressing (one field through `TargetPack`). Nothing else changes; the adjoint path is untouched. |
| `physmorph/pipeline/config.py` | `local_dress_iters` (0=off), `local_demand: bool`, reuse `lg_sweeps`, `lg_young`; every constant documented with its §7 rule. **No new weight scalars.** |
| `scripts/pipeline_run.py` | arms `dress`, `demand`, `lg2` (§10). |
| `physmorph/viewer/server.py`, `scripts/render_sequence.py` | consume dressing for display parity — what is seen is what was optimized (PhysGaussian contract). |
| `tests/` | stage-0 unit tests (§10). |

### 9.2 Cadence

| step | when | differentiated? |
|---|---|---|
| global window (`optimize_window`) | per commit | dFc, s through the MPM tape (unchanged); dressing frozen |
| outer promotion + guards | per commit | — |
| Tier D dressing solve | per **accepted** commit, on the promoted terminal state | dressing only, direct render backward, no MPM |
| Tier R band solve + demand | per **accepted** commit, after Tier D | u only, direct render backward, no MPM |
| assimilation (one call, `F_virt`) | per accepted commit (existing site) | never |
| outer merit gate | per commit, dressing-free components | — |
| c2f rebuild | as now; dressing resets to baseline (§11.6) | — |

### 9.3 Expected cost per commit

Tier D: ≤ `20 iters × 18 views` fwd+bwd rasterizations at gauss res (128–384px,
10–40k splats) — small next to a window (8 iters × up to 10 line-search rollouts of
T=20 MPM steps + one tape backward). Tier R: the dominant add — the retired pass
measured **3.3× wall-clock** with the same solver at similar budgets; expect 1.5–3× per
commit at `lg_sweeps=10` early, **annealing toward ~1× late** because the convergence
gate exits in 1–2 sweeps once the residual (and τ) shrink. The ladder's hero stage is
wall-clock-matched (G5 discipline); if lg2 cannot pay for itself at matched wall-clock,
that is a verdict, not a footnote.

---

## §10 Staged A/B falsification ladder (experiments.md style)

Baseline `b0` = flagship `render_full_dt_iso_nn` + gauss hybrid, same seed/budget per
stage. Discretisation for every number: `dx=0.5, dt=1/240, 64³, smoothing 0.955,
loss_res 64, T=20`.

| stage | arm | config delta | budget | primary metrics | kill criterion (pre-registered) |
|---|---|---|---|---|---|
| 0 | CPU tests | — | pytest | `u=0 ⇒ F_virt≡F` bit-exact vs current path; assimilation invariants (det Fp=1 to 1e-6, SV band) under random capped `A`; `A` SPD + cap bound; dressing zero-centroid + caps; determinism of the pass given fixed state | any failure blocks stage 1 |
| 1 | `d1` (Tier D only) | `local_dress_iters=20`, `local_demand=off` | N=20k, 120c | gauss L1 residual; raw chamfer/silIoU; G2; G6 | **K-D1**: raw chamfer regression >2% (dressing misleads the frozen-dressing window) or any guard ≠ 0. **K-D2**: gauss residual drop <10% at raw-chamfer tie — dressing has no explanatory power, drop Tier D. |
| 2 | `r1` (Tier R only) | `lg_sweeps=10`, `local_dress_iters=0` | N=20k, 120c | `lg2_realized_cos/frac`, `lg2_demand_energy`, window-start kin kick, chamfer/silIoU, out_nn, `F_invert_steps` | **K-R1**: realized_cos median <0.1 by c40 (dead channel again). **K-R2**: window-start kin > 2× baseline median (the displacement-assimilation spike signature). **K-R3**: chamfer regression >2% or any guard ≠ 0. |
| 3 | `lg2` (D+R) | both on | N=20k, 120c, + freeze-and-hold 40 extra commits | §8 P-float, P-osc, P-render; **held-phase ratchet probe**: per-held-commit ‖ΔFp‖ | each P's kill as written in §8. **K-R4 (ratchet)**: ‖ΔFp‖ per held commit not → 0 within 5 held commits — demand fails to self-terminate, Tier R is falsified. |
| 4 | `lg2_hero` | stage-3 winner config | N=40k, 300c, 4 pairs (bunny/armadillo/spot/A→C), wall-clock-matched vs `b0` | full gate battery G2–G6 + §8 predictions at hero scale | G5 rule: any pair with sil_iou ↓ or chamfer > +2% vs `b0` at matched wall-clock ⇒ not adopted; partial adoption (D without R or vice versa) allowed only along the stage-1/2 single-tier evidence. |
| 4b | `d1_cap` | amplitude cap A/B `{0.5, 1.0}·h_src` | N=20k, 120c | raw chamfer tie + gauss residual | pick-by-rule (§7); runs only if stage 3 passes. |

Verdict discipline: pre-registered thresholds above; no post-hoc metric swaps; every
number states its discretisation; adversarial REFUTE round (Codex + Opus) on the diff
before any hero run.

---

## §11 Adversarial self-review (REFUTE-style)

**11.1 "Two objectives, no joint guarantee — you rebuilt the lg mismatch."**
Partly true and owned. ARAP/PD get guarantees because both phases descend one energy;
lg2's local objective (gauss L1 + SNH coherence) ≠ the window objective (D_vol + hybrid
+ W1/nn + priors). The difference from the retired pass: the local output is a *demand*,
not a *state edit* — the global solve re-arbitrates it against every term it knows
(W1, fill, D_vol gradients all present in the next window), the line search can refuse
to realize it, and the outer fixed-merit gate can reject the resulting commit. The
mismatch can therefore cost budget (rejected commits), never correctness. Falsifier:
stage-2 K-R3 plus the rejection-rate telemetry (a demand arm whose outer-reject rate
doubles is fighting the data terms — that is the h15 "nothing to descend" answer
arriving as evidence, and the arm dies by K-R1/K-R3).

**11.2 "Ratchet through assimilation — a bad demand is baked before it is tested."**
Correct as stated: the demand at commit k acts through window k+1, so the gate at k
never saw its effect. Three answers. (a) The rollback dict gains `Fp_predemand`: an
outer rejection at k+1 restores `Fp` to *before* commit k's demand — the untested
mutation is stripped as part of the existing cold-restart rule, so a bad demand
survives at most one window and cannot compound. (b) The throttle τ bounds any single
demand by the physics' own current strain scale, and the isochoric projection + SV band
bound the cumulative excursion (the measured volume-ratchet forensic is the reason the
band exists). (c) The held-phase probe (K-R4) directly falsifies residual ratcheting at
convergence. Residual risk: a *slow* ratchet inside the band during descent —
`lg2_demand_energy` is logged per commit precisely so the REFUTE round can audit its
integral against the elastic energy scale.

**11.3 "Gauge freedom: dressing paints over real error; absorbing state at 'looks
right, is wrong'."** Three structural armors. (a) Capacity: tangent-only, ≤0.5·h_src,
scale in the `w_cov` band — dressing cannot represent (so cannot cancel) any mode the
particle sampling resolves; silhouette-scale error stays visible to the global channel.
(b) The freeze tracks and the outer gate consume dressing-free raw-state components
only; `metrics.py` never consumes the renderer — every gate number is dressing-proof,
so "converged because painted" is impossible. (c) No opacity DOF (floaters.md
contract). Residual risk: G6 (human visual QA) is the one gate dressing *does* touch —
by design; the QA checklist must add "toggle dressing off" to the viewer audit, next to
the existing "uncheck surface Gaussians".

**11.4 "Determinism of rejection loops."** The pass is a deterministic function of the
promoted state (fixed budgets, frozen assignments, full view set) up to CUDA atomic
noise — the same equivalence class as the existing replay check, and covered by the
same `ls_noise_rel` tolerance. On outer rejection, `Fp` (pre-demand) and dressing are
restored and the lineage cold-restarts — the s4 absorbing-state fix is inherited
unchanged. Null commits donate nothing (§7), so the "identical stale retry" loop cannot
accumulate demand.

**11.5 "Two clocks resonate: demand overshoots, next window overcorrects."** The
multirate literature's standing failure mode. Damping levers already in the loop: τ
shrinks when windows stop accepting strain; `anneal_stale` shrinks the global step on
non-improvement; the reversal gate latches. Dedicated telemetry: sign flips of
⟨u_k, u_{k+1}⟩ (demand-direction reversal across commits). Pre-registered response if
it fires: halve the demand cadence (every 2nd commit) — a structural change, not a
weight change — and re-run stage 3.

**11.6 "c2f rebuilds and window boundaries."** On the c2f rebuild the render targets
change resolution: dressing (whose objective just changed) resets to the PCA baseline
and re-earns, exactly like the outer latch re-earn at the same site; τ carries over
(it is physics-side). `Fp` needs no special handling — it is physical state and
legitimately survives rebuilds. Window boundaries: the demand's elastic kick enters at
window start — bounded by τ, watched by K-R2; the `mom_carry`/warm-start machinery is
unaffected because the control leaves see only a slightly different initial stress
field, the same class of change as any assimilation step today.

**11.7 "The frequency gap: modes between h_src (0.04) and dx (0.5)."** Honest gap:
dressing amplitude caps at 0.02 wu; band `u` lives on dx-scale nodes. Wavelengths of a
few particle spacings with amplitudes above the dressing cap are owned by *neither*
tier — they must still ride the global channel. The bet, falsifiable by P-osc, is that
the measured oscillators (1.2× spacing, sub-cap amplitudes) sit inside Tier D's band.
If P-osc fails while P-float/P-render pass, this gap is the first suspect, and the
pre-registered diagnostic is the 4b cap A/B — not an ad-hoc weight turn.

**11.8 "Tier R does nothing for a lone floater."** Correct: an isolated particle's
∇u ≈ 0 (pure translation demand), and strain-only extraction discards it — by design
(a translation demand through `Fp` would be exactly the inadmissible positional servo).
Lone-floater *transport* remains the near-band W1 term's job (adopted, fork 326→97→64);
Tier R attacks the floater *sources* (thin-feature under-coverage → squeeze ejecta),
Tier D + the support check own their visibility. P-float is written against the
combined stack accordingly.

**11.9 "You revived a falsified mechanism."** The h15 verdict ("lg parked") falsified
the *silhouette-objective, state-overwriting* pass — same-nullspace objective, tie at
3.3× cost. lg2 changes both falsified components (gauss objective; virtual demand) and
its re-activation condition is different in kind: not "PCGrad left a conflict gap" but
"the render channel's unique content measurably cannot reach dFc" (root_analysis,
work-share telemetry). The parked arm's park order stands; lg2 is a new arm with its
own ladder and its own kill criteria.

---

## §12 Non-goals

- No implicit-MPM integrator replacement (review option 1) — out of scope; lg2 is the
  engineering-scale route.
- No control-space H1/kNN preconditioner (measured regression; stays off).
- No opacity/deletion DOFs anywhere in a differentiable objective.
- No change to mass, momentum, P2G/G2P, the adjoint, `_state_ok`, guard semantics, or
  the metric battery.
- No new tuned scalar without its §7 rule; any constant that later needs "adjusting" is
  a design bug to be re-derived, not turned.
