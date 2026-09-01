# PhysMorph pipeline v2 — render feedback that *drives* the physics

Status: **design + implementation spec** (2026-09-01). This file is the source of truth for the
v2 pipeline; `docs/method.md` remains the equation contract for the MPM core (eq numbers cited
in `physmorph/mpm/*` refer to it). Results append here as they land.

---

## 1. Why v2 (diagnosis of v1, with file:line)

The audit of 2026-09-01 found four structural problems in the v1 loops:

1. **Two incompatible render-coupling types.** `morph.py` / `trajectory_opt.py` are true
   differentiable feedback (image loss → MPM adjoint → `dFc`), while `morph_physical.py` /
   `style_transfer.py` are displacement-space feedforward (render gradient computed at fixed `x`,
   never through the simulator). Only the first type supports the thesis; the second cannot.
2. **The render term was inert in the greedy loop.** Fixed `render_lambda` left
   `λ·D_img ~ 1e-3` vs `D_vol ~ 1e2` (measured; see `trajectory_opt.py` module docstring:
   "render term sat ~90x below D_vol and did nothing").
3. **Stability came from non-physical band-aids**, which also destroy measurement validity:
   `_pull_outliers`, `max_move` cap, `cohesion` penalty (morph.py), `_repel`, `_taubin`,
   overshoot clamp (morph_physical.py). Any "render beats physics-only" claim measured through
   these operators is confounded.
4. **Known formulation bugs**: greedy per-frame `dFc` reset (myopic), partial state promotion
   (fixed in `trajectory_opt.optimize_morph`), reflection-preserving `_condition_F` in morph.py
   (fixed version documented at `trajectory_opt.py:234`).

v2 keeps exactly one blessed path and makes the render signal a first-class *physical* input.

## 2. Objective (the user-stated acceptance criteria)

Render feedback must optimise the deformation-gradient control **and** touch other physical
quantities, such that the result is:

- **hole-free** (no interior background visible through the body),
- **rest-stable** (no in-place particle flicker / optimiser oscillation once converged),
- **metrically superior to physics-only** (same seeds, same budget),
- **visually clean** (per-frame QA rubric, AGENTS.md rule 2).

## 3. Formulation

### 3.1 State and control

Per particle: position `x`, velocity `v`, APIC affine `C`, total deformation `F`, plastic rest
state `Fp`, material multipliers `s = (s_λ, s_μ)`. Stress uses
`F_e = (F + dFc[t]) Fp⁻¹` (eq 3′), fixed corotated PK1 with per-particle Lamé

```
λ_i = λ₀ · exp(s_λ,i),   μ_i = μ₀ · exp(s_μ,i),   s clamped to [−s_max, s_max].
```

Control variables of one optimisation window (horizon `T`):

- `dFc[0..T−1]` — the control sequence (C++ CompGraph semantics, never reset inside a window),
- `s` — the material field (optional arm; `opt_material`).

### 3.2 The four render→physics channels

| # | channel | mechanism | what it buys |
|---|---|---|---|
| 1 | `dFc[t]` | image loss → warp tape adjoint → per-layer control grads | render shapes the *motion* |
| 2 | `s_λ, s_μ` | same adjoint, second leaf pair | render stiffens/softens *material* where images disagree |
| 3 | `Fp` | commit-time assimilation of the **optimised** displacement (§3.5) | removes stored-energy spring-back → no inter-commit oscillation |
| 4 | `v_T` | terminal kinetic loss (§3.3) | solutions must *arrive at rest* → no ballistic slam, no ringing, quiet promoted state |

Channels 3–4 are how render feedback "touches other physical quantities" without any
non-physical projection: both are ordinary loss/plasticity terms inside the same optimisation.

### 3.3 Loss (terminal, per inner iteration)

```
L(dFc, s) = D_vol(x_T)                                  # eq (13) mass matching (Xu et al.)
          + λ_R · D_render(x_T)                         # §3.4, balanced (§3.6)
          + w_kin  · mean‖v_T‖²                         # arrive at rest (anti-oscillation)
          + w_ctrl · Σ_t‖dFc[t]‖² / (T·N)               # running control cost (anti-slam)
          + w_box  · mean relu(|x_T| − r_box)²          # far-field leash (see below)
          + w_mat  · mean‖s‖²                           # material ridge (identifiability)
```

`D_vol` sees occupancy in 3-D (interior included); `D_render` sees projection consistency;
the kinetic and control terms see *how* the body was driven. Nothing else touches the state.

**The leash term exists because the render views are LOCAL**: a particle that leaves every
viewport (and the D_vol loss grid) receives exactly zero gradient from both data terms —
escaping far enough is indistinguishable from being deleted, so `w_spray` alone cannot
recover far ejecta (adversarial finding, verified: gradient is 0.0 at 1.2×extent). The
leash `relu(|x|−r_box)²` (r_box = the render extent) is differentiable everywhere, zero
inside the box, and hands escapees back to the pixel/grid gradients. It is an objective
term, not a projection.

### 3.4 D_render — multi-view asymmetric silhouette

Views: azimuth ring × elevation set `{0, +φ, −φ}` (v1's equator-only ring was blind to
y-axis concavities). Soft coverage per view via the CIC splat `α = 1 − exp(−k·w)` (existing,
differentiable). The per-view loss is **asymmetric**:

```
D_render = mean_views mean_px [ w_hole · relu(α_tgt − α)² + w_spray · relu(α − α_tgt)² ]
```

- `w_hole > 1`: deficit **inside** the target silhouette = holes / missing extremities — the
  exact failure the render channel exists to fix (D_vol at loss_res 32 is too coarse to see it).
- `w_spray`: excess **outside** = ejecta/floaters — the render term actively pulls strays back,
  so mass ejection is *penalised by the objective* instead of clamped after the fact.

Note the CIC-α saturates in the interior (`1 − exp(−k·w) → 1`), so pixel gradients concentrate
at the rim: the feedback is surface-dominant by construction, which is what a Gaussian-surface
deliverable needs (GS covariance rides F on the surface; interior is D_vol's job).

### 3.5 Commit-time plastic assimilation (channel 3)

After a window is accepted and its full state `(x_T, F_T, v_T, C_T)` promoted
(`trajectory_opt.optimize_morph` semantics — partial promotion was the v1 energy-re-injection
bug), an η-fraction of the **elastic stretch itself** is assimilated into the rest state
(`plasticity/assimilation.py::assimilate_elastic`):

```
F_e = F·Fp⁻¹ = R_e·S_e          # per-particle polar decomposition
Fp ← clamp_sv( S_e^η · Fp )     # ⇒ F_e_new = R_e·S_e^{1−η}   EXACTLY
```

`S_e` is symmetric, so it commutes with its own powers and the relation is exact: each
commit relaxes exactly η of the elastic stretch, leaves the rotation untouched (a rigid
motion is a strict no-op), and the fixed-corotated energy decreases monotonically
(tested). Because `F_T` is shaped by the render-optimised `dFc`, the render signal reaches
`Fp` through this channel. η < 1 keeps a fraction of the stress as elastic "glue". The
cumulative band is wide (`[0.2, 5.0]`): a saturated Fp stops tracking and re-arms
spring-back.

History of this design (both alternatives measured and rejected):
1. v1 `update_fp(J_sym, isochoric)` — fabricates strain from rigid rotation
   (sym(R−I) ≠ 0: energy 0 → 0.272 for a 90° commit) and is an exact no-op for dilation,
   the one mode `D_vol` drives (adversarial round 1).
2. Displacement-field polar assimilation (`assimilate_fp`) — objective in itself, but in
   THIS engine `dFc` is injected straight into F (`F ← (I+dt·C)(F+dFc)`), so F carries
   deformation the realised motion never had; migrating Fp toward the motion-stretch
   mismatches F and every commit boundary becomes a stress spike. Measured locally
   (N=5000, T=10, dx=0.5, dt=1/240): kin 66→509 across two commits, |v|max 290,
   line-search stall at commit 3; the exact elastic version at identical settings runs
   12/12 commits with kin 1.93→0.036 monotone.

### 3.6 λ_R balancing

`λ_R = α_λ · ‖∇_phys‖ / ‖∇_render‖` (the C++ `get_control_layer_grad_norm` rule), where
∇_phys is the gradient of all non-render terms. Two v2 rules, both anti-oscillation:

- λ_R is estimated **once per window** (from the first iteration's per-term gradient norms)
  and held fixed for the window, so the line search decreases a *single* objective — a
  per-iteration λ makes "monotone acceptance" vacuous (adversarial finding);
- across windows the estimate is **EMA-smoothed** (`λ ← (1−β)λ + β·λ_target`).

`α_λ = 0` turns the render channel off — that *is* the physics-only baseline arm (one code
path). The plateau/freeze detector never sees λ: it tracks the raw components
(`D_vol + w_kin·kin` and `D_render` separately), because a composite tracked under a
drifting weight can fake both improvement and stagnation.

### 3.7 Optimiser

Per window: hand-rolled Adam over the leaf list `[dFc, s]` with persistent moments,
**backtracking line search** (reject + restore all leaves and moments, halve α, C++
`max_ls_iters`) and **adaptive α** (`target_norm/‖g‖`). Acceptance requires a finite
**rollout state** `(x_T, F_T, v_T)`, not merely a finite scalar loss: the kernels'
`valid_pos` guard silently drops NaN particles from every splat, so a poisoned rollout can
show a finite and even *lower* `D_vol` by deleting mass (adversarial finding).

Across windows (runner): full-state promotion, plastic assimilation, plateau freeze on the
**raw** loss components (converged ⇒ hold still), and guard **counters** that must read
**zero** for a run to be valid (gate G2): domain clamp, non-finite x/v/C, F non-finite
reset, F reflection flip, and **any-step** F inversion across the whole window (an
inversion that recovers by `t = T` is still a failure). The accompanying sanitisation
(clip/nan_to_num) is *containment only* — it stops one poisoned state from cascading into
meaningless downstream telemetry; a fired counter already means the run is invalid.
Promotion repairs numerical pathologies of F only (non-finite rows, reflections — both
counted); there is **no silent singular-value projection** in the blessed path, and the
archived frames are the *promoted* states, so metrics, plasticity, and the next window all
describe the same trajectory.

### 3.8 What v2 deliberately does NOT do

No `_pull_outliers`, no `max_move` projection, no cohesion penalty, no `_repel`/`_taubin`,
no velocity override, no `v_max` clamp in the blessed path. If a gate fails, the fix goes into
the objective or the discretisation, not into a post-hoc projection of the state.

## 4. Code layout (v2)

```
physmorph/
  pipeline/                  ← the blessed path (NEW)
    config.py                PipelineConfig (all knobs; one dataclass)
    render_loss.py           views (azim×elev), asymmetric D_render, LambdaBalancer
    optimizer.py             optimize_window(): multi-leaf line-searched Adam over horizon T
    runner.py                run_pipeline(): commits, promotion, Fp assimilation, freeze, guards
  metrics.py                 chamfer, sil_iou, hole_frac, jitter, detF_min — raw state only (NEW)
  mpm/                       engine unchanged; function.py extended (material leaves, v_T out)
    conditioning.py          condition_F: SVD clamp + reflection repair (NEW; the fixed version)
  trajectory_opt.py          C++-parity reference implementation (kept verbatim for the oracle A/B)
  experimental/              quarantined v1 loops: morph.py, morph_physical.py, style_transfer.py
  losses/, plasticity/, render/, sampling/, surface/, viewer/   (unchanged)
scripts/
  pipeline_run.py            A/B/A′ arms: phys | render | render_mat, gates printed (NEW)
  morph_measure.py           updated import (experimental.morph); kept for old-number repro
  traj_opt_run.py            unchanged (parity gates vs C++)
```

`experimental/` is not deleted because the ablation figures ("displacement-space feedback
scatters texture", "fixed λ is inert") come from those loops; they are not part of any claim.

## 5. Acceptance gates (what "done" means)

| gate | test | threshold |
|---|---|---|
| G1a plumbing | constant `dFc` sequence ≡ shared control rollout | max&#124;Δx&#124; ≤ 1e-6·scale |
| G1b channels | dL/ds vs central finite difference (small subproblem, v_T in the loss) | rel err < 0.25, grad finite & nonzero |
| G2 stability | guard counters over full run (incl. any-step F inversion) | **all zero** |
| G3 rest | tail jitter over SIMULATED frames (held padding excluded) AND terminal-velocity drift `v̄·dt·T/diag` | both < 0.3% bbox diag |
| G4 holes | `hole_frac` (binary 3×3-footprint splat at the FIXED target extent) | ≤ 2% AND ≤ physics arm |
| G5 supremacy | render arm vs phys arm, same seed/budget | sil_iou ↑, chamfer within +2%, hole_frac ↓ |
| G6 visual | per-frame QA rubric (AGENTS rule 2) on server renders, FULL frame range | pass |

Metric independence (post-adversarial-round): metrics come from raw simulation state only
(AGENTS rule 3) and share **no operator with any loss** — `sil_iou`/`hole_frac` use a binary
point splat (not the soft CIC alpha the loss optimises), and every projected quantity uses
one fixed extent derived from the **target**, shared across frames and arms (a per-frame
autoscale let a single ejecta particle close holes: 7.7% → 4.4% from one stray, verified).
`outside_frac` reports ejecta beyond the leash box.

## 6. Verification protocol

1. `python -m py_compile` on every touched file (local, no GPU).
2. **Adversarial gate** (AGENTS rule 5): two independent refuters told to *refute*, findings
   cite `file:line`, the implementer answers every finding:
   - Codex CLI (`codex exec`, gpt-5.6-sol, xhigh reasoning) — cross-vendor reviewer;
   - Claude Opus subagent (high effort).
3. Server (hyde06): G1–G5 via `scripts/pipeline_run.py`; G6 via frame extraction + rubric.

**Round 1 (2026-09-01)**: Codex 15 findings (3 blocker / 9 major / 3 minor), Opus 11
findings (3 blocker / 6 major / 2 minor groups), strongly overlapping and several verified
numerically. All 26 accepted and fixed in the same commit; the substantive ones are
annotated inline in this spec (§3.3 leash, §3.5 objective assimilation, §3.6 per-window λ,
§3.7 finite-state acceptance + guard semantics, §5 metric independence). Both reviewers
independently cleared: the autograd bridge (material leaves, double-tape backward),
line-search restore completeness, elevation projection math, and the moved-module imports.

## 7. Run commands (hyde06)

```bash
ssh -J chayo@hyde01.dabh.io chayo@hyde06.dabh.io
cd ~/physmorph_v2
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
CUDA_VISIBLE_DEVICES=0 $PY scripts/pipeline_run.py --arms phys,render --out output/v2_ab
CUDA_VISIBLE_DEVICES=0 $PY scripts/pipeline_run.py --arms render_mat --out output/v2_mat
# QA spans the FULL morph: frames = animations*T + 1 (defaults -> 601, indices 0..600)
$PY scripts/quicklook.py --npz output/v2_ab_render.npz --frames 0,120,240,360,480,600 \
    --out output/v2_render_strip.png
$PY scripts/make_gif.py --npz output/v2_ab_render.npz --stride 3 --out output/v2_render.gif
```

## 8. Paper grounding (per stage, 2023–2026)

| stage | papers | what we take |
|---|---|---|
| render→physics learning | NeuMA (NeurIPS 24), PAC-NeRF (ICLR 23), GIC (NeurIPS 24), PhysDreamer (ECCV 24), OmniPhysGS (ICLR 25) | low-dim physical parameterisation under image loss (our `s` field); 2-D shape surrogates as render supervision (our D_render); physics prior + learned correction (our dFc as correction on MPM) |
| GS–physics binding | PhysGaussian (CVPR 24), Gaussian Surfels (SIGGRAPH 24), GauSTAR (CVPR 25), GASP (24), GausSim (ICCV 25) | sim representation = render representation; surface-only feedback via surfels; Σ = σ₀²FFᵀ rides F |
| optimisation stability | SAPO/Rewarped (RSS 25), Unrolled-training differentiability (TMLR 24), Diff-MPM control (25), FluidLab (ICLR 23), stabilised B-spline MPM (CMAME 23) | horizon within chaotic timescale; line search + trust region; **active damping as an objective** (our w_kin); checkpointing for long T |
| GPU execution | Newton (25), Rewarped, Warp diff-physics guide | tape adjoint now; O(√T) checkpointing when T grows; Newton solver port later |

## 9. Result log

*(state the discretisation with every number — AGENTS rule 4)*

### 2026-09-01 — local smoke A/B (RTX 4090 Laptop; discovered the machine HAS a usable GPU)

Discretisation: `dx=0.5, dt=1/240, grid 64³, smoothing=0.955, loss_res=32`; scale:
`N=5000` (isosphere→bunny), `T=10`, `iters=8`, `animations=12`, `w_kin=5, w_box=10,
assim=0.5(elastic), λ_auto=0.5, 4 azim × 3 elev views @ 48px`. Runtime ≈ 0.2–0.3 min/arm.

| arm | chamfer | sil_iou | hole | jitter_rel | drift_rel | guards | commits |
|---|---|---|---|---|---|---|---|
| phys | 0.2801 | 0.8476 | 1.56% | 0.00014 | 0.0008 | all 0 | 12/12 |
| render | **0.2522** | **0.9167** | **1.00%** | 0.00014 | 0.0008 | all 0 | 12/12 |

G1a/G1b/G2/G3/G4(abs+vs-phys)/G5 **all PASS**. λ_R self-anneals 553→91 as D_render
converges. Both arms end at rest (kin 1.93→0.036 monotone across commits).
Visual (quicklook, 2 az): solid closed body, no floaters, no crossfade ghost; render arm
visibly tighter to the bunny torso outline with less edge fringe than phys. **Not yet
reached at this budget: thin extremities (ears/paw)** — the full-scale question.

Tuning findings (each verified by ablation): `w_kin=0.5` too weak (momentum snowball:
kin 66→509, dead by commit 3); `w_kin=5, iters=8` stable. Displacement-based assimilation
destabilises (see §3.5 history); elastic-stretch version stable. First-run failure with
`iters=4` was budget, not formulation.
