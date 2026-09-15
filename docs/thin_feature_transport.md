# Thin-feature transport — the "scatter then return" dossier (2026-09-15)

Status: **measured, literature read, three remedies implemented and pre-registered; batch l
(coherence prior) running, batch m (bond bound / frontier loss) staged.** Companion to
`docs/floaters.md` (which owns the end-state floater census) and `docs/render_controls_physics.md`.

## 1. What the user saw and what it measures as

In the 40k GIFs the ears fill with a sparse spray of particles that later thickens
("scatter, then return"), and a few strays remain at the feet. Probes
(`scatter_probe2.py` on hyde06, raw state only, thin-feature target region = target points
with a low neighbour count within 0.9 wu and beyond the median radius: ears, feet, outer
shell — 19 % of the target mass):

| arm (20k unless noted) | thin-region SPARSE fraction, peak → end | mass on thin targets, end (target 0.194) | strays > 2 sp, end |
|---|---|---|---|
| flagship baseline | 0.58 (commit ≈5) → 0.38 | 0.165 | 1.3 % |
| 40k baseline | 0.58 → 0.39 | 0.147 | 1.6 % |
| 40k recipe (kinetic remedies) | 0.59 → 0.38 | 0.146 | 1.5 % |
| 40k `--ppc 8` + fine loss grid + recipe | 0.44 → 0.33 | 0.172 | 0.07 % |
| `render_ctrl` (coarse control basis 12³) | 0.51 → 0.28 | 0.180 | 2.0 % |
| ablation `w_nn 0` | 0.61 → 0.36 (froze at commit 9) | 0.136 | 5.2 % |
| ablation `w_dt 0` | 0.59 → 0.39 | 0.163 | 1.2 % |
| ablation `w_creg 1000` | 0.58 → 0.38 | 0.167 | 1.1 % |

Sparse = a particle whose 8 nearest neighbours are on average more than 2 target spacings
away. Reading:
- The vanguard forms in the first ~5 commits in every arm and is NOT produced by the
  per-particle cleanup pulls (nn-band, W1): removing them leaves the peak at 0.59–0.62; the
  control-smoothness penalty at 10× (creg 1000) changes nothing. It is produced by the core
  mass-matching descent acting on individual surface particles, which is exactly Xu et al.'s
  own diagnosis of their loss ("mass ejections lead to the quickest decrease of the loss
  function", TVCG 2025 §limitations) — the log form only flattens that direction.
- Once a particle has crossed an empty loss cell it has no grid neighbours: MLS-MPM cannot
  transmit tension across empty cells (numerical fracture, Yue et al. 2015 §2), so it coasts
  until later mass or the cleanup pulls re-couple it. That is the "return".
- The thin features end 15–25 % under-filled on dx 0.5; the two things that measurably help
  are the coarse control basis (coherent actuation: 0.180 delivered, end sparsity 0.28) and
  the finer MPM grid (`--ppc 8`: fracture threshold moves with dx). Kinetic remedies do not
  touch it. Volume is not involved (J p2p 0.008, isochoric assimilation, `w_jvol`).
- The 20k and 40k targets are the same mesh (bbox 4.99/6.46/6.35 vs 5.02/6.44/6.35, ear
  share 0.25); the "connected ears" in the az-0.6 view are the orthographic overlap.

## 2. Literature (read 2026-09-15; full list in `docs/related_work.md`)

- **Numerical fracture and its remedies.** Yue et al. 2015 (Continuum Foam, TOG): the grid
  keeps artificial connectivity only within the stencil; remedies are SDF-gated Poisson-disk
  resampling (insert where spacing exceeds the lattice), merging, and removing sub-grid
  geometry. Homel et al. 2016 (CPDI domain scaling): the particle's *influence domain*, not
  its position, must span the gap. Total-Lagrangian MPM (de Vaucorbeil 2020; A-ULMPM 2022;
  HLFEMP 2024): no fracture because connectivity lives in the reference configuration.
  CK-MPM 2024 (2412.10399): narrower kernels fracture MORE, 27 ppc restores integrity —
  stencil width and ppc set the threshold. Yao–Zhao 2026 (2603.03860): support-gated
  APIC — the affine/stress transfer is scaled by a smoothstep of the local particle count,
  so a depleted cell gets pure PIC transfer and cannot be driven ballistic. ASFLIP 2021:
  grid advection is itself a cohesion mechanism; spray means single-particle nodes are being
  driven ballistic by the stress on them. Jiang 2017 / Han 2019 (Lagrangian-energy MPM):
  forces on an explicit bond graph, scattered to the grid — the only mechanism that
  transmits tension across empty cells; cost = numerical cohesion when overdone.
- **Mass ejection in differentiable / controlled MPM.** Xu SCA 2023 + TVCG 2025: the log
  nodal-mass loss exists to suppress ejection; still reports "clumps ejected outward" as
  local minima; F-smoothing γ 0.93–0.965 hand-tuned. PlasticineLab / DiffTaichi: local
  minima and flat landscapes, nothing on separation. **No 2024–26 paper adds an ARAP /
  bond-stretch / neighbourhood-preserving regulariser inside a differentiable MPM
  controller.** The closest verified formulation is Dynamic 3D Gaussians (Luiten 2023):
  per-frame local rigidity + long-term isometry on frozen k=20 neighbours with weights
  exp(−λ‖Δx₀‖²), "the rigidity loss is both necessary and adequate on its own".
- **Morphing that preserves connectivity.** ARAP interpolation (Alexa 2000), Hamiltonian
  shape interpolation (Eisenberger 2020), volume-preserving neural morphing (Buonomo 2025),
  FLOWING (2510.09537): coherence by construction from ONE smooth velocity field — what a
  per-particle control lacks. Varifold metamorphosis (Hsieh–Charon 2112.04644): transport
  alone "cannot adequately match shapes with differing thin features"; add a mass
  creation term — morph the ear *in place*. Growth-tensor shape programming (Ortigosa
  2024, Zhou 2604.04984, van Rees 2017, Cislo 2302.07839): the control is a bounded,
  spatially smooth growth field and features form by frontier growth, not long-range
  transport. Entropic OT blurs thin features (Solomon 2015).
- **Coherence metrics.** Trustworthiness/continuity (Venna & Kaski 2010; our nbr_overlap),
  straightness / tortuosity (Batschelet, Benhamou), LTVE (2401.09222), topological OT
  distortion (2603.15683). Our per-frame "sparse fraction on thin targets" is the isolated-
  fraction-over-time analogue.

## 3. Mechanisms implemented (opt-in; code in `pipeline/optimizer.py`, `runner.py`)

| flag | what | where | evidence |
|---|---|---|---|
| `w_bond`, `bond_s0` | one-sided bond-stretch bound: `mean_i Σ_j w_ij relu(‖x_T,j−x_T,i‖ − (1+s₀)‖x_0,j−x_0,i‖)² / sp²`, frozen SOURCE neighbours (`coh_k`), weights `exp(−‖Δx_src‖²/(2(2sp)²))`, rest length = this window's start (a rate limit on separation; compression free) | objective | Luiten 2023 (strong for coherence), PD stretch criterion; no MPM precedent |
| `w_coh` | Laplacian of the window displacement over the same neighbours, `mean_i‖u_i − mean_j u_j‖²/sp²` (zero for locally affine motion) | objective | ARAP / Sobolev lineage; state-side twin of `w_creg` |
| `vol_frontier` | D_vol evaluated only on target cells within one loss cell (3³ dilation) of the window's start occupancy — no far-field pull into empty thin features; the ear fills by frontier growth | objective | Xu 2025 diagnosis; Hsieh–Charon; Cislo (medium) |
| already there | coarse control basis (`control_grid`), `--ppc 8` | control / discretisation | measured above |
| `gate_lo`, `gate_hi` (`MPMParams.gate_r_lo/r_hi/n0`) | **support-gated APIC** (forward model): `ω_p = smoothstep((n_p/n₀ − r_lo)/(r_hi − r_lo))` scales the affine term `m C (x_g − x_p)` in P2G; `n_p` = particle count in the 3³ cells around p's cell, `n₀` = median of that count over the source (fixed once per run). A particle in a depleted neighbourhood transfers PIC momentum only, so the steep affine field at a front cannot hand the empty-side nodes an outward velocity. ω is piecewise constant in x (computed outside the tape per step, read by the P2G adjoint as a constant; `tests/test_support_gate.py`: gate-off = plain APIC bit-for-bit, a lone particle keeps its affine field under APIC and loses it under the gate with its translation unchanged, gated adjoint = central FD within 5 %) | forward model | Yao–Zhao 2026 (2603.03860); ASFLIP's fringe diagnosis; Houdini's affine clamp |
| not implemented | Lagrangian bond forces (Jiang 2017), conservative split (Yue 2015) | forward model / post-hoc | held in reserve |

All three objective terms are differentiable through `x_T` → MPM adjoint → control, so the
premise (only physical variables move) holds; none edits state.

## 4. Pre-registration (batches l, m; 20k, flagship recipe, one seed)

Primary criterion: thin-region sparse fraction, PEAK and END (probe above); secondary:
mass delivered to thin targets; guards: chamfer within +2 %, hole ≤ 2 %, G2 = 0.

| arm | prediction | falsifier |
|---|---|---|
| `w_coh` 3 / 30 / 100 | peak < 0.45, end < 0.30 at 30 | peak ≥ 0.55 at every weight, or chamfer > +2 % |
| `render_ctrl` 24³ + `w_coh` 30 | best combined (control- and state-side coherence) | worse than either alone |
| `w_bond` 10 / 100, s₀ 0.3 | peak < 0.45 with LESS chamfer cost than `w_coh` (one-sided) | no reduction of the peak |
| `vol_frontier` | peak < 0.40 (no far-field pull) but slower fill; end mass on thin targets ≥ baseline | end mass on thin targets < 0.15 (fill stalls) |
| `w_bond` 100 + `vol_frontier` | peak < 0.35, end sparsity < 0.25 | — |
| `--ppc 8` + `w_bond` 100 (40k) | end sparsity < 0.25 and strays < 0.1 % | — |
| `render_ctrl` 24³ alone; 36³ (batch n) | 24³ alone reproduces the 24³+coh30 row (basis carries it); 36³ ≤ 24³ | 24³ alone back at the 12³ row (coh30 was the agent) |
| 40k `--ppc 8` + `render_ctrl` 24³ ± recipe (batch o) | the two levers add: end sparsity < 0.25, delivered ≥ 0.185, chamfer ≤ 0.12 | delivered < 0.172 (the ppc-8 recipe row) |
| support gate `(r_lo, r_hi)` = (0.05, 0.3), (0.1, 0.6); 24³ + (0.05, 0.3) (batch p) | if the affine fling is the ejection mechanism: peak < 0.50 on the per-particle control at no chamfer cost | peak ≥ 0.55 at both settings → the ejection is a translational push (control stress / loss pull on fringe particles), not an APIC artefact |

Reading rule: a mechanism that lowers the peak but not the end sparsity delays the spray;
one that lowers the end sparsity but delivers less mass trades fill for coherence — both
are reported, neither is adopted alone.

## 5. Results (hyde06, 20k, flagship recipe `--animations 300 --loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000`, dx 0.5, one seed, code 519617f)

Probe = `scatter_probe2.py` on the raw archives (renderer not consumed). Baseline seed noise on
chamfer at 20k is ±0.001 (batch j replicates), so ±2 % = ±0.003.

### 5a. Batch l — `w_coh` (state-side coherence prior) and the 24³ basis

| arm | chamfer | silIoU | hole | thin-region SPARSE peak → end | mass on thin targets, end (target 0.194) | strays > 2 sp |
|---|---|---|---|---|---|---|
| baseline `rcp_20k_a` | 0.1599 | 0.9655 | 0.04 % | 0.578 → 0.379 | 0.165 | 1.27 % |
| `w_coh 3` | 0.1597 | 0.9643 | 0.02 % | 0.580 → 0.372 | 0.160 | 1.09 % |
| `w_coh 30` | 0.1603 | 0.9640 | 0.03 % | 0.581 → 0.357 | 0.159 | 1.13 % |
| `w_coh 100` | 0.1591 | 0.9651 | 0.01 % | 0.577 → 0.371 | 0.170 | 1.01 % |
| `render_ctrl` 12³ (batch a) | 0.1617 | 0.9533 | 0.09 % | 0.508 → 0.281 | 0.180 | 1.97 % |
| `render_ctrl` 24³ + `w_coh 30` | 0.1604 | 0.9573 | 0.12 % | 0.503 → 0.226 | **0.194** | 1.23 % |

**Verdict on `w_coh`: FALSIFIED as a spray remedy.** The pre-registered falsifier ("peak ≥ 0.55
at every weight") fired: 0.577–0.581 at 3/30/100 against 0.578 baseline. End sparsity moved
by at most −0.02 (inside the run-to-run band), delivered thin mass ±0.005, chamfer within
noise. The prior does what it says (locally affine displacement is free), and the vanguard
IS a non-affine event — but by the time the window loss sees it the particle has already
crossed the empty cell inside the window, and the descent over the window's control finds
the same ejection with a slightly smaller Laplacian. A window-end penalty on x_T cannot
prevent an event that the forward model makes irreversible mid-window. The flag stays
(opt-in, default 0) as a documented negative.

**The control basis is the lever.** 24³ + coh30 delivered 0.194 (= the target share, first
arm to do so), end sparsity 0.226 (−40 % vs baseline), chamfer 0.1604 (within noise of the
baseline's 0.1599 and 0.8 % better than the 12³ basis), silIoU −0.8 pt vs baseline. The 24³
basis alone (without coh30) is batch n's first arm — until it runs, the split between "24³"
and "coh30" is unknown; the batch-l rows above say coh30 contributes nothing on the
per-particle control, so the prior expectation is that 24³ carries it.

### 5b. Batch m — `w_bond`, `vol_frontier`, and the bond bound at 40k `--ppc 8`

| arm | chamfer | silIoU | hole | thin-region SPARSE peak → end | mass on thin targets, end | strays > 2 sp |
|---|---|---|---|---|---|---|
| baseline `rcp_20k_a` | 0.1599 | 0.9655 | 0.04 % | 0.578 → 0.379 | 0.165 | 1.27 % |
| `w_bond 10` (s₀ 0.3) | 0.1601 | 0.9644 | 0.04 % | 0.581 → 0.377 | 0.165 | 1.06 % |
| `w_bond 100` | 0.1608 | 0.9649 | 0.01 % | 0.581 → 0.377 | 0.164 | 1.35 % |
| `vol_frontier` | 0.1591 | 0.9653 | 0.05 % | 0.586 → 0.368 | 0.164 | 1.14 % |
| `w_bond 100` + `vol_frontier` | 0.1596 | 0.9656 | 0.03 % | 0.580 → 0.379 | 0.162 | 1.19 % |
| 40k `--ppc 8` (batch i, legacy units) | 0.1164 | 0.9620 | 0.38 % | 0.481 → 0.295 | 0.188 | 0.23 % |
| 40k `--ppc 8` + density + recipe (batch j) | 0.1142 | 0.9735 | 0.52 % | 0.444 → 0.329 | 0.172 | 0.07 % |
| 40k `--ppc 8` + `w_bond 100` | 0.1296 | 0.9746 | 0.01 % | 0.541 → 0.331 | 0.148 | 0.01 % |

**Verdicts.** All three objective-side remedies are falsified on the pre-registered criterion:
- `w_bond` (one-sided stretch bound on frozen source neighbours): peak 0.581 at 10 and 100
  (baseline 0.578), end 0.377, delivered mass unchanged. At 40k `--ppc 8` it is **harmful**:
  chamfer +11 % (0.1296 vs 0.1164), delivered thin mass −21 % (0.148 vs 0.188), peak higher
  (0.541 vs 0.481). The bound does what it says — it stops the material from stretching —
  and at the fine grid the ear is filled by exactly that stretch of a connected stream; the
  bound trades fill for coherence, and the reading rule in §4 rejects it.
- `vol_frontier` (D_vol on target cells within one loss cell of current occupancy): peak
  0.586 — the spray forms identically with NO far-field volumetric pull, and the fill does
  not stall (0.164). So the far-field D_vol is not what sends single particles ahead.
- Their combination adds nothing.

**What this leaves.** The spray is unaffected by (i) the per-particle cleanup pulls (batch
k), (ii) control roughness (`w_creg` ×10), (iii) a window-end coherence prior (l), (iv) a
window-end bond bound (m), (v) the far-field D_vol (m). It IS reduced by the coarse control
basis (peak 0.50, end 0.23–0.28, batch l) and by the finer MPM grid (`--ppc 8`: peak 0.44–0.48,
batch i/j). Two hypotheses remain for the driver of the first ~5 commits: (a) the RENDER loss —
a handful of splats reaching the ear silhouette lowers the image loss more cheaply than moving
the ear's mass (this is the premise working as designed, at the wrong granularity), testable
by probing the physics-only arm of the same ladder; (b) an APIC fringe artefact — the affine
term of the front particles hands the empty-side nodes an outward velocity — testable with the
support gate (batch p). Both are pre-registered in §4.
