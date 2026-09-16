# Thin-feature transport — the "scatter then return" dossier (2026-09-15)

Status: **closed 2026-09-15 evening (batches k–p, code 13b9db7): the spray is the
mass-matching descent itself filling the ear cells with a stretched stream at sub-loss-cell
spacing — not volume, not the render loss, not an APIC artefact, not fracture, not the cleanup
pulls. Every window-end regulariser and the forward-model gate are falsified; the levers are
the control basis and the discretisation contract (`--ppc 8`, density units). §5e.** Companion to
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
- ~~Once a particle has crossed an empty loss cell it has no grid neighbours: MLS-MPM cannot
  transmit tension across empty cells (numerical fracture, Yue et al. 2015 §2), so it coasts
  until later mass or the cleanup pulls re-couple it.~~ **Corrected 2026-09-15 by the
  gate-coverage probe (§5d): at dx 0.5 / 20k the source has ~37 particles per cell and the
  vanguard's 3³-cell count is 0.31 of the interior median (~300 particles), so the sparse
  particles are NOT grid-isolated; "sparse" (8-NN mean > 2 target spacings ≈ 0.26 wu) is a
  SUB-CELL density deficit — the ear is filled first by a stream at ~1/8 of the target
  density, which a 0.5 wu loss cell cannot distinguish from a filled cell.** The "return" is
  the later arrival of the rest of the mass plus the particle-scale cleanup pulls.
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
| `w_coh` 3 / 30 / 100 | peak < 0.45, end < 0.30 at 30 | peak ≥ 0.55 at every weight, or chamfer > +2 % — **FIRED (§5a)** |
| `render_ctrl` 24³ + `w_coh` 30 | best combined (control- and state-side coherence) | worse than either alone |
| `w_bond` 10 / 100, s₀ 0.3 | peak < 0.45 with LESS chamfer cost than `w_coh` (one-sided) | no reduction of the peak — **FIRED (§5b)** |
| `vol_frontier` | peak < 0.40 (no far-field pull) but slower fill; end mass on thin targets ≥ baseline | end mass on thin targets < 0.15 (fill stalls) — **prediction failed (peak 0.586), fill did not stall (§5b)** |
| `w_bond` 100 + `vol_frontier` | peak < 0.35, end sparsity < 0.25 | — |
| `--ppc 8` + `w_bond` 100 (40k) | end sparsity < 0.25 and strays < 0.1 % | — |
| `render_ctrl` 24³ alone; 36³ (batch n) | 24³ alone reproduces the 24³+coh30 row (basis carries it); 36³ ≤ 24³ | 24³ alone back at the 12³ row (coh30 was the agent) |
| 40k `--ppc 8` + `render_ctrl` 24³ ± recipe (batch o) | the two levers add: end sparsity < 0.25, delivered ≥ 0.185, chamfer ≤ 0.12 | delivered < 0.172 (the ppc-8 recipe row) — **prediction failed: the levers do not add (end 0.319, delivered 0.173, chamfer 0.1175; froze at commit 135); falsifier not fired (§5f)** |
| support gate `(r_lo, r_hi)` = (0.05, 0.3), (0.1, 0.6); 24³ + (0.05, 0.3) (batch p) | if the affine fling is the ejection mechanism: peak < 0.50 on the per-particle control at no chamfer cost | peak ≥ 0.55 at both settings → the ejection is a translational push (control stress / loss pull on fringe particles), not an APIC artefact — **FIRED (0.569 / 0.569; §5e)** |
| physics-only `--lambda_auto 0` (batch p) | if the image loss drives the vanguard: peak < 0.45 | peak ≥ 0.55 → the driver is D_vol itself — **FIRED (0.560; §5e)** |
| 20k `--loss_units density --loss_res 128` (loss cell 0.25 wu < the 0.26 wu sparse threshold), per-particle control (batch q) | if "what the loss can see" is the lever: peak < 0.36 with fill ≥ baseline | peak ≥ 0.55 — **not fired; prediction met in direction, not magnitude (peak 0.493, fill 0.176 > 0.165; §5g)** |

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

### 5c. Batch n and the batch-c basis ladder — what the control basis does and does not set

All `render_ctrl` arms, 20k, dx 0.5, one seed (batch c rows re-probed from their archives).

| control basis / add-on | batch | chamfer | silIoU | thin-region SPARSE peak → end | delivered thin mass | strays > 2 sp |
|---|---|---|---|---|---|---|
| per-particle (flagship, for reference) | a | 0.1599 | 0.9655 | 0.578 → 0.379 | 0.165 | 1.27 % |
| 6³ | c | 0.1704 | 0.9483 | 0.506 → 0.371 | 0.138 | 4.39 % |
| 12³ | a | 0.1617 | 0.9533 | 0.508 → 0.281 | 0.180 | 1.97 % |
| 24³ | c | 0.1597 | 0.9570 | 0.506 → 0.250 | 0.191 | 1.27 % |
| 24³ (replicate) | n | 0.1605 | 0.9542 | 0.503 → 0.264 | 0.183 | 1.32 % |
| 36³ | n | 0.1601 | 0.9582 | 0.501 → 0.240 | 0.195 | 1.03 % |
| 24³ + `w_kin_var 50` | f | 0.1597 | 0.9589 | 0.502 → 0.232 | 0.194 | 1.12 % |
| 24³ + `w_coh 30` | l | 0.1604 | 0.9573 | 0.503 → 0.226 | 0.194 | 1.23 % |
| 24³ + `w_bond 100` | n | 0.1599 | 0.9570 | 0.503 → 0.246 | 0.189 | 1.17 % |
| 24³ + `vol_frontier` | n | 0.1599 | 0.9571 | 0.504 → 0.231 | 0.197 | 1.20 % |
| 12³ + `--ppc 8` + density units | c | 0.1660 | 0.8974 | 0.435 → 0.273 | 0.184 | 0.97 % |
| 12³ + density units | c | 0.1552 | 0.9625 | 0.360 → 0.231 | 0.135 | 1.10 % |

Reading (replicate band from the two 24³ rows: end ±0.01, delivered ±0.005, chamfer ±0.001):
- **The peak is set by the KIND of control, not its resolution.** Every basis from 6³ to 36³
  gives 0.50 (per-particle 0.58); the basis add-ons do not move it. The per-particle →
  basis step removes the part of the spray that a single-particle actuation can produce;
  what remains (half the arriving particles locally sparse at commit ~5) is produced by a
  spatially smooth control — i.e. by the loss asking for the ear before the ear has a
  connected stream to fill it with.
- **The END sparsity and the delivered mass are set by basis resolution, monotone and
  saturating:** 6³ 0.37 / 0.138 (cannot resolve the ear, silIoU −1.7 pt), 12³ 0.28 / 0.180,
  24³ 0.25–0.26 / 0.183–0.191, 36³ 0.24 / 0.195. 36³ is the best basis arm on every column
  and within noise of 24³ on chamfer.
- **The objective add-ons on the basis are noise-level:** coh30 0.226, frontier 0.231,
  bond100 0.246, kin_var50 0.232 against 0.250/0.264 for 24³ alone — at most −0.03 end
  sparsity (2–3 noise bands, no replicate), delivered within ±0.006. None is adopted on this
  evidence; `w_bond` is additionally harmful at fine dx (§5b).
- **Density units lower the peak most (0.36) but under-fill (0.135):** the resolution-
  invariant D_vol weighs the ear by its density deficit, not its cell count, so the ear is
  cheaper to leave empty; the finer MPM grid (`--ppc 8`) at 20k is silhouette-poor (0.8974)
  because 20k particles at ppc 8 give a 0.34 wu cell that the thin features do not fill.

Verdict on the pre-registration rows: "24³ alone reproduces the 24³+coh30 row" — CONFIRMED
(0.250/0.264 vs 0.226; coh30 is not the agent). "36³ ≤ 24³" — confirmed (0.240). What the
basis cannot do is remove the commit-5 peak; that is the driver question batch p asks
(render-loss granularity vs APIC fringe fling).

### 5d. Gate-coverage probe on the baseline archives (pre-registration for batch p)

`gate_probe.py` (hyde06 `/tmp/`): the kernel's 3³-cell count `n_p` recomputed per frame from
the archived raw state on the MPM grid (dx 0.5, 64³), `n₀` = median over the source, ω per
(r_lo, r_hi); "vanguard" = particles on thin targets AND locally sparse (§1 definition).

| archive | n₀ | (r_lo, r_hi) | vanguard at its peak: median n/n₀, mean ω, share ω = 0 | body: mean ω, share ω < 1 | on-thin mean ω |
|---|---|---|---|---|---|
| flagship `rcp_20k_a` | 1007 | (0.05, 0.3) | 0.31, 0.79, 0.00 | 0.99, 0.03 | 0.88 |
| `render_ctrl` 12³ | 1007 | (0.05, 0.3) | 0.31, 0.79, 0.02 | 1.00, 0.02 | 0.90 |
| flagship `rcp_20k_a` | 1007 | (0.1, 0.6) | 0.31, 0.42, 0.07 | 0.94, 0.20 | 0.55 |
| 40k `--ppc 8` (dx 0.2148, 149³) | 203 | (0.1, 0.6) | 0.43, 0.64, 0.03 | 0.97, 0.15 | 0.80 |
| 40k `--ppc 8` + density + recipe | 203 | (0.1, 0.6) | 0.39, 0.58, 0.04 | 0.97, 0.16 | 0.74 |

**What this changes.** `n₀` = 1007 means ~37 particles per 0.5 wu cell at 20k; the vanguard
sits in neighbourhoods of ~300 particles. At 40k `--ppc 8` (7.5 ppc, `n₀` 203) the vanguard
still has 0.39–0.43 of the interior count (~85 particles in its 3³ cells): not grid-isolated
at either discretisation. The numerical-fracture reading in §1 (a particle
beyond an empty cell has no grid neighbours) is therefore wrong at this discretisation and
is struck out above: the spray is a sub-cell density deficit of a connected stream, not a
set of detached particles. Consequences, stated before batch p reports:
- the (0.05, 0.3) gate leaves the vanguard at ω ≈ 0.8 and touches 3 % of the body — it is
  predicted to change nothing (a null result there is NOT evidence against the mechanism);
- the (0.1, 0.6) gate halves the affine transfer of the vanguard (ω 0.42) at the cost of
  gating 20 % of the body (mean 0.94) — the informative arm; if the peak drops there, a
  stronger, body-sparing gate (e.g. (0.2, 0.8) with n₀ measured on the ear) follows;
- the loss-side reading is now the leading one: a 0.5 wu D_vol cell and a 64³ silhouette
  cannot see a 0.26 wu spacing deficit, so the descent fills the ear's cells with the
  cheapest mass — a stretched stream — and only the finer loss grid (density units, `--ppc
  8`, peaks 0.36–0.44) and the particle-scale pulls act below the cell. The physics-only arm
  (`--lambda_auto 0`) in batch p separates the image loss from D_vol in this.

### 5e. Batch p — the two driver hypotheses (both falsified) and batch o2

| arm (20k, dx 0.5 unless noted) | chamfer | silIoU | hole | thin-region SPARSE peak → end | delivered thin mass | strays > 2 sp |
|---|---|---|---|---|---|---|
| baseline `rcp_20k_a` | 0.1599 | 0.9655 | 0.04 % | 0.578 → 0.379 | 0.165 | 1.27 % |
| support gate (0.05, 0.3) | 0.1593 | 0.9665 | 0.07 % | 0.569 → 0.396 | 0.165 | 1.05 % |
| support gate (0.1, 0.6) | 0.1588 | 0.9680 | 0.01 % | 0.569 → 0.405 | 0.163 | 1.00 % |
| **physics-only** (`--lambda_auto 0`, same code path, render OFF) | 0.1597 | 0.9561 | 0.06 % | 0.560 → 0.324 | 0.159 | 0.95 % |
| 24³ + gate (0.05, 0.3) | 0.1612 | 0.9576 | 0.07 % | 0.498 → 0.263 | 0.187 | 1.40 % |
| 24³ alone (batch n) | 0.1605 | 0.9542 | 0.08 % | 0.503 → 0.264 | 0.183 | 1.32 % |
| 40k `--ppc 8` + 24³, legacy units (batch o2; **froze at commit 46**) | 0.1337 | 0.9452 | 0.03 % | 0.436 → 0.245 | 0.183 | 0.36 % |
| 40k `--ppc 8` per-particle, legacy units (batch i) | 0.1164 | 0.9620 | 0.38 % | 0.481 → 0.295 | 0.188 | 0.23 % |

- **(b) APIC fringe fling — FALSIFIED.** The (0.1, 0.6) gate halves the vanguard's affine
  transfer in the gated run itself (`gate_probe.py` on its archive: vanguard mean ω 0.43,
  7 % at ω = 0, body 0.94) and the peak is 0.569 against 0.578 (noise band ±0.01); the end
  is slightly worse (0.405). The gate costs nothing on the metrics (chamfer 0.1588, silIoU
  +0.25 pt, strays −0.3 pt) but does not touch the spray. It stays opt-in, default off.
- **(a) the render loss — FALSIFIED as the driver.** With the image loss OFF on the same
  code path the peak is 0.560. The spray is produced by the physics-side objective — D_vol
  mass matching on a per-particle control. What the render loss does is fill the ears MORE
  (0.165 vs 0.159 delivered, +0.9 pt silIoU) at the price of a higher END sparsity (0.379 vs
  0.324): the silhouette pulls mass into the ear faster than D_vol alone, and that mass
  arrives at ~2× spacing. That is the premise working at the wrong granularity, and the
  granularity is the lever (§5c: basis; §5b/§5d: `--ppc 8`, density units).
- **Batch o2** (40k `--ppc 8` + 24³ in LEGACY units): the line search exhausted and the
  patience rule froze the run at commit 46 (the legacy cell-sum / fine-dx mismatch REFUTE-2
  F4 documented; the per-particle legacy arm survives it, the basis does not). On its
  delivered slice the peak 0.436 and end 0.245 are the lowest of any 40k arm so far, with
  chamfer 0.1337 (+15 % vs per-particle) — an early stop, not a result. The density-unit
  run (o1, `--loss_units density --warm_start --w_kin 5 --w_kin_var 200` + 24³) is the
  real test and is reported in §5f.

**Closing verdict on the user's question (volume, or something else?).** Something else,
and it is now pinned: the ears are filled first by the nearest particles at ~2× spacing
because the mass-matching loss cannot see spacing below its cell, and the per-particle
control lets it move single particles to do so. The particles are not detached (§5d), the
material is not over- or under-volumed (J p2p 0.008, `w_jvol` active), nothing is flung by
the transfer (batch p), and the image loss only amplifies the fill. Remedies that act at
window end on the state (`w_coh`, `w_bond`, `vol_frontier`) cannot undo a stream that the
descent has already stretched inside the window; remedies that change WHAT the descent can
move (a coarse basis: peak 0.50, end 0.24 at 36³) and WHAT the loss can see (`--ppc 8`:
peak 0.44–0.48, strays 0.01–0.2 %; density units: peak 0.36) are the ones that work. Mass
ejection at the END state is a discretisation problem and is solved by the contract
(`--ppc 8`: strays 0.07 % on the flagship candidate vs 1.3–2 % at dx 0.5).

### 5f. Batch o1 — the two levers together at 40k (`--ppc 8`, density units, recipe, 24³ basis)

| arm (40k, dx 0.2148, loss grid 149³ density units, `--warm_start --w_kin 5 --w_kin_var 200`) | commits delivered | chamfer | silIoU | hole | peak → end | delivered thin mass | strays > 2 sp |
|---|---|---|---|---|---|---|---|
| per-particle control (batch j, the flagship candidate) | 300 | 0.1142 | 0.9735 | 0.52 % | 0.444 → 0.329 | 0.172 | 0.07 % |
| + `render_ctrl --control_grid 24` (o1) | **135 (froze)** | 0.1175 | 0.9712 | 0.45 % | 0.424 → 0.319 | 0.173 | 0.22 % |
| + `render_ctrl --control_grid 36` (q2) | **119 (froze)** | 0.1151 | 0.9738 | 0.00 % | 0.433 → 0.344 | 0.173 | 0.11 % |
| 24³, legacy units, no recipe (o2, §5e) | 46 (froze) | 0.1337 | 0.9452 | 0.03 % | 0.436 → 0.245 | 0.183 | 0.36 % |

The basis on the fine grid does NOT add to the discretisation contract: within noise on the
spray (peak −0.02, end −0.01), chamfer +3 %, silIoU −0.2 pt, strays 3× (still 0.2 %), and
both basis runs at `--ppc 8` stop early through the outer-merit guard ("rejected candidate,
reversal −0.55" on consecutive windows → stale → freeze) that the per-particle recipe never
trips. At dx 0.5 the basis is the lever (§5c); at dx 0.215 the fine grid already gives the
control the granularity the basis was supplying, and the basis then only costs line-search
acceptance. The 36³ pair (q2) confirms it: froze at commit 119, chamfer 0.1151, silIoU
0.9738 (+0.03 pt), spray 0.433 → 0.344, delivered 0.173 — every column within noise of the
per-particle recipe, with 40 % fewer commits accepted. **The flagship candidate stays
per-particle at `--ppc 8` in density units with the kinetic recipe** (peak 0.44, end 0.33,
strays 0.07 %, chamfer 0.1142), and the 20k deliverables use the 24³–36³ basis.

### 5g. Batch q — making the loss see the spacing (20k, per-particle control, density units, 128³ loss grid)

| arm (20k, dx 0.5) | loss cell | chamfer | silIoU | hole | peak → end | delivered thin mass | strays > 2 sp |
|---|---|---|---|---|---|---|---|
| baseline (legacy units, 64³) | 0.5 wu | 0.1599 | 0.9655 | 0.04 % | 0.578 → 0.379 | 0.165 | 1.27 % |
| density units, 64³, 12³ basis (batch c) | 0.5 wu | 0.1552 | 0.9625 | 0.02 % | 0.360 → 0.231 | 0.135 | 1.10 % |
| **density units, 128³, per-particle** (q) | 0.25 wu | 0.1553 | 0.9634 | 0.02 % | 0.493 → 0.300 | 0.176 | 1.32 % |
| density units, 128³, **36³ basis** (q3) | 0.25 wu | 0.1558 | 0.9637 | 0.00 % | 0.487 → 0.311 | 0.177 | 1.31 % |
| 36³ basis, legacy units, 64³ (batch n) | 0.5 wu | 0.1601 | 0.9582 | 0.06 % | 0.501 → 0.240 | 0.195 | 1.03 % |

A loss cell below the sparse threshold, with nothing else changed, moves the per-particle
control from 0.578 → 0.493 at the peak and 0.379 → 0.300 at the end, delivers MORE thin mass
than the baseline (0.176 vs 0.165) and takes chamfer −2.9 % (silIoU −0.2 pt, strays
unchanged). This is the same peak the coarse basis reaches by the other route (0.50, §5c)
and, unlike the density/64³ basis arm, it does not under-fill. The pre-registered magnitude
(peak < 0.36) was not reached: the loss seeing the spacing halves the excess over the basis
arms but does not remove the transient — the remaining ~0.5 is what a mass-matching descent
produces when the ear must be filled from a distance in 20-step windows. The 40k 36³ pair
(second q arm) is reported when done.

**Recommendation update (2026-09-15 evening).** For the 20k deliverables: `--loss_units
density --loss_res 128` (per-particle) or `render_ctrl --control_grid 36` (basis), the two
levers measured to the same peak by different routes. **Their combination (q3) does not add:**
peak 0.487, end 0.311, delivered 0.177, chamfer 0.1558 — the density/128 numbers with the basis
on top, and the basis's own end-sparsity gain (0.240 in legacy units) is lost. The two routes
reach the same floor (~0.49 peak) because they remove the same thing — the single-particle
actuation that the coarse loss cannot see — and nothing measured so far goes below it. For
40k: the flagship candidate as recorded in §5f.

## 6. The ear–head "connection" (user report on the 40k flagship candidate, frame 1746/2545)

Measured on `rcpj_40k_ppc8_dens_rec` (raw state; `scripts/probes/web_probe.py`,
`ear_views.py`, `cover_diff.py`, the last two using the gallery's own splat renderer at 300 px
= 0.030 wu/px), mid frame 1745 and end frame 2545:

| quantity | mid | end |
|---|---|---|
| particles farther than 0.15 / 0.25 / 0.40 wu from ANY target point (over-mass in 3D) | 0.005 % / 0 / 0 | 0.003 % / 0 / 0 |
| target points with no particle within 0.15 wu (under-fill), ear / body | 1.56 % / 0.08 % | 1.20 % / 0.11 % |
| projected EXTRA (morph-only pixels, share of target area), az 0.6 / 1.4 / 2.2 / top | 1.04 / 0.92 / 0.70 / 1.01 % | 0.98 / 1.00 / 0.80 / 1.13 % |
| projected MISSING (target-only pixels), same views | 3.60 / 3.49 / 1.89 / 4.13 % | 3.30 / 2.84 / 1.77 / 4.02 % |

Reading (`cover_diff_*.png`, `ear_views_*.png`):
- In the az-0.6 gallery view the wedge between the lower ear and the head is covered by the
  TARGET's own projection — it is the far ear seen edge-on (the target render shows the same
  wedge, as a sparse band because a thin sheet sampled at 40k projects sparsely). The morph
  is not putting mass where the target has none: zero particles beyond 0.25 wu of the target
  surface at either frame, and the morph-only pixels are a ≤ 0.2 wu fringe along the ear base
  and rims (1 % of the projected area), not a bridge.
- At az 1.4, where the notch between the two ears and the head is visible, the notch is
  preserved in both the mid and the end frame (`40k_ppc8_loss_notch.gif` in the gallery).
- The genuine residual is the opposite sign: the ears are UNDER-filled — 3–4 % of the
  projected ear area and 1.2 % of the ear target points have no particle within 0.15 wu
  (body 0.1 %), the "sparse" 0.33 of §5f. What the eye reads as a web joining ear and head
  is the far ear rendered as a diffuse, coarser cloud (particles at ~2× spacing) instead of
  the target's dense thin sheet, overlapping the notch in projection.
- So the fix for the visual is the ear fill, not a separation term: the levers of §5 (loss
  granularity, basis at 20k), and — for a thin sheet whose thickness (~0.2 wu) is at the loss
  cell (0.215 wu at 40k `--ppc 8`) — a loss cell below the sheet thickness (density units,
  `--loss_res` above 149 at 40k) is the next pre-registered arm. A term that pushes mass OUT
  of the notch would act on mass that is not there.

**Addendum 2026-09-16 (target fix).** The user then pointed at the thin line above the ear and asked
whether the target discretisation itself was wrong. It was: trimesh's axis 'base' voxel fill
drew 1-voxel streaks on the non-watertight bunny (485 streak voxels), sampled as scattered
points at 20k and a dotted line at 40k (`docs/experiments.md` 2026-09-16, commit 0ccc43a:
orthographic fill + streak strip + `tests/test_sampler_fill.py`). Every number in §1–§6 was
measured against that target. Batch r re-ran the deliverable arms on the clean target: the
thin-feature picture is unchanged (20k per-particle 0.61 → 0.36 with 0.157 delivered; 36³
basis 0.52 → 0.23 with 0.191 / 0.191; 40k `--ppc 8` recipe 0.43 → 0.30 with 0.174 / 0.192,
chamfer 0.1135, ear under-fill at 0.15 wu 0.74 % vs 1.20 %, no over-mass near the ear), and
the "sparse band above the lower ear" in §6 was the far ear PLUS the streak — with the streak
gone the far ear renders as a lobe, and the ear–head "connection" question is closed on the
clean target as well (projected morph-only pixels 0.6–0.7 % of the target area, all fringe).
