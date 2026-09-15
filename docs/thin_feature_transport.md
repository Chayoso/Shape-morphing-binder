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
| not implemented | support-gated APIC (Yao–Zhao), Lagrangian bond forces (Jiang 2017), conservative split (Yue 2015) | forward model / post-hoc | held in reserve: forward-model changes need their own adjoint validation |

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

Reading rule: a mechanism that lowers the peak but not the end sparsity delays the spray;
one that lowers the end sparsity but delivers less mass trades fill for coherence — both
are reported, neither is adopted alone.
