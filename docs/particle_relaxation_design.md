# In-simulator particle relaxation — design (pre-registered, 2026-09-03)

Status: **design only, nothing implemented.** Branch `v3-grid-gs`. Scope: the sub-cell
particle-arrangement defect measured in docs/floaters.md ("Census", "Mechanism from the
census") and the x3/x5/x6 KDE-loss falsifications in docs/experiments.md. The remedy
designed here lives INSIDE the MLS-MPM substep (physmorph/mpm/kernels.py, traj.py), on the
`wp.Tape`, so the adjoint stays valid and no state is touched outside the rollout
(docs/engineering_review_20260902.md bans post-rollout overwrites).

Everything numeric below states its discretisation (AGENTS rule 4). New local measurements
(CPU, repo loader, no pipeline run) are marked **[measured 2026-09-03]**.

---

## 0. Two measured facts that change the reading of the census

Before the mechanism: two things about the premises were checked locally with the repo's
own loader (`physmorph.sampling.load_normalized`, `assets/bunny.obj`, `assets/isosphere.obj`,
N = 20k/40k, seeds 1/2 as in `scripts/pipeline_run.py`). Both matter for what "target
spacing" and "cluster ratio" mean, and therefore for the calibration rule.

### 0.1 The bunny target sample is a surface SHELL, not a volume [measured]

`sample_volume` voxelises the mesh at pitch = extent/110 and calls `VoxelGrid.fill()`.
`assets/bunny.obj` is **not watertight** (69,451 faces, `is_watertight=False`), and
`fill()` adds no interior voxels at any pitch tried (110, 80, 60, 40 cells across: filled
count == surface count every time). The 20k "volume" sample is therefore drawn from 38,140
surface voxels:

| quantity | isosphere source (20k) | bunny target (20k) | bunny target (40k) |
|---|---|---|---|
| bbox (normalised, diag 8) | 4.52 × 4.67 × 4.66 | 3.87 × 4.98 × 4.93 | same |
| voxel-fill centres / implied volume | 633,327 / **49.2** | 38,140 / 3.5 (a 1–2-voxel skin) | — |
| distance of samples to the mesh surface p50 / p99 / max | (solid) | **0.016 / 0.050 / 0.064** | — |
| neighbour-count scaling exponent at 1→2→4→8 NN-spacings | 3.03 / 2.95 / 2.87 (volume) | **2.57 / 2.25 / 2.11 (surface)** | — |
| median NN spacing (`tgt.nn_spacing`, the census "sp") | 0.0756 | **0.0341** | 0.0250 |
| volumetric mean spacing d = (V/N)^{1/3} | **0.135** | 0.108 for a SOLID bunny (V = 25.1, orthographic fill at pitch 0.062) | 0.086 |
| dx / spacing (dx = 0.5) | 6.6 | 14.7 (shell sp) | 20.0 |

Consequences:

- The census unit "sp" = 0.034 (20k) is the **shell** sampling spacing. It is 4× below the
  volumetric spacing of 20k particles filling the bunny (0.108) and of the source sphere
  (0.135). The docs' "D_vol cell ≈ 3.4 sp" uses the volumetric spacing (0.5/0.135 = 3.7),
  the out_nn / berth / KDE-h / display-sigma numbers use the shell spacing. Two different
  "sp" are in circulation.
- The solid bunny (25.1) is half the source volume (49.2). An isochoric body cannot fill
  the bunny; a body with the source volume cannot be inside the bunny.
- Chamfer floors against the shell target [measured, `metrics.chamfer`, symmetric mean
  NN]: shell-vs-shell (sampling noise, seed 3 vs seed 2) **0.071**; a 0.25-thick surface
  layer of a solid bunny sample **0.160**; a full solid bunny **0.340**; the undeformed
  sphere 0.522. The converged runs report **0.087 (20k) / 0.071 (40k)** — the shell-vs-
  shell floor. **The converged body is a thin layer wrapped on the bunny surface**, not a
  solid: 20k particles over 58.4 units² of surface, i.e. an areal spacing of 0.054 and,
  for a layer 2–3 particles thick, a 3-D spacing ≈ 0.05–0.10. `out_nn` for a solid bunny
  sample would read 75%, so the 8–12% reported is only possible for a layer.
- "J = det F ≈ 1 everywhere" cannot certify the cloud's volume: `k_update` blends
  `F_out = (1−s)·F_new + s·F_in` with s = 0.955, so F integrates **4.5% of each substep's
  deformation increment** (and of dFc). det F ≈ 0.97–1.0 is what a sphere flattened into a
  layer reports through that filter. This is the legacy morphing line's intended plastic
  behaviour, not a bug in the mechanism below, but the census inference "J≈1 ⇒ not a
  volume defect" does not hold. Check on the delivered state: (m/ρ_grid)^{1/3} per
  particle (the `k_volume` estimator on the current positions) versus Vp0^{1/3}.

### 0.2 The census "cluster ratio" mixes two things; the arrangement defect is Poisson clumping inherited from sampling [measured]

- The source sample is a jittered-voxel draw and is **exactly Poisson** in its NN
  statistics: mean neighbours within the median NN distance = 0.69 (Poisson: ln 2 = 0.693),
  p10/p50/p90 NN = 0.53/1.00/1.51 of the median (Poisson: 0.53/1.00/1.49). A Poisson
  arrangement has 41% of particles with a neighbour closer than 0.5 d and gaps up to ~2 d.
  Nothing in the pipeline can act on that arrangement (§1), so the delivered state carries
  the sampling's clumps and voids — the pinholes and filaments in the photoreal renders
  (display sigma 0.034–0.043 on a layer whose areal spacing is 0.054–0.076 is σ ≈ 0.5–0.6
  spacing: a Poisson layer shows holes at exactly that σ; a blue-noise layer does not).
- The census ratio "particle KDE / target KDE at particle positions, h = 2 sp = 0.068" is
  evaluated against a 1-voxel-thin shell. A body layer thicker than, or offset from, that
  shell reads ratio > 1 everywhere off the shell's mid-surface, with no clumping at all.
  The ratio therefore conflates (i) **layer thickness/offset normal to the shell** and
  (ii) **in-layer clumping**. The KDE losses (x3/x5/x6) minimised (i) — they thinned the
  layer toward the shell mid-surface and expelled the surplus (out_nn 8 → 14–22%), which
  is the documented failure mode ("exterior particles near full faces treated as
  surplus"). Only (ii) is an arrangement defect and only (ii) is addressed here.
- The "unfilled target points" numbers are close to what a Poisson layer at the body's
  areal density gives against a 4× finer shell: P(no particle within r) = exp(−π ρ_A r²)
  with ρ_A = 20000/58.4 → 54% at r = 1 sp, 25% at 1.5 sp (census: 65%, 33%). A perfect
  blue-noise layer at the same density still leaves ~25–35% of shell points without a
  particle within 1 sp = 0.034, because the body has 4× fewer points per area than the
  shell sample. **That floor is not a defect and must not be a falsifier.**

Pre-registration consequence: the primary falsifier for the mechanism is a **target-free
arrangement statistic** (§8), the census metrics are kept as continuity telemetry with
their floors stated, and chamfer/out_nn/guards are do-no-harm gates.

---

## 1. Why the grid cannot fix it (and why every loss-side remedy scattered the fringe)

MLS-MPM transfers are P2G: particles → 4³ cubic B-spline nodes, G2P: nodes → particles. With
≈ 40 particles per cell (dx = 0.5, spacing 0.135) the P2G map has a large null space: any
sub-cell rearrangement of the particles in a cell that keeps the nodal mass and momentum
moments is invisible to the grid, receives no restoring force from the continuum, and is
returned unchanged by G2P. This is the MPM "null space" of Gritton & Berzins and Tran et
al. (§14: errors in it are not damped and accumulate). APIC/PIC resets particle velocities
from the grid every substep (`k_g2p`: `v[p] = Σ w v_g`), so sub-cell relative velocities
are projected out each step — the arrangement is frozen in the null space and drifts only
with numerical noise. Cubic B-splines reduce cell-crossing quadrature error (Steffen et al.)
but do not touch this null space, and no higher-order transfer (PolyPIC) does either: they
recover velocity modes, not position modes.

The control `dFc` acts through stress → P2G → grid → G2P, i.e. entirely in the range of the
transfer: **the optimiser cannot produce sub-cell moves**. A loss that sees the sub-cell
scale (KDE, fine grids, the 96-px silhouette whose pixel is 0.57 spacing) therefore asks for
something the control cannot deliver; the only way it can lower such a residual is by
cell-scale mass motion — thinning the layer, expelling mass outward (x3/x5/x6), or noisier
CIC on an N-limited grid (x1). The fix must be a **particle-level operator inside the
substep**, orthogonal to the grid, conditioned on the density the grid owns.

---

## 2. Candidate evaluation against the literature (§14 for citations)

Criteria: (a) physically admissible here = pairwise antisymmetric (momentum-conserving)
and inside the rollout; (b) differentiable on the existing `wp.Tape`; (c) leaves F, Fp,
mass, Vp0 untouched (assimilation contract, isochoric Fp); (d) cost at N = 40k, T = 20,
8 iterations, hundreds of commits.

| # | candidate | mechanism | (a) | (b) | (c) | (d) | verdict |
|---|---|---|---|---|---|---|---|
| 1 | SPH near-pressure / Clavet et al. 2005 double density relaxation | pairwise displacement `D_ij = dt²[P(1−q) + P_near(1−q)²] r̂`, half to each particle, velocity re-derived from positions; P_near = k_near ρ_near is the anti-clustering term, always repulsive, cutoff h | yes (antisymmetric pairs; Clavet re-derives v so momentum is that of the moved positions) | yes (smooth pair function, static loops over a pair list) | yes | ~10⁵ pairs/substep, trivial | **ADOPT the near-pressure half** (§3). The rest-density half P = k(ρ−ρ0) is cell-scale density control — that is the grid's/D_vol's job here and would re-create the x3 outward push against a shell target. Monaghan 2000's artificial pressure is the same object (repulsive short-range term that removes tensile clumping). |
| 2 | FLIP position correction (Ando & Tsuruno 2011: weak springs; Ando, Thürey, Tsuruno 2012: anisotropic position correction); Zhu & Bridson 2005 PIC/FLIP blend | pairwise repulsive springs within a radius applied to positions each step, tuned to keep an even distribution in FLIP | yes, same structure as 1 | yes | yes | same | Same mechanism as 1 in a PIC/FLIP host — the closest precedent for MPM. Zhu–Bridson's PIC fraction regularises velocity noise only; it has no position term, which is why Ando had to add one. The anisotropic (sheet-preserving) variant is not needed: our layer is a closed surface, not a breaking sheet. |
| 3 | Particle resampling / splitting / merging (Yue et al. 2015 Continuum Foam resampling; Gao et al. 2017 adaptive GIMP; Wang et al. 2019) | discrete insertion/deletion of particles with F, mass interpolated from neighbours | changes N, mass, momentum bookkeeping | **no** — discrete topology change, not on a tape; kills the (T,N,3,3) control leaf layout | no (F assigned by interpolation) | — | **REJECT.** Note: Wang et al. 2019's abstract does not describe particle resampling (its contribution is damage + crack visualisation), so it is not a precedent; Yue et al. 2015 is the genuine MPM resampling precedent and confirms the practitioners' view that poor distributions are a real MPM defect repaired at the particle level. |
| 4 | Higher-order transfers (APIC, PolyPIC) | more velocity modes per particle | n/a | already used (APIC) | n/a | — | **No effect on arrangement**: they change what the grid returns to velocities, not the position null space. APIC's velocity reset is what makes candidate 1 equivalent to a displacement (§3.3). |
| 5 | Null-space filters (Gritton & Berzins 2017; Tran, Berzins, Sołowski 2019) | project particle quantities onto the range of P2G (SVD or cheaper) each step | not a force; acts on particle fields (velocity/stress), not positions | linear, differentiable | untouched | O(N·stencil) | Diagnoses the problem exactly but filters velocity/stress modes; it does not move particles. Not the remedy for a positional defect. |
| 6 | SPH particle shifting (Lind et al. 2012, Fickian) | shift along −∇C toward low concentration, with a free-surface correction | shift, not force; conservation only approximate | yes | yes | cheap | Rejected in favour of 1: shifting is proportional to the concentration gradient, so at a free surface it pushes outward unless corrected by surface detection — the x3 failure in another guise. A cutoff repulsion is exactly zero on any arrangement with spacing ≥ h, surface or bulk. |
| 7 | Position-based fluids artificial pressure (Macklin & Müller 2013) | Monaghan's s_corr inside a constraint solve; constraint averaging by neighbour count | as 1 | yes | yes | cheap | Supports 1; its neighbour-count averaging is the stability device adopted in §3 (near-density normalisation). |
| 8 | Colagrossi et al. 2012 particle packing | offline relaxation to a uniform packing before the run | offline | n/a | n/a | — | Would fix the SOURCE's Poisson arrangement only; the morph's flattening re-creates anisotropy. The in-loop form (1) subsumes it. Useful as the initial-state variant (§12, arm C). |

Decision: **candidate 1's near-pressure term, hosted as a per-substep pairwise displacement
of the FLIP position-correction type (2), normalised as in (7), with the cutoff length taken
from the grid density so it never competes with the grid.**

---

## 3. Mechanism specification

### 3.1 Local rest length (frozen per window, host side)

At the window start x0 (the state the pair lists are built on):

```
ρ_i   = Σ_g w_gi m_g / dx³          — the k_volume estimator at the CURRENT positions
d_i   = (m_p / ρ_i)^{1/3}           — local spacing implied by the grid density
d_med = median_i d_i
h_ij  = c · min(d_i, d_j, d_med)    — pair cutoff, symmetric in (i,j)     c = 0.75
```

`min` over the pair: the denser side sets the cutoff, so a free-surface density deficit
(ρ interpolated at a surface particle is ~0.5–0.7 of the bulk value → d inflated by
1.12–1.26) never inflates a mixed pair; the `d_med` cap bounds surface–surface pairs. The
cutoff follows the density the grid owns: in a thin layer it is the layer's spacing, in
the ears the ears' spacing. With c = 0.75 the hard-core packing fraction at the local
density is (π/6)c³ = 0.22 — far below random jamming (0.64) — so a force-free arrangement
with all pairs ≥ h always exists: **the term can always switch itself off**, it never
carries a residual pressure into the continuum. c = 0.75 is the blue-noise "relative radius"
regime (0.65–0.85 of the densest packing's spacing, Lagae–Dutré quality measure); it is a
geometric constant of point-set quality, not a tuned gain (§7).

### 3.2 Pair displacement (device, per substep t, on the tape)

For every frozen pair (i, j) with r = |x_j − x_i| and q = r / h_ij, at the current x[t]:

```
φ_ij = (1 − q)²   if q < 1, else 0                  C¹ cutoff: φ(1) = φ'(1) = 0
ñ_i  = 1 + Σ_{j∈P(i)} φ_ij                          near density (Clavet's ρ_near form)
w_ij = 2 / (ñ_i + ñ_j)                              symmetric normalisation (PBF averaging)
Δx_i = − α Σ_{j∈P(i)} h_ij · w_ij · φ_ij · r̂_ij       r̂_ij = (x_j − x_i)/r,  α = 1/4
```

Pairs with r < ε·h_ij (ε = 1e-3) are skipped (direction undefined; count them). Two
device kernels over pairs: `k_near_density` (atomic-add φ into ñ_i, ñ_j) then
`k_pair_shift` (atomic-add ∓ the pair term into Δx_i, Δx_j). Antisymmetry is exact per
pair: Δx_j receives the negated term, so Σ_i Δx_i = 0 up to float atomics.

### 3.3 Where in the substep; two modes

```
k_stress → [k_near_density, k_pair_shift on x[t]] → k_p2g → k_grid_op → k_g2p → k_update
```

- **Mode 2 — position shift (primary):** `k_update` becomes
  `x[t+1] = x[t] + dt·v[t+1] + Δx[t]`. Velocities and C are untouched; the next P2G
  scatters the rearranged positions. This is Clavet/Ando's displacement form.
- **Mode 1 — impulse (A/B alternative):** `k_g2p` adds `δv = Δx[t]/dt` to `v[t+1]`
  before `k_update` (which then uses the unchanged `x + dt·v`). Equivalent to a pairwise
  force `f_ij = (m/dt²)·α h_ij w_ij φ_ij r̂_ij`, stiffness `k_n = α m h/dt²`.

Why the two are the same mechanism here: `k_g2p` overwrites `v` from the grid every
substep (APIC/PIC), so an impulse survives exactly one advection and is then filtered; the
only residue is the pair's grid dipole `m δv (w_gi − w_gj)`, whose G2P return is
O((r/dx)²·m/m_g) ≈ (0.2)²/40 ≈ 0.1% of δv. Mode 1 additionally puts δv into the promoted
`v_T` (the w_kin term and the next window's P2G see a relaxation velocity of up to
h/(2dt) = 12 wu/s), which is why mode 2 is primary. Mode 1 is kept because it is the
literal "momentum-conserving pairwise force" and is the arm that tests whether letting the
continuum see 0.1% of the rearrangement matters (prediction: it does not).

Per substep, not per window: T = 20 relaxation steps per window keep the arrangement near
its equilibrium at all times (distributed correction, the same lesson as `k_add_guidance`).

---

## 4. Neighbour search plan

**Frozen per window, built on the host** (the pattern `kde_assign`/`nn_band_assign`
already use): at x0, `scipy.spatial.cKDTree(x0).query_pairs(r_list, output_type="ndarray")`
with `r_list = 1.5 · c · d_med`, then per-pair `h_ij` from §3.1, symmetrised as unique
(i < j) pairs → int32 arrays `pi, pj` and a float32 `hij`, uploaded once, no grad.
Expected pairs: neighbours within 1.125 d at density 1/d³ → (4π/3)(1.125)³ ≈ 6 → **≈ 3 N
pairs**: 60k (20k) / 120k (40k). [measured on the source sample: 1.16 N pairs within
1.5 median-NN, which is 0.83 d; consistent.]

Why not the MPM grid's cell lists: dx = 0.5 is 3.7–5 cutoffs; a 27-cell stencil holds
~1,000 candidates per particle at 40 particles/cell — 100× the useful pairs. Why not a
per-substep `wp.HashGrid`: it is the natural GPU alternative (`HashGrid.build` ≈ 0.1 ms at
40k, query loops are `while` iterations whose adjoint support is not something this repo
has validated), and it is pre-registered as the fallback if the frozen list misses pairs:
telemetry `rep_missed` = fraction of pairs with q < 1 at x_T that were absent from the
frozen list (one fresh `query_pairs` at commit time, CPU). Switch to a per-substep rebuild
(host or HashGrid) only if `rep_missed` > 1% in any window after the first ten. A missed
pair is a delayed relaxation, never an instability: only listed pairs generate motion.

Determinism: the same pair list is used by the tape rollout, every line-search candidate
(`eval_terms`) and the committed final rollout in `optimize_window`, so within a window the
forward map is a fixed C¹ function of dFc (the line search's Armijo test stays meaningful).
Float atomics add replay noise of the kind `replay_calibrate` already measures.

---

## 5. Invariants preserved

| invariant | mode 2 (shift) | mode 1 (impulse) | test |
|---|---|---|---|
| centre of mass Σ m_i x_i | exact (pairwise antisymmetric Δx, equal masses `m = 1.0`) | exact | Σ Δx = 0 to 1e-6·N·h |
| linear momentum Σ m_i v_i | exact (v untouched) | exact (antisymmetric δv) | Σ m δv = 0 |
| angular momentum | changes by Σ_pairs Δx_ij × m (v_i − v_j) = O(|Δx|·|C|·r): same-cell PIC velocities differ by C·r | exact for central forces | measured, bound reported |
| mass, Vp0, Fp, λ, μ | untouched | untouched | — |
| F | no direct term; changes only via the grid's response to rearranged positions (mode 2) or the 0.1% dipole (mode 1) | same | det F track unchanged within noise |
| finiteness | \|Δx_i\| < 2αh_max = h_max/2 per substep by construction (§7); NaN-free if x is (valid_pos guard reused) | same | guard counters 0 |
| `rep = 0` | bit-identical to the current kernels (the added terms are literally zero; kernels launched only when pairs exist) | same | parity test vs G1a fixture |
| free surface | zero force on any arrangement with all pairs ≥ h — no outward pressure at a surface unless the surface layer itself is clumped, and then bounded by one cutoff | same | ball test: p99 radius grows ≤ 2% |

---

## 6. Differentiability story

- Kernels needing adjoints: `k_near_density(x, pi, pj, hij, nd)` and
  `k_pair_shift(x, pi, pj, hij, nd, alpha, dxr)` (both dim = n_pairs, read `x[t]`, write via
  `wp.atomic_add`), and the one-line additions to `k_update` (mode 2) or `k_g2p` (mode 1).
  Warp's tape generates the adjoints; atomic adds are supported as adjoint accumulation
  ("in-place operations such as `x[tid] += 1.0` and `wp.atomic_add()` … the Warp graph
  specifically accommodates adjoint accumulation in these cases" — Warp differentiability
  docs, §14). Warp-lang 1.9.0 locally, 1.16 on hyde01/hyde06.
- No in-place read-write of a differentiable array (the repo's own `k_grid_op` rule):
  `nd[t]`, `dxr[t]` are fresh per-step arrays allocated zero in `Trajectory.__init__`; mode 1
  adds δv inside `k_g2p` while it is still writing `v[t+1]` from registers; mode 2 adds
  `dxr[t]` as an extra input of `k_update`.
- Loops: none over neighbours inside a kernel — the pair kernels are one thread per pair
  with integer index arrays as non-differentiable inputs. No `while`, no dynamic bounds, no
  hash-grid iterators on the tape.
- Gradient flow: ∂L/∂x[t] gains the transposed Jacobian of the shift; the pair term is C¹
  (φ = (1−q)² ⇒ C¹ force at the cutoff) so the window objective stays C¹ in dFc.
- Adjoint amplification: the shift is a damped-Jacobi step on a repulsive potential —
  contractive in the radial direction (eigenvalues of ∂Δx/∂x in [−1/2, 0] there) and
  expansive tangentially by the ratio r_after/r_before for a separating pair (symmetry
  breaking of nearly coincident particles is real sensitivity, not an artefact). The
  ε-skip caps it at 1/ε per pair; the source sample has ~16 pairs below r = 0.1 h
  [measured: Poisson estimate at 20k], and none after the first windows.
- Memory on the tape: `nd[t]` (N float) + `dxr[t]` (N vec3) = 16 B/particle/step → 13 MB
  at 40k × 20, against ~280 MB the trajectory already keeps (x, v, C, F, Fraw, P + grid
  arrays per step). The pair Jacobians are recomputed in the backward, never stored.
- Gates: G1a parity (rep = 0), a G1b-style finite-difference check of dL/ddFc on the CPU
  device with rep = 1 (T = 4, 128 particles, rel err < 0.25 as in tests/test_bridge_cpu.py)
  and a direct FD of the pair kernels (rel err < 1e-3 in float32 at h·1e-2 steps).

---

## 7. Calibration rule (pre-registered; no tuning)

The mechanism has one length and one rate.

**Length** `h_ij = c · min(d_i, d_j, d_med)`, `c = 0.75`, with d from the grid density
(§3.1). Reading of the ask's rule "equilibrium spacing = target NN spacing": the
equilibrium min-distance is 0.75 of the local spacing implied by the grid density; when
D_vol → 0 that density is the target's cell-scale density, so the equilibrium spacing equals
the target's **volumetric** spacing at the cell scale by construction — for any way the
target was sampled. The target sample's own NN spacing (0.034, a shell statistic, §0.1) is
explicitly NOT used: the body cannot reach it (its areal density is 4× lower) and a cutoff
there would act on 3% of the pairs [measured: 7,047 pairs closer than 1 shell-sp at 20k].

**Rate** `α = 1/4` (max per-substep move = h/2), derived, not chosen:

1. Row-sum bound. With the near-density normalisation,
   `|Δx_i| ≤ α h_max Σ_j φ_ij · 2/(ñ_i + ñ_j) ≤ 2 α h_max (ñ_i − 1)/ñ_i < 2 α h_max`
   for any crowding. The shift is a damped-Jacobi (PBD constraint-averaging) step whose
   row sums are bounded by 2α.
2. Isolated pair: ñ_i = ñ_j = 1 + φ, so Δr = 2αhφ/(1+φ). No overshoot past the cutoff
   ⇔ 2α(1−q)/(1+(1−q)²) ≤ 1 for all q ∈ [0,1) ⇔ α ≤ 1.
3. Many-body safety: a particle must not move more than half its cutoff in one substep
   (it can then never jump across a neighbour's exclusion zone it is not already inside):
   2αh ≤ h/2 ⇒ α = 1/4. This is the damped-Jacobi factor ω = 1/2 under the row-sum bound
   of item 1 — the standard convergent choice — and sits at 4× margin from item 2.

Equivalent stiffness for mode 1: `k_n = α m h / dt²` = 0.25·1.0·h·240² ≈ **1,440 h**
(≈ 145 at h = 0.10, in the run's mass/length/time units; m = 1.0 per particle as in
`RolloutSpec(m=1.0)`, dt = 1/240).

Relaxation speed at the rule (isolated pair): Δr/h = φ/(2(1+φ)) per substep: 0.10 at q = 0.5,
0.036 at q = 0.72 (the "2.7×" clump spacing), so a Poisson pair reaches q ≈ 0.9 in about
one window (T = 20). Once all pairs are ≥ h the term is identically zero.

The only run-time switch is on/off (`rep ∈ {0, 1}`). `c`, `α`, `r_list/h = 1.5`, ε are
fixed by this document; changing any of them is a new pre-registration, not a sweep.

Anchor values: source sphere d = 0.135 → h = 0.101 at 20k; a 0.25-thick layer on the bunny
d ≈ 0.088–0.10 → h ≈ 0.066–0.075; at 40k multiply by 0.79.

---

## 8. Predicted effects and falsifiers

Baseline for the first arm: `r3b_p0_full` (20k, pace-0 recipe: chamfer 0.0872, out_nn
8.1%, far 419, best d_vol 23.8 @a181, guards 0, conv 187). All metrics from raw state.

| metric | Poisson / current | prediction with rep = 1 | pass threshold | kill |
|---|---|---|---|---|
| **A1 arrangement: median NN_i / d_i** (d_i from the 16-NN ball, target-free; new `metrics.arrangement`) | 0.55 (Poisson) | ≥ 0.8 | **≥ 0.75** at the delivered commit | < 0.65 (mechanism ineffective) |
| **A2 arrangement: fraction with NN_i < 0.5 d_i** | 41% (Poisson) | ≤ 2% | **≤ 5%** | > 15% |
| census cluster ratio (docs/floaters.md definition, h = 2 shell-sp) | 2.0 (r3b) | 1.3–1.6 — the in-layer part relaxes; the layer-thickness part (§0.2) stays | ≤ 1.3 only if the thickness confound is removed (report both) | — (continuity telemetry, not a gate) |
| unfilled shell points @1.5 sp / @1 sp | 33% / 65% | < 10% / 25–35% (density floor, §0.2) | @1.5 sp < 20% | — |
| out_nn (> 2 shell-sp) | 8.1% | 7–9% (no outward force by construction; fewer Poisson stragglers) | **≤ 8.1%** | > 8.1% at 300 anims |
| chamfer | 0.0872 | 0.085–0.089 | **≤ 0.0889 (+2%)** | > +2% |
| guards (all counters, incl. F_invert_steps) | 0 | 0 | **0** | any |
| jitter G3 | 6–8e-5 | unchanged at the frozen tail; the relaxation's own per-frame move must decay to 0 | jitter_rel < 3e-3 AND mean \|Δx_rep\| per frame / diag < 1e-4 at the tail | limit cycle: Δx_rep does not decay |
| rep_missed (frozen-list misses) | — | < 0.5% | < 1% after anim 10 | > 1% → per-substep rebuild arm |
| rep_max_move / h per substep | — | ≤ 0.5 by construction | assert | violation = implementation bug |
| photoreal at display σ = 1.0 shell-sp (all particles, `render_photoreal.py`) | speckled, filament ears | no pinholes on the body; ears continuous where filled | per-frame QA rubric | pinholes persist |
| render channel: d_sil at matched chamfer | — | lower (the 96-px silhouette's pixel is 0.57 spacing: Poisson gaps are a residual the control could never fix) | report | — |
| best d_vol | 23.8 | within 5% | ≤ 25.0 | > +10% (cell-scale pressure leak) |

Kill the mechanism (not re-tune it) if any kill column fires. If A1/A2 pass but out_nn
fails, the pre-registered next arm is mode 1 vs mode 2 (not a change of c or α).

---

## 9. Integration plan (file:function)

| file | change |
|---|---|
| `physmorph/mpm/state.py:MPMParams` | `rep_alpha: float = 0.0` (0 = off; 0.25 = the rule), `rep_mode: int = 2`, `rep_eps: float = 1e-3` |
| `physmorph/mpm/pairs.py` (new, host) | `local_spacing(x0, m, prm, device) -> d_i` (mass-only P2G + `k_volume`, i.e. `compute_volumes` on a scratch state at the current x0 — Vp0 is NOT overwritten); `build_pairs(x0, d, c=0.75, list_k=1.5) -> (pi, pj, hij)` via `cKDTree.query_pairs`; `missed_pairs(xT, pi, pj, hij)` telemetry |
| `physmorph/mpm/kernels.py` | `k_near_density`, `k_pair_shift` (dim = n_pairs, atomic adds); `k_update(..., dxr, use_rep)`; `k_g2p(..., dxr, inv_dt, use_impulse)` |
| `physmorph/mpm/traj.py:Trajectory.__init__/step` | accept `pairs=(pi, pj, hij)`; allocate `self.nd[t]`, `self.dxr[t]` (requires_grad = rg); launch the pair kernels after `k_stress` when `prm.rep_alpha > 0 and n_pairs > 0`; pass `dxr[t]` into `k_update` / `k_g2p` |
| `physmorph/mpm/step.py:mpm_step` | mirror for the forward-only `MPMState` path (tests, `compute_volumes` unaffected: it runs with dt = 0 and no pairs) |
| `physmorph/mpm/function.py:RolloutSpec/_WarpMPM.forward` | `pairs` field, passed to `Trajectory` |
| `physmorph/pipeline/config.py:PipelineConfig` | `rep: float = 0.0`, `rep_c: float = 0.75`, `rep_list_k: float = 1.5`, `rep_mode: int = 2` (documented as fixed by this file) |
| `physmorph/pipeline/optimizer.py:optimize_window` | after `x0_t`: `d = local_spacing(...)`, `pairs = build_pairs(...)`; put `pairs` in `spec` AND in both no-tape `Trajectory(...)` constructions (`eval_terms`, final rollout); log `h_med`, `n_pairs` once per window; return `rep` stats in `stats` |
| `physmorph/pipeline/runner.py:run_pipeline` | per-commit telemetry next to the `d_kde_v` block: `rep_missed`, `rep_active_pairs`, `rep_move_mean`, `arrangement` metrics into `hist`; nothing enters the merit or the freeze tracks (the term is a simulator property, not an objective) |
| `physmorph/metrics.py` | `arrangement(x, k=16) -> {nn_over_d_med, frac_nn_lt_half_d}` (target-free; cKDTree) and its inclusion in `summarize` |
| `scripts/pipeline_run.py` | `--rep {0,1}` wired like `--w_kde` into every arm block; arm name suffix `_rep` |
| `tests/test_relaxation.py` (new, CPU warp) | parity (rep = 0 bit-identical); conservation (Σ Δx = 0, momentum, angular-momentum bound); isolated-pair monotone approach without overshoot at α = 1/4 and at the α = 1 boundary; Poisson cube → A1 ≥ 0.75, A2 ≤ 5% after 40 stress-free substeps (λ = μ = 0, no dFc); Poisson ball free-surface radius growth ≤ 2%; FD adjoint through a 4-step rollout with rep = 1 |
| docs | `docs/method.md` §3.3 gains eq. (9') for the shift; `docs/experiments.md` result log |

---

## 10. Cost estimate (N = 40k, T = 20, 8 iterations/window, ≤ 10 line-search rollouts/iteration)

- Device, per substep: 2 pair kernels × ~120k threads × ~60 flops + 2 atomic adds each →
  < 0.05 ms forward, < 0.15 ms adjoint. The existing P2G/G2P (64-node stencils × 40k
  particles + 262k-node grid ops) are ~1 ms per substep. **Overhead ≤ 5% per rollout.**
- Host, per window: `k_volume` pass on a scratch state (µs) + `cKDTree.query_pairs` at 40k,
  r = 0.12: 50–150 ms once per window ≈ +0.1–0.2 s per commit against windows measured in
  seconds. At 300 commits: ≤ 1 min per run.
- Memory: pair arrays 120k × 12 B ≈ 1.5 MB; per-step `nd`, `dxr` 13 MB on the tape (§6).
- No change to the number of rollouts, iterations, or commits.

---

## 11. Adversarial self-review

1. **Objective mismatch.** The term is not in L; the optimiser sees it only through the
   forward map. Can dFc exploit it? It cannot generate sub-cell moves, and the term is
   zero on every arrangement with spacing ≥ h, so the only cell-scale effect is a residual
   virial pressure in regions that stay jammed — impossible by construction at φ = 0.22
   with h tied to the local density. Watch: best d_vol (+10% kill).
2. **Interaction with the near-band pull.** The nn-band assigns each particle its nearest
   shell point (many-to-one allowed); several fringe particles pulled to the same point
   arrive clumped and the repulsion spreads them at h — a beneficial interaction and a
   plausible source of the residual clumps (hypothesis, pre-registered check: number of
   shell points with ≥ 2 assigned arrivals per window). The berth (1.0 shell-sp = 0.034)
   is smaller than h (≈ 0.07–0.10): a particle can sit inside its berth and still be
   pushed sideways by a neighbour — it is pushed along the surface, not off it, because
   its neighbours are in the layer. out_nn is the gate.
3. **Thin features.** Where the ears are under-filled (9–24% fewer particles) the local d
   is larger, h follows it, and nothing pushes; where a filament is clumped, particles are
   spread along/across it up to the cutoff — filaments become sheets at the same particle
   count. The mechanism cannot add material to the ears; it must not be read as a fill.
4. **Free surface.** Exact zero force at any spacing ≥ h; a clumped surface layer relaxes
   by at most one cutoff outward, i.e. < 0.1 wu, inside the out_nn threshold (0.068)
   only marginally — hence out_nn is a hard gate. Contrast with the KDE/shifting family:
   no equilibrium, force ∝ density mismatch, outward at any free surface.
5. **Stability / timestep.** No CFL: mode 2 is a geometric step bounded by h/2 per
   substep independent of dt; mode 1's impulse is filtered by G2P within one substep.
   The frozen list makes a within-window map that is fixed and C¹. Potential 2-cycles in
   pathological many-body configurations are excluded by the damped-Jacobi factor 1/2
   (§7); `rep_max_move/h` asserts the bound at run time.
6. **Adjoint memory / conditioning.** +13 MB; local tangential amplification for nearly
   coincident pairs only in the first windows (§6). Warp handles atomic adjoints; the
   G1b-style FD gate is mandatory before any server run.
7. **Density estimate bias.** ρ interpolated by cubic B-splines at a free surface is low
   by up to ~2× (half-space), so d is inflated by ≤ 1.26; the `min(d_i, d_j, d_med)` rule
   confines the effect to surface–surface pairs and caps it at the body median. In a
   layer 2–3 particles thick almost every particle is "surface": the cap therefore
   matters and is part of the rule.
8. **The shell-target confound (§0).** The mechanism fixes in-layer arrangement (ii), not
   layer thickness (i). If the census cluster ratio stays near 1.6 while A1/A2 pass, that
   is the expected signature of (i) and not a failure; the doc pre-registers this reading
   so it cannot be argued after the fact.
9. **Frozen list.** Misses only delay; `rep_missed` and the per-substep-rebuild fallback
   are pre-registered. The list must be rebuilt after any state rollback in
   `run_pipeline` (the window is re-run from the restored x0: the list is a function of
   x0, so rebuild in `optimize_window`, never cached across windows).
10. **Render subset.** `scripts/render_photoreal.py` splats ALL particles of a frame, so
    an all-particle relaxation is directly visible there. The viewer/`render_sequence.py`
    use the frozen source-material surface mask (outer half of the sphere); after the
    sphere is flattened into a layer that mask is a random ~50% thinning of the layer,
    which re-creates Poisson-like gaps at 2× spacing. Pre-register the photoreal QA on
    all particles; the subset's arrangement is a display-side selection problem (§13),
    not this mechanism's.
11. **What if the layer is denser than the shell can hold?** Irrelevant: h follows the
    layer's own density. What if the density field is strongly anisotropic (a 2-particle
    layer)? The 3-D density then over-estimates the in-plane spacing by (thickness/d)
    factors, h shrinks with it, the term under-relaxes in-plane. Detectable as A1 < 0.75
    with A2 passing; the pre-registered follow-up is a 2-D (tangent-plane) spacing
    estimate, not a larger c.
12. **Does APIC's C see the shift?** C is computed from grid velocities only; the shift
    changes positions, so the next P2G's affine term `m C (x_g − x_p)` uses the shifted
    x_p — consistent, that is exactly what happens to any advected particle.

---

## 12. First A/B arm and kill criteria

**Arm x7_rep20** — `render_full_dt_iso_nn --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000
--outer_merit --nn_berth_k 1.0` (the r3b recipe, unchanged) **+ `--rep 1`** (mode 2,
c = 0.75, α = 1/4, r_list = 1.5 h), N = 20k, T = 20, dx = 0.5, dt = 1/240, 64³, smoothing
0.955, loss_res 64, 300 animations, seed 1, hyde06 per the placement rule. Control =
`r3b_p0_full` (same seed/budget; ideally a fresh re-run in the same job for replay noise).

Pass = all of: A1 ≥ 0.75, A2 ≤ 5%, out_nn ≤ 8.1%, chamfer ≤ 0.0889, guards 0, jitter G3,
`rep_max_move/h ≤ 0.5` (assert), `rep_missed` < 1%, photoreal QA at display σ = 1.0 shell-sp
without pinholes on the body. Kill = any kill-column entry in §8. Second arm only after a
pass: **x7_rep40** on the 40k x2_berth1 recipe (control chamfer 0.0706, out_nn 9.5%).
Pre-registered alternatives, in order, each triggered by a named failure: (B) mode 1 if
out_nn fails with A1/A2 passing; (C) source-side offline packing (Colagrossi-type,
c = 0.75 on the sphere before window 1) as the cheapest ablation of "arrangement is
inherited"; (D) per-substep pair rebuild if `rep_missed` > 1%.

Before any server run: `python -m pytest tests/test_relaxation.py tests/test_bridge_cpu.py`
green, and the §0.1 check on the delivered r3b state ((m/ρ_grid)^{1/3} vs Vp0^{1/3}) to fix
the layer's real d, so the anchor values in §7 are replaced by measured ones in the log.

---

## 13. The display-side alternative — what `render_photoreal.py --adaptive_k` can and cannot do

`--adaptive_k K` sets each splat's sigma to `sigma_k × (mean distance to its K nearest
neighbours) / median`, i.e. isolated particles get larger Gaussians and clumped ones
smaller. It **can** close Poisson gaps in a picture at a chosen display σ and blur filament
ears into continuous ribbons; it is a legitimate viewer-side anti-aliasing of a random
arrangement and costs nothing. It **cannot** change any raw-state number (chamfer, out_nn,
A1/A2, cluster ratio, unfilled fractions are all computed from x), cannot put particles
where there are none (an under-filled ear stays under-filled, only fatter), inflates σ
exactly where the geometry is most uncertain (the fringe), is not in the objective (the
gauss loss uses a fixed sigma0 = 0.04032 = the target-surface NN at 20k), and if the
viewer keeps rendering the frozen material-surface subset, it hides a thinning artefact
rather than an arrangement one. Rule for the record: adaptive_k is a rendering choice to
be reported as such; the in-simulator relaxation is a state change measured on raw state.
A run must never be scored with one and shown with the other without saying so.

---

## 14. Paper list with one-line takeaways

Verified today by web search / fetch unless marked (s) = search-summary only.

1. Clavet, Beaudoin, Poulin, "Particle-based Viscoelastic Fluid Simulation", SCA 2005.
   https://dl.acm.org/doi/10.1145/1073368.1073400 — Double density relaxation:
   `D_ij = dt²[P(1−q) + P_near(1−q)²] r̂`, half to each particle, velocities re-derived
   from positions; the near-pressure (1−q)² term is the anti-clustering device (equations
   confirmed against the reference implementation kotsoft/particle_based_viscoelastic_fluid,
   sim_1.js: `density += (1−q)²; nearDensity += (1−q)³; D = dt²(P(1−q) + P_near(1−q)²)/2;
   v = (x − x_prev)/dt`).
2. Monaghan, "SPH without a tensile instability", J. Comput. Phys. 159(2):290–311, 2000.
   https://ui.adsabs.harvard.edu/abs/2000JCoPh.159..290M/abstract — Clumping under
   negative stress is removed by a small repulsive artificial pressure; a dispersion
   analysis, not tuning, fixes its parameters.
3. Ando, Tsuruno, "A Particle-based Method for Preserving Fluid Sheets", SCA 2011.
   http://diglib.eg.org/handle/10.2312/SCA.SCA11.007-016 (project:
   https://ryichando.graphics/sheetflip/) — FLIP + weak pairwise spring forces keep an
   even particle distribution; the direct PIC/FLIP precedent for a particle-level
   position corrector.
4. Ando, Thürey, Tsuruno, "Preserving Fluid Sheets with Adaptively Sampled Anisotropic
   Particles", IEEE TVCG 18(8):1202–1214, 2012. https://pubmed.ncbi.nlm.nih.gov/22411890/ —
   Anisotropic position correction for uniform spacing + adaptive sampling; anisotropy is
   for breaking sheets, not needed for a closed layer.
5. Zhu, Bridson, "Animating Sand as a Fluid", ACM SIGGRAPH 2005.
   https://dl.acm.org/doi/10.1145/1073204.1073298 (s) — The PIC/FLIP blend trades FLIP
   noise for PIC dissipation in velocity space only; it has no position-regularising term.
6. Jiang, Schroeder, Selle, Teran, Stomakhin, "The Affine Particle-In-Cell Method", ACM
   TOG 34(4), 2015. https://dl.acm.org/doi/10.1145/2766996 — APIC keeps PIC's velocity
   reset with an affine C; sub-cell relative velocities are projected out each transfer,
   which is why a pairwise impulse here is equivalent to a one-substep displacement.
7. Fu, Guo, Gast, Jiang, Teran, "A Polynomial Particle-In-Cell Method", ACM TOG 36(6),
   2017. https://dl.acm.org/doi/10.1145/3130800.3130878 — Higher-order transfers recover
   more velocity/vorticity modes; they do not act on the position null space.
8. Gao, Tampubolon, Jiang, Sifakis, "An Adaptive Generalized Interpolation Material Point
   Method for Simulating Elastoplastic Materials", ACM TOG 36(6), 2017.
   https://dl.acm.org/doi/10.1145/3130800.3130879 — Adaptivity via refined grids and
   particle handling: discrete, not differentiable, out of scope.
9. Yue, Smith, Batty, Zheng, Grinspun, "Continuum Foam: A Material Point Method for
   Shear-Dependent Flows", ACM TOG 34(5), 2015. https://dl.acm.org/doi/10.1145/2751541 —
   Large shear produces poor MPM particle distributions and non-physical holes; they
   introduce explicit particle resampling: the MPM precedent that arrangement is a real
   defect repaired at the particle level (discretely).
10. Wang, Ding, Gast, Zhu, Gagniere, Jiang, Teran, "Simulation and Visualization of
    Ductile Fracture with the Material Point Method", PACMCGIT 2(2), 2019.
    https://dl.acm.org/doi/10.1145/3340259 — Abstract describes damage/softening
    plasticity and crack visualisation (element splitting, Delaunay), not particle
    resampling; corrects the premise that it is a resampling precedent.
11. Steffen, Kirby, Berzins, "Analysis and reduction of quadrature errors in the material
    point method (MPM)", IJNME 76(6):922–948, 2008.
    https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.2360 — Internal-force quadrature
    error depends on the particle arrangement relative to the grid; cubic B-splines
    reduce it much but do not remove arrangement dependence.
12. Gritton, Berzins, "Improving accuracy in the MPM method using a null space filter",
    Computational Particle Mechanics 4:131–142, 2017 (s).
    https://www.researchgate.net/publication/308394629 — The particle→node map has a
    non-trivial null space; errors in it are not damped and grow over steps; a filter on
    the particle side is needed.
13. Tran, Berzins, Sołowski, "Temporal and null-space filter for the material point
    method", IJNME 120(3), 2019. https://onlinelibrary.wiley.com/doi/full/10.1002/nme.6138 —
    Extends the null-space filter to GIMP/DDMP at lower cost; confirms the mechanism for
    B-spline-class transfers.
14. Lind, Xu, Stansby, Rogers, "Incompressible smoothed particle hydrodynamics for
    free-surface flows: A generalised diffusion-based algorithm for stability and
    validations for impulsive flows and propagating waves", J. Comput. Phys. 231(4), 2012
    (s). https://www.researchgate.net/publication/220205958 — Fickian particle shifting
    from high to low concentration keeps spacing uniform but needs a free-surface
    treatment because it pushes surface particles outward — the failure the cutoff form
    avoids.
15. Macklin, Müller, "Position Based Fluids", ACM TOG 32(4), 2013.
    https://dl.acm.org/doi/10.1145/2461912.2461984 — Artificial pressure (s_corr) against
    clustering in a position-based solve; constraint averaging by neighbour count is the
    stability device behind the near-density normalisation used here.
16. Colagrossi, Bouscasse, Antuono, Marrone, "Particle packing algorithm for SPH
    schemes", Computer Physics Communications 183(8):1641–1653, 2012 (s).
    https://www.sciencedirect.com/science/article/abs/pii/S0010465512001051 — Relaxing the
    initial particle set to a uniform packing removes the numerical noise of particle
    resettlement; the offline analogue of the in-loop term (arm C).
17. Levi, "Cell-Constrained Particles for Incompressible Fluids", arXiv:2402.17088, 2024.
    https://arxiv.org/abs/2402.17088 — Bounds the number of particles per grid cell by a
    linear program: cell-scale (not sub-cell) distribution control, discrete; not
    applicable on a tape.
18. NVIDIA Warp, "Differentiability" (docs, Warp ≥ 1.x).
    https://nvidia.github.io/warp/modules/differentiability.html (s) — `wp.atomic_add`
    and in-place accumulation are supported as adjoint accumulation on the tape; the tape
    records kernel launches and replays their adjoints in reverse.

---

## 15. Summary

**Recommended mechanism.** A per-substep, pairwise, C¹ short-range repulsion of the
Clavet/Monaghan near-pressure type, hosted inside the MLS-MPM step as a position shift
(Ando-style FLIP position correction; impulse form as the A/B alternative), normalised by
the near density (PBF constraint averaging), with the cutoff `h_ij = 0.75·min(d_i, d_j,
d_med)` taken from the grid's own mass density so that it equidistributes particles at the
cell-scale density the grid and D_vol already own and is exactly zero on any arrangement
with spacing ≥ h — bulk or free surface. It is momentum-conserving by antisymmetry, touches
neither F, Fp, mass nor Vp0, lives on the `wp.Tape` through two atomic-add pair kernels over a
frozen per-window pair list, costs ≤ 5% per rollout and ~0.1 s per window, and converts
the Poisson arrangement inherited from `sample_volume` into a blue-noise one — the defect
the photoreal renders show, which the transfer null space makes invisible to the grid and
unreachable by dFc. Two measured facts qualify the premises: the bunny target is a surface
shell of a non-watertight mesh (its NN spacing 0.034 is not a volumetric spacing) and the
converged body is a thin layer (chamfer at the shell-vs-shell floor), so the census "cluster
ratio" conflates layer thickness with clumping and "J ≈ 1" is what F-smoothing 0.955 reports
for any deformation; the falsifiers are therefore target-free arrangement statistics plus
do-no-harm gates.

**Calibration rule.** Length: `h_ij = c·min(d_i, d_j, d_med)`, `d = (m/ρ_grid)^{1/3}`
frozen per window, `c = 0.75` (blue-noise relative radius; hard-core fraction 0.22, so an
equilibrium with the term switched off always exists). Rate: `α = 1/4`, i.e. a maximum move
of h/2 per substep — the damped-Jacobi factor 1/2 under the row-sum bound `|Δx_i| < 2αh`,
4× inside the isolated-pair no-overshoot bound α ≤ 1. Mode 1 equivalent stiffness
`k_n = α m h/dt²`. On/off is the only switch.

**First arm.** x7_rep20 = r3b pace-0 recipe + `--rep 1` at 20k/300 anims, control
r3b_p0_full. Pass: NN/d_local median ≥ 0.75 AND fraction(NN < 0.5 d_local) ≤ 5% AND
out_nn ≤ 8.1% AND chamfer ≤ 0.0889 AND guards 0 AND jitter G3 AND rep_missed < 1% AND
photoreal (all particles, σ = 1.0 shell-sp) without pinholes. Kill: any guard; chamfer
> +2%; out_nn > 8.1%; NN/d median < 0.65; relaxation move not decaying at the frozen tail;
best d_vol > +10%. Next only after a pass: 40k on the x2_berth1 recipe; on a named
failure: mode 1 (out_nn), source packing (arrangement-inheritance ablation), per-substep
pair rebuild (rep_missed).
