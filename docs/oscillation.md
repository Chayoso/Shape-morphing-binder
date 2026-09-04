# Near-optimum oscillation dossier

Status: **20k dynamic stability gate passed (2026-09-02); high-resolution render
representation is still under QA**.

## Symptom and acceptance criteria

The failure is a repeated reversal of surface motion near an optimum, not merely continued
descent. A result is accepted only when all of the following are true at its stated
discretisation:

- no NaN, clamp, reflection repair, or intermediate `det(F)<=0` guard fires;
- tail `jitter_rel < 3e-3` and one-window terminal-velocity drift `< 3e-3`;
- low-gain commit directions do not alternate (`reversal_cos < -0.2` is rejected);
- the fixed-scale outer merit reaches a plateau and the state is then held/null, rather
  than declaring an unfinished run converged.

## Root causes fixed in the current branch

1. Rest particle volumes were being reconstructed from the current state. `Vp0` is now
   computed once from the sampled source and reused by every rollout.
2. For `Fe=(F+dFc)Fp^-1`, the total first Piola stress must include the chain rule
   `Ptotal=Pe Fp^-T`; P2G then uses `Ptotal(F+dFc)^T = Pe Fe^T`. The previous expression
   was wrong after plastic assimilation and could inject non-symmetric stress.
3. Near stationarity, fresh Adam moments made each window behave like a sign step.
   Moments now carry across committed windows; stale/null windows anneal the step.
4. Inner steps use an Armijo-style backtracking test. Across windows, a fixed-scale merit
   and reversal gate roll back low-gain reversals, including plastic and optimizer state.
   The gate latches after both sufficient normalized target progress and a small-motion
   commit; it cannot switch itself off again when a rejected large-motion candidate
   appears.
5. Temporal first-difference control cost, terminal kinetic cost, and a covariance/volume
   band keep the accepted control trajectory smooth without damping or editing state after
   the MPM rollout.

These are optimizer and constitutive fixes. They do not apply a positional servo and do
not post-filter `x`, `F`, `v`, or `C`.

## Evidence so far

All simulation numbers below use `dt=1/240`, `dx=0.5`, MPM deformation smoothing `0.955`,
and zero velocity clamp.

- 5k, `T=10`, 80-commit surface-Gaussian ablations: `jitter_rel=1.7e-5..2.3e-5`,
  drift `2.0e-4..2.6e-4`, no guard event.
- 20k, `T=10`, 60-commit high-resolution sigma ablations: `jitter_rel=3.2e-5..3.4e-5`,
  drift `2.7e-4..2.8e-4`. In the last 20 commits, minimum reversal cosine was `0.84`
  (`sigma scale=0.85`) and `0.91` (`1.0`); neither run had a reversing commit.

The unrestricted 20k, `T=20` ablation was stopped after roughly 120 outer commits per arm:
render-on accepted 58 negative-merit candidates and physics-only accepted 54. Their fixed
merits repeatedly left the best basin (`0.369..1.162` and `0.112..1.257` respectively),
showing that the long wave was not caused by rendering alone.

With the progress-conditioned, persistent latch, the matched render-on run froze after 32
outer attempts. The latch activated at animation 10; thereafter no sub-threshold or
negative-gain candidate was promoted, the minimum accepted reversal cosine was `+0.335`,
and the last eight candidates were rolled back to the same state. At `N=20,000`, `T=20`,
`dt=1/240`, `dx=0.5`, smoothing `0.955`, the raw trajectory has
`jitter_rel=6.46e-5`, no guard event, Chamfer `0.11082`, and silhouette IoU `0.94572`.
The matched physics-only arm also froze (29 attempts) and had zero bad promotion after its
latch. This closes the optimizer-side oscillation gate; Gaussian footprint QA is tracked
separately in `floaters.md`.

## VBD / Projective Dynamics decision

The retired `surface_local_pass` changes `x/F` outside the rollout while leaving velocity
and APIC state inconsistent; it is not an admissible production fix. Vertex Block Descent
or Projective Dynamics would be valid only as either:

- a complete implicit integrator whose inertia, velocity update, constraints, and adjoint
  all replace the present explicit MPM step; or
- an SPD preconditioner on the **control gradient**, followed by the unchanged MPM rollout
  and line search.

The tested kNN/H1 control preconditioner reduced movement but worsened Chamfer, silhouette,
holes, and deformation quality, so production leaves it off. Surface render residuals are
instead masked at the terminal covector, pulled through the full MPM adjoint, and applied to
`dFc`; interior continuum response therefore comes from physics rather than a Gauss-Seidel
position correction.

## Addendum — late session 2026-09-02 (supersedes point 4's "cannot switch itself off")

**Driver #5 (per-window Adam restart zigzag)** — forensic on h17_bunny's last 40
pre-freeze commits: 37.5% of ALL particles reversed direction every commit (ears 54%,
but ear-and-untouched-by-cleanup identical → cleanup terms acquitted). Fixes:
`anneal_stale` (partial: rev-cos −0.523→−0.345, jitter halved) and `mom_carry`
(moments persist across windows like the warm-started control does).

**Pacing** — `pace_budget` derives the per-window loss cap from the budget
(1 − ρ^(1/A)); pace-bound windows are exempt from plateau accounting (the validated
flagship had frozen at anim 70/300 while descending perfectly on schedule).

**Driver #6 (flat-valley limit cycle under pacing)** — b4's λ trace: pace caps the
LOSS cut, not the MOTION; in the flat late valley an overshoot window moves 0.04 wu
(kin spike 0.86) while cutting 1%, the next commits pay it back (d_vol 62→215→62,
λ in antiphase 1160↔2280) and the run ends at a random cycle phase (final 0.146
despite a best-ever d_vol 62). Remedy = the gate's brake, which needed **gate v2**:
BRAKE rejects (fixed-merit regression beyond one pace budget) discard the candidate,
decay the step and cold-restart WITHOUT counting stale; only LATCHED low-gain rejects
are plateau evidence; the latch now RELEASES on a real track improvement (point 4
above is superseded — a permanent latch armed by a mid-run 3-commit stall killed b6 at
a94/450). Latch evidence itself went merit-threshold → small-move → sustained
stale-streak after three distinct early-freeze forensics (s1, s3, s4).

Ops lesson: three simultaneous launches with unbounded BLAS/OpenMP threads (28 cores
each on the 128-core node) stalled all three at zero commits for 85 min and stole CPU
from another user's jobs — every launch now sets OMP/OPENBLAS/MKL_NUM_THREADS=8 and
staggers by 45 s.

### Addendum 2 — gate v2 → v3, assimilation ladder, probes (2026-09-02, late)

- **b7 (gate v2, 450 paced)**: 333/450 BRAKE rejects, run pinned at a101–a244 (identical
  state), final chamfer 0.138. Forensic: the fixed merit's phys component was
  `d_vol + w_kin·kin`, so any motion from a settled state raised kin enough for a >5%
  merit regression — pacing demands motion, a kinetic merit punishes it. **Gate v3**:
  merit = shape terms only (d_vol, d_sil, d_dt); the brake still catches real shape
  regressions (b4's 62→215).
- **Best-commit truncation**: the deliverable trajectory now ends at its best
  shape-merit commit when the run ends worse (continuous up to the cut, no snap; the
  flat-valley tail stays in `history`). b4 would deliver a381's d_vol 69 instead of
  a450's 158.
- **E2 assimilation ladder (η 0.5 / 0.8 / 1.0, 300 paced)**: spring-back fraction 0.20
  at every η; η=1.0 final d_vol 70.9 (vs 97.6) but chamfer 0.1125 / out_nn 25.5% (vs
  0.1078 / 22.3%) — not a lever, not adopted.
- **Probes (docs/probes/)**: the adjoint attenuates the real render covector by only
  ~14%; Sobolev preconditioning is a no-op (balancer re-halves λ); F-smoothing s≤0.8
  falsified (J collapses); the render term descends at the same relative rate as
  physics per accepted step. The oscillation is therefore an optimizer/geometry
  property of the flat valley, not a render-transfer artifact — consistent with
  drivers #5/#6 and their optimizer-side remedies.

### Addendum 3 — r1 late-run cascade signature (40k, unpaced, anneal 0.7 + mom_carry 0.9 + E4)

Clean descent to d_vol 82.8 at a90 (λ≈850, kin 0.05, move 0.004), then:
a92 a ZERO-control window (dfc_absmax 0.000, 2 accepted iterations — warm start rejected,
cold fallback, early line-search exit) → a94 kin 0.17 → a96 d_vol 124, λ 1312 → a100
kin 0.62, move 0.058 → a102 d_vol 373, λ 2768 → a112 d_vol 489, λ 3254, kin 0.78, move
0.067; frozen at a112 (20 commits of all-track regression before patience fired).
Reading: an uncontrolled free rollout releases stored elastic energy, the body's
velocity grows 20×, the render term worsens so λ climbs toward the cap, and the tiny
controls (dfc ≤ 0.012) cannot arrest the motion — a physical runaway ignited by two
near-zero-control windows. The deliverable was rescued only by best-commit
truncation (a91: chamfer 0.0725, silIoU 0.966, out_nn 12.1%). Open questions under
test: r2 (mom_carry 0) isolates the optimizer-moment carry; the outer brake (>5%
shape-merit regression) would have rejected a96 onward; why patience took 20
commits (which track kept "improving"?) is the next forensic.
RESOLVED (same day): r1's provenance shows `pace 0.12` — the flagship arm always applies
`--pace` (default 0.12), so r1 was not unpaced, and the pace-bound plateau exemption
marked every runaway window "improved" (a 12% cut of the free-rollout loss is trivial
while the body runs away). The exemption now requires that no λ-free track regressed
versus the previous accepted commit. Truly unpaced runs need `--pace 0`.

### Addendum 4 — mom_carry RETIRED (r1 vs r2, 40k, 2026-09-03)

Same configuration except mom_carry (0.9 vs 0): r1 reached d_vol 82 at a91 then
cascaded to 489 by a112 (zero-control windows → elastic release → kin ×20 → λ ×4);
r2 reached 86.6 at a99, converged cleanly at a104 with only two >5% regressions,
delivered chamfer 0.0730 / silIoU 0.964 / out_nn 12.4% / jitter 8e-5. Carrying Adam
moments across windows pushes the iterate past the optimum near convergence; its
zigzag benefit was never validated in isolation (g5_mom was stopped). Retired from the
recipe: anneal_stale stays, mom_carry = 0. The b-series late regressions all ran with
mom_carry 0.9 and should be re-read in that light.

### Addendum 5 — pace 0.12 was the hidden brake (r3, 2026-09-03)

Every flagship run so far carried the arm default `pace 0.12`; with `--pace 0` (r3, 20k,
mom_carry 0, E4, gate v3) windows run their full 8 iterations, the run went 221 commits with
only 2 brake rejects and reached d_vol 23.4 — 2.7× below the previous best (62) — with
out_nn 8.6% and far floaters 417 at the (mis-)truncated a150. The 12% per-window cap
stopped windows early, left the plateau tracks starved and froze runs at ~100–160 commits
(r2/r4). New recipe: `--pace 0`, anneal 0.7, mom_carry 0, nn_far_k 1000, outer_merit
(brake as a safety net), best-commit delivery with a resolution-invariant merit and no frame
drops. The "no snap" requirement is met without pacing (first3 = 11–12%).

### Addendum 6 — the tail zigzag survives the sampler fix (v1 solid bunny, 2026-09-04)

On the first real-volume run (v1: out_nn 1.1%, guards 0) the last 40 commits still show
rev-cos −0.44 with 61% of particles reversing per commit at mean move 0.0068 — the same
signature as the shell era (h17: −0.52 / 64%). The oscillation is an optimizer property
(drivers #5/#6), independent of the target. mom_carry is retired; anneal is partial; the
gate's low-gain reversal reject only ran once latched. v4 tests it unlatched
(`outer_reversal_always`): a low-gain candidate that reverses the previous commit's
displacement is rejected and the next window cold-restarts.
v4 (unlatched reversal REJECT) froze at a56, v6 (α ×0.5 on reversal) froze at a71 with α
at its floor: commit-to-commit reversal is frequent during honest descent (valley
zigzag), so any reversal-triggered brake chokes progress. Both retired. What stands:
stale-based anneal, gate v3 (brake on real regressions), best-commit delivery. Next:
measure whether the residual zigzag is VISIBLE (per-particle per-commit displacement in
target spacings over the delivered tail) before spending more mechanism on it.
