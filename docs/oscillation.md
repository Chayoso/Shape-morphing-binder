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
| `pipeline/runner.py: outer gate v2` | reject-type split (brake vs latched low-gain), self-healing latch, cold restart after any reject | driver #6: the flat-valley limit cycle must be clipped by the brake without the brake's rejections killing the run through patience; mid-run spurious latches must not be permanent |
