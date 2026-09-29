# P308: motion reduction with data restoration in the displacement mode

P307 restored endpoint MSE but did not satisfy the movement and local-supply
gates. This is a bounded diagnostic, not a production policy. The implementation
must pass independent review before its first GPU execution.

Use a fresh real W20 callback at the existing N300k,T20,dt1/240,
dx.3062907543956724wu,loss36^3,budget8 raw/no-PIC/no-shift recipe. Preserve
P307's failed archive/C-repeat records; do not admit that restart. Generate the
fresh callback's own terminal05. Freeze terminal coefficients, stress/u,
initial state, reference, lambda and both window-start cohorts throughout.

Let R be the mean squared position-derived velocity across all20 steps and
the fixed start-arrived-free cohort, including x0->x1. Optimize displacement
coefficients to reduce R. Do not substitute endpoint distance or stored v_T.
Measure stored/geometric terminal speeds separately, since fixed terminal
coefficients do not fix the nonlinear terminal response.

At the current displacement coefficients, compute gradients of R, prepared
volume and prepared render. With baseline loss ceilings from the three
original baseline repeats, use affine restoration constraints:

```
g_volume dot delta <= baseline_volume - current_volume
g_render dot delta <= baseline_render - current_render
```

The existing homogeneous halfspace projection is insufficient: terminal05
already violates volume. A descent direction that merely does not increase
that loss does not restore its deficit. Keep the original fixed reference and
lambda; no new weighted surrogate or kinetic coefficient is introduced.

Bounded subproblem: minimize the linearized R over these two
halfspaces and a Euclidean trust ball whose RMS radius is the last ordinary
accepted displacement update. Enumerate the four possible active sets on GPU;
use a minimum-norm affine point and the remaining ball radius in its nullspace.
Check rank consistency and every inequality. For each of at most10 halvings,
solve again at the smaller radius: scaling an already feasible affine step can
destroy restoration. If no feasible linear step exists, record that result.
Use a direct thin SVD to determine row rank, QR/triangular solve for full-rank
active rows, and an SVD pseudoinverse for dependent rows. Do not form normal
equations: a nearly opposing-plane CPU counterexample exposed false failure
when the condition number was squared. Nonfinite/malformed inputs are errors.

Apply the existing residual joint coefficient bound with terminal05 fixed.
Recheck the actual final projected displacement step against the affine
constraints; a pre-projection certificate is insufficient. Evaluate the full
MPM trajectory, then require valid raw state, exact pins, actual volume/render
within the baseline ceilings, and R reduction resolved beyond the existing
repeat/float32 floor. Record failed linear/projection/nonlinear attempts. Use
at most4 accepted updates, with no fallback that weakens a constraint.
The projected affine dot check uses32 input float32 eps times
`||gradient|| ||actual_delta|| + |RHS|`; the trust check likewise accounts for
coefficient rounding. These are arithmetic checks, not nonlinear quality
tolerances. Actual prepared-loss ceilings and every final P306 gate are unchanged.

All final candidates still face unchanged P306 gates: both cohorts' net/step/
path motion; resolved stored/geometric terminal braking; independent raw
silhouette/coverage/Chamfer/tip/local-density measurements. Density counters
remain validation metrics, not objectives. This subproblem does not guarantee
their discrete gates. A failure is confined to this search and subspace.

Retain full accepted X/V/F/C/coefficient evidence, actual render components,
source/input hashes, private/accepted X/V/F/data closure, and exact production
state/Adam isolation after the callback. No candidate commit. Only a candidate
passing all gates can justify original-total-merit reconstruction and actual
coupled continuation. That continuation must carry the full state and normal
assimilation; it remains necessary for any persistent-rest claim.

## Implementation verification before running_repair1

Nineteen distinct CPU tests pass:14 affine/operator cases,2 actual CPU MPM
observer/derivative cases with layer control on/off, and3 existing braking
direction/P306 gate regressions. Affine tests include12 seeded independent
SciPy SLSQP comparisons, a feasible nearly opposing-plane counterexample at
angles1e-4/1e-6/1e-8, impossible constraints, zero/duplicate gradients,
nonfinite/malformed input, and explicit inclusion of the initial physical step.
The actual MPM cases compare the full running-motion displacement derivative
with central finite differences. This is not a full repository test-suite claim.

Independent review reproduced the opposing-plane and infinity-input failures,
verified their fixes, and cleared the complete noncommitting callback path.
`code_running_repair1` then passed its CUDA opposing-plane preflight on hyde06
GPU1 and began the fresh300k run. No physical result is available at this entry;
the old C-repeat/restart gate and production/4K-quality status remain unchanged.
