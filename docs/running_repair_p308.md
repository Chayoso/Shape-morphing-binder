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

## Completed running_repair1: tangent predictions miss the actual loss ceilings

The fresh run completed20 windows/160 accepted original inner updates with all
trajectory guards0 and exact post-callback production isolation. N300k,T20,
dt1/240,dx.3062907543956724wu,loss36^3,budget8 are unchanged. The fixed cohorts
are61,823 start-free and61,734 start-arrived-free IDs; this is a different fresh
realization from P307 and is compared only against its own three baselines.

No repair update was accepted. At halves0..7, both data planes are active, the
final projected coefficient step passes the affine/trust checks, and actual R
decreases. All eight forward candidates nevertheless exceed both nonlinear
prepared-loss ceilings. At halves8..10 the active-set solver found no feasible linear step inside
the smaller radius. The baseline ceilings are volume.0019056980963796377 and
render.0018326621502637863 and were never widened.

Independent recomputation of all P306 gates identifies h5 as failing only these
two prepared-data gates. Against its own baseline, arrived-free step RMS falls
3.0822%, net RMS2.8924%, path mean4.2938%, stored/geometric terminal speeds
30.0870%/29.8140%; all raw geometry/supply checks and both cohorts' motion gates
pass. Its fixed-source density/upper coverage/tip count equal baseline, overall
coverage gains1/300000 and independent silIoU gains2.44208e-5. These are
same-window measurements, not a full-morph hole/rest/appearance certificate.

H5 volume exceeds its ceiling3.3760443e-9 and render1.0244548e-8:9.67x and44x
their observed three-baseline ranges. Its projected linear residuals are about
-2e-14, so these actual excesses are not explained by that affine-dot error.
The observed model remainder is positive; it is not a certified curvature bound
or an isolated physical cause. All candidate states remain uncommitted.

Rendering at h5: silhouette decreases3.37837e-7 but PBR increases3.48198e-7;
the combined increase is1.04774e-8 relative to baseline0 (the frozen ceiling is
the maximum of all three baselines). Weighted increase is2.68654e-10 at
lambda.0256413714. Both render and volume constraints bind the linear proposal;
actual rendering still fails. Across the original160 accepted updates, median
nominal render-direction share is.5027565 and median lambda.061979; neither
is a causal displacement share. Exported4K appearance was not measured here.

Independent result review verified JSON/protocol/source bindings, cohort counts,
guards/isolation, all eight trials' gates and the scalar replay-range comparison.
Evidence: `docs/evidence/p308`; complete state/control/linearization sidecars
remain under server `work/p303/running_repair1`. Next is bounded observed-model
remainder correction, keeping the same ceilings and all final gates.
