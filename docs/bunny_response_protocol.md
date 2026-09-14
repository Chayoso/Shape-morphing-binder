# Sphere to bunny: fixed quality target

The user selected a **20% relative reduction in unseen-view silhouette error**
against the matched 3-D criterion on 2026-09-14. This is a pass requirement, not
a result that may be assumed. The current development uses raw-state binary
point silhouettes, so the reported metric is named point-silhouette error.

## Evaluation fixed before development results

For each view, compute binary 3x3 point-footprint IoU using all raw particle
positions, one target-derived orthographic extent shared by both arms, and 128x128
resolution. E is the mean of (1-IoU) across views. The paired relative reduction
is (E_3D-E_image)/E_3D; passing requires >=0.20. If E_3D<=1e-6, relative improvement
is undefined and cannot establish a pass. Absolute IoU change is also reported.
Sensitivity checks repeat the fixed metric at 96 and 160 pixels; neither may
reverse the improvement direction. These are sampling-dependent point metrics,
not an assertion of continuous mesh-surface accuracy.

The view sets, in radians, are defined in `scripts/bunny_response_benchmark.py`:

| Set | Azimuths | Elevations | Use |
|---|---|---|---|
| Train | j*pi/2, j=0..3 | -0.25, 0.25 | Native 1024px image objective |
| Development validation | pi/4+j*pi/2, j=0..3 | -0.1, 0.4 | Diagnose and select configuration |
| Sealed test | pi/8+j*pi/4, j=0..7 | -0.5, 0, 0.5 | Evaluate once after configuration lock |

Test evaluation is initially disabled in the development script. After the
configuration and checkpoint rule are frozen, a lock records their hashes before
enabling that evaluation. Inspecting final-test numbers or images and then tuning
would convert that set into development data; such results cannot remain sealed
evidence. Development uses seed 17. Confirmation uses paired seeds 29, 43 and 71
with the locked configuration. Report each seed and the reduction in mean E
across seeds; no general pass if an individual seed reverses the improvement.

Both arms use the same source/target/vol0, material, grid, horizon, strain basis,
control constraints, image observations and candidate validation code. Only the
objective switches between image and 3-D/core. All arrays and source hashes are
checked. Checkpoint selection is the final accepted iterate, independent of
validation and test metrics. Iteration and actual rollout costs are both reported.
A second physics run receives twice the iteration limit to check whether an
apparent advantage comes from a prematurely stopped reference. Its checkpoint
uses only the core objective and physical health.

## Additional gates

- At the final frame, raw symmetric Chamfer (mean of the two mean distances) may
  worsen by at most 2% relative to the stronger physics reference.
- Raw hole fraction at the fixed target extent may not exceed the physics result
  or target value by more than 0.005 absolute. Binary footprints stay fixed.
- No visible detached Gaussian component or through-hole is allowed in delivered
  frames. Every exported physical frame must be inspected. Raw isolated-particle
  fraction over the full trajectory must not increase over physics by more than
  0.001 absolute. These diagnostics do not prove topology preservation.
- Dynamics must remain finite, oriented and inside complete MPM grid support.
  No particle may leave the fixed projection canvas in any evaluation view;
  clipping cannot remove a false-positive footprint without failing a gate.
  Physical acceleration and terminal velocity are reported separately from
  frame-to-frame displacement; active morph motion must not be called jitter
  simply because it moves more than the reference. A settled animation is not
  approved until a zero-control continuation is checked for oscillation/ejection.
- Initial and final silhouettes must be visibly recognizable as sphere and bunny;
  a relative improvement between two failed morphs is not a finished-quality pass.

## Development design and scope

Source and bunny target are sampled from the repository meshes with equal
volume and centered mass. The legacy axis-fill sampler was later found to create
voids and artificial exterior bridges; the closed-surface family uses validated
derived meshes and direct interior sampling instead. Particle density is fixed at
1000 world-mass/world-volume and particle mass is adjusted with N, so increasing
sampling does not silently change wave speeds. The provisional development
discretization is N=6000, dx=.25, dt=1/120, 32^3 grid from (-4,-4,-4), T=64,
lambda=5000, mu=3000, drag=.9, F-model smoothing=.955. The mass observation uses
32^3 cells of width .25. Every recovered number must retain its own run metadata.

Overlapping compact patch fields on the source shell multiply six symmetric
strain components. Their span includes constant strain. The actual whitened
modes need not have compact support, but the span and its source-fixed spatial
variation are shared by both arms. Interior dFc remains zero in this diagnostic;
interior motion is still MPM. This limitation is not a claim that general MPM or
the eventual all-particle physics-control formulation has the same controllability.

The response optimizer retains the actual-rollout acceptance and finite-response
validation from `response_control.md`. No gradient gain, opacity fitting, free
Gaussian position or post-physics surface displacement is introduced. Every
development outcome, including failed thresholds, is retained. Parameters are
not frozen until development establishes a viable physical bunny morph.

## Development record (not sealed evidence)

The first paired runs reuse the exact legacy seed-17 fixture: N=6000,
source volume=4.28106175, particle mass=.713510292, 2400 selected shell particles,
dx=.25, dt=1/120, 32^3 grid, T=64, lambda=5000, mu=3000, 32^3 mass grid at dx=.25,
eight 1024px training cameras, sigma0=.064723756. Their scale came from sampled
bounds; this is shared within each pair but unsuitable as a fixed-geometry
N/seed study. Future fixture preparation instead uses mesh extents and filled
volume independent of random samples. Older fixtures remain identifiable by hash.

| Development variant | Image criterion E128 | 3-D criterion E128 | Outcome |
|---|---:|---:|---|
| v1: transported material covariance, 12 patches, constant in time | .26533724 | .17164452 | Failed; image criterion is worse |
| v2: fixed-bandwidth Gaussian observation, same fixture/patches/time | .22194186 | .16717059 | Failed; image criterion is worse |
| v3: two time phases and 80% predicted physics progress, fixed kernel | .19488373 | .16360467 | Failed; image criterion is worse |
| v4: screened image norm and linearized geometry constraints | .16444246 | .14857324 | Failed; image criterion is 10.68% worse |
| v5: same model with unit trust-region coordinates in SLSQP | .15488806 | .14486657 | Failed; image criterion is 6.92% worse |

The v2 solver also adds particle-peak constraints to the quadratic subproblem,
accepts feasible inexact subproblem proposals subject to actual-rollout checks,
and reduces Gram-construction memory. Thus the v1/v2 difference is not a strict
single-factor causal ablation of covariance transport. Neither result is a 20%
claim; neither used final-test views. Both runs stop near the geometric-condition
limit or a failed subproblem and are not converged bunny solutions.

The fixed-bandwidth observation is an explicit representation variant: Gaussian
centers remain exactly the MPM material-particle positions, while isotropic
covariance denotes a reconstruction kernel rather than a deforming material
ellipsoid. F_geom is still evolved and health-checked. Loss and displayed images
use the same choice. This variant does not claim to preserve the earlier
material-covariance rendering contract or to isolate its native covariance
gradient; the image still controls dFc through stress and particle motion.

v3 explores two independent time phases with time-weighted RMS units, and a
local-model sufficient-progress constraint. An uncommitted physics QP supplies
achievable predicted mass/core reductions; the image QP must preserve 80% of
those predicted reductions. There is only one committed physical candidate.
The 80% is a model-space task constraint, not an image-gradient multiplier and
not a guarantee of 80% of an independently realized physics rollout's progress.
The actual candidate still requires mass/core nonincrease and image improvement.

After peak constraints become active, diagnostic finite-difference probes may
cross the control bound to estimate a derivative. They cannot be committed and
must still pass trajectory health. Accepted controls and increments remain
bounded. Exact per-particle peak constraints are generated in the QP, avoiding
the old failure mode of shrinking forever toward an already saturated particle.

v4 additionally linearizes each particle's maximum-over-time log geometric
condition number divided by log(20), adding violated rows to the same QP.
The final actual rollout still has the original condition limit of 20. Diagnostic
probes can cross this optimization bound while retaining finite-state, complete
grid-support and orientation checks. The condition response is nonsmooth at an
isotropic initial state and at a change of the maximizing time. Its one-direction
error is reported, not treated as a derivative certificate or used to bypass
actual health validation.

v4 also changes the image residual to `(I-ell^2 Laplacian)^(-1/2)(I-I_target)`,
with mirrored boundaries and ell=.08 of the full image width/height. It is a
screened Sobolev image norm that preserves the DC/area residual. All cameras
still render at 1024px. Since the fixture's RGB channels are identically gray,
one normalized channel preserves the previous RGB-mean squared objective exactly
for the pixel variant, reducing stored response size by three.

The long-range image-optimization problem is also discussed by Xing et al.,
[Differentiable Rendering using RGBXY Derivatives and Optimal Transport (2022)](https://jkxing.github.io/academic/publication/DROT).
Their method uses point proxies and RGBXY derivatives with optimal transport;
our screened residual is a different proposed approach, not a reproduction of
that paper. CPU tests check a nonoverlapping translated Gaussian: the local
pixel objective gives effectively no translation direction, while the screened
metric points toward the target. A successful CPU direction does not establish
the required sphere/bunny quality advantage.

The user explicitly authorized different equations and gradient methods on
2026-09-14, provided rendering continues to guide physical state. The objective
changes above preserve that premise and do not change the registered test metric.

The v5 coordinate change uses delta=radius*u with complete chain rules. It
reduces numerical scaling problems without changing the mathematical objective
or feasible set. It is not a proof of solver stability: the image run ultimately
stops because its local image QP predicts no descent under the required physical
progress, while the physics run completes 12 iterations. Native v4 endpoint
inspection additionally fails visual quality because of floating-looking lobes
and incomplete ears. No final-test evaluation or successful quality claim follows.

The material-surface family is documented in `material_surface_control.md`.
Its preparation exposed 13,518 enclosed empty voxels in the legacy bunny fill
(256,499 occupied voxels at max-mesh-extent/110 pitch). Sealing enclosed background
did not remove artificial exterior bridges: v7's target had Euler -158 and visible
bars absent from the original bunny. v8/v9 use repaired closed geometry instead;
both arms get the same dense volume and source/target normalization. v8 stopped
before optimization because of a zero-control intersection-query false positive.
v9 adds independent separation certificates and dense-mass COM alignment. These
are changed development fixtures, not a re-evaluation of the legacy numbers above.
