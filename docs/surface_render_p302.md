# P302 shared surface rendering and influence reports

Status: experimental opt-in implementation, **no default or quality promotion**.
The current P300 before/corrected movies remain the retained results. This change
does not establish rest after individual arrival, consistent intermediate motion,
absence of morph-time holes, or removal of studio-lighting streaks.

## Objective and common primitives

The new `surface_gs_loss` adds a surface observation to the existing CIC/PBR-lite
render channel, evaluated at the **shared promoted PIC endpoint**:

    L_render = L_CIC + w_pbr L_PBR + surface_gs_weight L_surface.
    L_surface = mean_views(L_coverage + L_detail_coverage + L_detail_edge).

Coverage uses squared deficit/excess penalties with the existing `w_hole` and
`w_spray`. Detail edge uses mean squared Sobel residual, kernels divided by 8.
There is no new free opacity, size, position correction or physical actuator.
The current lambda balancer and gradient combiner see the composite render term.
Lambda adaptation means this is a whole-policy experiment, not a fixed-lambda
estimate of the incremental GS force.

`physmorph/render/surface_gaussians.py` supplies both the differentiable loss and
`render_splat_photoreal.py --surface-common`. Calibration (center, radius, nearest
spacing, median eighth-neighbor radius, density grid) is fixed from the final
target once; the intermediate paced cloud never recalibrates those quantities.
Every candidate refreshes KNN identities, support and normal donor selection.
Only KNN indices come from the detached GPU search; selected distances are
recomputed with live Torch positions. Density normals, distance-dependent sigma,
normal averaging and covariance retain their continuous derivative paths.

For a unit normal n and tangent radius sigma:

    Sigma = sigma^2 (I - 15/16 n n^T).

This is the historical tangent disc with normal radius sigma/4, passed directly
as packed covariance. The training path never differentiates arbitrary tangent
eigenvectors. A zero final normal uses the fixed-reference radial direction;
the exact center uses z. Sigma remains spacing times clamp(r8/reference_r8,1,4),
opacity remains 0.92 times live hard support. The new mode is **stateless** and
bypasses the historical pin appearance latch in both training and export.
It cannot certify that a pinned particle's appearance is temporally fixed.

Hard support, neighbor identities and donor identities are discrete active sets.
Particles at zero opacity have no GS recovery gradient through that indicator.
Existing volumetric/coarse guidance and independent raw-support checks remain
necessary. Merely masking unsupported splats is not physical hole repair.

## Cameras, targets, and acceptance

The default is four angularly separated cameras, including the studio camera
(azimuth35 degrees/elevation18 degrees). Additional cameras use greedy angular
separation from the existing view candidates, not array-stride selection. Losses
are averaged over cameras; camera count changes coverage, not nominal gain.

Global coverage uses height256 (width456). Detail uses full 3840x2160 images,
then crops a fixed target-selected 256-square edge patch per view. This retains
the native full-camera footprint and all-scene occlusion/tails. It is a bounded
first patch policy, not coverage of every thin feature or every artifact.
The target cloud is the current paced reference until the existing fixed-target
handoff. Targets, cameras and ROIs are frozen for the entire inner solve.

The same new scalar enters gradient, line-search merit, and final replay.
Unsafe trial positions are rejected before KNN/rasterization and can backtrack.
Shared endpoint guards and physical state checks are unchanged. Historical
`d_sil`/`d_render` remain CIC values; `d_render_total` identifies the composite
callback value. The outer fixed-target CIC/physics gate remains unchanged and
is **not** a GS convergence certificate. Inner accepted trials from an outer
rejected window are excluded from committed-window influence aggregates.

## Report contract

User request, 2026-09-28: report the degree of rendering-loss influence on future
runs. `pipeline_run.py` and `probes/gpu_pipeline.py` now write per-arm
`*.render_influence.json` and `.md`. Raw run histories preserve the step details.
Reports include discretization, available render/physics direction norm shares,
stress/body/surface-u update norms, observed render loss changes, and endpoint
changes between optimizer proposals. Those endpoint changes are not physical
substep velocities. Direction dot update is not physical work and may include
preconditioning/projection; it is not presented as an exact directional derivative.

The report distinguishes optimizer attempts from held/C2F metadata and actual
commits from rejected trials. Missing historical/per-step direction data stay
unavailable. Reporting never adds a gradient pass or changes the combined-only
gradient branch. A nominal share is **not a causal displacement percentage**.
Separate matched render-on/off raw trajectory comparisons are needed to measure
causal changes in motion, supply and completion. Image diagnostics are not raw
physical quality metrics.

On the retained P300 corrected cap24 run, N300k, T20, dt1/240,
dx0.3062907543956724wu, loss36^3, budget of eight inner iterations per window, coarse render64:
first-iteration nominal render share has median0.435868, range0.275480..0.582610.
Channel medians are stress0.567392, body0.336185, surface-u0.545490.
These are observational norms, not percentages of physical displacement.
Legacy archives lack the newly added per-step accepted-update fields.

## Validation and limits

Independent CPU tests use actual MPM with mocked GS observations to check the
additive adjoint seed/merit, accepted-buffer and forced-final-replay closure,
two-window weight-zero equality, and invalid-candidate backtracking. Separate
CPU tests cover analytic covariance at repeated eigenvalues, density-normal
directional derivatives, coincident particles, view averaging and report scope.
They do not substitute for real CUDA raster validation.

Hyde06 operator3 uses N300k, four native 4K views, a one-spacing translated source
reference, coarse256 and patch256. Packed versus eigen-decomposed raster output
differs by at most8.94e-8. A single primitive's interior-pixel directional tests
pass: best relative errors mean0.139%, covariance0.00936%, combined0.00743%.
The full live primitive gradient is finite (norm0.1514854; max0.0381999), repeated
loss agrees within1e-6; peak Torch allocation is3,548,174,336bytes. This peak
does not include an MPM tape or CUDA allocations owned outside Torch.

**The broad finite-difference check remains failed/open:** 48 splats over a full
image differ8.46..10.32%; full-cloud y-translation differs5.97..6.09% for steps
0.1/0.03/0.01 native spacings. Hard raster/active-set boundaries are a hypothesis,
not a demonstrated explanation. Local core-pixel agreement cannot certify the
global derivative. No threshold was relaxed to call the broad test passed.
Only a bounded cap1 execution comparison is permitted by this evidence, with
actual recomputed merit and existing physical checks on every trial.

Snapshots and logs live under server `/data/relcfd/chayo/physmorph_v2/work/p302/`.
The launch script uses the same 50-second shared launch lock, refuses existing
tags and refuses launch if project usage is already100GB (a launch precheck,
not an in-flight hard limit). Source/target inputs are the preserved
`repro/current_pair/source*` and prepared target reference; no resampling.

Full raw-trajectory quality gates and per-frame visual QA are still required
before promoting this loss or publishing a new rendered comparison.

## Bounded execution comparison

CUDA snapshot `code3`, numerical source SHA256
`523350aa6d3669009f214af95d66da0c4f7a4dbce704cf5d436b8ce37055610f`:
N300k, T20, dt1/240, dx0.3062907543956724wu, loss36^3, cap1, two inner
iterations. Same source/target arrays, prepared target, code, camera calibration,
ROI policy and actual ROIs; only `surface_gs_weight` differs. A control repeat
uses the exact control config. Each arm accepts two inner steps, one outer
commit, no backtracks and no state-guard interventions.

| Observation | GS weight0 | GS weight1 | Weight0 repeat |
|---|---:|---:|---:|
| Seconds | 11.43 | 11.58 | 11.41 |
| Peak Torch GB | 9.56 | 11.10 | 9.56 |
| Render lambda | 0.327159 | 0.014882 | 0.327159 |
| Global GS coverage loss | 0.0172585 | 0.0206362 | 0.0172585 |
| Native patch coverage loss | 0.0316570 | 0.0297090 | 0.0316575 |
| Native patch edge loss | 0.000333126 | 0.000318506 | 0.000333127 |
| Total GS loss | 0.0492486 | 0.0506637 | 0.0492491 |
| CIC silhouette loss | 0.0243921 | 0.0296387 | 0.0243921 |
| Raw endpoint binary IoU | 0.731116 | 0.726159 | 0.731116 |
| Raw endpoint Chamfer (wu) | 0.167825 | 0.170166 | 0.167825 |
| Minimum trajectory det F | 0.970745 | 0.973612 | 0.970745 |

The two local patch terms improve, while global GS coverage, CIC, binary IoU
and Chamfer worsen. Total scalar merits across arms are not directly comparable:
the objective and automatically fitted lambda differ. Weight1 changes the final
positions by0.0265775wu RMS versus control; the single control repeat differs
2.12787e-6wu RMS. These are cross-run endpoint differences across the first-window
cloud, not physical motion, oscillation or settled-drift measurements. The raw analysis runs on CUDA and consumes no
Gaussian render output. **Keep the new loss experimental.**

Next work is to investigate the broad raster derivative discrepancy and whether
separate coarse/detail gradient budgets can preserve global transport while
adding fine feedback. Adding more windows or declaring the current local patch
improvement a hole/artifact repair is not supported by this result.

Checked-in evidence:

- [Exact raw comparison and artifact hashes](evidence/p302/cap1_comparison.json).
- [Operator observations, including the failed global FD checks](evidence/p302/operator_observations.json).
- [Control influence report](evidence/p302/control1.render_influence.md),
  [candidate report](evidence/p302/candidate1.render_influence.md),
  [control repeat](evidence/p302/control_repeat1.render_influence.md).

The final raw comparison verifies exact source/start and endpoint/compact/F-sample
closure, accepted frame indices and finite arrays, and binds JSON/raw/compact
bytes and metric/helper sources by SHA256. Code snapshots are separate from the
current retained movies. The focused CPU verification set has110 passing tests
(109 initially passed; the remaining extracted-studio test fixture was corrected
to construct the new explicit legacy flag and its affected subset passed).
