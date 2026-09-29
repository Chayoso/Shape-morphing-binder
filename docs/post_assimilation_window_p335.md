# P335: bind the joint derivative to actual prepared successor state

P334 passed small conditional operator gates. P335 connects its two-window
bridge to the ordinary N300k W20-to-W21 handoff, before any changed candidate.
The first21 windows retain the existing raw mixed-body recipe, controls,
optimization budget, outer acceptance and pin/admission policies. The sole cap
override is21 windows; no physical or visual remedy is inferred from this cap.

The final W20 callback returns the original choice unchanged. While its lease
is live, save an owned private head replay, its complete merit and evaluator
binding. Compare that merit at the final fixed lambda to donor history. Save
the independent control owner and discard the callback's private graph. Never
retain or revive its merit callable after the runner closes the selection.

Capture the ordinary W21 prepared step-zero trajectory AFTER its optimizer
returns. This includes the OT-updated surface-u gate. After the full original
pipeline returns and production graphs are released, load the saved owner and
successor, then evaluate the private post-assimilation joint model. The adapter
owns actual successor pins/layer/bonds/viscosity; it rejects differing material,
mass, volumes, horizon, grid, time step or pin mode. Active support-gating and
layer-F reconstruction are excluded from this first adapter; the native recipe
uses neither. Check layer fractions/depth rather than assuming reconstruction.

Old head pins retain their original anchors. All next pins anchor at the head
endpoint and zero carried v/C. Fp is recomputed through P333, then checked against
the actual prepared successor. A fresh independent coast starts from the actual
W21 x/v/C/F/Fp, uses its captured policies and withdraws learned controls. It
must not replace its initial values with those of the joint model. Geometric F
is only an auxiliary observation when the ordinary successor did not track it.

Preregistered production scope: N300000, T20 per window, dt1/240,
dx .3062907543956724wu, grid/loss36^3, eight optimizer iterations per window.
Compare full joint head merit INPUTS to the saved live-merit witness, boundary
x/v/C/F/Fp to actual W21, and every coast x/v/C/F phase to the independent coast.
Use the existing32-FP32-epsilon elementwise rule with native position, velocity,
C and deformation units from P329. Report bit identity separately. Approximate
input closure plus live W20 scalar closure is NOT an independently recomputed
joint complete merit. Full candidate merit must still be evaluated under its
own live original reference before any selection.

Backpropagate mean squared geometric step speed and stored speed on the actual
surviving-free cohort to both reduced body modes. Require finite nonzero
covectors, fixed pins, all-step bounds and positive physical/effective F. These
are capability checks, not natural-rest requirements or a candidate search.
Retain owned coast arrays/gradients, original history and rendering influence.
Raw-state quality and all-frame4K QA remain separate later adoption gates.

Small CPU/actual CUDA adapter gates additionally cover source ownership, next-
pin anchors/projection, ordinary subset assimilation, independent coast, both
coefficient directions, stale/lifetime behavior and invalid successor policies.
The old pre-assimilation adapter must retain its current tests and behavior.
The reduced-control FD observation is weighted terminal coast velocity: the
older weak position-difference observation lost float32 resolution in the
terminal mode. The initial gate used both radii (1e-3,5e-4),2%/5e-6 bounds and
the same physical fixture; see the retained failure and radius revision below.
CUDA tests explicitly include both no-slip and native slip pin
modes; a passing no-slip test alone does not certify a different contact branch.

Initial implementation validation passes75CPU cases (46new adapter,3failure-
receipt tests,26existing adapter tests),11.30s. The full original velocity and
F histories are bound before private replay; the exact original archived W20
X/F sequence is checked after the run. A64KiB final failure receipt reservation
keeps output-cap or report failures explicit. At this initial checkpoint, actual
CUDA/production were pending; actual CUDA results are recorded below.
The shared evaluator refactor also passes55 existing selection and real CPU
multiwindow integration cases (17.85s). Six CUDA cases were initially registered
and skipped locally; local skips are not GPU evidence.

All numerical GPU execution is on hyde06; no local simulation outside CPU tests.
Use an isolated frozen source directory, a3GB output reservation, the100GB project
guard, idle-device check and process memory monitor. Preserve before/corrected.
An allocation/parity failure remains recorded evidence and blocks this gate;
do not relax a physical criterion or claim an accepted treatment.

Rendering influence: the ordinary21-window history is reported with lambda,
channel direction observations and image-loss changes. The private derivative
measurement adds no rendering objective or optimizer update. Its motion
gradients do not measure rendering-caused displacement; 64px feedback does not
supervise the exact4K footprint. Physical F and visible Gaussian covariance
remain separate acceptance requirements.

## First actual CUDA attempt: slip derivative fails

Frozen5d9d47a, `p335_cuda1`, passes4 of6 gates. Both no-slip reduced derivatives,
actual subset-assimilation/fresh-coast forward, and side-stream/lifetime pass.
Both slip-mode finite differences fail at1e-3: displacement AD -.299037859 versus
FD -.292962246; terminal AD -.161156504 versus FD -.156122348. Keep the failure
and original2%/5e-6 thresholds. This does not authorize a native production run.

The coast's pinned-mass raster `gmpin` had no gradient buffer. Newly pinned
anchors depend on the head controls, and this raster sets the collider normals
seen by free material. Thus freezing it omits a continuous boundary dependency
even when pin admission itself is held fixed. The targeted P334 correction makes
ONLY its coast `gmpin` gradient-bearing before seed registration and graph
capture; the ordinary single-window Trajectory default and all primal kernels
are unchanged. Do not infer that this alone cures production oscillation.

The missing edge does NOT explain the wide-radius discrepancy: corrected CPU
displacement AD is -.29906338, only about2.6e-5 from the detached value. Its
central FD changes from -.29296201 at1e-3 to -.30417561 at5e-4, -.30399548 at1e-4,
-.30293984 at5e-5 and -.29933996 at2e-5. Terminal AD -.16115535 compares with FD
-.15612156 at1e-3, -.15899210 at5e-4, -.16113646 at1e-4 and -.16102674 at5e-5.
This supports a finite-radius/nonlinear-contact explanation but does not prove
that all contact active sets remain unchanged.

After this CPU diagnostic, freeze a refined LOCAL slip FD gate at1e-4/5e-5
before the next CUDA run; keep2%/5e-6 bounds, both coefficient modes, and the
original1e-3/5e-4 no-slip gate. Preserve the failed original slip test rather than
calling its large-radius behavior repaired. The aggregate control observation
cannot distinguish the tiny gmpin contribution at this tolerance; use a separate
resolved boundary-anchor perturbation and detached-gmpin negative control.

The isolated new-pin-anchor check holds all other actual successor state fixed
and observes weighted first-step acceleration. On the N27/T20 CPU fixture,
full AD .000453370168 matches FD .000452877887 at1e-4 (0.109%); detaching only
gmpin gives exactly zero with identical primal values. Its5e-5 FD .000377522651
is unresolved and is not a passing second bracket. Repeated seeds agree and a
zero seed clears the pinned-mass covector. This identifies a boundary derivative,
not the source of the larger reduced-control discrepancy or production motion.
The updated adapter suite passes49 CPU cases in8.84s; ten P334 regressions also
pass. Seven actual CUDA cases are registered for the second run, including this
captured boundary negative and the separately refined local slip gates.

## Second actual CUDA attempt: isolated anchor FD fails

Frozen91f8bb9, `p335_cuda2`, passes6 of7 gates in5.72s. Both refined slip
control directions pass at1e-4 and5e-5, with the original2%/5e-6 bound. The
isolated anchor has AD .000453369836, detached AD0 and full-FP32 finite difference
.000494115164 at1e-4; their .0000407453 difference exceeds .0000098823 allowance.
All preceding actual-boundary/first-step primal, repeat and zero-seed checks
pass. This failed numerical witness still blocks the gate. CPU agreement at
one radius does not validate the CUDA finite difference or explain its error.

Next isolate the first-step collider path in an independent FP64 reference:
slip-mode pinned particles do not participate in the free P2G mass/momentum.
Hold those actual grids and all free particle states fixed, vary only the new
anchor, then recompute pin mass, collider normals/projection and free G2P.
Bind its unperturbed mass/grid/particle velocities to actual Warp values before
testing FD convergence at multiple radii. Keep the failed complete FP32 witness
and distinguish this path derivative from a complete full-precision simulator.

The test-only FP64 reference now matches actual unperturbed pin mass, grid
velocity and first-step particle velocity. At N27/T20/dt.002/dx.5/grid16^3 on
CPU, the same anchor direction gives Warp AD .000453370168, reference analytic
AD .000453370059 and FD .000453369904 at1e-4 / .000453370021 at5e-5. Detached
gmpin gives zero. Both unchanged2%/5e-6 comparisons pass; the two FD values also
agree within1e-6 relative or the explicit FP64 rounding bound. Contact, support,
cell, normal, wall and floor branch masks stay fixed across both brackets.
Free P2G mass/momentum is independently unchanged under the pin-only shifts.
All49 CPU adapter cases pass in9.62s. The same-device reference is now frozen
for a third actual CUDA gate; it is not a complete FP64 simulator or a repaired
full-FP32 FD witness, and does not establish the cause of the earlier mismatch.

## Third actual CUDA gate passes

Frozen c1a2b00, `p335_cuda3`, passes all7 cases,0skips,5.622s. The captured
anchor pullback .000453369836 agrees with independent GPU-FP64 analytic value
.000453371044 and FDs .000453370889 / .000453371005 at1e-4 /5e-5. The detached
negative stays zero; baseline/branch/seed/convergence checks pass. All reduced
body-mode, actual subset-assimilation/coast and lifetime/stream gates pass.
The physical kernels are unchanged from91f8bb9; this revision changes the
independent numerical witness, not the original failed FP32 finite differences.

The native read-only21-window `p335_native1` run uses the same frozen source.
The small CUDA gate authorizes this capability experiment only.

## First native run: coast state closure fails

N300000,T20,dt1/240,dx.3062907543956724wu,grid/loss36^3,iters8. All21 original
windows and168 inner updates commit; guards remain zero and the original W20
trajectory is retained. The run takes382.37s and retains989.1MB of evidence.
Live W20 full merit, private-head inputs and actual W21 x/v/C/F/Fp boundary
checks pass. Actual next pins total261872, including2898 new admissions.
All head/coast health and pin checks pass. These are producer observations
until independently checked against the available archives.

The complete coast gate fails: C first exceeds the registered bound at phase2,
reaching max absolute .000245318 and5.29589 times allowance at phase20. Velocity
first fails at phase10, reaching .0000223666wu/s and1.54183 times allowance at
phase20. Every position and physical-F phase passes (maximum allowance ratios
.36001 and .13901). Boundary C and Fp differences are already .0000272095 and
.00000447035 respectively, within their original bounds. Do not infer a kernel
bug, roundoff cause, or the cause of visible oscillation from these figures.
Do not enlarge tolerances or replace the actual coast input by the joint input
to pass the gate. No gradient/candidate is admitted.

The failure precedes coast/gradient archival, so only six of eight expected
archives exist. The missing joint path and covectors cannot be independently
reconstructed from producer comparison scalars. The bounded archive audit must
retain an overall failure. A separate CUDA diagnostic will compare actual
coast repeats, same-joint-input coast and isolated Fp versus other-boundary
changes, with explicit diagnostic labels and unchanged elementwise bounds.

Ordinary rendering influence in this same discretization uses18views64px and
shared GS loss remains off. Across168 updates, median nominal render-direction
share is .50702593 (body .35711240, stress .56421557, surface-u .74253830).
Across21 windows median lambda is .06224250; median observed image-loss change
per update is -.0001171824. These history observations are not causal movement
fractions. No render optimization occurs in the private coast, and these64px
terms do not supervise the exact4K Gaussian footprint.

## Saved-state localization: Fp response carries the discrepancy

The bounded CUDA diagnostic `p335_closure_diag1` completes11.14s on the same
N300k/T20/dt1/240/dx.3062907543956724/grid36^3 state, with source/input/output
hashes. It reruns no optimizer. The fresh joint reproduces C failure fromphase2
(maximum allowance ratio5.27020); its v remains just inside the bound this
time(.95858). This does not erase the native run's failed velocity witness.
Two actual-W21 coast replays agree within the original rule, including C .38988.

| Synthetic boundary change | Maximum C allowance ratio versus actual coast |
| --- | ---: |
| Joint Fp only; all other actual state retained | 5.03461 |
| Joint x/v/C/F; actual Fp retained | .65742 (all four state channels pass) |
| All joint boundary fields | 5.25886 |
| Joint x/v/C/F plus ordinary assimilation of the same joint F | 5.14404 |

The all-joint-input no-grad coast agrees with the captured joint(C .21920).
Replacing its Fp by the ordinary map of the SAME joint F also agrees(C .23811
versus the all-joint no-grad coast). Ordinary/P333 Fp is bit-identical on all
38128 free IDs at the same F; backend differences occur only in the new-pin
subset. Ordinary assimilation of the exact accepted W20 F recovers the actual
W21 Fp bit-for-bit for every ID.

Thus this case localizes the discrepancy to the Fp response to the small head-F
replay difference, rather than a differing free-particle assimilation backend
at fixed input. It does not establish why the head differs, a general stability
bound, or the cause of visible oscillation. Every replacement above is synthetic
and cannot pass the original actual-handoff gate. Test an explicitly labelled
FP64-internal/FP32-state assimilation counterfactual before changing production
precision; retain the original failed gate and all other physical parameters.

The partial archive audit independently checks3300 items, including available
head and exact ordinary handoff arrays and rendering bookkeeping. Only the four
expected producer/missing-coast conditions fail; no execution error remains.
Its first attempt rejected the legitimate declared device alias `cuda`; the
reviewed correction accepts `cuda`/`cuda:0` while retaining canonical captured
device checks. Earlier failed audit output is also retained. The diagnostic's
output arrays are hashed rather than archived, so a source/receipt audit cannot
independently reconstruct every diagnostic phase value from saved arrays.

## Precision counterfactual and opt-in actual recipe

`p335_closure_diag2` uses the same saved N300000/T20/dt1/240/dx.3062907543956724,
grid/loss36^3 state. It computes each assimilation call in FP64, stores FP32,
then feeds that stored state into the new-pin call. The two synthetic coasts
use their own original versus joint x/v/C/F states and respective high-precision
Fp. No optimizer or control change occurs. In13.54s their maximum allowance
ratios are x .19217, v .30020, C .40957 and F .13901; all phases pass.

The Fp response to the small replay-F difference has maximum3.57628e-7 and
component RMS1.57777e-9 after per-call FP32 storage, versus4.17233e-6 and
1.38675e-7 for the ordinary FP32 map. RMS includes all300000x9 components,
including unchanged old pins; it is not a material-motion metric. The response
difference itself fails the32-epsilon rule on4 IDs (maxratio1.09375), so the
two precision policies are not declared interchangeable.

Neither synthetic coast matches the original actual FP32 coast in C (maximum
ratios3.08825/3.02023). The fresh FP32 joint still fails v/C at1.03947/2.68331.
This is evidence to test a changed numerical recipe, not a pass for native1.
Independent receipt/source/scalar audit checks1275 rows and10200 worst-component
ratios exactly; numerical output arrays were hashed, not retained for reanalysis.

The default-off `assim_fp64` option now applies the same high-precision elastic
map in the ordinary runner and conditional derivative. Every call stores FP32,
including the intermediate first-call result before new-pin assimilation. The
same stable spectral pullback retains those casts. Growth and consensus are
explicitly unsupported. The existing FP32 path and frozen native1 are retained.
`post-assimilation-window-fp64` starts a fresh ordinary21-window prefix with this
option enabled from the beginning; it cannot reuse the old actual boundary as
a precision gate. CPU/CUDA/actual-native gates remain required before adoption.

The head replay's few-ulp difference is consistent with FP32 atomic reduction
ordering in separate trajectory allocations, but the first divergent reduction
and equality of all discrete masks have not been measured. No visible jitter,
hole, natural-rest or Gaussian footprint improvement follows from this test.
