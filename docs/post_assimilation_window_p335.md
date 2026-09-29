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
