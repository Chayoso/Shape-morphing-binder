# P312: matched test of an additional prepared-silhouette constraint

Status: diagnostic implemented;34 focused CPU cases and independent
implementation review pass. Not launched. P311 rejects both terminal strengths; no candidate
is committed. Its six data-restored candidates all lower running motion and
original merit, but fail raw silIoU while prepared silhouette worsens and PBR
improves. This motivates testing the component tradeoff, not declaring it causal.
P309's counterexample (prepared silhouette worsens while raw IoU improves) remains.

Use one fresh W20 at N300000,T20,dt1/240,dx.3062907543956724wu,loss36^3,budget8,
raw/no-PIC/no-shift. Both arms use the exact same freshly generated terminal05,
original displacement and immutable original three-repeat baseline. The .5
strength is fixed before execution: P311 restored data at four radii and produced
larger terminal braking, but it has no physical-quality acceptance certificate.
Do not search other terminal strengths in this experiment.

Because both arms use identical terminal05, prepare ONE owned terminal-origin
linearization before either arm: actual running value, volume/render/silhouette
values, all four gradients, original displacement trust radius, and one set of
three terminal-running noise observations. Compute the resolved-running threshold
once from this shared origin. Own and hash these arrays/scalars and the same
terminal-forward evidence; neither arm may regenerate or mutate them. The control
selects the existing volume/render gradient rows; the treatment selects those
exact rows plus silhouette. Include the shared package in before/after checks.
An invalid shared baseline/origin aborts this comparison; an invalid search
candidate remains an arm-local rejection. No per-arm noise recalibration.

Run the two-halfspace control arm first, then the three-halfspace arm, retaining
both even when the first succeeds or fails. The control arm is the unchanged
P310/P311 density+combined-render affine repair. The experimental arm additionally
requires prepared silhouette <= the maximum of the same three original baseline
silhouette values. Keep this component ceiling in shared immutable metadata;
do not recalculate it from either arm. All raw P306 gates, original data ceilings
and resolved-running thresholds remain unchanged. The extra silhouette gate
must pass on the actual candidate, not only in the affine model. Do not constrain
PBR separately or change its weight, lambda or reference.

Generalize the diagnostic trust-ball operator from at most two to at most three
halfspaces, enumerating all eight active subsets. Retain float64 SVD rank checks,
QR for full rank, input-precision projected-step arithmetic checks and all exact
nonlinear gates. Add independent analytic three-plane and dependent/contradictory
plane tests, scaled near-dependent three-plane cases (render includes silhouette)
and an independent constrained numerical oracle; ordinary orthogonal examples
alone do not verify this conditioning. Passing the old two-plane suite is
required. This is a diagnostic
operator extension, not a replacement for the production optimizer.

For each arm, keep one displacement origin,11 radii, at most two replaced
observed-model corrections per radius, at most33 search forwards and at most one
accepted update. The experimental model remainder has three entries. Its first
candidate passing original P306+resolved-running+extra-silhouette gets exactly
three fixed-control repeats, each checked against the same full set of gates.
Stop that arm even if a repeat or report-only original merit fails. The control
arm retains the original P311 stopping condition. Do not change the origin after
a raw-quality rejection. No adaptive extension after seeing results.

Bind both arms to the same original objective inputs, lambda/wu, initial spec,
controls/basis/gate, references and material-ID cohorts. Reuse P311's byte-owned
baseline and original-merit evaluator. Record separately: original P306 pass,
extra component pass, resolved-running pass, report-only original-merit pass and
fixed-repeat result. The added component gate must not be mislabeled an original
P306 condition. Preserve full raw/control evidence for original, terminal-only
and selected candidates, input/source hashes and exact post-callback isolation.

Report all evaluated trials, actual silhouette/PBR/combined changes, original
merit and the same raw binary silhouette/coverage/density metrics. Only this
within-callback pair can compare the added constraint on a shared realization;
separate P311/P312 W20 realizations cannot establish causal differences. A pass
still does not prove pixel-perfect4K appearance, full-morph no-holes or rest.
Original-merit/adoption and coupled-continuation requirements remain in
`candidate_commit_contract.md`; the callback remains read-only.

Implementation uses byte-owned `SharedRepairBaseline` and `SharedRepairOrigin`.
The original baseline includes one fixed silhouette ceiling; the terminal origin
contains the common data values, four gradients, actual terminal-forward report,
three running repeats and the one resolved-decrease threshold. Both arms decode
private copies and save their selected gradient rows for verification. Accepted
records now preserve the scalar measurements that selected the exact saved
forward, instead of redundantly remeasuring its losses before deciding to replay.
This removes a possible atomic-rounding change at that branch.

Validation:22 affine operator cases include analytic three-plane intersections,
scaled nearly dependent/opposing rows and random two/three-plane comparisons to
independent SLSQP. Eight existing paired/remainder cases still pass. Two new
component tests verify one common gradient forward, both-arm execution after
first-arm success, report-only merit, immutable owned gradients, and a repeated
candidate that passes P306 but fails the added silhouette gate. Two actual
Warp-CPU observer cases verify shared-origin construction, silhouette derivative
finite differences, original-merit closure and exact production isolation.
Only the bunny-specific raw quality report is stubbed in those small-cloud CPU
cases; their physical forward, losses and gradients are real. No CUDA quality
or production acceptance follows from these checks.

Independent review reran all34 cases and found no launch blocker. Its generic
three-row/two-dimensional edge case exposed an incomplete full-row-rank check;
the QR branch now requires as many retained singular values as active rows,
otherwise using the existing pseudoinverse/residual path. The actual large
control space was unaffected, and the added regression passes. The initial
small-cloud observer fixture correctly rejected an empty bunny-specific upper
cohort (NaN); its explicitly scoped quality-only test stub was then independently
retested. Production finite-data rejection was never loosened.
