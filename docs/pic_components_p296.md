# P296: partition the whole-window PIC correction before changing it

Status: preregistered read-only diagnostic. P295 found a large endpoint jump and
a smaller immediate response dominated by layer relaxation. It did not measure
which previous-window channel generated the jump. This experiment does that
partition before choosing a repair.

Use the exact P295 simulation-core bytes and immutable bunny300k inputs, T20,
dt1/240, dx0.3062907543956724wu, loss36^3, eight inner iterations, shared PIC,
shift_sub off, geometric-rest off, and the mixed body/stress/layer recipe.
The full attempt cap is60 under animations300. Select accepted windows1,24,30
before observing outcomes; no substitution for a missing selected window.
A cap1/window1 smoke checks the diagnostic first.

At the validated inner rollout, collect sums of the actual step components:

- A: dt times the stored physical velocity, summed over the window.
- B: pre-layer position minus previous position minus dt*v; includes bond/update roundoff.
- U: the applied normal u channel, including the actual gate and frozen masks.
- L: after-layer minus pre-layer minus U; relaxation plus arithmetic roundoff.

Take the EXACT owned endpoint operator H, start x0, pin mask Q, raw endpoint xT,
and this rollout's validated owned promoted endpoint y from the optimizer package. The
observer and package must agree on start/raw/pins. Require xT[pin]=x0[pin] and
A+B+U+L approximately equal xT-x0. The component corrections are
J_i=Q(H-I)i, with Q outside H. Their sum must reproduce y-xT. This is an algebraic
partition of one optimized rollout; A already depends on earlier direct position
updates, so this is not a causal channel-off experiment.

The helper reports every Gram cross term, signed component projection on the
actual J, A/R cancellation (R=B+U+L), and numerical closure. A repeated H(D)
evaluation measures one repeat difference. The separate roundoff allowance uses
64*dtype_epsilon times the coordinatewise maximum of start/raw magnitudes,
summed absolute components and one source spacing; operator identities also
allow10 times the measured repeat difference. These are explicit diagnostic
bounds, not a statistical confidence interval. Fractions near the numerical
floor are null. Original-map, component-linearity and repeat errors remain
separate; do not assign an unexplained residual to B or L.

Four endpoint clouds are compared without ever committing a candidate:

| Name | Endpoint |
|---|---|
|raw|xT, context only|
|current|the actual owned y|
|advection_only|xT+J_A, keeping all direct position corrections|
|preserve_relaxation|xT+J_A+J_B+J_U, keeping only L outside the filter|

Every candidate must retain shape/device/dtype, finite in-bounds positions and
exact start-pin anchors. Candidate jump size need not fall: when A and R cancel,
unfiltering R can expose a previously canceled J_A. Report both the candidate
jump and its change from current. Use identical current window-start all-free
and layer-free masks. These are not P295's next-window masks.

Independent raw geometry metrics reuse the fixed target extent, binary silhouette
and hole masks, target-neighbor coverage, upper-region coverage and tip count from
the established bunny audit. The density radius is median target k=9 distance
at index8, not twice source spacing. Upper y>2.3 and tip radius0.25wu are the
existing diagnostic regions, not physics parameters. Record exact candidate-
current differences; no newly fitted epsilon is introduced. The directional
screen requires non-increasing chamfer/hole fraction/far-out fraction and
non-decreasing silhouette IoU/target coverage/upper coverage/tip count/upper density,
and a non-increasing upper fraction below half target support. Missing measurements
are unresolved, not successful. These density directions screen against lost
support; higher density alone is not a quality certificate. Gap distributions
remain telemetry. Failed decomposition leaves geometry as diagnostic evidence
but comparison fields null and explicitly ineligible.

Only outer-accepted windows admit JSON and compact endpoint archives; rejected
attempts are discarded. A failed closure stops interpretation. If both late
windows support a smaller resolved correction with preserved endpoint quality,
that warrants a separately reviewed optimization/trajectory experiment. It does
not establish watertightness, continuity during the window, correct carried
F/v/C, or eventual rest. Tradeoffs remain explicit rather than converted to a
pass with the global final-fit threshold.

All numerical work runs on hyde06 CUDA. Local tests use CPU only. No production
default, optimizer equation or renderer is changed in this diagnostic.
