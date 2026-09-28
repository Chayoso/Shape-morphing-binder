# P296: partition the whole-window PIC correction before changing it

Status: completed read-only diagnostic; endpoint split candidates failed the
preregistered quality screen. P295 found a large endpoint jump and
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

## Result: September 28, 2026

Frozen source a785863 passed independent code review and21 CPU tests. The cap1
CUDA smoke accepted1/1 in18.025s, with all closures/admissibility checks passing.
Its density-above-y2.3 measurement is null because that region is still empty;
it is not a zero-hole result. Both endpoint candidates already lost target fit.

The full run used the same frozen simulation bytes as P295 and the discretization
above. Protocol began15:44:51UTC and ended15:54:05UTC:33 accepted/36 attempted
windows in554.573s, stopped by three consecutive outer-merit rejections. All
state guards were0. All fixed selections1/24/30 were present; numerical closure
and exact start-pin checks passed. No candidate endpoint was ever committed.

| Accepted window | Start pins | Layer-free IDs | Actual J RMS(sp) | A-only J RMS(sp) | Preserve-L J RMS(sp) | Signed A/U/L projection fractions |
|---|---:|---:|---:|---:|---:|---|
|1|0|20671|0.588981|0.238140|0.565489|0.230901 /0.521835 /0.247264|
|24|254352|5564|0.544351|0.322107|0.333660|0.336978 /0.019953 /0.643069|
|30|262698|5187|0.457883|0.203872|0.209574|0.189024 /0.007202 /0.803774|

These are current-window-start masks and native source spacing0.0349885366wu,
not P295's next-window masks. Fractions are signed projections onto J with cross
terms retained; they are not positive causal energy shares. Bond/update residual
projections are near roundoff. Layer-free A-only jumps shrink about41%/55% late,
but all-free A fractions are0.688/0.519: the claim must remain cohort-specific.

| Window / candidate | Delta IoU | Delta upper target coverage | Delta top density | Delta under-half support fraction |
|---|---:|---:|---:|---:|
|24 / A-only|-0.00176748|-0.00640021|+0.0134492|-0.0231641|
|24 / preserve-L|-0.00183460|-0.00731452|+0.0136029|-0.0212931|
|30 / A-only|-0.00136798|-0.00561651|+0.00959693|-0.0141354|
|30 / preserve-L|-0.00146328|-0.00600836|+0.00945370|-0.0130729|

All deltas use each current accepted endpoint as its comparator with fixed
target/extent. Chamfer also worsens for both late candidates; tip count is
unchanged. Binary projected hole_frac is0 for all these endpoints; that metric
does not establish 3-D coverage or watertightness. Local density improves while
target surface coverage worsens, illustrating why density alone cannot pass.
The density cohort itself is each candidate's y>2.3 set, so its IDs/count are not
fixed between candidates; upper target coverage uses the same15312 target IDs.
Both candidate screens fail at all three selected windows. The lower jump does
not justify deploying the split or claiming that morphing holes/rest are solved.
This static failure does not by itself refute a separately reoptimized policy.

Keep shared H(D) unchanged. A separate optional full-substep geometric-variance
experiment (P297) follows the alternate observation change documented in P295;
it needs its own gradient, CUDA and quality gates. P296 does not establish that
this alternate will improve the morph.

Evidence: server `/data/relcfd/chayo/physmorph_v2/work/p296/full60/`; local JSON
copies `output/p296/full60/`. Three compact endpoint archives total55,805,304bytes.
Protocol SHA256 `796a09fb70f743935696134b9219616a40cf5a93ffad0b085e260ced82651389`;
result SHA256 `40dff8d5a3219321fc3b63f2c707e385514e44763d3c799bb673615f6383fc7b`.
