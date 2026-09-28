# P299: one bounded proposal from the same prepared state

Status: bounded CUDA observation passed; both first proposals fail their merit
checks. No policy or quality promotion follows.
P298 measured a large pre-Adam change from the variance observable itself.
P299 tests whether that difference survives actual optimizer conditioning.
Geometric variance remains off by default and all earlier failed gates remain.

At the first inner iteration of baseline attempt24, share the exact prepared
state, loss references, plan, neighbors, pins, controls and optimizer moments.
Use original bunny300k, T20, dt1/240, dx0.3062907543956724wu, loss36^3,
eight inner iterations, shared PIC, no shift and committed-state outer merit.
Source sampling, target reference, coefficient and stopping contracts stay as
recorded by P298. Rerun through24; this is a newly executed state, not a claimed
bitwise replay of the earlier GPU window.

Construct A from the physical-variance gradient and its own render PCGrad
projection. Construct B from geometric variance and its own projection, at the
SAME positive baseline lambda. Freeze the first-trial alpha selected by A's
actual adaptive rule. The proposed control updates use the production shared
Adam/RPROP/surface projection/stress-u-body bounds helper, starting from cloned
identical parameters and moments. Alpha for B is deliberately fixed to A; do
not describe B as the geometric policy's native adaptive first trial.

Evaluate A, B, A on the existing prepared eval_terms closure, before any current
iteration candidate can be accepted. Each trial temporarily installs only its
candidate leaves, owns its outputs, and restores the original leaves and loss
telemetry in finally. Live moments and balancing state never advance. The
lease expires on callback exit. No accepted trajectory buffer may be present;
no continuity gate, material learning, Gaussian loss/cache, gradient
dump or outside-core objective is supported by this first diagnostic.

For each candidate report actual post-bound control deltas, both physical and
geometric scalar merits at the same lambda, predicted/required decrease, and
the existing finite/orientation/endpoint/pace/Armijo first-trial checks. These
checks are reused without relaxation. A failed first trial remains data: this
is not a line search, an outer acceptance or a completed window solve.

Independent geometry uses the same fixed target reference and raw proposed
particle states. Fix the source upper-boundary material IDs before the run and
intersect once with the selected window's start-free mask. Record all start-free
IDs separately. Report raw interior/final movement, PIC jump, saved final phase,
path/net displacement, fixed target coverage and fixed-ID neighbor supply. These
cohorts are not certified per-particle arrival sets; variance reduction alone
cannot certify rest. Numerical metrics stay on CUDA and consume no renderer or
optimization-loss operator. No full trajectory archive is written.

A/B/A must preserve exact proposed control deltas, pins and live controls/moments.
The actual prepared MPM initial state, material fields, layer/bond topology,
weights and gates are copied once and checked exactly after each candidate.
Record repeated state/merit differences separately without fitting tolerances
from the A/B difference. Require a real outer-accepted selected baseline window,
zero guards and exact reconstruction of its actual baseline lambda. Prior CPU
tests check observer-free continuation, exception restoration and expired leases;
the CUDA diagnostic does not by itself establish full-policy neutrality.

No new gain, default, after-arrival-rest claim, fit/supply repair or rendered
deliverable follows from a successful observation. Evidence from this single
step can only determine whether a controlled line-searched window is warranted.

The full CPU suite passed637 tests with22 skipped in278.13s after the proposal
extraction. The latest evaluator/driver checks passed15 cases, including immutable
prepared inputs and exception restoration. Independent checks passed12 helper,
4 observer and11 driver tests. These checks do not establish CUDA quality.

## First-proposal CUDA result, September 28

Frozen e39e0d6 ran on hyde06 GPU0 in415.726s with the original300k/T20/dt1/240,
dx0.3062907543956724wu/loss36^3 recipe above. All24 baseline attempts were
outer accepted with zero state guards. The observer selected the first inner
iteration of attempt24. Baseline lambda0.028738526231967466 reconstructs the
actual production value exactly. This newly executed state is not P298's
prepared state, so its scalar values need not equal P298's.

Observation ownership checks pass: all prepared inputs, controls and moments
are restored exactly, both raw and promoted anchors of243941 start-pinned IDs
are exact, and A/A-repeat control deltas are identical. The repeated promoted
endpoint differs by at most2.68221e-7wu; its physical merit difference is0 and
geometric merit difference9.31323e-10. These are descriptive repeat measurements,
not newly selected tolerances or clearance of the strict primitive failures.

Both A and B have native first-trial alpha0.02 at the shared baseline lambda;
adaptive-alpha scaling does not bind either. Both proposals are finite and
orientation preserving, but BOTH increase their corresponding initial merit
and fail the existing sufficient-decrease test:

| Evaluation | Physical merit | Geometric merit | Own first-trial merit passes |
|---|---:|---:|---|
| Prepared initial rollout |0.002655525416|0.002791497581|not a proposal|
| A: physical variance |0.005579575568|0.012776577051|no|
| B: geometric variance |0.005606942416|0.013754179307|no|

Compare each column against its own initial value, not A's physical value against
B's geometric value. A/B post-bound control deltas differ by joint L2 39.59985
in mixed leaf coordinates and rotate61.06 degrees; this is not a physical force
or distance. The distinction survives Adam and bounds, but these large rejected
steps are not representative of the accepted optimization path. The baseline
continuation uses first accepted alpha0.000625 (one32nd of the audited alpha),
and all eight accepted updates in that window retain that alpha.

For the same56059 window-start free IDs, saved-step RMS is0.0434629sp for the
initial prepared rollout,0.344553sp for A and0.361890sp for B. Raw-interior RMS
is0.0177591/0.192571/0.193609sp; saved-final RMS0.178292/1.292184/1.380972sp.
Native source spacing is0.03498853660707278wu. The same1602 source-upper free
IDs also have larger saved RMS with B than A (0.371162 vs0.339091sp).
Saved-path within-window reversal fractions fall slightly while path length
grows; raw-path reversals instead increase from A to B. This does not establish
rest. Neither cohort is certified arrived-free.

| Raw endpoint diagnostic | Prepared initial rollout | A | B |
|---|---:|---:|---:|
| Silhouette IoU |0.969890590|0.957151643|0.957494827|
| Fixed upper target coverage |0.962904911|0.825169801|0.829871996|
| Target-tip neighbors |45|26|24|

Fixed source-upper6712-ID density is0.734003 for A and0.736628 for B, while
under-half support rises0.227205 to0.227801. For the1602 free subset, density
is0.941948/0.949750 and under-half0.149189/0.150437. Initial-rollout supply was
not measured. A higher density does not offset lost target coverage; no hole
or thin-region repair follows from these rejected candidates.

This screen establishes a real conditioned proposal difference and failure of
both initial proposals to satisfy sufficient decrease. It does not compare accepted line-searched
updates, reject every geometric-variance step size, or explain P297's full-policy
regression. No new gain/default/full-policy run is selected. A bounded first-
iteration backtracking diagnostic would be a separate next experiment; it need
not introduce a second full optimizer or repeat prepared-state construction.

Evidence: local output/p299/proposal24/{protocol,proposal_audit}.json;
server /data/relcfd/chayo/physmorph_v2/work/p299/proposal24.
Result SHA256:55f27e2b79d2eb419e0cccbd45a15b24d6b5d87da4fda8ec7cd877d1a5721843.
Protocol SHA256:aaeb1c3470efcf648992250ee940742a30b895ebb9f475db1930cd6db1b04089.

## P299b: bounded first-update backtracking

The next diagnostic is opt-in and changes no production policy. Keep the same
single prepared state, initial controls/moments, gradients, baseline lambda,
initial merits and fixed material cohorts as P299. Require the two native
initial alphas to equal the shared initial alpha for this bounded comparison.
For each direction independently, follow the production half-step sequence
alpha/2^k for k=0,...,max_ls_iters-1. Each trial applies Adam and bounds from
fresh copies of the SAME starting controls and moments. Select the FIRST trial
passing its own objective's existing finite/state/pace/sufficient-decrease
checks, never the best geometry. Record every rejected scalar trial and any
exhaustion. The continuity gate remains unsupported and disabled.

Measure raw geometry, movement and supply only for the selected first feasible
proposal, or explicitly label the final rejected proposal if exhausted. Repeat
A at its selected index for ownership and descriptive numerical comparison.
The baseline continuation's first accepted alpha must exactly equal A's
diagnostic selected alpha. A mismatch invalidates this observation; GPU repeat
noise is not a reason to loosen that check. Both fixed-gradient directions
retain their own accepted alpha, which can differ and must be reported.

This is one first-update search, not another inner-iteration solve, outer
acceptance or native adaptive-lambda geometric policy. No additional gradient,
plan preparation, window state update, pin admission or renderer is introduced.
Prior primitive failures and whole-policy quality failures remain recorded.
