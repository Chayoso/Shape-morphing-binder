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

## P299b CUDA result, September 28

The bounded observation passed, but does not justify policy promotion or a new
full geometric-variance solve. Frozen9706141 ran on hyde06 GPU0 in467.743s,
using the original300k/T20/dt1/240, dx0.3062907543956724wu, loss36^3,
shared-PIC/no-shift recipe. This snapshot uses the prior constitutive
implementation; the separate P300 adjoint work is outside this observation.
All24 baseline attempts were outer accepted, with zero state guards. This is a
newly executed prepared state, not a matched-state comparison with the earlier
P299 run: baseline lambda is0.01908374959286506, and the fixed cohorts contain
66204 window-start free IDs and1801 source-upper free IDs.

Both native initial alphas equal0.02. Both directions reject indices0 through3
and first pass at index4, alpha0.00125. A's selected alpha equals the actual
baseline continuation's first accepted alpha exactly; all eight accepted inner
updates retain0.00125. Selection uses each direction's own merit, not geometry.

| Halving index / alpha | A physical merit | B geometric merit | Own merit passes |
|---|---:|---:|---|
| Initial prepared rollout |0.002645249863|0.002881857020|not a proposal|
|0 /0.02|0.005817247435|0.015082329859|neither|
|1 /0.01|0.005369909743|0.014023532782|neither|
|2 /0.005|0.003336223112|0.005722650853|neither|
|3 /0.0025|0.002744243614|0.003255634390|neither|
|4 /0.00125|0.002632946547|0.002805015354|both|

Compare each column with its own initial merit. At selection, A lowers physical
merit0.4651% but raises geometric merit0.8540%; B lowers geometric merit2.6664%
but raises physical merit0.7218%. Jmin is0.846537/0.846449 for A/B. These are
first-update merit passes, not independently admitted alternate outer windows.

All evaluated trials preserve the prepared inputs and live controls/moments
exactly. The233796 start-pinned IDs have exact raw and promoted anchors.
A-repeat at index4 has identical control deltas; maximum promoted-coordinate
difference is2.38419e-7wu and both merit differences are4.65661e-10. These
descriptive repeat measurements neither set new tolerances nor clear the
previous failed strict primitive gates. Independent isolated CPU review passed
24 cases (20 driver,4 observer) before this run.

The reference below is the initial prepared forward rollout and its promoted
endpoint, not the starting cloud x0. All motion columns use the same66204 free
material IDs, normalized by source native spacing0.03498853660707278wu.
Raw steps include the existing layer operator; saved final means raw final
step plus PIC, not PIC alone.

| Motion RMS (sp) | Initial rollout | A selected | B selected |
|---|---:|---:|---:|
| Raw steps1–19 |0.0218215|0.0236004|0.0234008|
| Raw final step |0.0208691|0.0276144|0.0472123|
| PIC jump |0.226904|0.238826|0.170210|
| Saved final phase |0.214979|0.232523|0.172924|
| All saved steps |0.0525659|0.0568549|0.0448927|
| Saved net displacement |0.300736|0.246761|0.191329|

B reduces saved-step RMS21.04% and PIC RMS28.73% relative to A, while raw-final
RMS rises70.97% and all-raw-step RMS rises5.52%. Raw-path reversal fraction
rises0.7623% to1.7344%; saved-path reversals fall2.6721% to2.4386%. The saved
path-length RMS falls12.52%, net-displacement RMS falls22.46%, and median
net/path falls0.4575 to0.2863. Thus the smaller saved motion accompanies less
net transport and more raw direction changes; it is not evidence of rest.
For the1801 source-upper free IDs, B similarly lowers saved RMS11.17% but
raises raw-final RMS51.49%, with raw reversals0.5728% to1.4407%. These fixed
free cohorts are not certified after-arrival cohorts, and this one-window
measurement does not observe the next-window boundary response.

| Promoted endpoint geometry | Initial rollout | A selected | B selected |
|---|---:|---:|---:|
| Silhouette IoU |0.969281767|0.969100325|0.969018017|
| Fixed upper target coverage |0.964668234|0.958398642|0.958202717|
| Target-tip neighbors |59|42|42|

Independent geometry is nearly equal between selected A/B, with B slightly
worse on silhouette and upper coverage. Both lose coverage and tip neighbors
relative to the initial rollout. The binary-projection hole metric is0 for
all three; it does not certify absence of physical holes. Fixed-source upper
6712-ID density is0.69973555/0.69953069 for A/B and under-half support is
0.23897497/0.23882598. For the1801 free subset, density is
0.82849806/0.82842865 and under-half0.15880067/0.15824542: the latter difference
is one ID. The fixed density radius remains target median eighth-neighbor
distance0.06898659982768912wu (target native spacing0.03493084911867985wu).
Initial-rollout fixed-ID supply was not recorded. Current-top density uses
position-dependent membership and is not a substitute for these fixed cohorts.

Backtracking removes the rejected-large-step confound and demonstrates that
both directions have a first merit-passing proposal at the same step size.
The observed tradeoff remains mixed: lower geometric merit/saved motion,
greater final raw motion and reversals, less net transport, and no thin-region
coverage improvement. No default, new weight, physical-hole/rest claim, or
clearance of P297's whole-policy failures follows. Constitutive/primitive
correctness must be resolved separately before treating this as a repair path.

Evidence: local output/p299/backtrack24/{protocol,proposal_audit}.json;
server /data/relcfd/chayo/physmorph_v2/work/p299/backtrack24.
Result SHA256:5152e561156407ad4eeae7017a9884b36c39a9e5c7e77dea264ee70ae0bcf187.
Protocol SHA256:0809b5050988507a8cbd05adec5277e69220734b3e540d776a23f17c88e6075e.

## Corrected-core recheck after P300

Frozen0ac2a2f reruns the same bounded first-update experiment on the corrected
constitutive adjoint. The historical failed primitive records and their False
flag are preserved with explicit historical scope; separately hash-bound P300
prerequisites pass52 original position-sequence checks and10 constitutive CUDA
tests. Every dependency recorded by those prerequisites matches the executed
source; the new proposal driver's own byte hash is recorded separately. The
protocol-only change passes25 new and20 existing CPU tests, with independent
review; objective, proposal, PCGrad, gain and backtracking calculations are unchanged.

This is a fresh baseline realization through accepted W24: N300k, T20,
dt1/240, dx0.3062907543956724wu, loss36^3, eight inner updates, shared PIC,
promoted outer render, shift off, physical variance. Motion accounting is off
as in this diagnostic's original protocol; P300's separate cap24 comparison
has that read-only accounting on. There is no shared realized trajectory or
arrival cohort between these runs. The baseline accepts24/24 in423.296s with
zero guards; the selected observation, restored state, baseline lambda and
actual first accepted alpha checks pass.

At the same prepared state, lambda is0.03044188680154145 and the fixed cohorts
contain63513 window-start free IDs and1621 source-upper free IDs. Both native
alphas are0.02; both A physical-variance and B geometric-variance directions
reject indices0..4 and first pass their own merit at index5, alpha0.000625.
A's selected alpha matches the production continuation exactly. Only A is
continued by the actual optimizer; B is a noncommitting first update.

| RMS on the same63513 free IDs (source spacings) | A | B |
|---|---:|---:|
| Raw steps1..19 |0.0175204|0.0142758|
| Raw final step |0.0165796|0.0255686|
| All raw steps |0.0174746|0.0150431|
| PIC jump |0.188093|0.156481|
| Saved final phase |0.181135|0.150499|
| All saved steps |0.0439558|0.0364156|
| Saved net displacement |0.209848|0.135940|

Source spacing is0.03498853660707278wu. B lowers saved RMS17.15%, all-raw RMS
13.91% and PIC RMS16.81%, while increasing raw-final RMS54.22%. Raw-path
reversal fraction rises0.00205180 to0.00613221; saved-path reversal falls
0.0307455 to0.0234201. Net-displacement RMS falls35.22% and median net/path
falls0.539703 to0.332290. Lower movement may also reduce transport; signed
delivery or mass flux was not measured. Motion is redistributed into the last
raw step, not demonstrated rest. Terminal stored-v L2 also rises33.9386 to
53.3728, despite the lower saved and full-path stored-velocity RMS.
On1621 upper-free IDs, saved RMS falls0.0841193 to0.0774162sp while raw-final
RMS rises0.0191799 to0.0252793sp; the tradeoff is not confined to the interior.

| Promoted geometry | Initial prepared rollout | A | B |
|---|---:|---:|---:|
| Silhouette IoU |0.969131|0.969318|0.969155|
| Fixed upper target coverage |0.958333|0.962709|0.961795|
| Highest-tip neighbors |50|48|48|

B's Chamfer is slightly lower (0.0588547 vs0.0588566wu), but IoU, upper
coverage and fixed6712-ID density are slightly worse than A. Fixed density is
0.717856/0.717744 and under-half fraction0.231377/0.231824, using the unchanged
target r8 radius0.06898659982768912wu. Both proposals improve coverage over
the initial prepared rollout and both lose two tip neighbors; retain both facts.
Trajectory det(F) minima are0.874815/0.875441. Binary projected hole_frac is0
for both, which does not certify transient coverage or watertightness.

Repeated A has identical control deltas and maximum promoted-coordinate
difference2.38419e-7wu. These observations do not set new tolerances. The
changed physics direction also changes PCGrad's reference; fixed lambda does
not isolate a pure regularizer effect. Constant uniform drift still has zero
temporal velocity variance. No next-window response, individual post-arrival
rest, full alternate solve or policy promotion follows from this one update.

Evidence: output/p300/proposal_corrected24.json and
output/p300/proposal_corrected24_protocol.json; server work/p300/proposal_corrected24.
Result SHA256:ba9a67d46f2634df91bbaf5ac231e836e1f40a69be427caa0cf3457ad0b18713.
