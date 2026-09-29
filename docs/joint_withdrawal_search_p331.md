# P331: bounded joint body-control search with withdrawal and shape guards

P329 establishes production-sized head/coast closure; P330 supplies the complete
original head-merit gradient. Neither changes control. This experiment searches
one fresh live window for an actual better pair of displacement/terminal body
coefficients, without replacing the accepted production state.

## Frozen execution and scope

Use the P329 recipe/input contract: N300000,T20,dt1/240,
dx.3062907543956724wu,loss36^3,iters8, source attempt20/inner8.
The only driver override is stop_after_windows=20. Earlier ordinary stops and
rejections remain authoritative. PIC/shift off, geometry-F/GS/material
optimization off, existing layer relaxation and control on. Original rendering
remains18views64px. Observe the P329 original-control capability first, preserving
its accepted/private/joint witnesses and original outer disposition.

No new pin admission, projection, material update or candidate adoption occurs.
Passive continuation uses the existing pre-assimilation frozen Fp/pins/layer/
bonds, not the runner's actual next-window preparation. A found direction still
needs candidate-specific actual handoff, whole-state rollback/continuation and
full-horizon/gallery gates before promotion. Arrival is not a rest label.

## One fixed origin and finite search budget

Let c0=(d0,b0) be the actual accepted M-node two-mode body coefficients. Evaluate
three fresh joint forwards at c0. Each must satisfy unchanged P329 accepted full
X/V, terminal F/C, original scalar merit and health gates. The final repeat owns
the graph used for derivatives; no earlier graph is differentiated after replay.
All other controls, basis/gates, reference, lambda and incoming state are frozen.

The primary cost G is mean squared geometric speed over all20 coast steps and
the fixed window-start-free material IDs. Include coast0->1. Track coarse-
arrived-free IDs separately; a population mean decrease cannot certify each ID's
rest. Empty primary cohorts or missing/nonpositive accepted update radius are
inconclusive, never zero-motion success.

Eight frozen first-order inequalities constrain head volume, combined render,
silhouette, complete original merit; coast-end volume, combined render,
silhouette; and mean stored coast speed squared on that same primary cohort.
Use the original rounded scalar evaluator for merit values and P330 tensor merit
only for its covector. PreparedReference supplies the separate image terms.
Ceilings are the maximum of three original observations plus
32*FP32eps*max(abs(ceiling),1e-12). The head-merit ceiling is additionally capped
by the original accepted merit with that same allowance. No new blend weight,
lambda/PCGrad update, terminal-strength override or data-reference rebuild occurs.

The initial global coefficient radius is
R0=sqrt(M)*sqrt(update_rms_displacement^2+update_rms_terminal^2), using the
actual accepted inner update's per-mode node-vector RMS. This is not per-scalar
RMS and has no extra factor3. For h=0..10 solve the existing affine trust-ball
linear problem at R0/2^h. Project the proposed per-node joint6-vector onto the
production unit ball, then recheck the actual projected coefficient difference
against affine bounds, radius and a resolved negative directional prediction.
Require -gradient dot actual_delta greater than the same32eps objective floor
used for the actual decrease below; a merely negative sub-floor prediction
does not consume a forward.
Keep c0/Jacobian/reference fixed; no rejected iterate becomes a new origin.

At most11 trial forwards are permitted. Actual nonlinear constraints and full
health must pass, and G must decrease below the minimum original G by the same
32eps precision floor. A nonlinear failure is recorded; raw geometry may be
marked not measured, never passed. The first candidate satisfying all measured
nonlinear and raw gates receives three same-control confirmation forwards. All
must pass. Stop after that confirmation even if one fails; do not shop for a
different candidate or widen a gate. Report per-ID energy changes and worseners
for both fixed cohorts alongside means. This is not an individual-rest certificate.

## Raw shape and material supply gates

The separate archive/observation callback consumes raw positions only, with no
renderer or loss operator. Evaluate every distinct head and coast position:
x0,20 controlled steps,20 further passive steps (41 positions). Derive all
cohorts/scales from fixed original source/target geometry. Three baseline
forwards establish per-phase envelopes; candidate checks require all three.

At each phase require per-view binary-splat IoU no lower than the baseline
minimum, absolute enclosed-hole pixels and clipped particle centers no higher
than their baseline maxima. Use fixed target extent,24views,128/256px and the
independent3x3 binary footprint. The azimuths are j*2pi/8 for each elevation
0,+.5,-.5 radians, without stagger, matching P329. Retain absolute hole counts
and report enclosed-hole pixels outside the target's enclosed-hole bitmap;
the latter differs from P329's max(body_count-target_count,0), is report-only,
and must not be compared as that old count. Count nonworsening does not imply
hole-mask inclusion or3D watertightness.

Target support uses a2-target-native-spacing radius. Every target ID covered in
all three baselines must remain covered; equal aggregate coverage cannot trade
away those IDs. Global/upper(y>2.3) support and tip-ball(.25wu) count must also
not fall below baseline minima. The fixed source-upper/sparse IDs use y>=source
bbox midpoint and2-source-spacing neighbor count<.6source median. Their
self-excluded neighbor count at target median8NN radius, divided by8, must have
no lower mean and no greater fraction below4 neighbors than baseline bounds.
This cohort includes pinned IDs; report their number separately.

Integer comparisons are exact. Floating raw aggregates permit only
64*FP64eps*(1+abs(bound)); original repeat variability is handled by the measured
envelope, not by a new pixel/particle tolerance. Preserve baseline envelopes,
stable target-coverage masks and source cohort IDs for independent replay.

## Evidence, limits and decision

Record each produced forward's controls, complete owned head/coast witnesses,
scalar/health/closure/raw decisions and per-ID energy evidence before another
forward can replace its buffers. A failure in scalar/raw evaluation must still
preserve the produced state. Bind all inputs, code, configuration and output
files before/after. Numerical work stays on CUDA; NPZ/JSON I/O uses the host.
Reserve12GB additional output under the100GB project cap. Worst-case uncompressed
payload exceeds this cap; compression is not guaranteed. A budget failure is
explicitly inconclusive, preserves prior witnesses and cannot adopt a state.

Report the original optimization's rendering-direction telemetry, fixed-lambda
head/coast silhouette/PBR/render changes for every evaluated candidate and their
scope. A feasible direction is not a full morph repair, exact post-handoff
derivative, independent causal rendering effect or4K footprint fix. No delivered
video/page is replaced on this experiment alone.

## Pre-execution verification

Independent adversarial review passes47 CPU cases:21 search,22 raw observation,
4 driver/archive-budget cases. Real small CPU rollouts preserve original X/F,
optimizer history, rendering telemetry and outer accepted state, including a
failed raw-envelope archive. Compile and shell syntax checks pass. This is not
the production CUDA/quality result.

Resolved review findings before execution: recheck both actual projected trust
radius and affine bounds; resolve the predicted decrease above the stated
precision floor; align the24 raw views exactly to P329; distinguish hole-count
gates from hole-mask inclusion; reserve actual UTF-8 JSON growth, keep rendering
summary inside guarded result serialization, and let only small failure records
consume the emergency reserve while still enforcing the absolute12GB cap.
The authoritative merit remains the original Python-float callback.

## Production result: no admissible candidate

Frozen a14fa74, work/p303/p331_search1, completed734.63s including evidence I/O.
The native discretization remains N300000,T20,dt1/240,dx.3062907543956724wu,
loss36^3,iters8.20 original windows/160 inner updates commit, all guards zero,
and the original callback state/returned state/outer endpoint checks pass.
This is a fresh run, not a bit-identical replay of P329's earlier optimized
prefix. Its fixed start-free and coarse-arrived-free cohorts contain41631 and
41429 IDs respectively. Neither label certifies individual rest.

All three original repeats pass. Ten candidate forwards(h00..h09) lower mean
coast geometric speed squared, but none passes the complete raw gates. The
last proposed h10 step fails its post-projection affine silhouette check before
forward: residual5.22075e-12 exceeds precision allowance3.17680e-12. No candidate
is selected, confirmed or adopted. The bounded one-origin search is not a proof
that no feasible joint-body solution exists.

G below averages all20 coast steps and the same41631 free IDs, in(wu/s)^2.
Percentages refer to G, not speed or net displacement. The maximum lost-ID count
is per phase relative to target IDs covered in all three baseline repeats.

| Candidate | G | G reduction | All8 nonlinear gates | Max lost stable target IDs |
| --- | ---: | ---: | --- | ---: |
| Original minimum | .0410806673 | -- | reference | 0 |
| h00 | .0146659586 | 64.30% | fail | 72 |
| h04 | .0369920341 | 9.95% | pass | 14 |
| h09 | .0409461128 | .328% | pass | 1 |

h04..h07 and h09 satisfy all eight nonlinear constraints, including complete
head merit and head/coast rendering terms, yet fail independent raw supply/
projection gates. h00..h05 also increase enclosed-hole counts in some phase/view
comparisons; later candidates still fail other raw gates. Reported gate
occurrences are not counts of distinct3D holes. The smallest one-ID violation
is not by itself a demonstrated visible artifact; it fails the preregistered
no-loss-of-covered-ID criterion. No threshold was widened after observing it.

Output is10.267GB, below the12GB reservation. The external process monitor
observes32906MiB process/32930MiB device peak, with no query errors; sampled peaks
are not exact high-water marks. No new video or quality promotion follows.

Rendering influence in the same discretization:18views64px, shared GS loss off.
Across160 accepted inner updates, nominal render-direction share median is
.50188177(body .36551093, stress .55989686, surface-u .75637753); adaptive lambda
median over20 window records .06813912, frozen checkpoint lambda .02161115,
median observed image-loss
change per update -.0001483256. These are optimizer observations, not causal
movement fractions. Search constraints reuse the fixed prepared reference and
lambda; passive coasts invoke no control optimization/render feedback. These
image terms do not supervise the exact exported4K covariance. F/footprint
limitations remain in gaussian_footprint_followup.md.

Independent archive audit passes6269 checks. It reconstructs all41 raw phases
for all13 produced forwards, coast/per-ID energies, full health/pins, original
closure and candidate decision arithmetic. Source/input/output identities are
bound before/after. Raw projections use a separately written binary scatter;
bounded KDTree/fill libraries are shared, with additional blocked FP64 samples.
Prepared image/density replay reuses frozen loss primitives. Complete physical
merit is not independently reconstructed because NN/DT/control-neighbor
auxiliary bindings were not archived; rounded recombination and available
kinetic/body terms are checked. Optimizer/outer isolation is corroborated
metadata plus the separate CPU identity tests, not a second optimizer replay.
There is no full-size finite-difference, individual-rest or watertightness claim.

The next prerequisite is an original-result identity gate for a runner-owned,
whole-trajectory experimental selection seam, before any candidate-specific
actual handoff/normal successor. Do not continue a least-bad failed candidate.
See candidate_commit_contract.md; actual Fp/pin/layer/bond changes cannot be
replaced by this frozen pre-assimilation coast.
