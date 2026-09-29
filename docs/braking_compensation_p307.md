# P307: preserve endpoint geometry while retaining body-mode braking

P306's trial05 reduced terminal speeds but failed the original density and
coverage gates. The terminal pulse has zero endpoint displacement only for a
free particle. The observed cross-response belongs to the full MPM/layer/pin
system; the experiment did not isolate elasticity or damping as its cause.

## Frozen-state capture prerequisite

P306 sidecars cannot reconstruct the entire prepared W20 state. Capture a fresh
realization of the same N300k,T20,dt1/240,dx.3062907543956724wu,loss36^3,
budget8 raw/no-PIC/no-shift run. Regenerate its own baseline/trial05; do not call
it the exact P306 continuation. Save every RolloutSpec field, body basis/gate,
stress/u and body coefficients, prepared reference, initial/accepted state,
cohorts, source/target and chosen terminal candidate in numeric NPZ members
with an explicit JSON tree. No pickle or executable classes are loaded.

Before reuse, loaded baseline and loaded trial05 must reproduce X/V/F, prepared
data terms and terminal-coefficient derivatives of brake/density/render. Terminal
C is also checked, with characteristic unit1/(T dt). Arrays
use32float32 eps times (physical unit+abs reference), data32eps relative, and
gradient error norm64eps times the reference norm (floor1e-12). These are
closure gates, not physical quality thresholds. The capture alone establishes
no compensation efficacy. Source/input/archive hashes bind the witnesses.

## Bounded compensation design

After valid capture, or in the fresh callback alternative below, hold that
realization's trial05 terminal coefficients fixed. Optimize
only displacement coefficients toward the original accepted endpoint through
the same actual MPM rollout; no post-rollout position replacement. Use radius
sqrt(max(0,1-|terminal05|^2)) for displacement, never a joint rescale that changes
the retained terminal coefficients. Stress/u, pins, initial state, references,
layer/bond topology and lambda remain fixed.

Use at most4 gradient-descent updates of full same-ID endpoint squared error.
Each trial starts at the RMS magnitude of the last original displacement-mode
update and halves at most10times; require a finite positive-orientation raw
trajectory, exact pins and endpoint-error reduction beyond the measured replay
floor. No density/brake weighted surrogate is substituted. Record every trial,
including rejections, then apply the unchanged P306 strict data, shape/supply,
both-cohort motion and resolved stored/geometric speed gates. A frozen terminal
coefficient does not guarantee the actual terminal speed stays reduced.

Only a candidate passing all gates can proceed to total-merit reconstruction
and actual coupled-window continuation. Failure is limited to this bounded
subspace/search; success is not passive equilibrium or a completed morph.

## Original capture1 failed

Frozen `code_braking_capture1` completed its new W20 trial schedule and wrote
`work/p303/braking_capture1/owned_window.npz`. The save/reload witness comparison
passed X/V/F checks, then failed the newly included C gate before the final
data/gradient assertions. No `result.json` was issued. The failed original
in-memory C witnesses were not preserved; the failure cannot be reclassified
using a later replay. Its log/protocol are retained in `docs/evidence/p307`.

Read-only `replay_noise1` reuses the retained archive, comparing three repeats
each for baseline/trial05 using Torch initial arrays and CuPy initial arrays.
All compared X/V/F, data and gradient ratios meet the declared gates. One
cross-representation trial05 pair has C maximum error5.0724e-5 s^-1 and RMS
1.9595e-7 s^-1, with two of2.7million C elements exceeding the original gate
(maximum ratio1.0763). Same-model Torch repeats have C ratios at most .7615.
This does not isolate serialization error from fresh-model floating-point
variation; the original failed comparison remains unavailable.

No C alias or stream-order error was found by static review. C is the direct
G2P affine velocity-gradient sum; F receives dt*C followed by smoothing, so
their floating-point sensitivities need not match. That is a hypothesis, not
a resolved cause. Thresholds remain unchanged and compensation is not launched.
The next bounded check must preserve all witnesses before asserting, compare
exact initial state/control/Warp buffers, test C snapshot immutability across
adjoints/next forward, and distinguish same-model from fresh-model repeat
variation. The existing archive can support that check without repeating W1-19.

### Preregistered identity/ownership discrimination

One fresh model from the retained archive is converted to CuPy initial arrays,
saved once, and reloaded. Compare every numeric spec/model/observation array
bitwise and all primitive/tuple fields exactly. Compare the resulting Warp
initial positions/velocities/C/F/Fg, material/mass/volume/Fp, pins, layer/bond
buffers and expanded controls between both arms. Three repeats of each arm's
baseline and trial05 expose C primal/snapshot and input-buffer immutability
across each adjoint and the next forward. Persist full X/V/F/C and all three
gradient vectors for every witness before closure decisions. Compare all15
pairs per control (within and cross instance) at the existing32eps/64eps bounds.
This is one new reuse validation, not a retry of the unavailable original
in-memory witness; failures and within-instance noise remain in the result.

### Identity1 result and fresh-callback alternative

`state_identity1` completed all12 witnesses and30 pair comparisons at the same
N300k,T20,dt1/240,dx.3062907543956724wu,loss36^3 discretization. All75 arrays
(313,741,143 elements) and213 primitive fields are exact; Warp/static initial
buffers, C primal/public ownership and non-vacuous next-forward checks pass.
X/V/F/data/gradient gates pass in every pair. C alone fails in8pairs, including
2of12 within-instance pairs. The largest C ratio is2.0384 (absolute9.6053e-5
s^-1); each failing pair has2of2.7million components outside the original bound.
This establishes that archive serialization is not necessary for this failure.
It does not prove an atomic cause or pass the failed capture/repeatability gate.

Preregistered alternative: run the identical four-update/ten-halving compensation
directly in a fresh real W20 callback. Generate that realization's own trial05;
do not load an old window or candidate. Keep the existing private/accepted
X/V/F/data closure, replay-noise endpoint acceptance and every P306 quality gate.
Archive full inputs for evidence only, not as an admitted restart. Verify the
production X/V/F/C, control leaves and Adam moments exactly after the callback.
The observer payload is owned and may itself be mutated; isolation checks use
separate private snapshots. No default, penalty, threshold or commit changes.

This alternative is cleared only because exact inputs/ownership and all non-C
checks passed, while same-instance C repeatability failed. Bind that complete
evidence before/after the run. The old C gate stays failed. If compensation
finds a feasible same-window candidate, test actual coupled continuation with
its complete x/v/C/F and normal assimilation before any persistence claim.

### Live compensation2: endpoint repair does not satisfy all gates

The first live launcher failed before simulation because one helper provenance
path was relative under run_path. Its code/log survive. Resolving that helper
path allowed `code_live_compensation2` to complete the fresh W20 experiment.
Same N300k,T20,dt1/240,dx.3062907543956724wu,loss36^3,budget8 raw recipe:
20 windows,160 accepted inner updates, all trajectory guards0 and exact
post-callback production X/V/F/C/control/Adam isolation. This fresh realization
has52,604 start-arrived-free and52,937 start-free IDs; no endpoint re-selection.

All4 endpoint updates are accepted from25 line-search trials. The final endpoint
MSE falls2.3549866e-7 ->4.9829588e-8 relative to its own terminal05, a78.84%
reduction. All4 compensated candidates nevertheless fail the unchanged gates.

| Fixed arrived-free / geometry observation | Own baseline | Compensation4 |
| --- | ---: | ---: |
| Stored terminal RMS (wu/s) | .200181067 | .147227928 |
| Geometric terminal RMS (wu/s) | .200474546 | .147948316 |
| Net RMS (sp) | .460356698 | .459135201 |
| Saved-step RMS (sp) | .0243444984 | .0243835735 |
| Path mean (sp) | .401150745 | .394993644 |
| Prepared volume loss | .001973961480 | .001973978477 |
| Fixed-source upper density | .963852056 | .963796186 |
| Independent silIoU | .962921666 | .962932240 |
| Upper target coverage | .934365204 | .934430512 |
| Overall target coverage | .993056667 | .993060000 |
| Tip count | 72 | 72 |

Speeds decrease26.45%/26.20%, net/path decrease, but saved-step RMS increases
.1605% and the volume/density gates fail. Compensation1/3 improve prepared
volume yet also increase net/step motion. A retrospective application of the
same P306 checks to all25 logged trials finds no overlooked feasible trial;
every trial loses fixed-source upper density. That is an audit of this bounded
search, not a proof that displacement control cannot satisfy the constraints.
The final density decrement represents3 aggregate neighbor-count units over
6712 fixed material IDs; the best rejected trial loses1. This strict gate
failure alone is not evidence of a substantial new visible hole.

Rendering: W20 lambda=.02134737552. Final compensation reduces silhouette
7.00937e-7 and PBR1.66474e-8, combined7.17584e-7 and weighted1.53185e-8.
Both components improve here, while the physical gates fail. Across20 original
committed windows/160 inner updates, median nominal render-direction share is
.492055 and median lambda=.0649134; these are not causal displacement shares.
Compensation directions themselves minimize endpoint error, and prepared render
loss screens candidates. Exported4K appearance was not evaluated in this probe.

No compensated state was committed. C repeatability remains failed as described
above. A smaller endpoint error alone is insufficient evidence of path/rest or
hole improvement. The next candidate must control within-window movement and
prepared data directly, while keeping independent supply/geometry gates intact;
no extra endpoint iterations, new stopping policy or default is justified here.
