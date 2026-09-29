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

After capture passes, hold its trial05 terminal coefficients fixed. Optimize
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

## Capture1 failed; compensation has not run

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
