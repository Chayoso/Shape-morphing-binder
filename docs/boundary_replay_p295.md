# P295: conditional first-step replay across a PIC commit

Status: pre-registered diagnostic, not a proposed controller or accepted repair.

P294 found unusually large displacements at the final archived step and many
direction reversals at the next window's first step. Its endpoint is PIC-filtered.
Older P292 no-PIC comparisons retained `shift_sub=True`; they did not remove every
external position correction. Neither result establishes a causal handoff defect.

## Fixed experiment

Use the immutable mixed60 bunny source: N=300000, T=20, dt=1/240,
dx=0.3062907543956724 wu, loss grid36^3, eight inner iterations. Start a fresh
instrumented run with the P294 control recipe: shared inner/committed PIC,
`shift_sub=False`, corrected outer render merit, geometric-rest term off.
Keep mixed body/stress control, layer control/relaxation and render guidance.
The cap is60 attempts under the original300-window schedule.

Before observing results, select accepted boundaries1,24,30. A cap2 run of
boundary1 tests the capture/replay plumbing first; it does not replace the full
run's selected boundaries. If a selected boundary or its next accepted window
is absent, report it as missing; do not choose a more favorable boundary.

An optional read-only optimizer observer copies the final validated inner
trajectory before gradient-dump diagnostics can overwrite its buffers. The
runner's outer accepted callback admits the packet. Rejected attempts are
discarded. Copy the next window's *actual* state and controls, after previous
assimilation, new pins and effective layer-u gating. The diagnostic writes only
selected complete first-step packets and metadata, not another full movie archive.

## Intervention and closure

Let x_raw be the prior raw endpoint, x_pic its promoted endpoint and j=x_pic-x_raw.
Replay the next accepted window with its original T; call only `step(0)`.
Changing T to1 would change the body impulse and layer/bond coefficients.

- A: actual next start x_pic.
- A repeat: identical input, to measure atomic/replay noise.
- B: x_raw on next-window free IDs; x_pic on **all** next-window pins.

Both branches retain identical F, Fp, v, C, rest volumes, material fields, dFc[0],
body modes, u, frozen layer data, bond graph/rest/fragment state and pin anchors.
Current-density bond activation and P2G weights are recomputed from each x0;
their differences are legitimate mediators, not a claim of identical forces.
Only first-step dFc is needed; the saved packet is not a full-window checkpoint.

First require A to reproduce the original accepted trajectory's x1, pre-layer
position, v1, F1 and C1. The elementwise diagnostic tolerance is
3e-6 + 3e-5*abs(reference). Also compare displacement-response RMS to both
original-closure RMS and A-repeat RMS; mark whether it exceeds10 times their
maximum, separately for all-free and layer-free cohorts, using vector RMS.
Every branch must also preserve pinned positions and pre-layer positions exactly,
zero pinned v/C, and equal pinned F. Numerical or pin-contract failure blocks
causal interpretation. A small signal
is reported as small or unresolved, not converted to a positive finding.

The induced next-step response is **(x1_A-x0_A)-(x1_B-x0_B)**.
The endpoint difference x1_A-x1_B includes the imposed initial j and is not this
response. Report same-ID all-free and layer-free cohorts; magnitudes in wu and
native source spacings; projections onto -j; and boundary-to-next-step reversals
when both step lengths exceed1e-4 spacing. Report both the whole alternative
boundary (its preceding vector also changes) and the next-response comparison
holding the preceding saved vector fixed. Components are dt*v1,
pre_layer-x0-dt*v1, and x1-pre_layer. The middle component includes update/bond
effects and must not be mislabeled a pure bond or pure physical term.

## Scope

This is a position intervention conditional on the actual post-PIC state, controls
and admitted pins. It is not the state or controller a full no-PIC history would
have produced. It cannot establish a state-consistent F/v/C correction, explain
all oscillations, certify hole removal or measure visual quality. No renderer
enters these metrics, and no production default is changed.

Capture and replay run on hyde06 CUDA with frozen source hashes. Local checks
are CPU tests only. Existing raw archives remain evidence; monitor the100GB
project-storage threshold before each launch.
