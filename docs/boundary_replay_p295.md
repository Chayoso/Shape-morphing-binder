# P295: conditional first-step replay across a PIC commit

Status: completed diagnostic. The endpoint jump is the dominant measured boundary
discontinuity; the immediate induced response is mainly layer relaxation. No
controller repair or hole-removal claim follows from this one-step experiment.

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
position, v1, F1 and C1. Protocol v2 compares dimensionless fields x/dx,
dt*v/dx, F, and dt*C, using elementwise tolerance
3e-6 + 3e-5*abs(normalized reference). Also compare displacement-response RMS to both
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

## Pilot closure correction, before the full run

The first cap2 capture (snapshot976a895,2026-09-27 03:42:10UTC launch) completed
two accepted windows with all state guards zero. Its replay is retained as a
**failed v1 closure check**, not silently reclassified. x1/pre-layer/v/F passed;
C failed the shared native-unit absolute tolerance (maximum ratio1.907).
For this same N300k/T20/dt1/240/dx0.3062907544wu/loss36^3 discretisation,
C's maximum component discrepancy was9.54e-6 1/s, while identical-input repeat
C had maximum Frobenius discrepancy8.44e-6 1/s. The original and repeat C
Frobenius RMS discrepancies were2.41e-6 and2.19e-6 1/s respectively.
Position closure maximum was2.38e-7wu.

A common absolute tolerance across position, velocity, strain and velocity
gradient is unit-dependent. Version2 fixes this by the dt/dx normalization above,
with unchanged dimensionless constants. This **changes native-unit strictness**:
at this discretisation the absolute C allowance becomes7.2e-4 1/s (240 times v1),
velocity2.2053e-4 wu/s (73.5 times v1), and position9.1887e-7wu (stricter than v1).
Relative tolerance is unchanged. Tests require invariance under consistent
length/time-unit changes and rejection of identical dimensionless errors in each
field. Keep the failed log/protocol/packet, review this revision, and perform a
fresh cap2 capture/replay before the preregistered full boundary selection. No
simulation equation or intervention is changed by this diagnostic correction.
The early boundary has zero pins, so its pin check is vacuous on GPU; later
boundary evidence must disclose actual checked pin counts.

The fresh v2 pilot uses snapshotbe2ef0e. Both accepted windows completed in28.20s
with all state guards zero. Every dimensionless first-step field comparison
passed (maximum tolerance ratio0.009954), while the position maximum remained
2.38e-7wu. Both all-free and layer-free responses exceeded their measured
original/repeat floors by more than10 times. This validates the diagnostic for
the preregistered late-boundary run, not a controller change or shape-quality claim.

## Full result (captured2026-09-27, recorded2026-09-28)

The same frozen be2ef0e snapshot completed33 accepted/36 attempted windows in552.25s,
ending after three outer-merit rejections. All state guards were zero, and the
three preselected boundaries were available without substitution. Capture ran
03:52:26–04:01:38UTC; comparison replay ran04:02:33–04:02:36UTC.

Every original first-step and pin-contract check passed. The maximum position
discrepancies were2.38e-7,2.38e-7,1.19e-7wu at boundaries1,24,30. Late comparisons
held257040/264662 active pin anchors exactly, with zero pinned velocity/C and
equal pinned F. Both free cohorts cleared the measured original/repeat noise
gate. Bond activation did not change in any contrast.

All numbers below are bunny300k, T20, dt1/240, dx0.3062907544wu, loss36^3,
source-native spacing0.0349885366wu. The layer-free cohort uses NEXT-window
unpinned IDs and its frozen layer mask; cohorts differ across boundaries.
RMS uses per-particle vector norms. Component RMS values are not additive shares.

| Accepted boundary | Layer-free IDs | PIC jump RMS (sp) | Induced next response RMS (sp) | dt*delta-v RMS (sp) | Induced layer update RMS (sp) |
|---|---:|---:|---:|---:|---:|
|1|21163|0.57712|0.02016|0.01289|0.02127|
|24|5150|0.56031|0.02085|0.000844|0.02107|
|30|4600|0.47987|0.02026|0.000403|0.02028|

At late boundaries24/30 the prior raw final-step RMS was0.03255/0.02030sp. The PIC
jump is much larger than either that motion or its induced next-step response.
The immediate response opposing the jump is primarily the positional layer term;
the advection response projects slightly in the opposite direction. The fixed u,
gate, normals and fraction are identical across branches, so the change in layer
update is attributable to its geometry-dependent relaxation plus arithmetic
roundoff. This does not identify how much of the PREVIOUS window's jump came from
its layer corrections; that requires a separate whole-window decomposition.

| Boundary | Reversal, original saved boundary | Whole alternative boundary (both vectors change) | Same preceding saved vector, altered next response only |
|---|---:|---:|---:|
|24|93.786%|0.408%|86.738%|
|30|97.391%|0.848%|93.957%|

Each row uses the same eligible IDs within its comparison, with both vector
lengths above1e-4sp. The dramatic whole-boundary difference mainly removes the
explicit preceding position jump. It must not be presented as suppression of
the next physical response, nor as the result of a full no-PIC optimization.
The next-response change is real and resolved, and is substantial relative to
the small next-step movement, even though it is small relative to the jump.

The small pre-layer residual component is near subtraction-roundoff scale; do
not give it a signed mechanical interpretation from the total-response noise gate.
This evidence weakens the hypothesis of a large immediate MPM elastic rebound.
It does not exclude later rebound, prove a complete repeated feedback loop, or
establish the cause of every drift/artifact. No hole, fit or final-rest quality
gate was evaluated by this diagnostic.

Artifacts: server `/data/relcfd/chayo/physmorph_v2/work/p295/full60_v2/`, local
JSON copies `output/p295/full60_v2/`. Protocol/capture/replay SHA256 respectively:

- `ffafb5b37f7e5d284b07ff11ea207a6588296edafefc90c08d91c5a4a0921f66`
- `821d998b24fec8ad142802102ec71c0103d98f010709643a78dff90f7b83ce2b`
- `22d770671e0ea1a21a977e9600ac40650002e4c73216a71c68f34d3185b22ca6`

The three first-step packets occupy2,077,233,764bytes. Project usage after replay
was98.205GB, below the100GB cleanup trigger; no complete movie archive was added.

## Next decomposition and possible repairs (not implemented here)

Before choosing a repair, decompose the previous whole-window displacement as
D=A+R, where A=dt*sum(V) is accumulated advection and R contains bond, u and layer
position corrections. Apply the SAME frozen H and output pin mask Q separately:
J_A=Q(H-I)A and J_R=Q(H-I)R. Their sum must reproduce the actual PIC jump. Measure
their cross term and each direct-position channel; the next-step layer response
above does not prove that J_R dominates the prior jump.

If this test supports it, a more direct candidate is to retain R at commit and
filter only A: y=x0+Q(HA+R), with pins restored exactly. It avoids filtering away
the subcell position corrections that the layer channels were introduced to
produce. It does NOT guarantee a smaller jump: A and R may cancel under the old
map, and separating them can expose a larger J_A. It also does not make F/v/C
state-consistent by itself. Thin-region coverage, target fit and signed
cancellation must be checked before any optimization comparison. The raw
pullback would be Qg and each velocity pullback dt*(H^T-I)Qg; Q stays outside H.

Another bounded candidate is the temporal-observable change below.

If late evidence also points primarily to the explicit endpoint jump, the next
bounded candidate is to make the existing temporal variance term observe the
actual positional path. Current `w_kin_var=200` uses physical V; it does not
observe every layer/bond/PIC position change. Define U_t=(y_t-y_{t-1})/dt,
where y_t=x_t before T and y_T is the shared promoted endpoint, and change only
that variance input in a matched one-window solve. Retain physical V for physical
kinetic/contact/continuity terms; U is a geometric rate, not reconstructed momentum.

This would require the full position-sequence adjoint and validation against
actual archive states. Hold target/arrival/pins and lambda fixed for the first
matched solve, check independent thin-region supply and fit, and retain separate
raw-step and remap telemetry to expose cancellation. Constant drift is not
penalized by variance, so this alone cannot guarantee rest. At T=20 an isolated
terminal jump has variance coefficient200*(T-1)/T^2=9.5, compared with P294's
remap coefficient5 before common unit scaling. Keeping the nominal weight does
not keep the effective regularization strength equal. A full optimization pair
with adaptive lambda would be a whole-policy comparison.

The first-step packets saved here omit later controls and optimizer/target state;
they cannot restart that full matched solve. No velocity kick J/dt, post-render
interpolation, renderer modification or new production default is applied.
