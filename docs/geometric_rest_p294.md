# P294: terminal geometric-rest diagnostic

Status: implemented and independently reviewed; the CUDA bridge and integration
gates passed. The subsequent full comparison fails the combined shape, supply
and rest gates. This remains an opt-in experiment, not an adopted quality repair;
the formulation and limitations are in method section10.41.

The objective separately penalizes the final raw position step and the endpoint
remap on a frozen arrived/free material cohort. It retains stored-momentum kinetic
loss, uses the existing `wu*w_kin` coefficient as an explicit experimental choice,
and normalizes by all particles. An empty cohort contributes zero. No additional
pin-admission rule, post-render smoothing or position averaging is added. The
ordinary settlement policy can still admit new pins. Automatic render lambda
can change because the new term participates in the physical gradient norm.

Both whole-cloud and conditional component means, eligibility, nominal w_kin,
the unit multiplier and the effective coefficient are logged. This term sees
terminal geometric motion, not the whole trajectory. Remap/dt depends on temporal
discretization; the initial physical comparisons must keep T20 and dt1/240 fixed.

## CPU validation

The previous-position interface preserves the existing five-output APIs. Its
opt-in sixth output is a cloned actual x[T-1], with an independent adjoint seed.
T1 returns constant x0; stale persistent contexts are rejected. Independent
review passed25 bridge tests and7 helper tests. Correctly CUDA-disabled legacy
regression passed40 tests with3 CUDA skips. The local CUDA-exclusion incident
and discarded incomplete result are recorded in the maintenance log.

Twelve CPU integration tests cover legacy/density units, taped gradients,
accepted-buffer and overwritten-buffer replay, direct objective reconstruction,
positive rendering lambda, conditional/whole-cloud telemetry, existing pins,
empty and missing cohorts, stale previous-state rejection, and preservation of
the physical F/v returned by the optimizer through runner promotion. These are
N160/T3 integration fixtures, not evidence about the300k morph's quality.
The complete CPU-only suite subsequently passed455 tests with22 skipped in89.31s
(477 collected), with `CUDA_VISIBLE_DEVICES=-1` set explicitly.

## Frozen CUDA bridge validation

Snapshot: `/data/relcfd/chayo/physmorph_v2/work/p294/geometry`.
The local ZIP/manifest under `output/p294/` contains148 verified files, including
57 numerical Python files with aggregate SHA256
`fd7c0ba18463ed6ddc232a800850372134e540c40af4000698b83b4053192f6a`.

GPU0 executed the bridge probe on2026-09-27 00:49:09.913--00:49:10.361 UTC.
It selects64 original source particles deterministically on CUDA, uses T3,
dt1/240, dx0.3062907544wu and the original36-cubed MPM grid/material. Rest volumes
are estimated for this subset, not copied from the full body. The synthetic
case includes stress, two body modes, layer u and pins. It has no rest objective.

- Ordinary and captured persistent forward/backward graphs passed declared
  comparisons. xT and x[T-1] differences were exactly zero in this run.
- The worst gradient error/tolerance ratio was0.89727. Repeated and absent
  previous-position seeds passed; the last stress control has zero effect on
  the earlier x[T-1] output, as required.
- Independent ordinary-forward finite differences for u and body coordinates
  had absolute errors0.000469685 and0.000821531, within predeclared tolerances.
- Pins, stale-context rejection and the separate T1 constant-output case passed.

Evidence is local `output/p294/previous_gpu.json`, SHA256
`adad3ac0282d49d090a0746ee331fefa848015d99242678b09dca7cd802188ab`.
The reviewer matched all executed dependency hashes to the frozen ZIP and closed
this bounded evidence gate. This is neither a300k stress test nor a rest result.

## Physical comparison protocol

Both arms use the original bunny N300000/T20/dt1/240/dx0.3062907544/loss36^3,
eight inner iterations, the300-window schedule, shared XPIC objective, shifting
disabled and corrected outer render merit. Only `geometric_rest` differs.
One-window integration precedes an archived eight-window prefix; full runs
require the corresponding execution and prefix checks. No output is promoted
by this report, and every result must retain pin/cohort and raw-phase controls.

The one-window pair has identical frozen numerical hashes and differs only in
`geometric_rest`. Both accepted8 inner updates and1 outer commit, with0 rejected
updates and all guards0. Both promoted the owned accepted endpoint with exact
objective/commit agreement. The candidate's previous-position agreement is also0;
CUDA final replay was not exercised by this case.

Candidate eligibility was53205 particles (17.735%). Its whole-cloud raw/remap
squared speeds were0.187759757/0.506113112 wu^2/s^2, total0.693872869. Delivered
squared speed0.567782164 equals total plus cross term-0.126090705 within float32
rounding. The effective coefficient is0.000151499747, explicitly5 times the
density-unit multiplier0.0000302999494. These are candidate-only diagnostics,
not a between-arm rest improvement. They include material still being redistributed.

The control and candidate recorded15.804/16.761s and Torch allocation peaks
7.947/8.009GB on different GPUs; this is not a performance benchmark. Launches
were00:50:43 and00:51:41UTC, checked against the prior launch's server timestamp.
The JSONs are `output/p294/geom_control1.json` and `geom_rest1.json`.

## Eight-window screen and longer comparison

The [completed prefix comparison](geometric_rest_prefix_p294.md) retains the
same numerical snapshot and changes only geometric_rest. Both arms complete8
accepted windows with zero guards and exact owned endpoints. Early shape and
upper-region supply improve modestly, but the same14468 free IDs show only a
2.88% reduction in raw-step median and more commit-to-commit direction reversals.
Pin selection also differs. Independent review closed the result/report gate;
this screen does not establish late rest or disappearance of holes.

The next bounded experiment keeps the exact same two formulations, source,
target, discretization and8 inner iterations. Only the stop cap changes from8
to60; the animation schedule remains300. The frozen launchers are
`work/p294/geom_control60.sh` and `geom_rest60.sh`, with outputs of those names.
Before launch, completed raw archives are offloaded with full-hash verification
to leave server project usage below89GB for the worst-case pair. The control
uses GPU0 and the candidate GPU2, separated by at least50s on the server clock.

Evaluation must retain actual termination, equal accepted commits, first
crossings of fixed geometric-progress thresholds, early thin-region supply,
final silhouette (existing0.971 gate), tip retention and state guards. The
source-defined cohort and the same free material IDs in both arms remain
separate. Raw-step, accepted-window and boundary-phase motion are all reported;
increased pinning or a smaller changing eligible cohort cannot establish rest.
Archive hold frames must not dilute motion. An encouraging longer comparison
would still require visual QA and broader-shape validation before adoption.

## What arrival and visual matching do not establish

The active density, silhouette and density-normal shading terms compare aggregate
fields. They do not assign a unique final position to each material ID. Tangential
rearrangement that leaves those fields alike is not identified by the matching
terms alone; finite sampling, dynamics and control/kinetic/strain penalties can
still respond to it. This is not a claim that the entire objective has an exact
tangential nullspace.

The current arrival test is distance to this window's frozen full-plan OT image
within max(plan blur, loss-cell width). It contains no speed, gradient, stress or
local-optimality test. P294 freezes this eligibility at window start and penalizes
all three components of terminal raw motion and remapping. It does not establish
stationarity of the actual position path over every substep. The active running
kinetic weight is0; velocity variance has weight200 and uses stored physical V.
Neither observation makes a position correction equivalent to a velocity update.
These distinctions remain necessary even if the full comparison improves.

The [completed full comparison](geometric_rest_full_p294.md) does not improve the
requested combination: both arms miss0.971 IoU, tip retention is worse with the
new term, and identical free material has larger accepted-window displacement
and more window-to-window reversals despite smaller raw/boundary motion. The
term stays disabled by default. The original rendered deliverables are unchanged.
