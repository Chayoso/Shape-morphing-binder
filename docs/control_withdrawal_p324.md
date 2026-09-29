# P324: controlled segment and control-withdrawal continuation

Design inspection, 2026-09-28. No simulation, objective change or tail bridge has
been implemented by this task. The decision must use the actual post-assimilation
handoff before treating a continuation penalty as a next-window remedy.

A bounded differentiation capability is feasible: two 20-step `Trajectory`
segments under **one Warp tape**, with the second segment reading the first
segment's complete final state. With the original Fp and frozen layer/pin policy,
this is a **pre-assimilation, learned-control-free continuation**. It is neither
the actual next window nor force-free dynamics. Gravity/contact/damping, material
stress, bonds and the chosen layer relaxation remain active.

## Smallest correct capability

Keep the controlled horizon T=20 and dt=1/240. Construct both segments before
recording/capture, with explicit source rest volumes. Before the tape runs, bind
the coast's `x[0], v[0], C[0], F[0], Fg[0]` to the controlled segment's arrays at
20; then record `controlled.rollout(); coast.rollout()` under one tape. The
boundary arrays must be the same gradient-bearing Warp arrays, not values copied
through NumPy/CuPy or a second ordinary Torch bridge. Materials, masses, rest
volumes and original Fp are shared constants in this limited capability.

- `RolloutSpec` initial states are configuration, not Torch leaves; existing
  bridges return control/material gradients only (`mpm/function.py:31`, `:199`,
  `:242`). Chaining existing bridges loses the head-control-to-coast derivative.
- Preserve APIC C. It contributes to subsequent transfer, and resetting it changes
  the dynamics (`mpm/traj.py:163`, `:320`). Preserve the controlled/smoothed physical
  F, separately from Fg: `k_g2p` incorporates dFc into F and `k_update` smooths it;
  Fg receives only the C-based kinematic update (`mpm/kernels.py:384`, `:403`,
  `:488`). Withdrawing dFc means zero **future increments**, not subtracting its
  accumulated contribution from F.
- Coast dFc and u are explicit non-leaf zero buffers; body control is absent.
  Head body coefficients, the displacement/braking pulses and the head control
  basis still use T=20. Simply setting T=40 changes the body normalization, u/T,
  bond/T and default relaxation coefficient (`mpm/traj.py:119`, `:250`, `:342`;
  `pipeline/optimizer.py:401`, `:453`).
- Keep the head's frozen pin mask and anchors, layer mask/normals/neighbors/weights,
  relaxation coefficient and material bonds for this continuation. Dynamic bond
  activation still follows the existing per-step rule (`mpm/traj.py:290`). Keep
  support-gate normalization fixed as prepared for the head. No new pins or
  neighbors are silently inferred at step20.
- Retaining layer relaxation is an explicit policy choice, not a claim of pure
  elastodynamics. Its positional projection changes x independently of stored v
  and, in the current model, relaxation is outside F (`mpm/kernels.py:477`, `:546`).
  Removing the optimized u channel alone does not remove that motion.

Return owned head endpoint outputs plus coast positions/velocities. Seed both
segments in one backward; sum seeds for any shared boundary output exactly once.
Reset missing/repeated seeds and deduplicate aliased gradient buffers. Persistent
capture must own both segments, invalidate stale forwards and obey the existing
Torch/Warp stream context (`mpm/function.py:333`, `:363`, `:466`).

Keep the existing endpoint merit at step20, including its head-only running and
variance terms, control normalization and render reference
(`pipeline/optimizer.py:1792`, `:1918`). A separate tail term would change the
physics direction, PCGrad and lambda balance and requires its own experiment.
Temporal variance alone is blind to constant drift; full post-layer displacement
rates expose that limitation. No weight or cohort policy is selected here.

The smallest capability returns a **lookahead**, not extra delivered frames or a
replacement step40 commit. Delivering/promoting the coast would change the
physical clock, assimilation timing and archive contract. Gradient evaluation,
line-search evaluation, accepted-buffer reuse and fallback replay must use the
same coast model if a later objective includes it (`optimizer.py:1803`, `:2878`).

## What an actual post-commit joint gradient additionally requires

The runner first promotes/guards x/F/v/C/Fg, then updates Fp using elastic-stretch
assimilation (`pipeline/runner.py:706`, `:766`). Old pins can keep their prior Fp;
new pin admission later performs its own full assimilation (`:800`, `:1672`).
The next optimizer prepares new layer data (`pipeline/optimizer.py:377`) and the
runner can rebase bonds or apply other enabled commit policies. None of these
operations is reproduced by keeping the original Fp and neighborhoods.

For a bounded **fixed-admission post-commit map**, minimum correct differentiation
is the full composition

`controls -> head state -> declared commit map -> coast state -> coast loss`.

1. Capture the actual continuous head state and the declared old/new pin masks,
   source Fp, layer/bond policy and assimilation settings. Freeze discrete choices
   deliberately. A frozen admission/neighborhood branch is conditional on those
   choices; it does not differentiate admission switches or re-preparation.
2. Reproduce the actual Fp map and connect its VJP to head F. With
   `Fe=F20 @ inverse(Fp0)`, this includes the fractional stretch, isochoric
   normalization, cumulative singular-value bounds/log-volume projection, and
   old/new-pin exceptions. The current array-returning assimilation API is not a
   Warp-tape handoff (`plasticity/assimilation.py:34`, `:63`, `:170`, `:198`).
   Reusing its detached Fp value is not the derivative of this composed map.
3. Supply a gradient-bearing post-commit Fp buffer to coast stress evaluation;
   retain every x/v/C/F/Fg dependency as well. On a single tape, the commit map
   needs a differentiable kernel/custom adjoint producing these state buffers.
   Alternatively, a larger bridge redesign could expose all initial-state VJPs
   and compose a differentiable commit in Torch. The existing bridge cannot do
   either by merely adding a second call.
4. Establish stable spectral derivatives at repeated positive stretches and the
   derivative on declared clamp active sets. Generic SVD-vector autodiff is not
   justified by P300's corrected **rotation** adjoint. Fail closed on unmodeled
   repair/local-pass/PIC/shift/reattachment/rest-reset policies; do not silently
   approximate them. Guard rejection stays separate from differentiable dynamics.

Even exact fixed-branch assimilation is not a guarantee of next-window invariance:
the next solve can change controls, targets and neighborhood/admission decisions.

## Decisive bounded gates before a repair experiment

First measure withdrawal from the **real post-assimilation handoff** with its full
v/C/F/Fp, pin and layer policy, alongside the explicitly pre-assimilation coast.
Determine whether the latter predicts the same residual drift/reversals; no new
optimization or arbitrary stopping threshold is needed for that comparison.

If a joint-gradient capability is pursued, require unchanged head-prefix outputs;
agreement with a separately evaluated zero-control coast from the same complete
state; finite differences of coast loss through dFc/body/u with nonzero incoming
v/C/F/Fp; simultaneous head/coast seeds; zero/missing/repeated-seed isolation;
owned-output/stale-buffer checks; and ordinary/captured CUDA parity. Include
nontrivial layer relaxation and bonds, unchanged T20 pulse normalization, exact
pinned anchors and a case where dropping C or detaching the boundary fails.
An actual-assimilation version additionally needs independent Fp-map VJP tests
and forward equality to the runner on the declared branch.

Independent design refutation agrees that the two-segment tape is the smallest
pre-assimilation capability, subject to these gates. It has not established a
quality fix, natural rest, hole prevention or actual next-window invariance.

## Implemented read-only handoff observation

`scripts/probes/control_withdrawal.py capture` wraps the ordinary full-horizon
raw arm without changing its controls, losses, acceptance or stopping. Requested
source attempts are 6,20,28 (one-based attempts, not accepted-window ordinals).
It captures a validated source endpoint before commit, then the step-zero input
of the immediately following optimizer attempt after preparation finishes. Only
source attempts actually accepted by the runner qualify. The successor need not
be accepted; missing successors, including an ordinary final stop, are reported
as missing. Full-horizon trace binds actual archive rows and raw x equality.

`OwnedWithdrawal` owns full x/v/C/F/Fp/Fg, source volumes, mass/material/viscosity,
pin/collider, permanent bonds, prepared layer data and resolved support count.
It reconstructs an independent forward-only T20 trajectory. Future dFc and u are
zero, body control is absent; accumulated F and APIC C are retained. It performs
no assimilation or new preparation. Archival I/O is explicit; numerical replay
and reductions execute on the GPU. This is detached observation, not a gradient
capability or optimizer change.

The separate `analyze` process replays both snapshots only after capture has
finished. It verifies the frozen producer code, input, configuration, trace,
render report and captured-state hashes. Pre/post differences include the whole
handoff: assimilation, pin admission/zeroing, bond rebase and layer re-preparation.
Report common-free, newly pinned, old-pinned and initially-free cohorts, plus
endpoint-arrived/transit common-free cohorts using the source's actual frozen
plan and arrival radius. Pins are imposed rest, not natural rest. Reversal
eligibility uses the existing P317 length floor, 1e-4 source spacing.

Full v/C/F/Fg finiteness, positive detF, runner containment and pinned paths are
checked without repair. Invalid tails are archived and flagged, without ordinary
motion metrics. All21 positions of valid tails and per-ID reductions are retained.
Raw endpoint IoU/target coverage are renderer independent. There is no rendering
loss in passive replay; the originating run still writes its rendering-influence
report. No post-commit differentiable map or tail penalty is implemented here.

## Completed observation, frozen73fdb63

Independent27 CPU tests pass, including exact observed/unobserved two-window
position/F/Fp/history equality and full-state snapshot ownership/oracle tests.
The actual CUDA reconstruction/ordinary/captured-forward gate passes1 test with
no skip in21.59s. This is a forward capability, not an adjoint test.

The source run uses N300000,T20,dt1/240,dx.3062907543956724wu,loss36^3,iters8,
source spacing.03498853660707278wu. It accepts36 of38 attempts and archives721
physical positions plusone held row. All guards are zero. It stops on
`outer_rejection_patience`, with individual rest explicitly not evaluated.
568.35s includes observation/archive overhead and is not a timing comparison.
Allthree requested source attempts commit and have real prepared successors.

Both coasts use the SAME free-ID cohort within each row. Numbers are RMS net
displacement over20 steps, in source spacings; cohorts differ between rows.

| Source attempt (1-based) | Common-free IDs | Pre-handoff coast | Actual post-handoff coast | Newly pinned |
| --- | ---: | ---: | ---: | ---: |
| 6 | 267575 | 1.535233 | 1.508385 | 24002 |
| 20 | 40021 | .430643 | .463103 | 4918 |
| 28 | 22511 | .300400 | .303633 | 814 |

At attempt28 all22511 common-free IDs satisfy the source plan's coarse arrival
predicate. Their post-handoff coast still moves; its last-step geometric speed
RMS is.148466wu/s, versus stored speed.148237wu/s. The814 new pins have zero
post-coast displacement by construction. Similar RMS magnitudes do not establish
per-ID position/velocity consistency. This is an intermediate/late-state
observation, not proof that these IDs had reached individual optima or a test
of the ordinary final commit after attempt36. Retained layer relaxation remains
part of the forward model; this does not isolate pure elastodynamics from it.

Every coast passes complete-state finiteness, positive detF, containment and
exact pin-path guards. Across each captured handoff x and F are unchanged; Fp,
new-pin v/C, layer data and bond-rest data change. The difference between columns
is their combined effect, not an assimilation-only ablation. Neighbor-ID numeric
differences in the archival report are identity differences, not physical lengths.

Raw24-view128px IoU at source20 is.962316, versus.963004/pre and.963229/post
coast. At source28 it is.962969, versus.962756/pre and.962768/post. Two-target-
spacing coverage at source28 is.992827, versus.992880/pre and.992877/post.
Quality changes are mixed; no passive tail is adopted or exported. There is no
new physical-hole or4K appearance certification from this comparison. In
particular, penalizing all withdrawal motion could suppress useful fitting at
source20; the coarse arrival mask does not justify a blanket rest penalty.

The independent CUDA artifact audit passes in20.97s, rehashing provenance,
validating actual attempt/archive/pin lineage and recomputing all six saved
position paths, cohorts and position-derived reductions. Full valid-tail v/C/F
sequences were not saved, so their health cannot be independently reconstructed;
stored-speed/min-detF summaries were checked against per-ID evidence. IoU and
coverage are source-bound, not independently recomputed by that audit. Exact
scope, source hashes, reports and auditor are retained in `docs/evidence/p324`.

Interpretation: real commit processing does not remove the observed residual
motion. A frozen pre-assimilation coast matches the order of the observed cohort
RMS at these three heads, not exact magnitude, reversals or a gradient through
changed controls. The
next capability remains a joint head-plus-withdrawal derivative with all full-
state paths connected, followed by actual post-commit continuation validation.
Do not introduce a detached Fp handoff and call it an exact post-commit gradient.

Rendering influence in the originating optimization:18 views,64pixels,GSoff;
288 accepted inner updates in committed windows (304 including rejected outer
attempts). Median nominal render-direction share.456425, adaptive lambda.0259867,
and within-reference render-loss change-1.43735e-5. These are observations, not
causal displacement shares. The passive comparisons optimize nothing and invoke
no rendering loss. Residual positions therefore move without a renderer call,
while their starting states were obtained with rendering guidance.
