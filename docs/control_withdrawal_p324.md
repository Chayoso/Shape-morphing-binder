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
