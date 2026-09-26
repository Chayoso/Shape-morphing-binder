# P292: residual motion, sparse transport and high-resolution artifacts

The user authorized fixes on 2026-09-26 after the CUDA migration cba9d29.
The starting bunny300k path still has sparse growth and unpinned surface motion.
All admitted pins are exactly stationary, which does not certify all-particle rest.

## First physical candidate

`body_rprop` conditions displacement-mode updates using raw RPROP scales on active
arrived particles, protects any node used by active unarrived material, and keeps
the terminal-force update unchanged. It changes neither objective nor lead/pinning
policy. The precise contract is method.md10.37. This addresses a verified difference
between stress update scaling and the newer body's boolean control gate. It is not
an asserted explanation of every oscillation or hole.

Compare identical archived mixed60 source/target, N300000, T20, dt1/240,
dx0.3062907544wu, loss36^3, seed1, eight inner updates, animations300/cap60,
CUDA backend, with the single flag difference. Preserve both actual time and
accepted-commit indexing; compare first crossings of fixed initial-Chamfer fractions
and disclose their progress mismatch. Use raw geometry metrics, exact pin checks,
fixed material cohorts as well as separately selected endpoint cohorts. Faster
pinning or a slower trajectory alone cannot count as a fix. Report late normal/
tangential motion, direction changes, net/path ratio, terminal velocity, thin-region
density, tip occupancy/attrition, silhouette and detF. Existing coarse hole_frac
cannot certify continuous material coverage; Gaussian-splat movies need all-frame QA.

## Rendering candidates

On the unchanged original mixed60 trajectory, separately test compact continuous
current-density support against the existing hard neighbor count. Support must
vanish outside its existing physical radius, never exceed the corresponding hard
count, and retain active-pin position/normal/radius invariance. This may expose
more sparse coverage; it cannot be presented as filling physical gaps. Separately
test a resolution-scaled screen-space normal filter while retaining original
coverage. See render_artifacts_p292.md for precise implementation and evidence.

No candidate is promoted before its measured gates and adversarial review pass.

Implementation checks before the follow-up material-shading experiment: full CPU
reference suite with CUDA disabled, **331 passed / 22 skipped** in 80.40 seconds.
The adversarial body-update review closed after an end-to-end nonunit-scale test
verified that displacement updates shrink while the separate braking update does
not, away from joint-bound saturation. This is execution/algebra evidence, not
proof of better morph quality. The GPU comparisons use immutable source snapshots.

## Mixed body plus existing stress taper screen

P290's taper experiment preceded the body actuator. A separate eight-window screen
uses the current two-mode body controller with the existing `ctrl_taper_sp=2`,
`body_rprop=False`, identical inputs and animations300 schedule. Compare the first
eight accepted commits of the new baseline as well as matched progress. Increased
density caused only by slower extension or a shorter/blunter tip is a failure.

This is a whole-feature ablation, not a clean boundary-traction intervention. The
existing taper also reduces the stress-control bound and tapers inherited expanded
dFc again during warmstart. That interaction must be separated before promoting any
positive result. The old claim that taper alone separates skin and bulk remains
refuted. Pin invariance, raw motion, terminal velocity, guards, silhouette and detF
remain gates; a prefix cannot certify final shape or final rest.

## Follow-up: actual displacement accounting

Both physical candidates failed their intended quality gates; see
[the raw comparison](quality_comparison_p292.md). Rather than choosing another
damping coefficient from those endpoint results, `motion_accounting` is a read-only
diagnostic of the final validated trajectory that the runner is about to promote.
It neither changes a loss nor reruns a counterfactual. It records whether this
trajectory came from an accepted buffer or a validated replay.

Each rollout step is decomposed algebraically into stored MPM velocity times dt,
the bond/update residual before the layer projection, analytic surface-u motion,
and the remaining layer projection. The latter two residuals include floating-point
subtraction error. Their reconstruction error is reported. Last-substep geometric
velocity `(x_T-x_(T-1))/dt` is compared with stored vT; window-average transport is
not mistaken for terminal velocity. Commit PIC and subgrid shifts are recorded
separately, with any other pre-PIC position correction retained as its own component.

The same frozen full plan defines arrival at the window start and actual promoted
end. Cohorts are free particles arrived at both endpoints, all other free particles,
window-start pins, and the full cloud. Per-step component RMS and signed contributions
to actual motion expose cancellation; RMS magnitudes must not be added as percentages.
This is bookkeeping within the hybrid model, not causal intervention evidence and
not a new rest criterion. Only accepted outer records may support accepted-motion
claims. The diagnostic run retains the baseline N/T/dt/dx/loss/schedules above;
its opt-in flag is the sole intended behavior change and no full raw archive is
needed. A new GPU run may differ through the existing atomic nondeterminism.

Two CPU checks passed: actual zero-elasticity Warp motion driven by u while stored
momentum is exactly zero (plus pin/commit accounting), and exact pipeline trajectory/F
equality with accounting off/on, including layer, PIC and shift operators.
The full CPU suite after accounting and material-shading integration passed
**341 tests / 22 skipped** in 74.90 seconds with CUDA disabled. The independent
accounting review confirmed final-buffer ownership, no state mutation, correct
cohort timing and terminal-step comparison before the GPU diagnostic was launched
on GPU0 at 23:06:38 UTC.

The separate raw-phase audit localizes almost all sampled direction reversals to
the endpoint boundary, without isolating its components. In parallel with stage
accounting, a bounded existing-feature ablation disables only `commit_pic` on the
original immutable P292 numerical snapshot (`body_rprop=False`, shift retained),
with an eight-window execution cap and unchanged animations300 schedule. This is
a causal test of removing the whole endpoint filter, not a claim that its transfer
is the sole cause or that arbitrary correction interpolation is legitimate. The
existing full baseline supplies the first eight accepted commits. Admission and
future trajectories can change; supply, shape, raw-frame reversal and deformation
must all be measured. A prefix cannot establish late rest or final morphology.

The PIC-off prefix is mixed rather than a definitive failure: density and boundary
normal return improve, while early silhouette and target coverage decline. With
the component accounting now confirming substantial opposing PIC displacement,
a bounded 60-window continuation from the same source on the unchanged numerical
snapshot is warranted to resolve final shape and late free-particle motion. Shift
stays on; body RPROP and the new outer bookkeeping option stay off. No recipe is
promoted from the prefix alone.

Separately, `outer_render_committed` implements method10.38. It evaluates a fixed
target on the actually proposed commit and prevents rejected candidates from
overwriting the previous accepted plateau tracks. The helper tests fixed-target
independence from paced telemetry, changed promoted positions, inactive rendering
and device/target consistency. Real CPU pipeline tests cover PIC/shift integration
and an accepted/rejected/accepted sequence whose last candidate regresses only
against the correct accepted reference. Historical behavior remains the default
pending paired quality validation.
