# P299: one bounded proposal from the same prepared state

Status: CPU and independent review passed; bounded CUDA diagnostic pending.
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
