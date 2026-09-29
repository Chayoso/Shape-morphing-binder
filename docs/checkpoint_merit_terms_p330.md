# P330: differentiate the complete prepared head merit

P329 closes the production-size joint head/coast capability at N300k,T20,
dt1/240,dx.3062907543956724wu,loss36^3,iters8. It does not propose a different
control. A subsequent body-control step needs the gradient of the entire
existing head objective so that reducing future motion does not evade its
material, control, kinetic, variance or fitting terms.

The existing `packet['evaluate_merit'](values)` is the authoritative value-only
candidate evaluator. Its acceptance value combines separately converted Python
floats: `float(physical)+fixed_lambda*float(render)`. Changing it to a tensor sum
would change rounding. `common_positions` fixes F/v/V and control energy and
therefore cannot provide the missing complete candidate covector.

The opt-in checkpoint now exposes `packet['evaluate_merit'].terms(values)`.
Both entry points reuse the same input validation, frozen stress/reference,
candidate body-energy override, `losses_of`, `phys_total`, and cache restoration.
The new method returns graph-connected physical/render/volume/kinetic/variance/
body-energy terms and a tensor merit for differentiation. It never substitutes
that tensor for the rounded scalar acceptance value. Silhouette remains the
existing scalar telemetry; its separate gradient is available from the frozen
`PreparedReference`. No PCGrad, balancer update or control installation occurs.

Returned tensors own their scalar storage and carry the checkpoint's expiring
backward lease. Cloning a returned tensor cannot extend that lease; independent
caller inputs do not receive these hooks. An older checkpoint does not revive
when a later checkpoint opens. Invalid inputs and exceptions must leave the
original caches, production state and optimizer history unchanged.

This is a callback capability with the existing unsupported-mode guards. It
changes no default loss, coefficient, accepted state or rendered deliverable.
It is not an actual post-assimilation derivative or an adoption API.

## Validation

The CPU regression passes39 cases:11 new merit-term cases,26 existing adapter
cases and2 existing original-merit observer cases. The2 explicit opt-in CUDA
cases are collected but skipped locally; they must run on hyde06 before a CUDA
capability claim. Active PBR, layer on/off and their untouched original solver
outputs are included in scalar-parity coverage.
The real small MPM fixture exercises both reduced modes through the joint
adapter, analytic F/J-volume and candidate body-energy sensitivity, complete
stored-V running/variance sensitivity, original scalar parity, callback lifetime,
error recovery and unchanged original pipeline results with layer control off/on.
Finite differences test the complete scalar evaluator; they do not test only
prepared volume+render or a surrogate penalty. Each tested directional derivative
must be resolved above the fixed absolute error allowance.

At N160,T3,dt1/240,dx1wu,grid32^3,loss12^3,iters2, the CPU complete-merit
directional derivatives are displacement AD.005865249972 versus centered FD
.005864893322/.005863824516 and terminal AD.000139332174 versus
.000139374498/.000139937728, at coefficient radii.003/.0015. The preregistered
gate is3% relative or1e-6 absolute error and |AD|>50e-6. This is a small smooth
fixture, not a finite-difference result for P329's production checkpoint.

## Next control experiment

Start at the original accepted pair of displacement and terminal body modes;
do not first impose a terminal-strength reduction and repair only displacement.
A bounded first-order proposal can reduce the passive geometric mean speed
squared while constraining complete original head merit, head fitting and coast
fitting. Recheck the actual jointly projected node coefficients and actual
nonlinear merit. A zero or infeasible step is not an improvement.

Prepared loss constraints remain insufficient: P329 raw IoU/support improved
during coast while its prepared losses worsened and tip count fell. Candidate
checks must therefore retain the original-covered target IDs, fixed source
supply and raw projections at every saved coast phase, and separate per-ID
worsening from a mean motion decrease. A successful mean decrease is not
individual rest. Candidate-specific actual assimilation/pin/layer/bond handoff,
whole-state rollback, subsequent optimized windows and full-horizon/gallery
validation remain required before adoption. No4K artifact claim follows from
this derivative; active exported covariance still has its separate gate.
