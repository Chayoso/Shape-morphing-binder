# P309: bounded observed-model-remainder correction

P308 h5 passed all raw shape/supply and cohort motion gates but failed actual
prepared volume/render ceilings. Its affine residuals did not predict the
observed excesses. This protocol keeps those ceilings and the entire P306 gate.
It is a noncommitting diagnostic, not a new production stopping policy.

Use a fresh W20 callback, N300000,T20,dt1/240,dx.3062907543956724wu,
loss36^3,budget8,raw/no-PIC/no-shift. Do not admit P307's failed restart or
reuse the previous experiment's selected particle state. Generate this run's
own terminal05. Fix terminal/stress/u, initial state, reference/lambda and
window-start cohorts. P308's objective, displacement trust scale, final joint
coefficient projection, max4 accepted updates and halvings0..10 remain.

For each accepted displacement origin D0, freeze running gradient, prepared
data f0 and Jacobian G. Each halving starts with e=0 and its own trust radius.
After solving the affine problem and projecting into the residual coefficient
ball, measure the actual data at the final float32 coefficient candidate:

```
delta_k = projected_candidate_k - D0
e_k = actual_data_k - f0 - G @ delta_k
new_RHS = original_baseline_ceilings - f0 - e_k
```

For a valid candidate missing the actual data ceilings, permit at most two
additional solves at this halving. Replace e, never accumulate it. Retain D0,
G, running gradient and radius through all rejected attempts. Re-solve the
same whole trust-ball problem; do not add a correction outside that ball.
Each new halving resets e to zero. Only an accepted displacement update may
establish a new origin. Invalid states, infeasible linear subproblems or failed
projected checks terminate this halving. No restoration is attempted when data
already passes but the running decrease is unresolved.

Record the original and shifted-model affine checks separately. Passing the
shifted model is not an original affine certificate. The nonlinear volume and
render ceilings stay exactly the original max of three baseline replays.
The running decrease must pass the existing observed-repeat/32eps floor;
validity and exact pins are mandatory. The final independent raw shape/supply
and both cohorts' motion gates remain decisive and are not objectives.

At the first accepted running-repair candidate passing every P306 gate, replay the identical fixed
coefficients three more times while the same private model is live. Apply all
gates and the resolved running decrease against the already frozen baseline
and origin. Stop this search after those three repeats even if one fails; do
not select the best repeat or widen the ceilings. Preserve full X/V/F/C/control
sidecars and exact post-callback production/Adam isolation. The model remainder
is observed error, not a certified curvature bound, physical cause or global
feasibility guarantee. Original C-repeat/archive admission remains failed.
This replay check does not apply to the preliminary terminal-only schedule.

Report lambda, actual render components and direction norms separately from
raw motion/geometry. A positive same-window result still needs original-total-
merit reconstruction (including candidate body energy), coupled continuation,
gallery checks and full-frame/4K QA before any production or visual claim.

## Implementation checks

Twenty-two distinct CPU tests pass:3 new analytic remainder/callback cases,
14 affine-operator cases,3 braking-direction/final-gate regressions, and2
actual CPU MPM observer/derivative cases with layer control on/off. The callback
fixtures require all three held-out replays to execute; a deliberately perturbed
second replay must reject repeated feasibility even while the other two pass,
and no further candidate search may run. This is not a full-suite claim.
