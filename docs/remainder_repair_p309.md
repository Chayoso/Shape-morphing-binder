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

## Completed remainder_repair1

Frozen0999f4d ran on hyde06 GPU1. The fresh20-window run completed160 accepted
ordinary updates, all guards0 and exact post-callback production isolation.
Discretization remains N300000,T20,dt1/240,dx.3062907543956724wu,loss36^3,
budget8. This realization has48,033 start-free and47,738 start-arrived-free IDs;
only within-run comparisons are meaningful here.

Observed-model correction accepted4 data/running repairs from68 forward trials.
All four satisfy the exact frozen volume/render ceilings and resolved running
decrease. No candidate passes all independent P306 gates, so the held-out
fixed-candidate replay branch is not reached. No candidate state is committed.

| Versus own baseline0 | Repair1 | Repair4 |
| --- | ---: | ---: |
| Arrived-free running mean-square change | -4.3371% | -11.4905% |
| Net RMS change | -1.9617% | -5.7678% |
| Saved-step RMS change | -2.1926% | -5.9205% |
| Path mean change | -3.3985% | -8.0687% |
| Stored terminal RMS change | -25.8909% | -24.6422% |
| Geometric terminal RMS change | -25.7996% | -24.5587% |
| Upper target coverage change | -0.0000653083 | -0.000457158 |
| Overall target coverage change | +0.0000033333 | -0.0000233333 |
| Fixed-source upper density change | +0.000111740 | +0.000279350 |
| Tip count change | 0 | -1 |

Repair1 fails only upper-target coverage (one reference target point).
Repair2 fails upper/overall coverage; repair3 also fails independent silIoU and
tip count; repair4 fails upper/overall coverage and tip count. Source density
improves, so the previous local-supply failure is not universal. These strict
discrete failures alone do not certify a substantial visible new hole.
A retrospective gate check of all68 evaluated trials finds no overlooked
feasible candidate. This is not a global impossibility result.

Rendering at lambda.0209818792: repair1 silhouette increases8.83592e-8 while
PBR decreases8.84756e-8; combined render changes-2.32831e-10 and weighted render
-4.88522e-12 against baseline0. Repair4 silhouette increases3.03378e-7 while PBR
decreases3.04542e-7; combined changes-1.16415e-9 and weighted-2.44261e-11.
The combined losses remain near the frozen ceiling; component changes oppose
one another. Median nominal render-direction share over160 ordinary steps is
.5072956, median lambda over20 windows.0654024. This is not a causal movement
share or exported4K appearance measurement.

The search accepted the first restored-data/running candidate before checking
raw quality as a final gate. Its four accepted proposals are at halves5,5,4,5,
with correction rounds2,2,2,1. Consequently it never evaluated smaller
same-origin radii after each accepted proposal. A bounded next diagnostic can
make the unchanged final quality gates part of backtracking acceptance, so a
failed coverage candidate cannot advance the origin. This changes acceptance
order, not a loss or tolerance. Evidence is retained in `docs/evidence/p309`;
full state/linearization sidecars remain in server `work/p303/remainder_repair1`.

Independent review reapplied all gates to all68 valid trials and verified the
JSON/protocol linkage,69 numerical source files and7 helpers. The frozen server
copy contains an extra older `physmorph/volumetric.py`, absent from the local
package; its exact19150 bytes are retained as `.py.txt` with SHAaca76c8...5c84
alongside the evidence. Active imports use `physmorph/losses/volumetric.py`.
The review checks recorded scalar arithmetic and bindings, not an independent
remeasurement of the server-only raw-array sidecars.
