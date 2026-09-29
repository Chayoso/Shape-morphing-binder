# P306: terminal-body-only braking feasibility

Preregistered, noncommitting diagnostic. Original300k raw/no-PIC/no-shift
recipe, T20,dt1/240,dx.3062907543956724wu,loss36^3, eight inner iterations.
W1-19 remain unchanged. At W20 accepted iteration8, hold initial physical
state, stress,u,body displacement mode, pins, layer/bond operators, prepared
density/render references and lambda fixed. The separate rollout owns its
state; no diagnostic candidate enters the production trajectory or optimizer.

On the same start-arrived-free IDs, minimize the equal average of stored
terminal speed squared and actual last position difference/dt squared. Project
its negative gradient onto the two linearized density/render non-increase
halfspaces. Normalize by per-node vector RMS and test scales4,2,1,.5,.25 of
the last accepted terminal-mode coefficient update. Only terminal coefficients
are projected into radius sqrt(max(0,1-|displacement|^2)); displacement is never
rescaled. Three independent original-coefficient replays define an observed
numerical range, not a confidence interval.

Before trials, private baseline X/V/F must match every accepted element within
32float32 eps times (characteristic unit + absolute reference). Units are dx,
dx/(T dt), and1 respectively. Prepared density/render must agree within32eps
relative to the accepted history. No full-C closure claim is made. Resolved
braking additionally requires each terminal speed reduction to exceed10times
max(repeat range,32eps times baseline speed); a one-ULP decrease is insufficient.

Positive feasibility requires both terminal speeds below that repeat range;
no regression beyond it in prepared density/render, independent fixed-target
silhouette/Chamfer/coverage, tip supply, fixed-source upper density, or net,
step and path motion of either start-free or start-arrived-free IDs. All raw
states must be finite, inside the padded MPM box with positive stored/effective
detF, and start pins exact. All candidates and failure gates are reported.

Failure of this one direction/subspace does not prove braking impossible.
Success does not prove passive equilibrium after controls/assimilation or a
complete-morph rest policy. Original total merit is not reconstructed from an
old body-control closure; report candidate data terms and body energy
separately. Rendering terms, lambda and accepted production update influence
remain distinct. No visual/full-morph promotion follows this diagnostic.

## Conditional integration gate

Only a resolved positive example warrants an opt-in post-solve terminal-control
polish. It must recompute the existing total merit with the candidate body
energy and promote the complete physical rollout through ordinary state gates.
Require existing-merit nonincrease initially; a different lexicographic policy
would need its own explicit formulation and evidence.

Track the same preselected IDs through subsequent coupled windows with actual
assimilation, plan refresh and neighbors. New pin admission is a separate
outcome: the current pin policy explicitly zeros v/C and may assimilate Fp.
Do not credit that as natural rest, or silently remove these IDs from the
cohort. Stored/geometric speeds, all raw phases, F/Fp/C changes, plan relabels
and fixed-target fit/supply all remain observable. An isolated zero-control
hold may diagnose recoil but cannot replace the coupled continuation.

The300-window recipe's cap24 is only2seconds at this T/dt. Its stale-fit stop
and best/delivered endpoint do not certify complete-morph rest. A matched full
horizon, raw-state geometry and all-frame visual QA remain required after the
local continuation passes; held presentation frames cannot dilute motion.

## Completed result: no candidate passes all gates

`work/p303/terminal_braking3` uses frozen `code_terminal_braking3`. All three
replays and five candidates have valid raw states and exact start pins. The
production W20 is accepted (8inner accepts,0rejects, frame_end401), all guards0.
Same50,957 start-arrived-free IDs, with the declared discretization above:

| Observation | Baseline | Trial .5 |
| --- | ---: | ---: |
| Stored terminal RMS (wu/s) | .203410 | .150848 |
| Geometric terminal RMS (wu/s) | .203083 | .150757 |
| Net RMS (sp) | .430451 | .424593 |
| Step RMS (sp) | .023152 | .022866 |
| Path mean (sp) | .364004 | .353720 |
| Fixed-target silIoU | .96250233 | .96250392 |

The ~26% speed reductions clear the repeat/roundoff threshold, and both
preselected cohorts' raw motion improves. However, density rises3.5623e-8,
upper coverage loses4of15,312 target points and overall coverage loses3of300,000.
Trial .5 and every other tested scale therefore fail the preregistered gates.
No terminal-only feasibility or production-quality promotion follows.

Rendering lambda is .01889300448. Combined rendering decreases9.5228e-8
(weighted1.7992e-9), while its silhouette component worsens9.0338e-8 and PBR
improves1.8545e-7. Independent silIoU improves only1.581e-6. These are separate
observations: shading improvement does not certify preserved physical supply.
The full accepted-step render influence report is retained with the run.

Private/accepted maximum X/V/F differences are4.77e-7wu,3.25e-6wu/s,5.96e-7;
all closure gates pass. The independent review checked protocol and67numerical
source/3helper hashes,16,669 finite JSON floats and every gate result. Evidence
is in `docs/evidence/p306`; trajectories remain on hyde06. A failed earlier
launch stopped before physics due a relative helper path; it is not a numerical
run. Next: [bounded endpoint compensation](braking_compensation_p307.md).
