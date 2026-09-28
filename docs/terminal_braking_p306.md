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
