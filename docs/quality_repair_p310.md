# P310: raw-quality filtering before advancing the repair origin

P309 accepted four restored-data/running improvements but all failed final raw
quality. After accepting a half5 proposal it never tried smaller steps from that
same origin. This bounded diagnostic changes only that acceptance order; it
does not change the objective, thresholds, model, cohort or replay gates.

Use another fresh live W20, N300000,T20,dt1/240,dx.3062907543956724wu,
loss36^3,budget8,raw/no-PIC/no-shift. Generate its own terminal05 and three
baseline repeats. Preserve P309's two observed-model correction rounds at each
of11 radii (halves0..10), the original control-step trust scale, residual joint
coefficient projection, exact data ceilings and resolved running decrease.

Before any displacement origin advances, apply every unchanged P306 raw
geometry/supply and both-cohort motion gate. If data/running pass but quality
fails, record the rejection and proceed to the next smaller radius from the
same origin. Do not spend remainder correction rounds on a data-passing
quality rejection; these corrections model only prepared data. Each halving
still resets the remainder; no rejected step changes the origin or Jacobian.
Raw validation remains nondifferentiated; no new loss is added. These metrics
now filter candidate selection, so they are not independent post-selection
validation. Their definitions, reference sets, denominators and thresholds stay
unchanged; they continue to consume raw state only.

The first accepted all-gate repair is replayed with identical coefficients three
times against the original frozen baseline gates, then the search stops whether
the repeats pass or fail. Thus this protocol has at most33 search forwards and
three fixed-coefficient repeat forwards, with at most one accepted update.
The preliminary terminal schedule and baseline repeats are separate from this
search budget. No production commit or old-archive admission.
Preserve full sidecars, actual render components and exact observer isolation.

A negative result is bounded to this terminal choice, displacement direction
model and trust schedule. A positive result would still require the original
total-merit and coupled continuation checks in
`candidate_commit_contract.md`, then full-morph and4K validation.

Four CPU callback/operator regression cases pass, including the new required
data-pass/raw-fail case: the first candidate is rejected without changing the
origin, a smaller-radius candidate is evaluated and passes three fixed replays.
The preserved P309 fixtures still reject one deliberately perturbed repeat.

## Completed quality_repair1: no joint feasible candidate

Frozen b5809b5 ran on hyde06 GPU1;20 ordinary windows/160 accepted updates,
all guards0 and exact post-callback production isolation. N300000,T20,
dt1/240,dx.3062907543956724wu,loss36^3,budget8 remain unchanged. Fixed cohorts
are53,447 start-free and53,287 start-arrived-free IDs. This is another fresh
realization, so comparisons are against its own baseline only.

There are27 search records:24 full forward evaluations at halves0..7 with up
to two correction rounds, and3 smaller radii at which the active-set solver
finds no feasible step. No repair is accepted and no fixed-candidate repeat
branch runs. The origin stays fixed. No production state changes.

Only h6/correction2 passes the actual data ceilings and resolved running
decrease. Against its own baseline0, arrived-free running mean-square falls
3.9988%, net RMS1.7322%, saved-step RMS2.0198%, path mean3.6347%, and
stored/geometric terminal RMS27.0523%/27.0217%. But raw silIoU falls2.29483e-5,
upper coverage loses one reference target point (1/15312), and fixed-source
upper density falls1.86234e-5. Overall coverage/tip count equal baseline and
Chamfer improves8.97664e-7wu. These discrete failures do not alone establish a
substantial new visible hole, but they correctly reject the candidate.

The new acceptance policy then tries half7 at the unchanged origin; all three
correction attempts fail data and raw silIoU. A retrospective application of
all unchanged gates to all24 evaluated trials finds no feasible candidate.
This rejects this finite search, not all body/displacement control formulations.

At h6/correction2 prepared silhouette increases1.34460e-7 while PBR decreases
1.34693e-7; combined render changes-2.32831e-10 and weighted render-5.95445e-12
at lambda.0255741819. The combined channel can offset silhouette worsening with
PBR improvement; here raw silIoU also worsens. This is an observed tradeoff,
not an isolated cause of every failed gate. Median ordinary render-direction
share.5010839 and median lambda.0668287 are not causal motion shares. Exported
4K appearance is unmeasured.

Evidence is retained in `docs/evidence/p310`; full state/control sidecars remain
on the server. The candidate original-merit API was added later in85e4b5d and
did not participate in this frozen run. No adopted state, persistent rest,
full-morph no-hole or4K-quality result follows from this diagnostic.

Independent review verifies the result/protocol binding,69 numerical source and
8 helper hashes, every trial's gate arithmetic and rendering recombination.
The unchanged server-only volumetric source is retained in P309's evidence.
This is not an independent remeasurement of the raw-array sidecars.
