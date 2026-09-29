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
