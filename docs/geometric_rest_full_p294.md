# P294 geometric-rest full-horizon comparison — results pending

This is a follow-up to the [eight-window prefix](geometric_rest_prefix_p294.md),
whose modest shape/supply improvement did not establish free-particle rest.
The pair extends the same formulation to a cap of 60 windows to observe
later transport and settling. No full-horizon quality result is reported yet.
Root launched control on GPU0 at 2026-09-27 01:09:30 UTC and geometric rest on
GPU2 at 01:10:34 UTC, 64 seconds apart, using the unchanged geometry snapshot.

Both runs must use the frozen P294 numerical source, identical original mixed60
300k source/target arrays, T=20, dt=1/240, dx=0.3062907544 wu, loss grid 36³, 8 inner
iterations, the 300-window schedule capped at 60, shared PIC objective, no subcell
shift, corrected committed-state outer rendering, the same positive finite
kinetic weight and active rendering throughout. Only `geometric_rest` may differ.
The serialized `auto` mode requires a unique recorded OT resolution event,
matching resolved modes in both arms and finite full-plan arrival evidence in
every accepted record. The logs' hashes and exact resolver lines are retained.

`geometric_rest_full` preserves both complete delivered runs, including unequal
stopping times. Endpoint quality is labeled by each arm's actual stopping point;
equal-accepted-commit comparisons, fixed Chamfer first crossings, and fixed-ID
motion cohorts are separate. The full raw pin audit covers each complete archive.
Common late motion uses at most the last 10 shared accepted window transitions;
the phase audit consumes the exact material IDs and interval from that comparison.
Interior null-held rows and presentation holds are excluded from comparison
motion. The legacy arm-level tail audit may include interior holds and is labeled
when present. Pin fractions, outcome selection and dynamic geometric-rest
eligibility remain necessary interpretation limits.

Expected artifacts are `work/p294/geom_control60` and `work/p294/geom_rest60`.
The separate audit environment is `work/p294/audit_full`, built from the immutable
`output/p294/geometry_snapshot.zip` with only the reviewed audit-script overlays.
Its numerical source is the same 57-file aggregate
`fd7c0ba18463ed6ddc232a800850372134e540c40af4000698b83b4053192f6a`.
The already executed prefix environment `audit_v2`, its results and its prepared
probe copies remain unchanged. Root owns all GPU launches; this audit preparation
does not start a simulation or extend a run.

Local preparation files and receipts reside in `output/p294/` with the
`audit_full`/`prepared_full` names. GPU0 launchers are `quality_geom60.sh` and
`phase_geom60.sh`; intended outputs are `quality_geom60.json` and
`phase_geom60.json`. Remote deployment verification checked all 148 manifest
entries and all 57 numerical files against the simulation snapshot; its receipt
is `audit_full_provenance.json`. Quality/phase SHA256 values are
`3288a4deb0b75b1d13544bad1107b0f7ffa2c982d54361437fadea020cfe067f` and
`b09c12e69a7764aca603c99625f3316faca6bb99150875243628a5e98cff3617`.
The full-mode code review and 21 independent CPU scope tests passed. The separate
deployment readiness review closed after checking every ZIP member, the exact
two overlays, numerical/probe hashes and launcher paths. Measured outcomes require another
numerical/report review before any quality conclusion.
