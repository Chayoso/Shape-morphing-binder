# P314: preserve identified material support while braking

Status: CUDA comparison completed and independently audited. All48 valid
forwards fail original raw IoU; no repair is adopted. Design and implementation
were independently reviewed;62 distinct focused CPU cases independently pass. The eight-row
operator and explicit original-row metadata mapping pass33 independently
reviewed CPU cases. P313 localizes a support
change in the saved terminal05 origin, not the unsaved P312 repair candidates.
It motivates testing a local support constraint; it does not establish a remedy.

Use a fresh live W20 at N300000,T20,dt1/240,dx.3062907543956724wu,loss36^3,
budget8,raw/no-PIC/no-shift. Do not restart from the P312 archive. Retain the
failed C-repeat admission gate and original read-only callback isolation.

Prepare one byte-owned original baseline triplet and one byte-owned terminal05
origin, including all common gradients, trust radius and running-noise threshold
as in P312. Extend the original baseline package to own its three endpoint X
arrays. Calculate exact CUDA target-to-body nearest distances with the existing
raw metric's radius2*target_spacing. Freeze the set of target IDs covered in
all three originals but uncovered in the actual common terminal origin. Use
all such IDs, sorted, with no location/cohort/margin-based selection. Save the
full endpoint coverage bitsets so gains, losses and baseline ambiguity remain
separate. Do not reuse P313's particular IDs in a new realization.

For each protected target, use the repeat0 nearest supplying material ID as a
fixed witness; verify that same ID covers the target in all three originals.
If any witness lacks that common certificate, abort this bounded experiment
and retain the reason. This conservative witness is sufficient for preserving
that target's coverage, not a proof of unique supply or global feasibility.
Freeze the witness ID and target coordinate before forming any repair step.

Add the differentiable scalar s_i=(||x_T[p_i]-q_i||^2-r^2)/r^2, constrained
to s_i<=0. Compute distances in float64 like the raw KDTree; differentiate
through float32 physical X and the actual MPM displacement control. This changes
control coefficients through a physical rollout; it applies no position patch.
There is no added loss weight, coverage-radius change or new stopping threshold.
Use the identical FP64 scalar for linearization, actual candidates and every
fixed-control repeat, and also check each protected target's actual nearest
distance against the unchanged raw radius. Do not accept a boundary disagreement.
The running-motion objective remains over all saved steps of the original
fixed arrived-free cohort; the added constraints concern endpoint support only.
They cannot certify all-phase or full-morph hole absence.

Prepare these extra values/gradients in the SAME common origin forward as the
existing running/volume/render/silhouette gradients, not a later rerollout.
Both arms own exactly the same first three constraint rows. The control arm
uses volume, combined-render and silhouette. The treatment adds every selected
support row. Both use the same fixed data/silhouette ceilings, original-merit
evaluator, terminal coefficients, trust radii and decrease threshold.
Bind target/witness IDs, q/r, baseline endpoints and every gradient row in the
immutable package; never reselect after rejection. Require actual-MPM support
directional finite differences and common-row/ownership checks before launch.

Bound the diagnostic operator at eight total halfspaces (at most256 active
subsets). If more than five support targets are selected, stop without running
either search; do not truncate, batch or choose convenient targets. This is a
declared computational limit, not a physical parameter or feasibility claim.
Zero selected targets would produce identical constraint sets: report no
intervention and skip both searches. Generalize active-set enumeration only with analytic and
independent constrained-oracle tests, including rank-deficient rows and
near-dependent scaled constraints; retain existing two/three-plane tests.

Each arm keeps one origin,11 radii, up to two replaced model-remainder corrections
per radius, at most33 search forwards and at most one accepted repair. Both
arms run regardless of the first arm's result. The actual support scalars must
pass with no tolerance on the treatment candidate and all three fixed-control
repeats. Original P306, original prepared constraints, resolved-running and
report-only original-merit decisions remain separately reported. A failure
does not move the origin or start another strength/weight search.

Persist every valid search endpoint (X_T and fixed witness positions/scalars),
plus full X/V trajectories and terminal F/C for the first candidate in each arm
that restores that arm's prepared constraints and for any selected candidate
and its repeats. This removes P312's inability to localize rejected endpoint
support. Keep arrays from the actual evaluated forward; never regenerate
purported evidence afterward. Record checksum-bound actual control parameters.
For all targets, record separate candidate losses/gains relative to each
original baseline, plus repeat ambiguity. Protecting the selected targets can
create losses elsewhere; an unchanged aggregate P306 pass alone is not a
per-ID preservation certificate. A fixed-witness failure cannot disprove
feasibility under supplier reassignment.

All CUDA numerical work stays on hyde06; archive/hash/report I/O stays explicit.
Report actual silhouette/PBR/weighted-render changes and original lambda/norm
share separately from raw support and material motion. These raw coverage
measurements now help select a repair and cannot serve as independent evidence
that the method generalizes. Production adoption still requires the original
merit/complete-trajectory contract, held-out repeats, actual coupled continuation,
gallery checks and framewise visual QA. This experiment neither admits archives
nor adds an adoption/pinning/rest policy or modifies the export renderer.

Implementation extends `SharedRepairBaseline` only when explicitly requested:
it owns the three actual original endpoint arrays. The common origin owns the
selection, radius, target coordinates, witness IDs, all endpoint coverage bits,
distances, scalar values and gradients. Twelve helper CPU cases check selection,
common-witness failure, cap/no-intervention, real supplier replacement, inclusive
boundary and directional derivative, and malformed/nonfinite input/output.
The operator's `active_input_rows` maps past removed zero rows; existing `active`
continues to index compressed rows.

Analytic callback tests check common-row equality, one gradient forward, all-target
loss/gain reporting, actual rejected endpoint persistence and support-only
held-out-repeat rejection. They also check both-arm skip paths. Their raw quality
report is a stated stub, not physical evidence. The injected uncertified-witness
status tests orchestration; a separate helper case verifies the actual changing
supplier geometry. Two real Warp-CPU observer cases (surface channel off/on)
check physical support directional finite differences and production isolation.
Their sparse common-origin fixture is explicitly expected to select zero support
targets: its common-package test does not certify multiple support adjoints in
an actual CUDA300k solve. The separate real-MPM derivative check uses a fixed
target/material pair. No physical or visual acceptance follows from CPU checks.

The first held-out replay fixture also crossed the original render ceiling due
to FP32 rounding; its injected synthetic position was adjusted to put that
unrelated component strictly inside its original ceiling. The final test fails
support alone while passing the unchanged original gates. No production
tolerance, cutoff or objective was relaxed. Independent review reran all62
distinct cases, including the final uncertified-witness orchestration case.

## Completed CUDA comparison: support restores locally, raw silhouette still fails

Frozen `db241d029356dd4593ecc2551450d8372b579549`, W20,N300000,T20,
dt1/240,dx.3062907543956724wu,loss36^3,budget8,raw/no-PIC/no-shift.
Twenty ordinary commits/160 inner accepts; all recorded physical guards zero
and production callback state remains exact. Fixed free54,716/arrived-free54,529
IDs. Original-merit closure passes all three original repeats. The common
origin selects exactly target53946, witness250580 at radius.0698616982373597wu;
no cap/uncertified/no-intervention abort. The three baseline coverage bitsets
are identical. The original target coverage is297798/300000; the common origin
loses one target and gains three. A positive net count does not preserve all IDs.

| Arm | Search records / valid forwards | Prepared constraints restored | Raw P306 pass | Accepted / repeats |
|---|---:|---:|---:|---:|
| Volume/render/silhouette |24 /22|4|0|0 /0|
| Same + fixed material support |28 /26|2|0|0 /0|

All48 valid forwards fail raw silIoU. Treatment h4/c2 restores its prepared
constraints but loses six original-covered targets, gains five and fails upper
coverage. The selected witness remains covered; preserving one witness does
not prevent deficits elsewhere.

Treatment h8/c1 is the nearest measured joint candidate: no original-covered
target IDs are lost and three are gained (61596,187210,188522). It passes all
original gates except raw silIoU, .9606576095811922 -> .960616020068067
(delta-4.1589513125162014e-5). Stored/geometric terminal RMS decrease30.3650%/
30.0497%, saved-step RMS1.5984%, net1.5643%, path2.9942%, all on the same fixed
arrived-free cohort. Original merit decreases1.0090932054543714e-6 relative to
baseline0. These are one-window reductions, not final rest.

Its support scalar is-2.428727167001314e-6, only about8.48e-8wu inside the
unchanged radius. The raw quality rejection prevents fixed-control repeats;
there is no repeated or robust support certificate. Control h8/c0 loses53946
while retaining the same three gains and has identical raw IoU. This saved pair
shares a radius but uses different correction counts; it is not a matched
single-update direction ablation. Both remain rejected.

Rendering at this W20 prefix uses18 views at64pixels. At lambda.022316043302899786,
treatment h8/c1 improves prepared silhouette6.845220923423767e-8 but worsens PBR
6.82193785905838e-8. Combined render decreases only2.3283064365386963e-10;
weighted render decreases5.1958587240119765e-12. Thus better combined image loss
does not certify raw silhouette preservation. The160 ordinary accepted updates
have median nominal render-direction share.4905880; the20 lambda values have
median.0616080. These are direction/weight telemetry, not causal displacement
shares and not4K appearance evidence.

The independent result audit checks every decision, common row/bound identity,
all48 saved endpoint controls/witnesses and reconstructed projected-step norms
and affine remainders. A separate read-only server audit matches64 sidecars,
79 source/helper files and3 inputs (146 stable file identities/hashes). The
inherited unused server-only volumetric.py matches its previously retained
bytes; CRLF/LF differences versus Git are explicitly recorded, not ignored.
No MPM rerollout is used to manufacture rejected-state evidence.

Persistence limitation: P314's full-state sidecars contain complete X/V histories
but only terminal F/C. They do not contain the full per-step F sequence required
for consistent export/adoption. The later optional full-F capture is separately
CPU-reviewed and was absent from this frozen run; no state is eligible for
production adoption on the P314 archive alone.

Evidence is in [evidence/p314/result.json](evidence/p314/result.json), its bound
protocol, independent audit and remote hash receipt. Full arrays remain under
the server's `work/p303/support_repair1`. [P315](silhouette_pixels_p315.md)
localizes saved raw mask/supplier changes before another repair formulation.
Eventual stopping-window analysis, coupled continuation, gallery/all-frame
hole/rest QA and4K appearance remain open. No physical/default policy changes.
