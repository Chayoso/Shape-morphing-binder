# P301: individual arrival and subsequent material motion

Status: read-only observer implemented;16 CPU tests passed independent review,
including exact two-window observer-on/off parity on160 particles/T3. Production
cap24 CUDA capture is pending. No new
physical objective, control gain, arrival radius, pin admission rule or renderer.

P300 corrected a constitutive derivative but did not repair free-particle rest
or transient coverage. Its old/new cap24 archives lack per-ID arrival evidence
and the raw final position before PIC. An accepted-endpoint arrived fraction
cannot identify which particles subsequently leave, and pin admission is not
arrival. This capture closes those observation gaps on the corrected recipe.

## Frozen scope

Use the original bunny300k inputs, T20, dt1/240,
dx0.3062907543956724wu, loss36^3, eight inner iterations and cap24. Keep shared
PIC, promoted-state outer render, motion accounting and the current physical
velocity variance. Shift, geometric rest and geometric variance stay off.
All numerical observation work runs on hyde06 CUDA. Serialization and the
existing public commit callback's host transfers remain output I/O.

The optimizer's validated rollout observer supplies an owned temporary copy of
the actual position sequence. The return supplies the frozen full-plan images,
arrival radius and start-arrival mask. The observer checks these against the
owned shared endpoint and admits evidence only after the corresponding outer
accepted commit. Rejected attempts and null/held frames do not advance the
observation clock. No additional objective evaluation is required.

## Identity and time conventions

Arrival uses the executed policy's full-plan distance test and radius. It is
not a uniquely optimized point or a rest tolerance. Plans can change between
windows. Store each actual plan and both endpoint masks instead of reconstructing
a new plan after the run.

IDs initially inside the first accepted window's plan radius form a separate cohort,
anchored at the source, whose W1 motion is observed. Other IDs acquire a first
arrival anchor at their first accepted endpoint satisfying that window's policy
test; their post-arrival motion starts at the next physical step. If qualification
persists to the next endpoint, detection can lag by T*dt=1/12s. A transient
intra-window entry followed by exit before that endpoint can be missed entirely.

Keep IDs after escape and reentry. Distinguish a membership change between the
previous endpoint and the next start under a different plan/radius from movement
between start and end under one frozen plan. Excursion from the first-arrival
position remains meaningful independently of later plan membership.

Measure position-derived path, net displacement, step squared sum/count and
maximum, excursion from the arrival anchor, and direction-reversal counts with
their eligible-pair counts. Keep raw physical/layer x[T] and the promoted endpoint
separate, so the last raw step and PIC jump are not conflated. Reversals across
accepted-window boundaries retain the immediately preceding saved displacement,
including a zero step; both members must exceed the existing movement floor.

Separate free observations from observations under a window-start pin. The
commit callback precedes new pin admission; use subsequent accepted starts and
the final admission record for that timing. A final arrival or pin with no later
physical observation is unobserved, not settled. Enforced zero movement must not
dilute the free-material statistics.

## Storage and interpretation

Avoid another full movie archive. Per accepted window retain the frozen full
plan, raw x[T], promoted endpoint and three boolean masks, approximately11.7MB
for N300k. Together with source coordinates and final per-ID counters, the
preflight enforces a350MB total sidecar budget (346.3MB planned, including10MB
for headers and JSON). Temporary path copies stay
on CUDA and are discarded after accepted statistics are accumulated.

These sidecars permit independent recomputation of endpoint arrival masks and
raw-final versus promoted displacement. The accumulated full-path statistics
are tested observations, not a complete stored trajectory from which every
intermediate position can be reconstructed. The existing full archives remain
the evidence for P300's all-saved-frame supply audit.

This is a fresh observational run, not the same realized trajectory as the
earlier candidate24 or a separate proposal audit. Compare its settings and
endpoint telemetry explicitly; do not transfer material cohorts between runs
or assume bitwise replay. Neither a completed capture nor a high policy-arrival
fraction establishes no holes or natural rest.
