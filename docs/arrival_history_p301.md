# P301: individual arrival and subsequent material motion

Status: read-only observer implemented;16 CPU tests passed independent review,
including exact two-window observer-on/off parity on160 particles/T3. The
cap24 CUDA capture and saved raw-final/promoted support audit completed. No new
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

## CUDA capture result

Frozen d3d7f26 accepts24/24 in464.808s with zero guards,481 delivered physical
frames, no held suffix or trimming, and337,158,728bytes of evidence. All24
observed start masks match the independently recomputed start predicate; all
owned/promoted and start-pin equality checks pass. Numerical arrays use CUDA;
no new simulation policy or objective is introduced. The discretization is
the N300k/T20/dt1/240/dx0.3062907543956724/loss36^3 scope above.

The arrival radius is0.30629074573516846wu in every window, or8.754031 native
source spacings. Initially53205 IDs qualify;246631 first qualify at a later
accepted endpoint. Of299836 ever-qualified IDs,299825 have a subsequent
physical observation and11 do not. Another164 never qualify;18 previously
qualified IDs are outside the last endpoint's current plan radius. These are
policy classifications, not exact optimum or settled-position certificates.

The full history contains85175 plan-departure and1956 plan-reentry events at
unchanged accepted/start coordinates, versus984 geometric-departure and84185
geometric-reentry events within a frozen plan. These are repeated event counts,
not distinct particles. Plan reclassification is much less frequent late
(only4 departure events at W24); its large whole-history count does not explain
all late motion. Escaped IDs remain in subsequent motion observations.

| Previously qualified and free at the W24 window start (56220 IDs) | RMS in source spacings |
|---|---:|
| Last raw physical/layer step |0.0242151|
| Final PIC position correction |0.254279|
| Saved final step, including both |0.241968|
| All20 saved steps |0.0584255|

Source spacing is0.03498853660707278wu. The first three rows are separate
vector magnitudes, not additive fractions of motion. W24 has68689 reversals
over1124293 eligible adjacent pairs (6.1095%), including the previous accepted
window boundary. This does not measure periodic-oscillation frequency or arrival
at an exact optimum. It establishes nonzero saved motion after the recorded
policy arrival, with a large final position correction on those same IDs.

Across the entire post-arrival history, free-observation RMS is0.119765sp over
62861480 particle-steps, with varying observation durations per particle over
a2s simulated horizon. This is not a
matched late-motion A/B metric. Previously admitted pins have exact zero path
and displacement on243605 observed IDs;246309 are finally admitted, leaving
2704 without a later physical window. Nonzero pinned-bank excursion from the
first-arrival anchor records travel BEFORE pinning and must not be called pin drift.

This fresh realization differs from the earlier corrected P300 cap24. At W24,
pin fraction is0.82103 versus0.79970, adaptive lambda0.0392263 versus0.0309309,
and trajectory det(F) minimum0.866774 versus0.872049. The small CPU observer
parity test is not proof of bitwise CUDA replay or equivalence of these cohorts.
The output retains per-window scalar comparisons rather than claiming an
observer-induced improvement or deterioration.

## Raw-final versus promoted support

Frozen2d57751 reads the completed capture on CUDA without rerunning physics.
The archive-only probe passed5 independently run CPU tests and binds the exact
capture, quality report, helpers and all24 sidecars before loading arrays. It
uses the same6712 fixed source IDs and15312 fixed upper target IDs as P300,
target spacing0.03493084911867985wu and density radius0.06898659982768912wu.

At W24, upper target coverage changes from95.1803% at raw x[T] to96.0423%
after PIC, versus96.2056% at the previous accepted start. PIC covers175 target
IDs and uncovers43 (net132); of14641 IDs covered at BOTH window endpoints,
155 are uncovered at raw x[T]. Their maximum raw gap is2.80732 target spacings.
This closes the historical missing-raw-final observation gap for this run.

The same operator reduces fixed-source density from0.718750 to0.696514.
Support falls below the four-neighbor threshold for126 of5182 raw-supported IDs, while
22 of1530 raw-unsupported IDs gain support. Eight IDs among those126 losses
have no other neighbor within the diagnostic radius afterward. These source IDs
need not be free or moving: neighbors can leave a pinned point's neighborhood.
Across this run, mean fixed-source density decreases in24/24 windows and target
coverage increases in23/24 (W1 raw/promoted upper-target coverage is zero). The operator improves
target coverage here while spreading local material support; it is not a
uniform hole-removal operation. Its net effect cannot be judged from either
the average density or target coverage alone.

These are direct geometric changes under the final position operator at each
observed state. The audit has no phase19 state, full-window minimum, later
dynamical intervention, watertightness test or new renderer result. No pin rule,
geometric-variance weight, momentum kick or physical default is promoted.

Evidence: local output/p301/{protocol,result,boundary_supply24}.json; full
sidecars remain at server work/p301/arrival24. Result SHA256:
c94c3ba8b3deb6e3c6d86519e86005a46946487414c8d08a5c5d5387b90f0208.
Protocol SHA256:2cf2072f9767b3131b41cbd9603a861a04b842d192942b277cea1a9cfce8bb3b.
Boundary audit SHA256:94bc54b19e9258c02026bc65877bc7173de8d12d4cd76b804a8cb8c51aa62bf4.
