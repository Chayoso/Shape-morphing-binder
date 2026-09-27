# P294 geometric-rest full-horizon comparison — rejected as a quality repair

This is a follow-up to the [eight-window prefix](geometric_rest_prefix_p294.md),
whose modest shape/supply improvement did not establish free-particle rest.
The pair extends the same formulation to a cap of 60 windows to observe
later transport and settling. The current formulation is rejected as a default
quality repair: final silhouette remains below the 0.971 gate, tip supply and top
coverage regress, and identical free particles move farther and reverse more often
between commits. Raw-step and boundary-motion reductions are real observations in
this pair, but they do not establish rest or eliminate the thin-region deficit.
The opt-in implementation remains a diagnostic; no renderer promotion follows.
Root launched control on GPU0 at 2026-09-27 01:09:30 UTC and geometric rest on
GPU2 at 01:10:34 UTC, 64 seconds apart, using the unchanged geometry snapshot.

## Completed simulation metadata

| Completed run | Control | Geometric rest |
|---|---:|---:|
| Runtime (s) | 635.7866 | 630.3027 |
| Torch peak allocation (GB) | 8.3613 | 8.4396 |
| Delivered accepted windows / attempts | 35 / 38 | 34 / 37 |
| Last delivered attempt | 35 | 34 |
| Final pinned fraction | 0.914323 | 0.788900 |
| Final accepted full-plan arrival fraction | 0.999970 | 0.999650 |

Both logs terminate after three consecutive outer-merit rejections at attempts
38 / 37, retaining the prior accepted state. Their `converged=true` field labels
this stopping condition; it does not establish rest or accurate fit. Both report
no truncation and all state guards zero. All 35 / 34 accepted endpoint packages
come from accepted buffers and have exactly zero objective/commit position gap.
Rendering is still paced at the final delivered states. Different GPUs and
different numbers of accepted/attempted windows preclude a simple timing claim.

The complete JSONs and logs are local `output/p294/geom_control60.*` and
`geom_rest60.*`. The exact serialized configuration difference is only the absent
legacy-false `geometric_rest` flag becoming true. The numerical aggregate and
complete MPM dictionaries match. Both accepted histories contain full-plan arrival
evidence throughout; the formal raw audit additionally validates inputs and
recorded `auto` resolution. Only the first 34 accepted endpoints are shared.

## Shape and supply at equal accepted progress

All values here use N=300k, T20, dt1/240, dx=0.3062907544 wu and loss36³. Motion
uses source-native spacing 0.03498853660707278 wu; target coverage uses target-native
spacing 0.03493084911867985 wu. These are raw particle measurements, independent
of the display renderer and optimization-loss operators.

| Common accepted endpoint 34 | Control | Geometric rest |
|---|---:|---:|
| Raw silhouette IoU | 0.967615 | 0.969363 |
| Symmetric mean nearest-neighbor Chamfer (wu) | 0.0588583 | 0.0587372 |
| Global target coverage within 2 target spacings | 0.994797 | 0.994873 |
| Top target coverage (y>2.3 wu) | 0.964929 | 0.960554 |
| Top relative density | 0.948111 | 0.947998 |
| Top under-half-density fraction | 0.111457 | 0.116920 |
| Target-tip ball particle count | 52 | 42 |
| Endpoint trajectory minimum det F | 0.861916 | 0.855959 |
| Stored physical mean speed (wu/s) | 0.00857255 | 0.0200691 |
| Pinned fraction | 0.913060 | 0.788900 |

At the control's later endpoint35, IoU is 0.967479, top coverage 0.964799 and tip
count52; the candidate stops at34. These unequal final endpoints are retained,
not treated as equal elapsed transport. The minima of accepted trajectory det F
over each complete run are 0.861703 / 0.855959; all guards remain zero.
Density counts source neighbors within the target median eighth-neighbor radius,
excludes self and divides by8. Coverage and density do not prove watertightness.

| Accepted commit | IoU control / candidate | Top density control / candidate | Top coverage control / candidate | Tip count control / candidate |
|---|---|---|---|---|
| 6 | 0.894816 / 0.895022 | 0.608457 / 0.629042 | 0.611546 / 0.610044 | 0 / 0 |
| 10 | 0.933252 / 0.934761 | 0.729620 / 0.739925 | 0.828109 / 0.823994 | 17 / 14 |
| 20 | 0.967595 / 0.968685 | 0.883316 / 0.886197 | 0.952194 / 0.951672 | 43 / 42 |
| 30 | 0.967815 / 0.969264 | 0.934724 / 0.940878 | 0.964407 / 0.958595 | 49 / 42 |
| 34 | 0.967615 / 0.969363 | 0.948111 / 0.947998 | 0.964929 / 0.960554 | 52 / 42 |

The fixed Chamfer first crossings at 75%,50%,25%,22.5% of initial distance occur
at the same commits1,2,7,19. Neither arm reaches22% or lower. First crossings only
bound progress; the two states and actual distances still differ.

The radius0.25 wu target-tip ball peaks at56 particles (control rawframe228) versus
63 (candidate rawframe238), ending at52 versus42. Of the IDs present at each
arm's first peak,33/56 versus30/63 remain;23 versus33 leave, while19 versus12 new IDs
appear. Peak-to-end retention intervals differ and transient overshoot can raise a
peak, so retention alone is not the shape verdict. The lower common34 tip count
and top coverage provide the matched endpoint evidence.

## Same free material: smaller raw steps, worse commit movement

The primary comparison uses exactly2,640 material IDs on the union of common34
sparse boundaries that are unpinned in BOTH arms. They stay free throughout the
24→34 endpoint interval under monotone pin admission. The ID SHA256 is
`caf0b1cd63ef833611ac7ebf64a03b96131cfa0768fd48bd65a12d9e8cb0b3b2`.
There are10 accepted window displacements,200 raw steps and201 raw states; no
interior null-held rows or trailing presentation hold enter these motion values.
Both normal/tangent bases are frozen from their respective arm's endpoint geometry.
This is an outcome-selected population and is descriptive, not an unbiased
population causal estimate.

| Same2,640 free IDs, endpoints24→34 | Control | Geometric rest |
|---|---:|---:|
| Raw step median / p95 (sp) | 0.0141297 / 0.0555370 | 0.0132556 / 0.0455785 |
| Absolute normal raw-step median (sp) | 0.00717889 | 0.00640984 |
| Tangential raw-step median (sp) | 0.00915466 | 0.00907496 |
| Raw reversed pairs / eligible pairs | 38,304 / 525,358 | 28,657 / 525,358 |
| Raw reversal fraction | 7.29103% | 5.45476% |
| Commit step median / p95 (sp) | 0.135939 / 0.396835 | 0.150093 / 0.447952 |
| Absolute normal commit-step median (sp) | 0.0491932 | 0.0568221 |
| Tangential commit-step median (sp) | 0.105998 | 0.119374 |
| Commit reversed pairs / eligible pairs | 1,900 / 23,760 | 3,880 / 23,760 |
| Commit reversal fraction | 7.99663% | 16.3300% |
| Median commit net displacement / path length | 0.830765 | 0.865581 |

Raw median motion falls6.19%, while commit median motion rises10.41% and its
reversal fraction roughly doubles. This cannot be attributed merely to mixing
new pins into the displayed motion median: these exact IDs stay free in both arms.
It nevertheless does not isolate the mechanism from changed geometry, controls,
render balancing or GPU trajectory variation. It rules out interpreting smaller
raw jumps as successful settled motion in this pair.

The6,712-ID pre-treatment source upper-surface cohort has zero median motion in
both arms because most members become pinned. Pin fractions over24→34 are
78.9482%→86.9041% in control and67.8933%→78.5757% in candidate. Raw p95 is
0.0257474→0.0220986 sp, but commit p95 is0.176984→0.198224 sp. Raw reversal
fractions are16,141/218,325=7.39311% versus20,506/387,421=5.29295%; commit fractions
are1,244/9,574=12.9935% versus2,568/17,324=14.8234%.

The endpoint-free UNION has7,365 IDs selected at each arm's own stopping endpoint.
At common34 its pin fractions differ47.2641% /17.1079%, so its raw median
0.00607250→0.00799280 sp and commit median0.0545363→0.0814581 sp mix pin status.
The original arm-specific tail cohorts likewise differ (3,147 versus5,400 IDs)
and are not used as the primary same-material comparison.

## Phase localization

The phase audit uses the identical2,640 IDs and24→34 interval above. Phase20
combines the last raw rollout step, including layer effects, with PIC remapping;
subcell shift is disabled in both arms. These archived positions cannot separate
the contributing operators.

| Phase statistic | Control | Geometric rest |
|---|---:|---:|
| Phases1–19 median displacement (sp) | 0.0134818 | 0.0126741 |
| Phase20 median / p95 displacement (sp) | 0.0885156 / 0.686159 | 0.0493047 / 0.542055 |
| Phase20 absolute-normal median (sp) | 0.0480344 | 0.0288839 |
| Phase20 tangential median (sp) | 0.0486180 | 0.0326365 |
| Phase20 share of total raw path length | 38.1246% | 31.9294% |
| Reversals19→20 | 19,535 / 26,400 (73.9962%) | 13,958 / 26,400 (52.8712%) |
| Reversals20→next1 | 18,470 / 23,760 (77.7357%) | 13,839 / 23,760 (58.2449%) |
| Interior-phase reversals | 299 / 475,198 (0.06292%) | 860 / 475,198 (0.18098%) |

Boundary motion and reversal rates decline materially but remain nonzero and
frequent. At common34 alone, phase20 median is0.0882850→0.0529442 sp and path
share41.1499%→37.8513%. Across all particles at that endpoint, the final promoted
displacement/dt mean speed is0.109838→0.0970413 wu/s; recorded physical v_T mean
speed is0.00857255→0.0200691 wu/s. These distinct definitions and different pin
mixtures prevent interpreting their ratio as a damping fraction. Direction
reversals are not by themselves proof of periodic oscillation or image flicker.

## Pins, objective telemetry and interpretation limits

The full raw archive audit reports exactly zero drift for274,297 /236,670 admitted
pins. Each archive contains one duplicate held state after the last accepted
endpoint (702/682 delivered archive frames versus701/681 accepted rollout states).
The final379 control pins and20,559 candidate pins therefore have only that held
state as a later observation; this is not evidence of stability under additional
physics. The other273,918 /216,111 pins have subsequent accepted rollout states.
The common free-motion analysis excludes the held state entirely.

At candidate endpoint34 the frozen arrived/free objective cohort contains83,790
particles. Whole-N components in(wu/s)² are raw0.001925217, remap0.169148624,
total0.171073839, delivered0.156041518 and cross−0.015032311. Conditional raw/remap
are0.006893006 /0.605616269. The existing nominal kinetic weight is5; the recorded
unit multiplier is3.0299945792e−5 and effective weight0.000151499729. Cohort membership
changes by window, and the control does not record this optional decomposition;
these are not an on/off reduction measured on identical objective-eligible IDs.
Endpoint-work telemetry also excludes direct raw/previous-position paths of this
term, so it cannot be used as the term's complete work.

Both logs retain paced rendering through all38/37 attempts; no fixed-reference
handoff occurs. Median accepted render lambda is0.0360305 /0.0878195. This adaptive
response is part of adding the physics penalty, not an independently held-constant
render force. The full runs also diverge from their earlier8-window executions:
at window8 control IoU is0.913331 versus0.912640 in the separate prefix run, and
candidate IoU is0.915098 versus0.915179. Both use the same frozen numerical core
and initial recipe but different stopping caps; this comparison is context for
trajectory variation, not a repeat-noise confidence interval or significance test.

The decision concerns this fixed formulation and discretisation. Lower raw and
boundary motion does not satisfy the required combination of final shape, thin
supply and stationary free particles. No display-rendering result is promoted,
and the user-facing hole/flow/oscillation objective remains unresolved.

## Measurement contract and deployment

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

Completed artifacts are `work/p294/geom_control60` and `work/p294/geom_rest60`.
The separate audit environment is `work/p294/audit_full`, built from the immutable
`output/p294/geometry_snapshot.zip` with only the reviewed audit-script overlays.
Its numerical source is the same 57-file aggregate
`fd7c0ba18463ed6ddc232a800850372134e540c40af4000698b83b4053192f6a`.
The already executed prefix environment `audit_v2`, its results and its prepared
probe copies remain unchanged. Root owns all GPU launches; this audit preparation
does not start a simulation or extend a run.

Local preparation files and receipts reside in `output/p294/` with the
`audit_full`/`prepared_full` names. GPU0 launchers are `quality_geom60.sh` and
`phase_geom60.sh`; completed outputs are `quality_geom60.json` and
`phase_geom60.json`. Remote deployment verification checked all 148 manifest
entries and all 57 numerical files against the simulation snapshot; its receipt
is `audit_full_provenance.json`. Executed quality/phase script SHA256 values are
`3288a4deb0b75b1d13544bad1107b0f7ffa2c982d54361437fadea020cfe067f` and
`b09c12e69a7764aca603c99625f3316faca6bb99150875243628a5e98cff3617`.
The full-mode code review and 21 independent CPU scope tests passed. The separate
deployment readiness review closed after checking every ZIP member, the exact
two overlays, numerical/probe hashes and launcher paths. The separate final
numerical/causal report review is closed: it checked the result tables, fixed-ID
cohorts, held-tail pin limitation, termination and provenance against the JSONs.
It supports rejection of this formulation as a quality repair.

Root launched the quality audit on GPU0 at2026-09-27 01:22:54 UTC and the phase
audit at01:24:27 UTC,93s later. Result JSON SHA256 values, distinct from the script
hashes above, are `668f11106491f4243d3c87e6a2eba5b2b0fed84c907668389140d1bea0db0bab`
for `quality_geom60.json` and
`a9f6d86603eacfe63fc9cf0400e5aa6581b4ae489749c600a4f5ec6726bc4730`
for `phase_geom60.json`. The phase result references the exact quality-result hash.
Both are retained locally in `output/p294/` and on hyde06 under `work/p294/`.

The prepared1080p overview remains unlaunched; neither this report nor the raw
comparison establishes a new rendered deliverable. At01:27:53UTC the server
project used94,698,953,688bytes (94.699GB), below the user's100GB cleanup threshold.
