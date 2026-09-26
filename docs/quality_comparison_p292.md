# P292: raw quality comparison for body displacement RPROP

Status: both matched full simulations and both CUDA audits completed on hyde06
on 2026-09-26. **Do not promote body RPROP as a rest/holes fix.** It leaves the
shared free material moving more, despite a small endpoint silhouette increase.
The separate two-spacing stress taper also fails its eight-commit supply screen.

The intervention applies the existing particle RPROP scale to the body's
displacement-control mode through active, unpinned interpolation weights. Nodes
with a contribution from material still in transit retain scale 1. The independent
terminal-velocity control mode remains unchanged. The audit measures the resulting
whole pipeline, including any different accepted windows or stopping decisions.

Both arms use bunny source and target clouds of 300000 particles, MPM T=20,
dt=1/240, dx=0.3062907544 wu, loss grid 36 cubed, eight optimizer iterations,
animations=300 and cap=60. Source-native median NN spacing is approximately
0.0349885366 wu. Both retain the render channel and identical stress/body/surface-u
controls. The only permitted configuration difference is `body_rprop` off to on.
The comparison refuses unequal input clouds, MPM parameters or simulation code hashes.
The loaded numerical audit code must match that same simulation hash.

An explicitly named `--intervention stress_taper` supports a separate exploratory
screen. It permits only `ctrl_taper_sp: 0 -> 2` and `stop_after_windows: 60 -> 8`,
with body RPROP off in both arms and the same strict input/code/MPM checks. Its
analysis is limited to the first common accepted prefix, at most eight commits.
The baseline's later states never enter its geometry or cohort metrics. Original
metadata is retained; output records the scope and original run lengths explicitly.
Prefix endpoints and full-run runtimes must not be compared as final quality or
convergence. The default body-RPROP audit does not permit these extra changes.

After the boundary-phase diagnosis, a third explicit mode, `commit_pic_off`, was
added for a separate eight-commit screen. It permits only `commit_pic: true -> false`
and `stop_after_windows: 60 -> 8`; body RPROP remains off and subgrid shifting must
remain unchanged. It uses a newly selected common-free cohort at commit 8, not the
1374 late-run IDs. This is an early whole-operator ablation, not evidence for late
rest or attribution among the remaining position changes. Results are below.

The subsequent `commit_pic_off_full` mode requires cap 60 in both arms and permits
only the PIC flag change. It preserves each complete run for endpoint, pin and
tip-history audits; comparisons still use common accepted commits and fixed
progress, with the late phase interval taken from the last ten shared intervals.
It does not reuse the eight-commit scope or cohort.

## Measurement

`scripts/probes/quality_compare.py` runs numerical geometry and motion on CUDA.
It reuses the reviewed archive loader and raw audit without modifying their prior
measurement scripts or artifacts. Host work is limited to archives, hashes and
metadata. The probe records its own hash and the imported probe dependency hashes.

At each accepted commit it measures independent binary silhouette IoU, symmetric
mean nearest-neighbor distance, target NN coverage globally and in the fixed bunny
region y>2.3 wu, density in that region, tip-ball mass, det F and terminal velocity
telemetry. Coverage means target particles within two target-native spacings of the
current cloud. Density means neighbors inside the target's median eighth-neighbor
radius, excluding self, divided by 8. Neither is a watertightness test.

The tip ball has radius 0.25 wu around the target's highest-y particle. Its count is
measured at every archived simulated frame, excluding held padding. Peak and final
counts are accompanied by material-ID retention: particles that leave the peak tip
ball are distinguished from particles that enter later. A transient overshoot may
inflate the peak, so retention alone is not a quality verdict.

Comparisons use the same accepted commit ordinal and first crossings of fixed
fractions of the initial raw Chamfer distance. The actual mismatch in Chamfer at
each crossing is retained. Movie time is never normalized to hide different run
lengths or stalled transport. Endpoint metrics are also reported separately.

## Motion cohorts and limitations

Three common material cohorts are fixed across arms:

- **Source upper surface:** selected only from the source before either treatment.
  A particle lies at or above the source bounding-box y midpoint and has fewer than
  0.6 times the median number of source neighbors within two source-native spacings
  (this cohort count includes self).
  This source-only density boundary is reproducible; it is not guaranteed to identify
  exactly the material that will become a bunny ear.
- **Endpoint free-surface union:** union of each arm's final sparse-surface,
  unpinned IDs, using the same density rule. An ID may be pinned in one arm. This is
  selected from outcomes and is explicitly descriptive rather than a pre-treatment
  sample.
- **Free in both arms:** at the same last common accepted commit, form the union of
  the two sparse-boundary masks, then keep only IDs unpinned in both arms at that
  time. With monotone pins these IDs remain free throughout the measured interval.
  This prevents a lower motion median caused solely by more pins from being called
  a fix for free-particle motion. This cohort is still selected from outcomes.

All IDs are used up to 20000; larger cohorts use a deterministic evenly spaced
selection from sorted IDs. The selected-ID hash is recorded. Motion is compared
over the last ten common accepted-commit intervals and every raw frame inside that
same interval. Only archive stride 1 is accepted. Actual accepted rollout spans
are reconstructed from T and each accepted `frame_end`; interior null-commit holds
and trailing padding are excluded before motion differencing. Absolute displacement and direction
reversals use the exact same material IDs. Normal/tangential decomposition uses
each arm's own frozen 33-NN centroid-offset direction at the common endpoint;
these directions need not agree between arms and can be noisy in the interior.
They are local geometric diagnostics, not a shared pre-treatment normal basis.
Each cohort also reports its pin fraction at the interval start and end: reduced
motion may come from more pin admissions rather than quieter free material.

A reversal is a negative dot product of successive displacements, both larger
than 1e-4 source-native spacings. Eligible pair counts and net/path ratios are
reported alongside rates. This detects sampled direction changes, not a periodic
vibration mode. The existing raw audit separately checks admitted pins over every
delivered raw frame and reports each arm's own final unpinned-surface tail.

Reduced late motion is insufficient for promotion if the candidate loses early
thin-region supply, tip material, target coverage or valid deformation. Exact pin
invariance does not establish that every particle has reached its correct resting
position. Rendered output, if produced later, still requires its separate frame QA.

## Body RPROP result

The baseline accepted 32 of 35 attempted windows in 567.18 s; body RPROP accepted
35 of 37 in 607.40 s. All recorded safety guards were zero. They share numerical
source SHA-256
`eee25aec5c769c9037852f1cc49c0a50da0b2b9b816e39c10550e7c12c823dbe`.
The only configuration difference is absent/default-false versus true
`body_rprop`. Endpoints below are each run's own stopping point; the next table
holds accepted time and material IDs fixed.

| Endpoint measurement | Baseline | Body RPROP |
| --- | ---: | ---: |
| Binary silhouette IoU | 0.973039 | 0.973517 |
| Raw Chamfer, wu | 0.055922 | 0.055913 |
| Global target coverage | 0.998053 | 0.998307 |
| Top target coverage | 0.989877 | 0.988963 |
| Top mean density | 0.90748 | 0.91902 |
| Top fraction below half density | 0.09174 | 0.09303 |
| Tip-ball particles (target: 89) | 53 | 50 |
| Minimum accepted trajectory det F | 0.76790 | 0.78072 |
| Terminal mean speed, wu/s | 0.01409 | 0.01474 |
| Terminal maximum absolute velocity component, wu/s | 1.32521 | 0.78372 |
| Admitted pins checked for drift | 263917 | 251715 |
| Pin drift, maximum wu | 0 exactly | 0 exactly |

For the **same 1374 sparse-boundary IDs free in both arms**, commits 22 through 32
give 201 raw states and 200 raw displacements. No ID is pinned anywhere in this
interval, and neither archive contains an interior null-held row. All `sp` units
below use source-native spacing 0.0349885366 wu, not a rendered frame spacing.

| Shared free material, commits 22–32 | Baseline | Body RPROP |
| --- | ---: | ---: |
| Per-commit displacement median / p95, sp | 0.17573 / 0.45179 | 0.21309 / 0.66814 |
| Per-raw-step displacement median, sp | 0.02391 | 0.02687 |
| Per-raw-step absolute normal displacement median, sp | 0.01628 | 0.01666 |
| Per-raw-step tangent displacement median, sp | 0.01309 | 0.01561 |
| Commit direction reversals | 740 / 12366 (5.98%) | 515 / 12366 (4.16%) |
| Raw direction reversals | 21981 / 273426 (8.04%) | 22783 / 273426 (8.33%) |
| Per-particle net/path median, using commit positions | 0.88021 | 0.92962 |

At the accepted-window scale this is mostly directed drift with some direction
changes, rather than rest. Raw direction changes coexist with that drift; the
counts do not establish a periodic vibration. The normal bases differ by arm as
defined above, but total displacement and direction tests do not use normals.

The source-only cohort contains 6712 IDs. Its per-commit median is zero in both
arms because most of this source population is pinned; p95 increases from 0.21743
to 0.25659 sp. Pin fractions at the common interval start/end are 0.73808/0.84952
versus 0.69875/0.82464. The outcome-selected endpoint-free union contains 3370
IDs; its per-commit median also increases, 0.08874 to 0.10825 sp. These checks do
not support a quieter population. Each arm's separate final free-surface tail
does show lower medians in the candidate, but those use different IDs and later
candidate times. That descriptive result cannot replace the matched comparison.

The actual conditioning remains weak for typical arrived material: across accepted
windows, the median node-mean multiplier is 0.96754 and the median arrived-particle
effective median is 0.99314. The smallest arrived median is 0.84868 at commit 8;
it recovers to 0.99619 at common commit 32 and ends at 0.99201. The minimum transit
multiplier is 0.99999988, consistent with protection at one within float roundoff.
Late commits 29–35 have no transit-protected nodes, so transit protection alone
does not explain their near-one multiplier.

The existing RPROP trigger reads accepted-window displacement, averaged over eight
source neighbors, halves only arrived reversals and raises same-direction steps
by 1.2 up to one (`runner.py`, control sign-history block). Its movement threshold
is 1e-4 **MPM cells**, whereas this audit's direction-test threshold is 1e-4
**source spacings**. It cannot detect a within-window return that cancels at the
accepted endpoints or sustained one-way drift. Node averaging is another possible
dilution. These are code-level limitations consistent with the measured weak
conditioning; the aggregates do not reconstruct each audited ID's trigger history
or causally separate those mechanisms. The independently controlled braking mode
is not directly multiplied by body RPROP; changed measured braking updates are an
indirect result of different evolving optimization states.

Thin-region supply changes are mixed, not a cure:

| Accepted commit | Top density, baseline / body RPROP | Top target coverage, baseline / body RPROP | Tip count, baseline / body RPROP |
| --- | ---: | ---: | ---: |
| 3 | 0.24088 / 0.24529 | 0.33111 / 0.33275 | 0 / 0 |
| 6 | 0.58177 / 0.55580 | 0.60684 / 0.61964 | 0 / 0 |
| 10 | 0.67977 / 0.69159 | 0.87004 / 0.87213 | 28 / 29 |
| 20 | 0.83487 / 0.84125 | 0.98433 / 0.98041 | 65 / 55 |
| 30 | 0.90494 / 0.90249 | 0.98922 / 0.98890 | 54 / 51 |
| 32 | 0.90748 / 0.91039 | 0.98988 / 0.98870 | 53 / 51 |

The baseline tip peaks at 72 particles at raw frame 377 and ends at 53: 53 of the
peak IDs remain and 19 leave. The candidate peaks at 64 at raw frame 437 and ends
at 50: 49 peak IDs remain, 15 leave, and one enters. Better fractional retention
from a smaller peak is not a fix for reduced final tip mass.

Both arms first cross 0.75, 0.5, 0.25, 0.225, 0.22 and 0.215 times the initial
Chamfer at the same commits 1, 2, 6, 9, 10 and 16. At the 0.215 crossing, actual
Chamfer is 0.0562383/0.0562132 wu and top coverage is 0.96839/0.97342; tip count is
54/46. Neither reaches 0.2 times initial Chamfer. Some shape metrics improve at
matched progress, but no measured gate supports simultaneous thin-feature
preservation and free-material rest. This is one matched pair, with no repeated-run
uncertainty envelope; the 0.000478 endpoint IoU gain is not a significance claim.

## Stress taper prefix result

This is only the first eight accepted commits, with body RPROP off. It is not a
comparison of final run quality. Tapering stress over two source spacings reduces
early top coverage and delays transport despite a higher local mean density at
commit 8.

| Commit-8 measurement | Baseline prefix | Taper prefix |
| --- | ---: | ---: |
| Binary silhouette IoU | 0.91640 | 0.88973 |
| Raw Chamfer, wu | 0.05912 | 0.06840 |
| Global target coverage | 0.98560 | 0.97706 |
| Top target coverage | 0.74608 | 0.57426 |
| Top target gap p95, target spacings | 9.2717 | 30.0095 |
| Top mean density | 0.67609 | 0.69175 |
| Top fraction below half density | 0.29891 | 0.34132 |
| Minimum det F in this accepted window | 0.88483 | 0.87626 |
| Mean terminal speed, wu/s | 0.30053 | 0.31633 |

At commit 6, top density falls 0.58177 to 0.52023 and top target coverage falls
0.60684 to 0.49687. The baseline crosses 0.25 times initial Chamfer at commit 6;
the taper does not cross it within eight commits. Neither has populated the tip
ball by commit 8, so zero tip counts are not a final tip verdict. Both prefix pin
audits find exactly zero drift. The full baseline runtime and the short taper
runtime have different horizons and are not compared as performance evidence.
The supply/shape screen rejects promotion of this taper and does not justify a
full gallery run on its own.

## Raw phase diagnostic: reversals concentrate at the commit transition

An additional archive-only CUDA measurement uses the **same 1374 IDs and the same
accepted endpoints 22 through 32**. It reconstructs accepted windows 23–32, with
20 steps per window, and checks the cohort ID hash against the original audit.
No simulation, optimizer or renderer is rerun. The normal basis is the same
arm-specific frozen endpoint estimator. Tangent components use an explicit fixed
orthonormal basis recorded in the JSON; tangent length is basis independent within
that plane.

The saved phase-20 displacement contains the final rollout step **and** all commit
position corrections. In these runs `commit_pic` and `shift_sub` are both on. The
archive has no separate pre-correction final state, so this diagnostic localizes
the jump to that combined transition and cannot apportion it to PIC, shifting,
layer motion or the last physical step.

| Same free cohort, ten accepted windows | Baseline | Body RPROP |
| --- | ---: | ---: |
| Phases 1–19 displacement median / p95, sp | 0.02293 / 0.05856 | 0.02584 / 0.06159 |
| Phase 20 displacement median / p95, sp | 0.33328 / 0.88520 | 0.33974 / 0.89678 |
| Phase 20 share of summed raw path length | 43.21% | 41.46% |
| Phases 1–19 signed normal median / p95, sp | -0.00959 / 0.02306 | -0.00904 / 0.02500 |
| Phase 20 signed normal median / p95, sp | 0.16744 / 0.84939 | 0.15830 / 0.84178 |
| Phases 1–19 tangent length median / p95, sp | 0.01255 / 0.03095 | 0.01499 / 0.03954 |
| Phase 20 tangent length median / p95, sp | 0.08467 / 0.35245 | 0.09503 / 0.36356 |
| Reversals from phase 19 into 20 | 11040 / 13740 (80.35%) | 11533 / 13740 (83.94%) |
| Reversals from phase 20 into next phase 1 | 10719 / 12366 (86.68%) | 11155 / 12366 (90.21%) |
| Reversals at all other adjacent phases | 222 / 247320 (0.0898%) | 95 / 247320 (0.0384%) |
| Per-particle net/path median, using all raw steps | 0.20119 | 0.26765 |

The two boundary transitions account for 98.99% and 99.58% of all recorded raw
reversals. This is a strong localization in the measured cohort, not evidence that
99% of image flicker is caused by one operator. Directed drift at accepted
endpoints coexists with substantial within-window out-and-back motion; the raw
net/path ratio is much lower than the commit-only ratio in the earlier table.
The RPROP trigger reads only the latter endpoints, so it does not directly observe
these dominant substep reversals.

At common commit 32, the mean **geometric** last-step displacement divided by dt,
over all 300000 particles, is 0.21930/0.24756 wu/s. Recorded terminal-state mean
speed is 0.01409/0.02123 wu/s. These are different velocity definitions: the
geometric value includes commit position corrections and is not a physical
velocity-state estimate. Their disagreement motivates separate stage accounting,
rather than treating the recorded terminal speed as proof of a quiet archive.

All 20 phase distributions, signed tangent components, per-window measurements
and exact counts are in `output/p292/raw_phase.json`. The source probe is
`scripts/probes/raw_phase.py`, executed SHA-256
`9a17479407363caa224f0b6cdd749a331d7391c468415b086dc0b7b300c9eae8`.
It launched on GPU 2 at 23:05:37 UTC and completed at 23:05:43 UTC. Its prelaunch
adversarial gate checked indexing, provenance, cohort identity and attribution
scope; the phase clock regression and five prior comparison tests all pass.

## PIC-off prefix: less boundary normal return, mixed shape and supply

The new candidate completed eight accepted windows in 164.81 s, with all safety
guards zero. Exact source and target, numerical source hash, discretisation and
all other configuration fields match the baseline. `body_ctrl` remains **on**,
`body_rprop` remains off and `shift_sub` remains on. The only configuration changes
are `commit_pic: true -> false` and cap 60 to 8. Both arms are analyzed only through
accepted commit 8, raw frame 160; later baseline frames do not enter any result.

| Commit-8 prefix measurement | Baseline PIC on | PIC off |
| --- | ---: | ---: |
| Binary silhouette IoU | 0.91640 | 0.91316 |
| Raw Chamfer, wu | 0.059119 | 0.059331 |
| Global target coverage | 0.98560 | 0.98423 |
| Top target coverage | 0.74608 | 0.72969 |
| Top mean density | 0.67609 | 0.76023 |
| Top fraction below half density | 0.29891 | 0.23207 |
| Top target gap p95, target spacings | 9.2717 | 9.9514 |
| Minimum accepted det F within prefix | 0.88483 | 0.88570 |
| Terminal mean speed, wu/s | 0.30053 | 0.28502 |
| Pinned fraction at prefix endpoint | 0.35136 | 0.39984 |

At commit 3, top density rises 0.24088 to 0.34028 while top target coverage falls
0.33111 to 0.28383. At commit 6 they are 0.58177/0.59794 and 0.60684/0.60600.
The denser visible material does not imply that the missing target region is being
filled faster. Tip-ball counts remain zero in both prefixes and are not a final
tip-preservation verdict. The prefix pin check covers 161 archived states and
74888/78321 particles with post-admission observations; both have exactly zero
drift. Particles admitted at the final prefix frame have no later observation in
this scope.

The new outcome-selected common-free cohort contains **9314 IDs** and is selected
at common commit 8. It is not the late 1374-ID cohort. The interval is accepted
endpoints 1 through 8, i.e. accepted windows 2–8, with 141 archived states and no
null holds. All these IDs remain unpinned in both arms throughout the interval.

| Same early free cohort | Baseline PIC on | PIC off |
| --- | ---: | ---: |
| All raw displacement median, sp | 0.14575 | 0.14788 |
| All raw absolute-normal median, sp | 0.06483 | 0.05779 |
| All raw tangent median, sp | 0.09804 | 0.10520 |
| Raw reversal pairs | 37917 / 1294646 (2.929%) | 17600 / 1294646 (1.359%) |
| Phase-20 displacement median / p95, sp | 0.33887 / 0.95127 | 0.24813 / 0.64587 |
| Phase-20 absolute-normal median, sp | 0.19268 | 0.07999 |
| Phase-20 tangent median, sp | 0.19870 | 0.20131 |
| Phase-20 share of raw path length | 9.9568% | 7.1182% |
| Reversals 19 -> 20 | 20265 / 65198 (31.08%) | 9273 / 65198 (14.22%) |
| Reversals 20 -> next 1 | 16982 / 55884 (30.39%) | 7553 / 55884 (13.52%) |
| Reversals at other adjacent phases | 670 / 1173564 (0.0571%) | 774 / 1173564 (0.0660%) |
| Per-particle raw net/path median | 0.70676 | 0.73575 |

This matched whole-operator experiment supports a PIC contribution to the early
boundary normal return: both boundary reversal rates and the final-step normal
movement are substantially lower with it disabled. Optimization and subsequent
states also change, so the difference is not the isolated per-step PIC correction
vector. Tangential motion and much of the boundary return remain. It does not
establish a remedy for late rest, nor that disabling PIC improves every shape or
supply gate. The observed silhouette and coverage regressions prevent unconditional
promotion on the density and reversal gains alone. The subsequently completed
[component accounting](motion_accounting_p292.md) supports a bounded full follow-up
with the same gates; this mixed early screen does not itself reject that experiment.

New JSONs are `output/p292/quality_no_pic8.json` and `phase_no_pic8.json`, also under
the server P292 directory. Their probe hashes are respectively
`7b0bfdbdac22f7b61039b94127650f1147a9b214a2cdec02a8dbccddb1207cc4` and
`4c0f6117b703f46a76fc2377f07153b43429999b9472374d2f4454bcd0f26d96`.
The prefix audit launched on GPU 2 at 23:14:53 UTC and completed at 23:15:13; the
phase audit launched at 23:15:56 and completed at 23:16:02. The extension code gate
and final numeric report gate are closed, with seven CPU tests independently passing.

## Artifacts and verification

Server base: `/data/relcfd/chayo/physmorph_v2/work/p292`.
Simulation artifacts are `baseline60`, `body_rprop60`, `taper8`; immutable numerical
code is in `physics`. The audit uses a separate `audit` snapshot with identical
numerical source and added probe scripts. Result JSONs are `quality_body_rprop.json`
and `quality_taper8.json`, also copied locally to `output/p292/`.
Both record probe SHA-256
`0e9410b6826390ca0552d20665c5db43c783ca9232eb151c25a3bbb8c076fa7c`.
No rendered image or rendering-loss operator enters any measurement.

The later `commit_pic_off` probe extension changes the current probe source hash.
The exact scripts that produced the original body/taper and phase results remain
in the unchanged server `audit/scripts/probes` snapshot, with local copies
`output/p292/executed_quality_compare.py` and `executed_raw_phase.py`. Earlier JSONs
and numerical snapshots remain unchanged. The new comparison uses a separate
`audit_pic_off` snapshot and records its own hashes; no earlier result is attributed
to a later probe version.

Before adding the full PIC-off mode, the exact prefix probe versions were also
preserved as `output/p292/executed_quality_compare_no_pic8.py` and
`executed_raw_phase_no_pic8.py`; server `audit_pic_off` remains unchanged. The full
comparison will use a separate `audit_pic_full` snapshot and new recorded hashes.

The probe compile check and five CPU regressions pass. An independent adversarial
review checked exact config/code provenance, prefix scope, null-held frame removal,
the identical material cohorts and pin fractions, and reran all five regressions.
The executed taper audit launched on GPU 0 at 22:56:39 UTC; the body audit launched
on GPU 2 at 22:57:39 UTC and completed at 22:58:12 UTC. The independent final
body/taper report-to-JSON review and the appended phase numeric review are both
closed with no remaining blocker.
