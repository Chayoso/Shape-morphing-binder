# P294 geometric-rest prefix audit

At accepted window 8, geometric rest modestly improves silhouette and thin-region
supply in this pair. Same-material free motion falls by only 2.9% per raw step,
while its commit-to-commit reversal fraction rises. The candidate also admits more
pins. These results do not demonstrate free-particle rest or absence of holes;
they support only further bounded investigation. The comparison changes only
`geometric_rest=False → True`; both arms already use the shared PIC endpoint
objective. Raw phase measurements still show reversals concentrated around the
window boundary. A longer matched run is needed to evaluate late rest; this
prefix is not a quality promotion.

## Completed prefix and raw geometry

Both runs accepted all 8 outer windows and all 8 inner updates per window. All
state guards are zero. Shared endpoint objective/commit position gaps are exactly
zero in all 16 accepted records, all from accepted buffers; this pair does not
exercise a final replay. Runtime is 155.286 / 167.600 seconds and Torch peak memory
8.362 / 8.440 GB (control / candidate); the arms used different GPUs, so these are
observations rather than an isolated performance comparison. All 8 render references
in each arm are paced. The median recorded render lambda is 0.210776 / 0.212721;
rendering remains active, and its balancing response is part of this intervention.

All geometry values below use N=300k, T20, dt1/240, dx=0.3062907544 wu and the 36³
loss grid. The audit's motion unit is source median nearest-neighbor spacing
`sp=0.03498853660707278 wu`; target coverage uses target spacing
`0.03493084911867985 wu`. Neither is a renderer-derived spacing.

| Accepted endpoint 8 | Control | Geometric rest |
|---|---:|---:|
| Raw silhouette IoU | 0.912640 | 0.915179 |
| Symmetric mean nearest-neighbor Chamfer (wu) | 0.0620060 | 0.0614899 |
| Global target coverage within 2 target spacings | 0.982270 | 0.982727 |
| Top-region target coverage | 0.717542 | 0.725183 |
| Top-region relative density | 0.709097 | 0.717681 |
| Top-region under-half-density fraction | 0.266951 | 0.256929 |
| Minimum accepted trajectory det F across prefix | 0.909861 | 0.904094 |
| Stored physical mean speed (wu/s) | 0.263452 | 0.267274 |
| Stored maximum absolute velocity component (wu/s) | 3.10649 | 4.42220 |
| Pinned fraction after endpoint | 0.356220 | 0.430713 |

The fixed top region is y>2.3 wu. Density counts neighbors within the target median
8th-neighbor radius, excludes self and divides by 8; it is not a watertightness
test. Silhouette IoU is computed from raw state, not the display renderer.

| Accepted commit | IoU control / candidate | Top density control / candidate | Top coverage control / candidate |
|---|---|---|---|
| 3 | 0.853580 / 0.853026 | 0.354425 / 0.356211 | 0.323994 / 0.321055 |
| 6 | 0.894241 / 0.894687 | 0.621109 / 0.630755 | 0.603710 / 0.608673 |
| 8 | 0.912640 / 0.915179 | 0.709097 / 0.717681 | 0.717542 / 0.725183 |

The first crossings of 75%, 50% and 25% of initial Chamfer occur at the same commits
1, 2 and 7. Neither arm reaches 22.5% or lower in this prefix; identical first-crossing
indices are not identical geometric states. The target-tip ball remains empty in
all 161 raw states in both arms, so this prefix cannot establish tip retention.

## Same-material motion and pin context

The strongest free-motion comparison uses exactly 14,468 IDs on the union of the
two sparse endpoint boundaries, unpinned in BOTH arms at common endpoint 8. They
remain unpinned throughout the interval because pin admission is monotone.
The ID hash is
`8d0c8ff5ec7ed18725e98e31720a96d608c5dceba3157dcfc4c570a3edf57108`.
This is an outcome-selected population, not an unbiased causal population.
The interval is endpoint 1→8: 7 window displacements, 140 raw steps and 141 raw
states, with no interior null-held rows. Normal/tangent bases are separately frozen
from each arm's endpoint geometry.

| Same 14,468 free IDs | Control | Geometric rest |
|---|---:|---:|
| Raw step median / p95 (sp) | 0.136071 / 0.507389 | 0.132152 / 0.510024 |
| Absolute normal raw-step median (sp) | 0.0574663 | 0.0559615 |
| Tangential raw-step median (sp) | 0.0932816 | 0.0892162 |
| Raw reversed pairs / eligible moving pairs | 38,688 / 2,011,052 | 38,346 / 2,011,052 |
| Raw reversal fraction | 1.92377% | 1.90676% |
| Commit step median / p95 (sp) | 2.510765 / 9.605863 | 2.412361 / 9.777992 |
| Commit reversed pairs / eligible moving pairs | 5,766 / 86,808 | 6,619 / 86,808 |
| Commit reversal fraction | 6.64224% | 7.62487% |
| Median net displacement / path length | 0.845747 | 0.849279 |

Median raw and commit motion decrease 2.88% and 3.92%; both p95 values rise slightly.
The interval is still purposeful early transport. Reversal counts describe local
direction changes and do not establish periodic oscillation or image flicker.

The pre-treatment source-defined upper-surface cohort has 6,712 identical IDs.
Raw median motion changes 0.106527→0.103321 sp, but its endpoint pinned fraction
changes 26.1919%→35.6228%. Raw reversal pairs are 22,792/890,228→20,902/875,288
(2.56024%→2.38801%); commit pairs are 6,792/38,135→7,979/37,388
(17.8104%→21.3411%). This population is fixed before treatment, but changing pin
status remains part of its outcome. The endpoint-free UNION has 19,199 IDs and also
mixes pin states (9.4588%→15.1831% pinned at endpoint); its lower raw median
0.118706→0.113948 sp cannot alone establish a free-motion improvement.

All admitted pins with later raw observations have exactly zero drift:
65,501 checked control particles and 91,743 checked candidate particles. Total
admitted counts are 106,866 / 129,214; particles first pinned at the last endpoint
have no later observation and are not included in the zero-drift claim.

## Geometric-rest telemetry

The candidate's frozen arrived/free cohort counts by window are
53,205; 97,529; 216,426; 244,143; 258,377; 271,156; 234,879; 194,189.
At window 8, whole-N squared-speed components in (wu/s)² are raw 0.1008192,
remap 0.5554175, sum 0.6562366, delivered 0.5288385, and cross −0.1273981.
The conditional raw/remap values are 0.1557542 / 0.8580571. Nominal w_kin is 5,
the density-unit multiplier is 3.0299949378e−5 and effective weight is
0.0001514997469. The remap component remains nonzero. The control does not record
this optional decomposition, so these telemetry values are not an on/off reduction
measurement on the same eligible IDs. Existing endpoint-work telemetry excludes
the direct raw/previous-position paths of this term and is not its full work measure.

## Raw boundary phases

The independently executed phase probe uses the same 14,468 IDs, endpoint interval
1→8, source-native spacing and arm-specific normal bases as the quality report.

| Raw phase statistic | Control | Geometric rest |
|---|---:|---:|
| Phases 1–19 displacement median (sp) | 0.133411 | 0.129669 |
| Phase 20 displacement median / p95 (sp) | 0.192929 / 0.738117 | 0.178183 / 0.692412 |
| Phase 20 absolute-normal median (sp) | 0.0936264 | 0.0873840 |
| Phase 20 tangential median (sp) | 0.123810 | 0.116764 |
| Phase 20 share of summed raw path length | 7.25047% | 6.90483% |
| Reversals 19→20 | 20,954 / 101,276 (20.6900%) | 20,519 / 101,276 (20.2605%) |
| Reversals 20→next 1 | 17,341 / 86,808 (19.9763%) | 16,930 / 86,808 (19.5028%) |
| Interior-phase reversals | 393 / 1,822,968 (0.02156%) | 897 / 1,822,968 (0.04921%) |

The combined boundary step becomes smaller but does not disappear. Window 8 alone
has phase-20 median 0.115686→0.115287 sp and path share 11.0882%→10.5493% on this
cohort. Across all particles, window 8's promoted last-step displacement divided
by dt has mean speed 0.680131→0.533295 wu/s, whereas stored physical v_T mean speed
is 0.263452→0.267274 wu/s. These definitions differ, and the former includes the
changed pin mix; their discrepancy is not an isolated physical damping effect.
Here subcell shift is disabled, but the archive still does not separate the last
physical/layer step from PIC remapping. Boundary reversal remains present despite
the modest reduction, and the commit-to-commit reversal fraction above increases.

## Intervention and scope

The completed pair is `work/p294/geom_control8` and `work/p294/geom_rest8` on hyde06.
Both use the original 300,000-particle mixed60 source/target, T=20, dt=1/240,
dx=0.3062907544 wu, loss grid 36³, 8 inner iterations per window, the 300-window
animation schedule capped after 8 windows, active rendering guidance, shared PIC,
no subcell shift, and the corrected committed-state outer render gate. The actual
input arrays, complete MPM dictionary, numerical source hash and configuration
difference must pass the audit before any measurement is accepted.

The new term uses the frozen full-plan-arrived, unpinned cohort at each window
start. It adds the whole-N mean of raw final-step speed squared plus the whole-N
mean of PIC-remap speed squared at the existing kinetic weight. It also records
delivered speed squared and the cross term; these are diagnostics, not additional
penalties. The sum does not allow cancellation between raw motion and the remap.
The nominal weight, unit multiplier and effective weight are reported separately.
This is a fixed-dt objective choice, not a claim of physical remap energy or
timestep invariance. It can also change the physics gradient norm used by the
render-lambda balancer.

The audit compares equal accepted commits and first crossings of fixed Chamfer
thresholds. It retains thin-region supply, target-nearest-neighbor coverage,
tip-ball material retention, determinant minima, physical stored velocity,
accepted endpoint telemetry and raw pin checks. Raw positions drive every geometric
metric; no rendered image or optimization-loss operator supplies quality scores.
Conditional geometric-rest statistics describe changing per-window eligible
cohorts and cannot be interpreted as a fixed-material trend on their own.

Motion comparisons retain a pre-treatment source-defined surface cohort and
the same material IDs from common endpoint cohorts in both arms. Endpoint cohorts
are selected after the intervention, so their comparisons are descriptive. Raw
motion excludes interior null-held archive rows and trailing presentation holds.
Phase T includes the last physical step and endpoint remap; phase-localization
does not by itself apportion motion between operators. Eight windows are unfinished
transport and cannot establish late settling, absence of oscillation, or closure
throughout the full morph. Same-code GPU runs can diverge numerically before any
material policy effect; the paired difference is not a repeat-noise estimate.

## Frozen deployment and provenance

The simulation snapshot is
`/data/relcfd/chayo/physmorph_v2/work/p294/geometry`, built from local
`output/p294/geometry_snapshot.zip`. The corrected audit snapshot is
`/data/relcfd/chayo/physmorph_v2/work/p294/audit_v2`.
It contains all 148 frozen files with only `scripts/probes/quality_compare.py`
and `scripts/probes/raw_phase.py` replaced by the reviewed P294 audit versions.
All 57 numerical Python files were verified byte-for-byte against the simulation
snapshot on hyde06; the complete audit manifest was also checked.

The numerical aggregate SHA256 is
`fd7c0ba18463ed6ddc232a800850372134e540c40af4000698b83b4053192f6a`.
The prepared quality probe SHA256 is
`acdc56e7cd2b558007da8bff3a8b5845bf977bd2ac8f0a01e5ada1a618037192`;
the prepared phase probe SHA256 is
`66fce74fa76da4ba8b2736b1d8a8ee2e918c539713a5a5881c3b34f0062ee736`.
The frozen archive loader and raw QA probe remain unchanged. Deployment receipt,
manifest, verifier, prepared probe copies and launchers are preserved locally in
`output/p294/`; the corrected receipt is `audit_v2_provenance.json`.

Executed GPU0 launchers are `work/p294/quality_geom8_v2.sh` and
`work/p294/phase_geom8_v2.sh`. The phase launcher consumes the exact quality JSON and
checks the executing quality-probe hash. Root controls launch authorization,
GPU occupancy and the global launch interval. Preparing these files launches no
simulation or GPU audit. Root launched the successful quality audit at
2026-09-27 01:04:13 UTC and the phase audit at 01:05:26 UTC, a 73-second interval.
The compact results are local `output/p294/quality_geom8.json` and
`output/p294/phase_geom8.json`, also retained under remote `work/p294/`.
Their SHA256 values are respectively
`ba8877b2fdeaf0c15d3564cfb90a36a08d99fe91dc252c3771514e7413d33e19` and
`ca7257de65ebe9a88da29a4ee39072640834b9f39c7afa38f68df3ccfbd912aa`.
The phase result verifies the exact quality-result and quality-probe hashes.

The first audit launch at 2026-09-27 00:58:09 UTC failed before measurements:
both completed runs serialize `phys_loss='auto'`, while the first validator required
an explicit OT mode. Both immutable logs record resolution to `ot_pace`, and all
eight accepted records in each arm carry `accepted_full_plan` arrival evidence.
The corrected validator requires exactly one OT resolution event plus finite
full-plan arrival evidence in every accepted record, and stores the original log
SHA256 and resolver line. The failed `audit` snapshot and `quality_geom8.log` remain
preserved. No simulation or completed JSON was rewritten. A regression fixture
copies the actual serialized configs and relevant history/log evidence.

## Validation before measurement

`tests/test_geometric_rest_pipeline.py` passed 12 CPU tests, independently repeated
by the reviewer. They exercise real N160/T3/render-on OT runs, accepted-buffer and
forced-replay paths, independent position-based reconstruction, previous-position
gradients, legacy and density-unit weights, empty cohorts, pin exclusion and
ownership, unchanged physical F/v through endpoint promotion, and fail-closed
invalid weights/missing arrival/changed previous-position contracts. The pin test
proves the optimizer preserves its supplied pin set; the runner's existing
settlement policy may still admit pins.

The corrected P294 audit extension and raw-phase scope tests passed 18 CPU tests in the
independent review. The strict contract permits only the geometric-rest flag
change, with T20/dt1/240/cap8, equal positive finite kinetic weights and active
rendering. Both the integration-test and audit-extension adversarial gates closed.
These tests establish implementation and measurement contracts, not physical
quality. The separate numerical report review closed after independent checks of
both result files, ownership/telemetry records, all cohort and phase denominators,
observed pin counts and provenance. No full-morph quality promotion was made.
