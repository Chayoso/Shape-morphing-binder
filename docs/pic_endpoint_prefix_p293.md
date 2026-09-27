# P293: shared-PIC objective prefix audit

The shared promoted-endpoint contract works in this eight-window run. Raw quality
is mixed: upper-region density improves modestly, silhouette IoU is lower at
commit8, and boundary direction changes persist. The prefix does not demonstrate
a clear transport-quality advance and does not justify promoting the option.
It provides no final-rest or hole-free-morph conclusion while transport is unfinished.
The separate implementation gates are in [the shared-endpoint report](shared_endpoint_p293.md).

## Fixed comparison

Server run prefixes are `work/p293/pic_legacy8` and
`work/p293/pic_objective8`, both using the original300k mixed60 input pair and the
same immutable numerical snapshot `work/p293/endpoint`. Verified discretization
is N=300000, T=20, dt=1/240 s, MPM dx=0.3062907543956724 wu and loss grid36 cubed,
with eight optimizer iterations and an eight-window cap (`animations=300`). Both
arms accept8 of8 outer attempts, with8 accepted inner updates and no inner rejection
in every window. All recorded state guards are zero. Each archive contains161
physical states, including the initial state, with no null or held rows.

The `commit_pic_objective_prefix` contract allows only
`commit_pic_objective=False` (or absent) to `True`. Both arms require:

- `stop_after_windows=8`, `commit_pic=True`, and `shift_sub=False`.
- `outer_render_committed=True` and positive `lambda_auto`.
- No `render_until` cutoff within the eight attempts.
- Exact equality of source/target arrays, numerical source hash and MPM parameters.
- Strict CUDA compute, archive stride1, preserved particle IDs and monotone pins.

All other serialized configuration fields match. The experiment asks whether
evaluating the shared promoted PIC endpoint changes early transport and raw boundary
motion, while PIC remains active in both arms. It is distinct from the earlier
PIC-on/off comparison and the rendering-reference handoff.

The analysis uses the common eight-commit prefix. Both arms retain body displacement
and terminal-velocity control, stress and outer-layer channels, `lambda_auto=0.5`,
and paced64x64 rendering throughout. `use_gauss_loss=False`; there is no arrival
render handoff in this experiment. No final-rest, full-morph, gallery or hole-free
conclusion is available from this unfinished transport prefix.

## Endpoint identity and run cost

All eight candidate commits record `endpoint_contract.space=promoted_xpic`,
`source=accepted_buffer`, and objective-to-commit maximum difference exactly0 wu.
Both arms use the accepted rollout buffer throughout. This execution therefore
does not exercise the final-replay ownership branch; its separate tests are described
in the implementation report. The legacy objective still evaluates raw rollout
positions before its commit PIC map. No extra pins were introduced by the new option.

| Run | Accepted / attempted | Wall time | Torch allocation peak |
| --- | ---: | ---: | ---: |
| Legacy PIC objective | 8 / 8 | 141.991 s | 8.114 GB |
| Shared PIC objective | 8 / 8 | 167.101 s | 8.362 GB |

These are single runs on different GPUs, not a controlled speed benchmark. The
pair differs from its first solve; median same-ID positional separation is0.12588 sp
at commit1 and1.38085 sp at commit8. Without repeated runs this includes unmeasured
GPU/numerical variation, so small differences are observations rather than a
statistically established policy benefit. Unlike the preceding handoff experiment,
there is no inactive prefix that supplies a within-pair noise comparison.

At commit8, recorded `pic_null_share` is0.202932/0.155504 for legacy/candidate.
This is the removed-displacement RMS fraction, not proof of an exact null space.
The endpoint contract is ownership/evaluation telemetry, not a quality metric.

## Raw shape and early supply

Geometry, neighbors, cohorts and motion are computed on CUDA from raw simulation
archives; archive/JSON/hash I/O and reporting use the host. No renderer or optimization
loss supplies a quality metric. Source-native spacing is0.03498853660707278 wu;
target-native spacing is0.03493084911867985 wu. Top means y>2.3 wu. Target coverage
uses a two-target-spacing nearest-source threshold. Density counts neighbors within
the target median eighth-neighbor radius0.06898659982768912 wu, excludes self and
divides by8; under-half means fewer than four neighbors. These counts do not certify
watertightness. The radius0.25 wu tip ball contains89 target particles.

| At accepted commit8 | Legacy | Shared objective |
| --- | ---: | ---: |
| Silhouette IoU | 0.915918 | 0.912857 |
| Symmetric NN Chamfer, wu | 0.06168016 | 0.06167987 |
| Global target coverage | 0.983367 | 0.983103 |
| Top target coverage | 0.721526 | 0.722440 |
| Top density | 0.689241 | 0.716024 |
| Top under-half fraction | 0.269959 | 0.258427 |
| Minimum accepted trajectory detF | 0.890095 | 0.907235 |
| Recorded terminal mean speed, wu/s | 0.285916 | 0.322194 |
| Pinned fraction | 0.366543 | 0.351777 |
| Full-plan arrival-mask fraction | 0.958490 | 0.957997 |

IoU falls by0.003061 while Chamfer is nearly unchanged. The candidate's top density
and under-half count improve modestly, but target coverage is nearly unchanged.
This is not a final-fit gate: both arms are still transporting material. The arrival
mask uses a radius and must not be read as exact geometric convergence.

| Commit | Legacy / candidate IoU | Legacy / candidate top density | Legacy / candidate top coverage |
| --- | ---: | ---: | ---: |
| 3 | 0.848964 / 0.853344 | 0.340084 / 0.355556 | 0.322362 / 0.322688 |
| 6 | 0.890471 / 0.891308 | 0.619691 / 0.598886 | 0.590583 / 0.605016 |
| 8 | 0.915918 / 0.912857 | 0.689241 / 0.716024 | 0.721526 / 0.722440 |

Neither arm puts a particle in the final tip ball during any of its161 raw frames.
Peak and final counts are both zero, so retention/attrition ratios are undefined;
this short prefix cannot test the earlier late tip-loss problem.

Fixed initial-Chamfer fractions0.75,0.5 and0.25 first cross at commits1,2 and7 in
both arms. Candidate-minus-legacy actual Chamfer differences there are−0.000320330,
−0.0000566985 and−0.000310136 wu. Neither reaches0.225. These are threshold crossings,
not exactly equal progress or an independent endpoint shape match.

## Same material and window-boundary motion

The common-free cohort contains all15375 eligible IDs, unpinned in both arms at
commit8 and within the union of their sparse boundary masks. Monotone pins make
this cohort free throughout the earlier interval. It is outcome-selected and
descriptive. Its material-ID hash is
`19c0f02666e26698b1059c7229d9b80f8e36c3fc3c61c83ab88964ab5e74bc73`.

Motion uses endpoints1–8: seven window displacements and140 physical substeps per
ID, starting after the first accepted window. All141 archived states in this
interval are physical; no held padding enters the statistics. Normal bases are
arm-specific frozen33NN centroid-offset normals at commit8. Total displacement
and reversal are independent of those normal bases. Reversal requires a negative
successive-displacement dot product and both lengths above1e-4 source spacing.

| Same15375 free IDs, endpoints1–8 | Legacy | Shared objective |
| --- | ---: | ---: |
| Commit displacement median / p95, sp | 2.54628 / 9.79834 | 2.67058 / 9.70263 |
| Raw-step displacement median, sp | 0.140840 | 0.143120 |
| Raw absolute-normal median, sp | 0.060528 | 0.059636 |
| Raw tangent median, sp | 0.097902 | 0.098357 |
| Raw reversed / eligible pairs | 38525 / 2137125 (1.80266%) | 38200 / 2137125 (1.78745%) |
| Commit reversed / eligible pairs | 5864 / 92250 (6.35664%) | 5702 / 92250 (6.18103%) |
| Commit net/path median | 0.85483 | 0.84196 |

These displacements include intended transport. Their nonzero size is not a
late-rest failure, and their small between-arm change is not a substantial
vibration reduction. The independent pre-treatment source-upper-surface cohort
has6712 IDs; raw median displacement is0.109008/0.107951 sp and raw reversal is
2.39463%/2.43033%. Its final pinned fractions are27.9499%/29.0524%, so it mixes
transport and progressively fixed material. The endpoint-free union has20398
eligible IDs, deterministically sampled to20000; its final pin fractions differ
(14.115%/10.605%), and it remains descriptive context in the JSON.

Prefix pin checks find zero exact drift among76834 legacy and67071 candidate pins
with a later observed frame. Total admitted counts by commit8 are109963 and105533;
pins first admitted at the final endpoint have no later frame and are not included
in the drift denominator. No future trajectory is inferred from those checks.

The phase audit independently reconstructs these same15375 IDs and endpoints1–8.
Phase20 combines the final physical/layer rollout step and commit PIC correction;
subgrid shifting is off. The saved archive cannot separate individual contributions.

| Phase statistic on identical IDs | Legacy | Shared objective |
| --- | ---: | ---: |
| Phases1–19 displacement median, sp | 0.137681 | 0.140240 |
| Phase20 displacement median / p95, sp | 0.20572 / 0.80807 | 0.19777 / 0.74662 |
| Phase20 absolute-normal median, sp | 0.10029 | 0.09624 |
| Phase20 tangent median, sp | 0.13730 | 0.13055 |
| Phase20 share of total raw path | 7.59018% | 7.17027% |
| Reversal19→20 | 21121 / 107625 (19.6246%) | 20419 / 107625 (18.9724%) |
| Reversal20→next1 | 16100 / 92250 (17.4526%) | 17124 / 92250 (18.5626%) |
| Interior reversals | 1304 / 1937250 (0.06731%) | 657 / 1937250 (0.03391%) |

Pooled boundary amplitude is modestly lower, but reversal after the boundary
increases. At window8 alone, phase20 path share is11.3148%/11.5336% and displacement
median0.136279/0.134812 sp. Thus there is no consistent disappearance of the boundary
pattern. These fractions describe material trajectory directions, not periodic
oscillation frequency or pixel flicker. They must not be compared directly with the
late P292 cohorts as if their transport stage and selected IDs were the same.

## Deployment and provenance

The isolated audit snapshot is `/data/relcfd/chayo/physmorph_v2/work/p293/audit`.
All56 `physmorph/*.py` files, recursively, match the frozen source manifest byte
for byte; no simulation-core file was edited for this audit. Numerical code hash:
`e269dc2570f4cf9d10b8e4196cc75626ddec3dfd72a91cec3afef82a4c959084`.

Executed probe SHA-256 values:

- Quality: `cacc8d1f906370887dba08f52dc714abc1d28994562424f2f80f6239ccb49395`.
- Phase: `cc621f478166a7958fd79c6d824d5001cf4635ad5f146bc96605e89df6b5294d`.

Local `output/p293/audit_provenance.json` records the preparation-time source, audit,
numerical hash and probe/dependency hashes. Exact executed probe copies are
`executed_quality_compare.py` and `executed_raw_phase.py` in that directory.
Earlier P292 executed probes/results are unchanged. Result files are
`output/p293/quality_pic8.json` and `phase_pic8.json`, beside both run JSONs.
Quality JSON SHA-256 is
`5e65bd1655ec40c9f7137f3d4406c65a617e20dc3c5059526a6c2ccf15ec6503`;
phase JSON is
`9169e0c1aab15708804b670d552b2bf700009685cc5753a3f49060ebdef71840`.
The phase result checks the exact quality result/probe and cohort hashes.

Launchers `work/p293/quality_pic8.sh` and `phase_pic8.sh` source only the audit
snapshot. Root launched the quality audit on GPU0 at2026-09-27 00:35:10 UTC;
it completed00:35:26. The phase audit launched00:36:11 UTC, after quality completion
and61 seconds after its start. The source simulation launches were00:30:48 on
GPU0 and00:31:23 onGPU2; their35-second gap violates the45-second operational rule
and is recorded in the implementation report. Later audit launches used an explicit
server-time spacing check. No additional GPU job was launched by the reporting agent.

The narrow audit-contract review is closed, with14 independently run CPU tests
passing. This validates readiness and scope, not the proposed pair's quality.
The separate deployment/document review is also closed: the reviewer recomputed
the56-file numerical aggregate from the frozen ZIP and checked current probe
hashes and launcher inputs/environment. Remote byte verification is the recorded
preparation step, not a remote repeat by that reviewer.
After any numerical snapshot change, create and verify a new audit snapshot
rather than attributing its outputs to this manifest. The final independent
numeric/causal result gate is closed: result hashes, source/configuration evidence,
shape/progress tables, all cohort denominators, phase counts and pin exclusions were
checked against the JSONs. No GPU replay was performed by that reviewer. The prefix
supports implementation consistency and only mixed early quality evidence; it does
not establish a fix for holes or motion after arrival.
