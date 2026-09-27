# P292: accepted-arrival render-reference handoff

The new handoff activates as intended. This candidate has better endpoint silhouette
and upper-target coverage than its paired control, and its tip-ball population reaches
the target count. It still fails the prior 0.971 silhouette gate, retains sparse regions,
and leaves unpinned material moving. Keep the option experimental; this is not a
hole-free or vibration-free result.

The trajectories already differ substantially before the new policy acts. Consequently,
neither the final quality difference nor the pre/post change below is an isolated causal
effect of the handoff. This is one matched-configuration pair, with numerical trajectory
variation and different stopping times, not a repeated treatment-effect estimate.

## Contract and actual activation

Both runs use the same original mixed60 source/target arrays and immutable numerical
snapshot `/data/relcfd/chayo/physmorph_v2/work/p292/handoff`. Both disable commit PIC,
enable corrected committed-position outer rendering, and retain the same mixed body,
stress, outer-layer position, layer-relaxation and subgrid-shift channels. Render guidance
is active with `lambda_auto=0.5`; body RPROP is off. The sole configuration change is
`render_paced_arrived=False` to `True`. The audit checks exact input equality, MPM
parameters, configuration difference and numerical source hash, including its own
loaded numerical dependencies.

Discretization is N=300000, T=20, dt=1/240 s, MPM dx=0.3062907543956724 wu,
MPM/loss grid36 cubed, eight optimizer iterations per window, cap60 with
`animations=300`. Source-native spacing is0.03498853660707278 wu; target-native
spacing is0.03493084911867985 wu. Archive stride is1. Geometry, neighbor queries,
cohort construction and motion calculations execute on CUDA; archive/JSON/hash I/O
and reporting use the host. No renderer or optimization-loss operator supplies a
reported quality metric.

| Run | Accepted / attempted | Runtime | Actual solve reference |
| --- | ---: | ---: | --- |
| `outer_no_pic60` control | 44 / 47 | 714.653 s | Paced for all47 attempts |
| `handoff_no_pic60` candidate | 37 / 37 | 629.026 s | Paced1–26, fixed27–37 |

All recorded guard counts are zero. Control attempts45–47 are rejected; no future
accepted state is trimmed from either result. The candidate stops after five stale
windows. These runtimes include different numbers of attempts and are not a speed
comparison. There is no C2F event within either run.

Candidate commit26 supplies exact accepted evidence:300000 arrived particles.
That solve still uses the paced target; the fixed target starts with solve27. At the
latch,83.6553% are pinned and49034 are free. Commit27 temporarily has two particles
outside the arrival mask; the accepted-evidence latch persists, as designed. The
candidate finishes with284922 pinned and15078 free particles. The control finishes
with295358 pinned and4642 free particles. Neither complete log records the older
render-convergence switch; the candidate's fixed-reference history comes from the
new accepted-arrival latch.

The rendering objective remains64x64. Its96x96 rebuild is scheduled beyond this run;
`use_gauss_loss=False`. This policy changes the image reference for physical control
optimization. It does not optimize exported high-resolution Gaussian attributes.

## Divergence precedes the intervention

The table uses identical material IDs at identical accepted commits, before candidate
commit26. Position differences are source-native spacings, over all300000 IDs.

| Commit | Position difference median / p95, sp | Control pinned | Candidate pinned |
| --- | ---: | ---: | ---: |
| 1 | 0.00335 / 0.01577 | 0% | 0% |
| 3 | 0.50523 / 1.10920 | 0.1060% | 0.0647% |
| 6 | 1.09179 / 2.24478 | 9.2423% | 11.4773% |
| 10 | 1.41399 / 3.23097 | 56.7897% | 54.9570% |
| 20 | 1.72010 / 5.80182 | 85.5637% | 79.0277% |

Both arms still use paced guidance throughout these observations. The candidate
already has75 tip-ball particles versus59 at commit20. GPU reduction/order and
device-dependent trajectory variation are plausible contributors; this experiment
does not isolate their cause. The differing prefix rules out attributing all later
differences to the reference switch. The single-flag/core/input gates do not remove
this limitation.

## Raw shape and material supply

Top-region measurements use the fixed spatial region y>2.3 wu. Target coverage means
nearest source distance within two target-native spacings. Density counts neighbors
within the target median eighth-neighbor radius0.06898659982768912 wu, excludes self,
and divides by8. Under-half means fewer than four such neighbors. The tip ball has
radius0.25 wu at the maximum-y target point, with89 target particles. These measures
detect sparse supply and target gaps; they do not certify topology or watertightness.

| Own final accepted endpoint | Control44 | Candidate37 |
| --- | ---: | ---: |
| Silhouette IoU | 0.966321 | 0.968164 |
| Symmetric NN Chamfer, wu | 0.0562570 | 0.0562047 |
| Global target coverage | 0.996787 | 0.996967 |
| Top target coverage | 0.971983 | 0.977077 |
| Top density | 0.968541 | 0.955225 |
| Top under-half fraction | 0.062861 | 0.066510 |
| Tip-ball count | 62 | 89 |
| Minimum accepted trajectory detF | 0.821072 | 0.812343 |
| Recorded terminal mean speed, wu/s | 0.001597 | 0.006180 |

At common commit37 the control has IoU0.966288, top coverage0.970024, density0.970585,
under-half fraction0.058276 and62 tip particles; the candidate is as above. The fit
difference therefore survives equal-commit comparison, but both remain below0.971.
The candidate has better top coverage but slightly worse local density/under-half
counts. A count of89 in the tip ball is not proof that the correct surface is recovered.

| Same accepted commit | Control / candidate top density | Control / candidate top coverage | Control / candidate tip count |
| --- | ---: | ---: | ---: |
| 3 | 0.32959 / 0.35252 | 0.28742 / 0.28507 | 0 / 0 |
| 6 | 0.62422 / 0.62181 | 0.61037 / 0.60345 | 0 / 0 |
| 10 | 0.78014 / 0.75510 | 0.84966 / 0.86305 | 37 / 17 |
| 20 | 0.94558 / 0.94441 | 0.95624 / 0.96212 | 59 / 75 |
| 26, last candidate paced solve | 0.96519 / 0.97022 | 0.96415 / 0.96624 | 62 / 74 |
| 30 | 0.96886 / 0.94969 | 0.96754 / 0.97616 | 62 / 89 |
| 37 | 0.97059 / 0.95522 | 0.97002 / 0.97708 | 62 / 89 |

The control's tip population peaks at73 on raw frame337, then ends at62:61 peak IDs
remain,12 leave and one new ID enters. Candidate peak93 occurs on raw659;86 peak IDs
remain, seven leave and three enter, ending at89 on raw740. Peak-to-end durations are
different, so the ratios are not comparable retention-rate estimates.

Fixed initial-Chamfer fractions0.75,0.5,0.25,0.225 and0.22 are first crossed at the same
commits1,2,6,9 and10. Fraction0.215 is first crossed at control23/candidate20, both before
the handoff; the actual Chamfer difference there is−7.57651e-6 wu. Neither reaches0.2.
The JSON records the nonzero progress mismatch for every threshold; the0.25 crossing
differs by+0.000633818 wu. These are first threshold crossings, not identical shapes.

## Same unpinned material still moves

The principal cohort contains423 identical material IDs: the union of each sparse
boundary at common commit37, intersected with IDs unpinned in both arms there. This is
an outcome-selected descriptive cohort. Monotone pin admission makes its pin fraction
zero throughout the earlier intervals. It is not a randomized or representative
population. Endpoints27–37 yield ten accepted-window displacements and200 raw-step
displacements per ID, with no held frames. Normal/tangent bases are frozen33NN
centroid-offset normals at that interval endpoint, separately for each arm; total
displacement and reversal are basis-independent.

| Same423 IDs, endpoints27–37 | Control | Candidate |
| --- | ---: | ---: |
| Commit displacement median / p95, sp | 0.15812 / 0.44127 | 0.17894 / 0.47698 |
| Raw-step displacement median, sp | 0.009170 | 0.010582 |
| Raw normal displacement median, sp | 0.003326 | 0.003337 |
| Raw tangent displacement median, sp | 0.007750 | 0.009068 |
| Raw reversed / eligible pairs | 4361 / 84177 (5.1808%) | 4313 / 84177 (5.1237%) |
| Commit reversed / eligible pairs | 458 / 3807 (12.0305%) | 286 / 3807 (7.5125%) |
| Commit net/path median | 0.87899 | 0.86971 |

Reversal means negative consecutive-displacement dot product, with both magnitudes
above1e-4 source spacings. It is not by itself proof of periodic oscillation. The
candidate has fewer commit reversals but greater displacement on this shared free
cohort. Neither arm is at rest; the large net/path ratios also show persistent travel.

For the same423 IDs, the audit additionally separates endpoints16–26 (paced windows
17–26) from26–36 (candidate fixed windows27–36). It verifies actual reference history,
rather than inferring it solely from the trigger. Each band contains ten windows and
uses its own arm-specific endpoint normal basis.

| Same material band | Control / candidate commit median, sp | Control / candidate raw median, sp | Control / candidate raw reversed pairs, out of84177 |
| --- | ---: | ---: | ---: |
| Before trigger16–26 | 0.34537 / 0.40245 | 0.019544 / 0.022356 | 3203 / 2328 |
| After trigger26–36 | 0.16406 / 0.18460 | 0.009515 / 0.010778 | 4251 / 4280 |

Both arms slow down as the morph proceeds; raw reversals increase in both bands.
This descriptive pre/post change cannot be assigned to the handoff, particularly
given the observed pre-trigger divergence. Identical IDs prevent a changing-cohort
explanation for this table, but do not remove trajectory/stage confounding.

The independent pre-treatment source-upper-surface cohort contains6712 fixed IDs.
Over27–37 its commit p95 is0.11139/0.13278 sp, while pinned fractions increase from
89.18% to95.69% in control and85.13% to93.15% in candidate. Its zero median reflects
many pinned particles. The1144-ID union of endpoint-free cohorts has a different pin
mix in each arm and is retained in JSON as descriptive context, not evidence that
free-particle motion has been suppressed.

Full-archive pin checks find exactly zero motion for all295358 admitted control pins
and284922 candidate pins. This validates the implemented pin constraint, not the
criterion used to classify physical arrival or rest. The candidate's own800-ID final
free surface tail still has median normal/tangent motion0.003645/0.009796 sp per raw
step and reversal fraction5.5999%. That tail uses a different cohort/time range from
the principal comparison and is not an unbiased arm comparison.

## The window boundary remains a motion source

The independent phase audit reconstructs the same423 IDs and endpoints27–37, preserving
all20 true substeps per accepted window. Phase20 includes the final rollout step and
all commit position corrections. It cannot isolate the final physical step, layer
relaxation or subgrid shift; PIC is off in both arms.

| Shared-cohort phase statistic | Control | Candidate |
| --- | ---: | ---: |
| First19 phases displacement median, sp | 0.008865 | 0.010198 |
| Phase20 displacement median / p95, sp | 0.06832 / 0.27575 | 0.06309 / 0.23974 |
| Phase20 absolute normal median, sp | 0.02033 | 0.01970 |
| Phase20 tangent median, sp | 0.05747 | 0.05371 |
| Phase20 share of total raw path | 32.5650% | 28.9504% |
| Reversal19→20 | 2285 / 4230 (54.0189%) | 2248 / 4230 (53.1442%) |
| Reversal20→next1 | 2038 / 3807 (53.5330%) | 2043 / 3807 (53.6643%) |
| Interior reversals | 38 / 76140 (0.0499%) | 22 / 76140 (0.0289%) |

The reference correction does not eliminate this boundary pattern. These are
geometric direction changes in raw material trajectories, not renderer flicker or
an attribution of pixel variance. The next physical design still needs a consistent
promoted-endpoint objective and a rest criterion that observes actual material motion;
this experiment does not establish that either proposed change will suffice.

## Provenance and decision

Numerical source SHA-256:
`cb48fed53d053c7cf09c535967c0956a0c46dd03be3ed4062d0548d686e6286b`.
The current checkout may contain later comment/documentation changes; it is not the
source attributed to these runs. GPU0 control launched2026-09-26 23:52:09 UTC; GPU2
candidate launched23:53:11 UTC. GPU0 quality audit ran2026-09-27 00:07:37–00:08:16 UTC;
the phase audit ran00:08:37–00:08:43 UTC. No further simulation or render was run for
this report.

Compact local evidence is `output/p292/{outer_no_pic60,handoff_no_pic60}.json` and logs,
`quality_handoff.json`, and `phase_handoff.json`; compact originals remain under
server `work/p292/`. The two completed raw archives were subsequently verified
and offloaded to `C:/dev/physmorph_archives/p292_outer_no_pic/` and
`C:/dev/physmorph_archives/p292_handoff_no_pic/`; see the durable receipts and
[maintenance record](maintenance_20260926.md). The isolated audit snapshot is `audit_handoff`. Executed
probes are also preserved locally as `executed_quality_compare_handoff.py` and
`executed_raw_phase_handoff.py`.

- Quality probe SHA-256: `cf913ad3916311d9c670f222e931e21888a095600922b5323682ca2f842408c6`.
- Phase probe SHA-256: `f35224b41e027fd7b2e582006871a712035e48b597edddd7eeeeb3e20436a60e`.
- Quality JSON SHA-256: `46f9543119baefa41bd4a81c496abea8a0d588ae245d7f8b9121a012d66ab905`.
- Phase JSON SHA-256: `71867b4aaccc89d4574c25d1a54e316462b2c3f8ef5ff7382e53b1c77d43f822`.
- Shared423-ID SHA-256: `321d167d91760f5121a8b85c9f807e5fce13f84eda95145af2604de6a108fc24`.

The configuration/source/scope audit and pre/post extension passed adversarial code
review and12 independent CPU tests. The independent final numeric/causal report gate
is closed: endpoint/common-commit geometry, reference history, pre-trigger divergence,
cohort intervals, pin and reversal counts, tip retention, progress crossings and
probe/result hashes were checked against the compact evidence. The reviewer did not
rerun GPU work.
The decision is unchanged by the partial endpoint improvement: do not promote this
candidate as meeting the shape, hole-free morphing or rest requirements.
