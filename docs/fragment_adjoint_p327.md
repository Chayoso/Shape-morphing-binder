# P327: production effect of retained fragment activity

P326 found an incorrect lifetime for the per-step fragment mask in reverse. Its
small changing-mask fixture demonstrates an opposite-sign old gradient with an
unchanged forward path. P327 measures the effect in the actual N300k morph before
claiming any improvement in holes, residual flow, arrival oscillation or fitting.

Both arms use one frozen checkout and the unchanged full-horizon raw recipe:
N300k,T20,dt1/240,dx.3062907543956724wu,loss36^3,iters8, maximum300 attempts,
ordinary stopping, no endpoint PIC, no shift_sub. Rendering guidance is18views
at64px with GS loss off. No withdrawal objective or new rest policy is enabled.

`fragment_adjoint_compare.py --mode legacy` aliases every gradient trajectory's
fragment-mask slots to one buffer before capture, reproducing the old reverse
lifetime. `--mode retained` uses the corrected per-step arrays. The original
allocation remains owned in both arms; no-grad evaluation is unchanged. Both
arms separately record actual masks into identical per-step observation buffers
inside the captured forward. Reading the legacy array list after the rollout
would incorrectly report the final mask at every time, so that is not used.

After every actual persistent-adjoint forward, device reductions report per-phase
active and dynamic-only counts, differences from the final mask and temporal
flips, for all/free/layer-free cohorts. One scalar packet is copied to the host.
These are attempted gradient evaluation points, including repeats; they are not
automatically accepted inner updates or committed outer endpoints. Warmup/capture
construction is not counted as an executed optimizer forward. Diagnostics add
24MB per N300k,T20 bonded trajectory plus temporary reductions and small JSON;
do not treat measured runtime as uninstrumented performance.

Runs are serial, with the shared GPU launch lock/interval and30GB reservation
per arm. The wrapper/mode protocol and all completed observer data are hashed
against the full-horizon trace, whose input/physical-code/output bindings remain
unchanged. An observer failure does not grant quality or provenance clearance.
Both arms have identical physical code/config digests because the intervention
is a runtime wrapper. Pair analysis must consume each activity report's bound
fragment protocol and validate legacy then retained modes; ordinary digests alone
do not identify this intervention.

Use existing P317 chronology/arrival/free-ID analysis and P319 every-phase raw
shape observations. Compare the common physical clock separately from each own
actual endpoint and held padding. Preserve original acceptance/quality gates;
no new rest-speed threshold. Projected openings alone are not3D watertightness.
Use per-arm rendering-influence reports: norms, balance, accepted update dot
products and reference-local image loss changes. A changed gradient from this
fix may affect both physics and rendering paths; norm share is not a causal
motion fraction. A pair alone cannot establish universal improvement or replace
the19-case gallery and all-frame visual QA before a new deliverable is adopted.

The producer frozen6a525c3 passed3 CPU observer cases (including full C parity)
and2 actual CUDA graph cases. The same actual GPU test launch additionally passed
all6 stronger P326 derivative/stream/fragment checks, for8 passed and0 skipped
in2.21s. The serial production pair is `p327_legacy1` then `p327_retained1`, GPU0,
from `work/p303/code_fragment_adjoint1`. Outputs remain diagnostic; the retained
before/corrected4K exports are unchanged.

The pair comparator now supports `fragment_adjoint_retained_full` with identical
configs bound to the exact native raw recipe, explicit ordered modes, full
producer/wrapper/input hashes, every attempt's disposition and archive frame
span, exact cohort basename, and before/after output/log checks. Existing full
raw metrics/thresholds are reused. Final40 metadata cases passed independently;
broader existing quality/phase/layer checks also passed. This does not independently
recompute the activity packet counts. For analysis, clone the frozen6a525c3
producer and overlay only `quality_compare.py` and `raw_phase.py`; its producer
ops/docs/physical files must remain byte-identical. An external launch wrapper
may supply the ordinary GPU lock/interval/storage guard without replacing those
bound producer files. No analysis job may silently relabel the two equal-config
arms from code hashes alone.

## Completed physical pair and common-clock comparison

Both runs finished with zero guards. Legacy accepted30/31 attempts and stopped
on outer rejection patience; retained accepted33/33 and stopped on accepted-track
plateau. Archives contain601/661 physical states plusone held suffix row each.
Observed times441.65/459.35s include the activity instrumentation; they are not an
uninstrumented speed comparison. Neither termination evaluates individual rest.
All following numbers use the N300k,T20,dt1/240,dx.3062907543956724wu,loss36^3,
iters8 discretization above, with source spacing.03498853660707278wu.

Legacy records22/248 attempted adjoint forwards with masks differing from their
final phase, only at zero-based attempts12,13,14; maximum26 particle-time
differences in a forward. Retained records2/264, at attempt15, maximum5
particle-time differences. Each arm's peak instantaneous difference is two
particles. These are producer observations, not independently replayed activity
or a bound on downstream causal influence.

The first12 endpoint NPZ arrays differ bytewise between arms, starting at
attempt0. Both activity reports contain no final-mask differences in those
attempts. At accepted commit1 the matched-particle position difference is
median.0004459sp,p95.0049879sp,max.0266862sp; by commit6 it is median.75656sp.
Thus the optimized paths already diverge before the recorded temporal-mask
exposure. Do not attribute the full pair difference solely to the mask fix or
claim that this pair isolates its causal quality effect.

At the common accepted commit30:

| Raw measure | Legacy | Retained |
| --- | ---: | ---: |
| 24-view128px silhouette IoU | .96439204 | .96370623 |
| Target support fraction | .99292333 | .99286667 |
| Fixed upper-target support fraction | .94083072 | .93945925 |
| Fixed source-upper cohort density | .98430051 | .96573302 |
| Particles in target-tip ball | 83 | 77 |

Same2025 IDs unpinned in both arms at the common endpoint, selected from its
sparse surface, have raw-step RMS.01408774/.01366174sp over commits20->30
(-3.0%). Their between-commit reversal fractions are.060027/.030123 and
within-raw-step fractions.001663/.001216. Direction changes are not by themselves
periodic oscillation. This is an outcome-selected cohort, not a general rest
result: the fixed6712 source-upper IDs instead have raw-step RMS
.0056572/.0059890sp (+5.9%), with different pin fractions between arms.

Own final endpoints differ in physical clock (commit30 versus33): IoU
.96439204/.96379333, target support.99292333/.99293000, tip count83/76.
The final-free cohorts differ (20803/28429 IDs); their own last-window terminal
geometric RMS speeds are.0830102/.0713536wu/s, with exactly zero endpoint position
correction. Those unmatched speeds cannot establish an improvement for the same
material. Both are nonzero even though all source-plan arrival predicates hold
at the end. Existing held frames and newly pinned particles are imposed stillness,
not natural rest.

The raw quality result is mixed, so the correction is retained as an adjoint
correctness fix without promoting a hole/rest/4K remedy. Existing before/corrected
deliverables remain unchanged. Full-phase projection observations and their
bounded independent audits are recorded separately below.

P319 examines every physical archive state and each raw optimizer endpoint:
631 samples in legacy,694 in retained,24 views at128 and256 pixels. Maximum
projected internal-hole pixels outside the target's own holes are77/54 at128px
and275/279 at256px. The maxima occur at different frames/views; they are not a
matched-time causal reduction. Neither run is hole-free under these projections,
and no3D watertightness or4K appearance conclusion follows. Full reports remain
under `work/p303/p327_{legacy,retained}_shape1/result.json` on hyde06.

Independent source/archive and motion audits pass for both arms. The motion
audits recompute the actual last accepted raw interval and saved per-ID path
reductions; they do not replay every earlier raw path or arrival predicate.
Shape audits pass9876/10769 stored-mask/coverage reductions across631/694
samples, plus7/6 selected source-state reprojections, with zero violations.
Independent target-distance checks remain bounded to128 IDs per selected state.
Pair metadata checks bind172 files and exact phase/cohort/reversal partitions;
four already independently full-hashed large archives are reused by current
identity, rather than rehashed twice. Activity scalar bounds are checked, but
the masks themselves are not replayed. Audit scope and hashes are retained in
`docs/evidence/p327/independent_audit_index.json`. These checks validate the
reported evidence and its limits, not a quality or natural-rest claim.

## Rendering influence in this pair

Both arms use18-view64px guidance with GS loss off. Legacy/retained committed
windows contain240/264 accepted inner updates. Median nominal render-direction
share is.445574/.447220; body share.305606/.308935, stress.438834/.458243,
surface-u.656442/.622558. Median adaptive lambda is.0399058/.0262601; median
reference-local image-loss change is-1.85489e-5/-1.33174e-5. References and
adaptive lambda evolve, so these do not compare one fixed image objective.
Shares are not causal movement fractions; no render-off matched experiment was
run here. The new withdrawal capability has no active loss weight, and neither
this guidance nor this mask fix supervises the exported4K covariance footprint.
