---
name: photoreal-3dgs-goal
description: "Standing directive (2026-09-01) — iterate autonomously until floaters are fully solved, generalized across pairs, and ONE photorealistic 3DGS result exists"
metadata: 
  node_type: memory
  type: project
  originSessionId: f83ab3dd-27cd-4a63-8354-f08614b6a754
  modified: 2026-09-02T16:35:24.822Z
---

The user (2026-09-01): "이 문제 해결 및 일반화 해서 photo realistic 한 3DGS 결과 한 개가 나올 때까지 계속 iterative하게 진행 해" — and earlier the same day: "bunny ear 사이에 있는 입자가 아예 없도록" (ZERO particles between the ears), per-particle instrumentation of gradient/physics relations while doing it.

**Why:** The deliverable is a hero morph rendered photorealistically with real 3DGS (diff_gauss build exists on hyde06), with no floating Gaussians, generalized beyond sphere→bunny (armadillo/A→C/spot suite).

**How to apply:** Keep the iterate loop: instrument per-particle → mechanism (lit-anchored, pre-registered falsifier) → adversarial review → 4-GPU A/B (attribution discipline) → visual + honest metric verdict → docs.

**State as of 2026-09-02 (both problem dossiers CLOSED — docs/floaters.md, docs/oscillation.md):** flagship = `render_full_dt_iso_nn` (+fill v3) = h19 config: converged records chamfer 0.0652 / silIoU 0.9766, freeze at ~commit 270-360 (production budget ≥300c), fork floaters 326→78, J∈[0.92,1.09]. Stack: isochoric assimilation + sKL w_jvol=50 + kNN-gated DT-W1 0.2 + near-band nn-W1 0.2 + fill v3 α0.1 + trajectory-det guard; PCGrad removed, lg parked. Ear-fill mechanism class CLOSED after 5 falsifications (fill v1-v4, growth v1) — cov<15% criterion retired as over-spec by visual audit. Live viewing: pipeline_run --live_port + /quad (4-GPU). Remaining roadmap: real diff_gauss LOSS tranche, photoreal art direction (face-on framing), teapot/letters at target-relative gates, Tampubolon volume ledger + IPC step-length filter queued. Related: [[papers-before-ambiguous-design]], [[structural-fixes-only]].

**State as of 2026-09-02 late (dossiers REOPENED by the viewer standard, then root cause re-measured):**
The "render gradient is diluted by the MPM adjoint" root hypothesis (docs/root_analysis.md §2) was
REFUTED by measurement (docs/probes/transfer_function.md: real silhouette covector reaches dFc at
gain 0.89 vs physics 1.03; the 1000x is loss scale; F-smoothing is a temporal EMA). Sobolev
preconditioning no-op, F-smoothing s<=0.8 falsified, material carrier (Tier M) NO-GO, Tier D
dressing failed K-D2 (docs/probes/, docs/local_global_design.md §14). Observability probe: floaters
DO get silhouette gradient (97%) but with median cos ~0.3 toward the target; the surface-parent gauss
loss is blind to the interior half of floaters; the unowned class = floater CLUMPS beyond the nn-band
far cap (E4 = nn_far_k → ∞ tests it). Optimizer side: driver #5 (per-window Adam restart) → anneal +
mom_carry; pacing via pace_budget (glidepath 1−ρ^(1/A)); driver #6 flat-valley limit cycle → outer
gate v3 (shape-only merit, brake vs latched rejects, self-healing latch, cold restart) + best-commit
deliverable truncation. Assimilation η is not a lever (E2). Replay-noise-calibrated commit tolerance
(E3). Pending when written: b8 (gate v3, 450 paced) and E4 verdicts; hyde06 ssh key was being
rejected at the end of the session (auto-retry loop armed). Related: [[hyde06-ops-etiquette]].

**Session end 2026-09-03 (user: "작업 여기까지만"):** mom_carry RETIRED (r1 vs r2 isolation). Current
best recipe = render_full_dt_iso_nn + anneal 0.7 + nn_far_k 1000 (E4) + outer_merit gate v3 (brake,
shape-only merit) + best-commit truncation, mom_carry 0, pace left at the arm default 0.12 (or --pace 0).
Left RUNNING unattended on hyde01 GPU1 (outputs ~/physmorph_v2/output there): r3_unpaced_gate (20k,
--pace 0) and r4_record40 (40k record attempt; g2 reference 0.0701, r2 0.0730). Watchers were stopped;
read their logs/json next session, copy json+log to hyde06 /data/relcfd/chayo/physmorph_v2/output, then
delete hyde01 files (user wants hyde01 clean). Open bottlenecks: late-run runaway cause (elastic memory /
lambda near optimum) and floater direction/ownership (E4 modest); pacing line failing; photoreal not yet
produced with the new pipeline. Related: [[hyde06-ops-etiquette]].

**2026-09-03 (continued, "무조건 해결"):** r3 (20k, `--pace 0`, anneal 0.7, mom_carry 0, nn_far_k
1000, outer_merit gate v3) reached d_vol 23.4 / out_nn 8.6% / far 417 — 2.7x below any prior run:
the arm-default `pace 0.12` was a hidden convergence cap in EVERY earlier flagship run (r2/r4/g2
froze at 100-230 commits). NEW RECIPE = `--pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000
--outer_merit` (no pace_budget). Delivery truncation now uses d_vol+d_dt only (d_sil jumps at the
c2f rebuild) and never drops frames (npz stores all frames + deliver_n). Running when written: r5
(40k, this recipe) and r3b (20k re-run for a full-frame archive) on hyde01 GPU1 (hyde06 saturated).
Next: observability probe on r3b's delivered state, then photoreal render on hyde06 (diff_gauss).
**2026-09-03 late:** berth 1.0 adopted (x2, 40k chamfer 0.0706 / out_nn 9.5%, best 40k so far).
Clustering diagnosis (particle/target local density 2-2.7x, ears 9-24% under-filled, J~1) drove a
particle-scale KDE density term: v1 one-sided (x3) and v2 two-sided (x5) fixed the interior
(cluster 2.0->1.25, unfilled 33->27%) but scattered the fringe (out_nn 8->16-22%); v3 (squared
target-support weight on the particle side) is x6. loss_res 96 falsified (N-limited). Armadillo
generalizes (out_nn 1.6%). Photoreal at display sigma 1.0 shows the clustering as speckle.

**FOUNDATIONAL BUG FOUND 2026-09-03 (late):** `assets/bunny.obj` is non-watertight and trimesh's
default `voxelized().fill()` added ZERO interior voxels, so `sample_volume` silently returned a
SURFACE SHELL; every bunny run morphed a solid sphere into a hollow shell target (verified: 98% of the
converged body within 0.1 of the shell, NN spacing collapsed 0.0756 -> 0.0293). "Floaters" were the
layer's thickness; "clustering", "under-filled ears", the KDE losses and the loss_res ladder were all
fighting the sampler. Fix: axis-based fill ('base'/'orthographic') with an interior-voxel check +
target volume matched to the source (isochoric particles cannot change total volume) in
physmorph/sampling/mesh.py + pipeline_run. ALL bunny metrics before this fix are shell-target
numbers (chamfer 0.070 was the shell-vs-shell floor); the ladder must be re-run on real solids.
Optimizer-side conclusions (mom_carry cascade, pace 0, gate v3, best-commit delivery, E4/berth)
stand on their own evidence. Related: [[hyde06-ops-etiquette]].
**2026-09-04:** first real-volume runs: v1 solid bunny 20k out_nn 1.1% / far 28 / uncovered 16% (floor 11%)
/ guards 0; v2 solid armadillo out_nn 1.6% / far 12. The floater problem is at the sampling floor on
real targets with the adopted recipe (pace 0, anneal 0.7, mom 0, E4, berth 1.0, gate v3, best-commit
delivery). v3 = 40k solid bunny (hero). Open: photoreal rendering of SOLIDS (render_photoreal was tuned
on shell layers; surface extraction needed), oscillation re-check on solids, hero-scale replicates.
v3 (40k solid bunny hero baseline): out_nn 1.5%, far 100, hole 0%, guards 0, conv 164 — floaters solved at
the sampling floor; remaining defect = ears under-reached (ear-region particle fraction 0.069 vs 0.100)
**State as of 2026-09-04:** v5/v5b density-J prior FALSIFIED (a prior at parity cancels the
data term; v5b looped on an identical rejected candidate → gate fix: consecutive brake rejects
count as stale). v4/v6 reversal-based brakes FALSIFIED (reversal is not a plateau signal); the
solid-hero tail zigzag measured SUB-SPACING (0% of particles move >0.5 sp per commit) → the
oscillation line is CLOSED with a reopening rule (docs/oscillation.md Addendum 7). The real
residual on solids is a VERTICAL UNDER-STRETCH (mid band +35..67%, both poles −20..40%, inside a
correct silhouette): every shape term is local, L2 D_vol is blind inside a uniform surplus band
(band_push probe). Fix = H⁻¹ (FFT-Poisson) norm of the density residual, `--w_h1 1` (equal-norm
to D_vol at the source; docs/method.md): v7a 20k reached d_vol 12 by commit 24 (v1 floor 30 at
139). v7a/v7b running on hyde06 GPUs 1/3 (placed at 0%/35% util; nurshat's jobs joined later).
REFUTE (Codex xhigh, docs/reviews/refute_h1_codex_20260904.md) fixed: H⁻¹ moved OUT of phys_core
(W1 precedent: no λ/PCGrad coupling), calibration survives the c2f rebuild, replay-signature stale
rule (not a bare streak). v7a/v7b ran the in-core version; isolation arm v7c (out-of-core, 20k) on
hyde01 GPU0 (/tmp/pm42; hyde06 was 100% on all GPUs per the placement rule). Lateral decompression
measured (+0.50 radial in the band) — v7 census out_nn/floater_frac is the arbiter.
v7a VERDICT (2026-09-04): all pre-registered criteria met — best d_vol 4.69 (v1 30), bands 1.01/1.04/1.04,
ear frac 0.105, uncovered 12.5/1.7 (floor 11.2/1.0), out_nn 1.31%, guards 0, chamfer 0.1571; paired census:
cluster ratio 1.18 vs 1.23 (no x3-style trade). Opus REFUTE → P3M self-energy subtraction in d_h1 (lattice
attractor), back in-core, h1_ratio logged; v8a (self-corrected) running hyde01 GPU1, v7b 40k hyde06 GPU3,
v7c out-of-core hyde01 GPU0. H⁻¹ provisionally ADOPTED.
Next: v8a census → adopt v8 recipe → 40k hero (v7b) → pacing re-test
(spread the fast morph over the frame budget) → photoreal surface rendering. Ear width is 2.5
cells at dx 0.5: resolution is a deferred second suspect.
