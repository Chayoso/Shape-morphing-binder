# Experiments

What to run next, what is running, and what has been measured. Every experiment runs on the GPU server (hyde06).
Before reading a result, add its entry here with the prediction and the pass criteria; then add the result, including
failures. Stamps use the server clock (CDT). The full record of the earlier pipelines is on the `v3-grid-gs` branch
(tag `v3-grid-gs-final`).

## Running now

- **S3, GPU-only refactor equivalence (2026-09-29 12:41 CDT).** The settled path rebuilt as GPU-only modules, every
  file under 500 lines (README, layout). The 19-mesh gallery at 40k and the 300k bunny and dragon, seed 97, with the
  new code (`output/gpu/g/`), against the S2 settled runs. Pass: every mesh's silhouette IoU within ±0.002 of its S2
  run; a mesh outside that band gets a second run of the old code at the same seed, and passes if the new result lies
  within the old code's own run-to-run spread. Prediction: all pass, and the 40k runs are faster (the per-window host
  round trips are gone).
  **S3b (pre-registered 2026-09-29 13:45 CDT).** The 300k dragon: both new runs (0.9686, 0.9764) lie below the old
  code's three (0.9769–0.9787). Three more runs of each code, seed 97, the adopted recipe. Pass: the new median lies
  inside the old runs' range, and a one-sided rank test (Mann–Whitney, new below old) gives p ≥ 0.05. Prediction:
  pass; the spread at 300k is set by where each run stops (26–36 windows), not by the code.
- **R2, loss resolution and support floor at 300k (pre-registered 2026-09-29 14:05 CDT).** Two switches, both off by
  default: `--loss_follows_n` (the transport grid and its blur refine as (N / 40000)^(1/3) above 40k particles, D/26
  → about D/51 at 300k; the MPM grid, dt, render and the transport gate's MPM-cell radius unchanged; at N ≤ 40k the
  loss grid is the MPM grid, so the 40k gallery is untouched) and `--support_target_ref` (the support floor is half the
  target's own kernel density at each particle's nearest target point, instead of half its median: the target itself
  pays nothing, a particle off the target still compares with the nearest target density). A 2×2 on the 300k dragon
  (two runs per arm) and bunny: A (both off; the S3/S3b runs), B (support), C0 (loss grid), C (both); plus the dragon
  with a long budget at D/26 (`--reject_stop 20 --patience 20`), the stopping control. Census as R1d (both
  thresholds), with λ, g_share, wall time, windows and E. Predictions: C0 lowers the bunny's own-threshold uncovered in
  the 1–2, 2–4 and ≥ 4 bins beyond A's spread, with anisotropy p90 ≤ 1.2; on the dragon it changes little unless the
  run finishes (the stopping control decides that part). B lowers the < 1 and 1–2 bins by a few points and leaves the
  ≥ 4 bin alone (the target meets the global floor there: 0.6 % of its points). λ moves by itself under C0, because it
  is calibrated on the first window's physics gradient. Pass for adoption: lower own-threshold uncovered than A,
  beyond A's spread, in at least three of four bins on both meshes; anisotropy p90 ≤ 1.2 and det F min, reversals
  and centre-of-mass drift within S2's gates; silhouette IoU not below A's range; for B, the 40k gallery within
  ±0.002.
- **R2b, arm C before adoption (pre-registered 2026-09-29 14:55 CDT).** (1) The 40k gallery, 19 meshes, seed 97,
  with both switches (the loss grid is the MPM grid at 40k, so only the support floor changes; the log must show
  loss_res = grid). Pass: every mesh within ±0.002 silhouette IoU of the new code's gallery run; a mesh outside gets
  a second C run and passes if both lie within the new code's same-seed spread; the median surface gap does not rise.
  (2) The render-off twin of C on the 300k dragon: the render effect against the spread of the two C runs.
  (3) Generalisation at 300k: A and C on nefertiti and V. Pass: C's surface gap below A's on both, silhouette IoU not
  below A's by more than 0.004 (the dragon's same-code spread was 0.009), reversals and COM drift within the gates.
- **R3, the support penalty as a missing-mass fraction (pre-registered 2026-09-29 16:23 CDT, before launch at 16:24).** `--support_form
  ratio`: the per-particle penalty radius²·relu(1 − s/f)² in place of radius²·relu(log f − log s)². The missing
  fraction of the floor's local mass is at most 1 by definition. It agrees with the log form as s → f, and under it an
  isolated particle adds at most radius²/N and gets no gradient, which leaves it to the W1 cleanup (SF: the stray
  particles of the dragon have W1 gate 1.00). No new constant. The outer bound E + E·wB/(E + wB), MPM, N, render and
  λ are unchanged. Every committed window now records E, B, the support-gradient weight w(E/(E + wB))² and the
  per-particle penalty's max / p99 / median.
  Stage 1 (in parallel): A_R = A + ratio (global floor), the 300k dragon twice with `--ls_probe` (the causal test;
  its coupling weight moves only 7.3 → 7.8 by SF); C_R = ratio + target floor + `--loss_follows_n` on the 300k bunny.
  Stage 2, only if stage 1 passes: C_R on the 300k dragon twice, and C_R on the 40k gallery (19 meshes; at 40k the
  loss grid is the MPM grid).
  Pass, stage 1: (P1) continuity: in the failed small-step trials the objective change shrinks with the step
  (no step-independent offset above the replay noise), and no window's accepted step falls below 1e-6 after window
  3 or ends in three rejections from collapsed steps; (P2) progress: both runs keep improving after window 20, and
  their end E and thick-bin uncovered share at the fixed world distance lie beyond A's five-run spread (E below
  1.2e-2; ≥ 4-cell bin below 7.2 %), or the run reaches the transport level of the 40k dragon; the largest
  per-particle penalty stays ≤ radius². Pass, stage 2: the same on C_R's dragon runs; 40k gallery: nefertiti does not
  collapse, median surface gap ≤ 7.1 % (A), every mesh within ±0.002 silhouette IoU of A or within the new code's
  same-seed spread on a repeat.
  Predictions: P1 and P2 pass on the dragon. The bunny changes little (it has no tail; SF). At 40k the support
  gradient grows 1.7–3× through the bound (SF), and with the target floor this should not cost the thin bins. The
  isolated particle's discontinuous end position stays; only its weight in the objective goes.
- **R3c, the ratio form alone at 40k (pre-registered 2026-09-29 17:56 CDT, before launch).** A_R (ratio form,
  global floor) on the 40k gallery, 19 meshes, seed 97. It separates the two changes C_R makes at 40k. Through the
  bound, the support gradient at 40k rises 1.7–3× (SF). With the global floor, which the targets' thin features
  already violate (R1b P3), the prediction is: A_R costs the thin-feature meshes (median surface gap above C_R's
  7.2 %, and silhouette IoU below C_R on more meshes than above). If A_R matches C_R, the target floor is not needed
  at 40k.
- **R3d, follow-up to R3 stage 2 (pre-registered 2026-09-29 18:50 CDT, before launch; a follow-up criterion set
  after the stage-2 result, not a pass of the original).** (1) Run-to-run spread at 40k: a second run of A (the new
  code's defaults) and a second run of C_R on the 19-mesh gallery, seed 97. Follow-up pass: C_R's median surface gap
  (the mean over its two runs) is not above A's two-run mean by more than A's own run-to-run difference in median
  gap, and no mesh has both C_R runs below both A runs by more than 0.002 in silhouette IoU. The four meshes above the
  band (beast, C, cheburashka, teapot) count as improved only if both C_R runs lie above both A runs. (2) Render
  influence under C_R: the 300k dragon with C_R and `--render_weight_scale 0`, against the spread of the two C_R runs
  (silIoU 0.9833–0.9838; census bins within 1 point). Prediction: the median-gap difference (0.1 point) is inside
  A's run-to-run spread; the render keeps its effect under C_R (render off costs more silIoU than the C_R spread,
  mostly in the thin bins).
- **V1, are C_R's late reversals visible? (pre-registered 2026-09-29 19:35 CDT, before launch).** C_R again on the
  40k meshes whose second half reversed most (dragon, cow, teapot, spot, armadilo; run 3, archives kept), against
  A's first-run archives of the same meshes and the 300k A_R dragon (run 2). `window_reversal.py`: per window the
  outer layer's median step (target spacings) and the cosine with the previous window's step. `video_flicker.py`
  on the quick two-view splat video: per frame the alternating (ALT) and drift (DRIFT) parts of the image change,
  over the whole run and its last 20 %. Visible oscillation = reversing windows whose step is comparable to the
  windows before them (≥ half their median) and a tail ALT above A's with ALT/DRIFT > 1. Prediction: not visible.
  The reversing windows move the layer by well under a tenth of the earlier windows' step, and the tail ALT matches
  A's.
- **T2, what keeps thin features uncovered (pre-registered 2026-09-29 19:58 CDT; read-only, existing archives).**
  At the end states of the 300k dragon (A_R run 2) and the 40k A dragon and bunny, take the outer target points of
  the two thin bins (< 2 MPM cells) left uncovered, and ask of each one:
  (a) Resolution: the training silhouette deficit clamp(a_t − a, 0) at its pixel, with the training operator (18
  views, extent 1.25·max|target|, k = 1.5, CIC), at 64 and 96 px (the training resolutions) and at 256 px.
  (b) Occlusion: in how many views the silhouette depends on it at all. It is exposed at a resolution if every target
  point in its pixel lies within two MPM cells of it in depth, so that the pixel would be empty without the feature.
  It is front-visible if it lies within two cells of the nearest target depth there (the shading term can see it).
  Reading: uncovered points that are exposed but show no deficit at 64/96 px and a deficit at 256 px → the render
  resolution is the limit. Points exposed in no view → no silhouette term reaches them at any resolution, and a 3D
  term is needed (shading can still reach front-visible ones). Points with a clear deficit at the training
  resolution → the render sees them but does not fix them, and the limit is elsewhere (gate, weighting).
  Prediction: at 300k mostly the resolution case (a pixel is 2.7 spacings and the alpha saturates at a few particles
  per pixel); at 40k a mix of resolution and occlusion.
- **R4, the support made two-sided (pre-registered 2026-09-29 20:35 CDT, before launch).** The support is one-sided.
  It asks each body particle for enough body around it, so it cannot see target material with no body near it
  (T2: no term sees an empty thin region in 3D at the particle scale). `--support_two_sided` applies the same
  estimator (32 nearest, h = r8 / 2), the same floor and the same ratio form at every target point as well, with the
  body's kernel density there against the floor there, and averages the two sides. Weight, bound and channels are
  unchanged, and there is no new constant. A new measurement, not in the objective (`physmorph/thin.py`), is
  recorded in every run: the thin set is the outer target points thinner than two MPM cells, with its uncovered
  share (`thin_uncovered`, 1.5 target spacings; `thin_uncovered_world`) at the end and after every committed window.
  Stage 1: the 40k gallery (19 meshes, seed 97) with C_R and with C_R + `--support_two_sided`. Pass: (primary,
  thin) `thin_uncovered` lower under two-sided on at least 14 of 19 meshes (one-sided sign test p ≈ 0.03) with a lower
  median; (secondary) silhouette IoU median difference ≥ −0.0005, no mesh below C_R by more than 0.004, median surface
  gap not above C_R's, no collapsed window. Stage 2, if stage 1 passes: the 300k dragon and bunny.
  Prediction: the thin share falls on most meshes by several points (the 40k gaps are 1.6–2.6 spacings, inside the
  kernel's reach); the thick bins and the silhouette change little.
- **D3, what limits thin features: the objective, the controls, or an eroder (diagnostics, pre-registered
  2026-09-29 21:26 CDT, before launch).** Both on the 40k gallery (19 meshes, seed 97), against R4's C_R run
  (`thin_uncovered` recorded). Neither is a recipe candidate.
  D3a, capacity: C_R + `--diag_coverage 1`. It adds the thin set's missing-mass fraction (the support estimator at
  every thin target point, times r8², in the transport's length² units) outside the support's bound. If
  `thin_uncovered` falls on ≥ 14 of 19 meshes with a median drop ≥ 3 points, the controls can cover thin features when
  an objective asks for them at the particle scale, and the limit is the objective. If ≤ 11 of 19, or a median drop
  < 1 point, the controls or the dynamics are the limit.
  D3b, erosion: C_R + `--no_layer_relax`. The relaxation projects each outer particle's normal offset toward its
  neighbours' mean every step, which can round thin tips. If `thin_uncovered` falls on ≥ 14 of 19 meshes, the
  relaxation erodes thin features. Otherwise it is not the limit.
  Prediction: D3a lowers the thin share clearly; D3b changes it little.
- **D3c, both at once (diagnostic, pre-registered 2026-09-29 22:26 CDT, before launch).** D3a gave a consistent but
  small thin gain, and the partial D3b (13 of 19) none, though its silhouette IoU rose on nearly every mesh. The
  per-particle control u can place material below the MPM cell only if something asks for it and nothing undoes it.
  C_R + `--diag_coverage 1 --no_layer_relax` on the 40k gallery. If `thin_uncovered` falls on ≥ 14 of 19 meshes with a
  median drop ≥ 3 points, u can close thin gaps, and the relaxation is what blocks a particle-scale objective. If the
  drop stays ≤ 1.5 points (about D3a's), the limit is the dynamics' resolution (the MPM cell), not the controls' reach.
  Prediction: the drop stays small (≤ 2 points).
- **D4, beast's collapse under target-side coverage (diagnostic, pre-registered 2026-09-30 00:04 CDT, launched
  00:04).** Beast froze at window 13 twice (R4 two-sided, D3c coverage without relaxation) and never under C_R or
  D3a. Reruns of both with `--ls_probe` (run 2). If the freeze recurs, the probe shows where the failed trials'
  objective change sits (transport with coverage, support B, kinetic, render) and whether it shrinks with the step. If
  it does not recur, the freeze is trap timing (as S3b was).
- **R5, the relaxation moved onto u as a preconditioner (pre-registered 2026-09-30 00:23 CDT, before launch).**
  Literature (7 primary sources): the relaxation is Taubin's averaging step without the un-shrinking step. Its
  2-spacing stencil is the size of the thin gaps, so no linear smoother in the forward model can tell u's roughness
  from the target's thin shape. Nicolet, Jacobson & Jakob 2021 move smoothing into the step's parameterisation,
  which leaves the objective's minimiser unchanged and keeps large steps stable. `--u_precond`: u = S v with
  S = (I + 2 (I − W))⁻¹ on the layer graph, where W holds the relaxation's own weights and 2 is its strength over a
  window (2T steps of 1/T). The forward relaxation is then off. No new constant.
  R5a: C_R + `--u_precond`. R5b: C_R + `--u_precond --diag_coverage 1` (D3c's coverage signal, still a
  diagnostic). 40k gallery, seed 97, against C_R, D3b and D3c.
  Pass R5a: silhouette IoU median ≥ C_R + 0.003 (D3b gave +0.0069); gallery wall ≤ 2× C_R (≤ 100 min, D3b took 155);
  collapsed windows summed over the gallery ≤ half of D3b's; thin share median not above C_R's + 0.5 point.
  Pass R5b: `thin_uncovered` lower than C_R on ≥ 14 of 19 meshes with a median drop ≥ 2.5 points (D3c: 14 of 19,
  −3.0); collapsed windows summed ≤ half of D3c's (163); gallery wall ≤ 120 min (D3c 170); no mesh frozen by null
  windows. Prediction: R5a keeps most of D3b's silhouette gain with far fewer collapses; R5b keeps D3c's thin gain
  (−2 to −3 points) at lower cost.
  Restarted 2026-09-30 02:33 CDT on 04665e0: the first launch (00:23) ran 25× slower per gradient because the
  preconditioner's gather used index 0 for every off-layer row, which serialised its backward on one address. The
  fix runs the preconditioner on compact layer indices. The mathematics is unchanged, and the four runs finished
  before the fix are kept aside (`output/gpu/r5_aborted_slow`) and not used.
- **D5, what makes the sub-cell tail slow (diagnostic, pre-registered 2026-09-30 10:12 CDT, before launch).** Without
  the forward relaxation (R5a) the runs keep improving by about 1 % of the merit per window for a hundred windows and
  more, and collapse late on merit. Both controls share one Adam step length and one backtracking search. dFc is a
  dimensionless deformation increment that acts through the grid on the whole body; u is a normal offset in world
  units. In R5a's tails the accepted step is about 1.5e-4, so u moves 0.2 % of a spacing (0.07 wu) per iteration, and a
  1.6–2 spacing thin gap would take about a hundred windows. At 300k (D1) the dFc-only step raised the objective in
  87–96 % of failed trials, while the u-only step mostly lowered it. H_block: the dFc block sets the shared step, and
  the sub-cell block is carried along at a step far below its own scale. Run: C_R + `--u_precond --ls_probe` on bunny,
  cheburashka and homer (40k, seed 97). The tail is the committed windows after a run first reaches C_R's end
  silhouette loss. Over the failed trials in the tail, H_block is supported if the u-only step lowers the objective
  while the dFc-only step raises it in ≥ 70 % of them. It is refuted if the u-only step raises the objective in at least
  half of them. A discontinuity (as in D1b) is indicated if, at steps below 1e-5, the joint change stays above 1e-7
  relative whatever the step. Prediction: H_block supported. If it is, the fix is structural (a step length per
  control block), designed after a literature pass. No step constant changes.
- **R6, a step length per control block (pre-registered 2026-09-30 10:33 CDT, before launch; repo_r12).**
  `--block_steps` gives each control block its own step length, as D5 and its literature pass indicate. The Adam
  moments take the gradient once. From the current point, dFc and then u each run their own backtracking search
  (the other block held), starting from their own step memory, under the same acceptance test. When both blocks
  find a step, their sum is tried at full and at half length (for two coupled blocks, the halved sum of descending
  block steps descends). The lowest accepted candidate is taken. No new constant.
  `--u_uniform_adam` gives the u block one second-moment scalar, as Nicolet et al. prescribe for a preconditioned
  parameter.
  Arms on the 40k gallery (19 meshes, seed 97), with `surf_rough` (`physmorph/surface.py`) recorded in every run:
  - CR2: C_R again, for the roughness baseline and a second C_R sample.
  - PCB: C_R + `--u_precond --block_steps`.
  - PCBU: PCB + `--u_uniform_adam`.
  A candidate passes against CR2 if all of these hold:
  - `thin_uncovered` is lower on ≥ 14 of 19 meshes, with a median drop ≥ 2 points.
  - Silhouette IoU median ≥ CR2 + 0.003.
  - `surf_rough` median is not above CR2's.
  - The gallery's summed run time is ≤ 2× CR2's.
  - No mesh is frozen by null windows.
  Mechanism check: the u block's tail step (median `alpha_u` after a run reaches CR2's end silhouette loss) is ≥ 10×
  R5a's shared tail step (1.5e-4).
  Predictions: the tail shortens (PCB reaches CR2's end thin share in a median ≤ 25 windows, against R5a's 54), thin
  drops by 2–3 points, and PCBU is smoother than PCB.
- **D1, why the line search collapses late (diagnostic, pre-registered 2026-09-29 14:50 CDT).** At 300k the dragon
  stops unfinished (R1d). In r2B_dragon_1 the accepted step fell from 4.6e-3 to 1.2e-6 inside window 20, and the
  fresh 0.02 starts after rejections collapsed again (windows 23–25) until three rejections stopped the run. Ruled
  out: the adaptive-step factor (‖g‖ ≈ 1e-3 against a target norm of 0.039, so the factor is 1). Hypothesis H_u: Adam
  restarts every window (`mom_carry` 0), so its first steps are sign steps. On u this moves every layer particle's
  normal offset by the full step, with the sign of a noisy per-particle gradient, and it roughens the surface once
  the transport gate is open on most of the layer (late: 90 % and more). dFc acts through the grid and is smoothed.
  Run: the 300k dragon, default flags, `--ls_probe` (every failed trial re-run on dFc alone and on u alone), two runs.
  H_u is supported if, in the failed trials of the windows where u_gate > 0.8, the u-only step raises the objective
  while the dFc-only step lowers it, in most of those trials. H_u is refuted if the dFc-only step fails as often, or if
  the state check causes most failures. The fix follows from the mechanism, not from a step constant.
- **D2, the transport value's stopping rule (pre-registered 2026-09-29 15:02 CDT).**
  `scripts/probes/settled/sinkhorn_jump.py` on the 300k dragon's end state (A run 2 and C run 2) evaluates positions
  x + δ·d for δ from 1e-10 to 1e-4 with the solver as it is and with the sweep schedule of the unperturbed solve
  replayed. D2 is supported if the adaptive value changes by a δ-independent amount (≥ 1e-5) wherever the block
  schedule changes, while the replayed-schedule value changes smoothly (∝ δ). D2 is refuted if the adaptive value
  is smooth. If supported, the fix is structural: one transport function per comparison. The line search compares
  the current point, its trials and the commit rollout under the sweep schedule of the current point's solve (the
  gradient's solve, converged to tol). Deterministic schedules are how GeomLoss makes the Sinkhorn divergence a
  smooth function of its inputs (Feydy et al. 2019). No tolerance or budget change.
  **Result (15:05 CDT): not supported at the probed states.** At both end states (A run 2, 43³; C run 2, 85³) no δ
  up to 1e-4 changed the block schedule. The value changes stay at float32 rounding (1e-9), identical for the adaptive
  and replayed schedules. The Sinkhorn stopping rule does not produce the +9.5e-5 offset there.
- **D1b, what carries the two-state transport value (pre-registered 2026-09-29 15:06 CDT).** The probe also records,
  per variant: the transport without the support bound, the support penalty B (mean and largest per particle), and
  the particles flagged decoupled at the last step. Hypothesis H_frag: the rollout's per-step decoupling test
  (`k_frag_step`: no other particle in the 3³ cells around it) flips for a borderline isolated particle. Its bond
  projection then switches on or off, and so does its unbounded support penalty, which the bound E·wB/(E + wB)
  passes into the objective up to a fraction of E. Supported if, in the collapsing windows, the transport without
  support changes smoothly with the step while B's largest per-particle value takes two levels that match the
  offset and the decoupled count. Refuted if the offset sits in the transport without support.
- **Gradient dumps** of a 300k bunny run (`$OUT/viz/`), for a video of the per-particle render and physics gradients
  and the losses over the morph (large loss red, small green).

## To run next, in order

1. **The gallery at 300k.** The 19 meshes at the delivery resolution, with 4K renders. Same measurements as S2.
2. **Render influence across meshes.** Render-off twins (`--render_weight_scale 0`) and a second seed on five meshes
   (bunny, dragon, C, V, nefertiti) at 300k: the render's effect against the seed spread, per mesh.
3. **End-state sharpness (the main open defect).** At 40k settled transport leaves less of the target surface
   uncovered than v3 on every mesh; at 300k it leaves more (bunny 11.7 % against 7.8 %, dragon 34.5 % against 11.7 %;
   the dragon's end state looks melted). Hypothesis R: the transport term's resolution is its loss cell (the MPM
   cell, shape diagonal / 26) and blur (one cell), which do not refine with N; at 300k the particle spacing is about
   a sixth of a cell, so relief finer than a cell is seen only by the 64 px render terms. **R1 (diagnostic,
   pre-registered 2026-09-29 12:58 CDT):** the dragon at 40k, 100k and 300k, seed 97, the adopted recipe. Measure the
   uncovered target surface (farther than 1.5 target spacings) and, for every uncovered target point, the local
   feature thickness (twice the largest inscribed radius of the target within two loss cells of the point). R is
   supported if the uncovered share grows with N and at least 70 % of the uncovered points at 300k sit on features
   thinner than two loss cells; it is refuted if the uncovered share does not grow with N or the uncovered points
   sit on thick body. Then design the fix from the result and the literature survey (4–5 papers; candidates: a blur
   annealed toward the particle spacing at the end, a particle-level transport term on the surface layer). No pins
   or position edits. Pass for a fix: the 300k gap at or below the v3 form on bunny and dragon, 40k gallery within
   ±0.002 silhouette IoU, momentum unchanged.
   **R1b (thin-region census, pre-registered 2026-09-29 13:02 CDT; `scripts/probes/settled/thin_regions.py`, no new
   runs).** The target's local thickness h (max inscribed ball) in loss cells, bins <1, 1–2, 2–4, ≥4; every body
   particle takes the h of its nearest target point. Per bin: uncovered outer target, silhouette holes, sparsity,
   stretch, anisotropy, det F, the per-particle gradients of the transport, local-support, render and cleanup
   terms at the end state, and the support penalty of the body and of the target itself. Archives: 300k dragon
   (settled, v3, new code), 300k bunny (settled mj300, v3 ours300); 40k bunny, dragon, teapot, fandisk (settled, v3).
   Predictions: (P1) settled at 300k leaves ≥ 3× more of the thin bins (< 2 cells) uncovered than of the ≥ 4 bin,
   and more than v3 in the thin bins; (P2) at the 300k end states the support gradient in the thin bins is < 10 % of
   the transport and of the render gradient (the bounded form has switched it off); (P3) the target itself violates
   the support floor (penalty > 0) on ≥ 30 % of its thinnest-bin points. Reading: P1 with P2 → the end-state
   defect is the transport's resolution and a support rewrite alone cannot fix it; P3 → the global floor asks thin
   features to be denser than the target and a target-referenced floor is indicated; P1 without P2 (support active
   in thin bins) → a surface-aware (tangent) support neighbourhood is the first candidate.
   **R1c (render on/off by thickness, pre-registered 2026-09-29 13:13 CDT, at the user's request).** Render-off
   twins (`--render_weight_scale 0`) of the new code: 300k dragon and bunny, 40k bunny, dragon, teapot, fandisk; the
   old code's 300k bunny twin (mj300r0) too; the same census. Prediction: without the render term the uncovered share
   grows mostly in the thin bins (at the end states the render gradient is the largest one there), and little in the
   thickest bin. Added 13:45 CDT, after the on/off census was read and before the spread census ran: the same census on two spread pairs, the old
   code's 300k bunny at seed 98 (against seed 97) and the new code's first 300k dragon run (against the second, same
   seed). The render effect counts only where it exceeds that spread.
   **R1d (pre-registered 2026-09-29 13:58 CDT, census running, not read).** The uncovered share is measured at 1.5
   target spacings, a threshold that halves in world units from 40k to 300k, so "300k leaves more uncovered" may be the
   threshold. The census adds `uncovered_40k`, the same world distance at every N (1.5 spacings of a 40k sampling),
   for the dragon and bunny at 40k and 300k (settled, new code, v3). The v3 form ran on the same loss cell (0.30 wu =
   8.8 spacings at 300k) and still covered the thick body at 300k (1.5 % against 23 %), so the loss cell alone cannot
   be the cause. Predictions: (a) bunny: no inversion at the fixed distance (300k ≤ 40k in every bin); (b) dragon: the
   thick-bin gap stays at 300k (its transport stopped unfinished, E 1.2–2.7e-2 against 2.8e-4 at 40k), so it is a
   stopping problem there, not only resolution. Reading: (a) and (b) → resolution and stopping are separate levers
   and both get tested; an inversion on the bunny too → resolution is the first lever.
4. **Ear tips.** The ear tip holds 11–12 reference particles of the 13 required. Same approach as item 3.
5. **Constants.** `support_weight = 8` and `ot_iters = 1600` are validation values. Either derive them from the
   discretisation or show the result does not depend on them (a factor-of-two sweep each way on three meshes).
6. **Stopping.** After convergence the same rejected step can repeat identically each window until the patience runs
   out (the bunny with a longer budget: windows 50–68). Stop when a rejected step repeats unchanged, and confirm the
   end state is unaffected.
7. **Scaling.** Wall time and memory at 40k, 100k, 300k and 1M particles on one GPU.

## Results so far

**D5, 2026-09-30 10:28 CDT — the dFc block sets the shared step; u is carried along (pre-registered 10:12).** C_R +
`--u_precond --ls_probe`, bunny, cheburashka, homer at 40k (114, 209 and 246 windows). `tmp/d5_eval.py`,
`tmp/d5_parts.py`.
- H_block is supported on all three meshes. In the tail (from windows 13, 15 and 24), the u-only step lowers the
  objective while the dFc-only step raises it in 85, 82 and 82 % of the failed trials (needed ≥ 70 %). The u-only step
  raises it in only 12–16 %. The accepted step is 1.5–1.8e-4 in the tail, and u moves by exactly that (max |Δu| ≈ a):
  0.28 % of a spacing per iteration.
- Most dFc failures are ordinary curvature. Only 2–9 % of the failed trials rise through a jump of the support penalty.
- The discontinuity reading applies in part: the joint change at steps below 1e-5 stays above 1e-7 relative in 90–97
  % of those trials, in two ways:
  - Down to steps of about 5e-7, one particle's support penalty jumps and holds its size whatever the step. On the
    bunny, B's largest per-particle value goes from 0.00098 to 0.00368, or from 0.00578 to 0.00664, while the transport
    without support changes in proportion to the step. The target-referenced floor reads the target density at the
    particle's nearest target point, so it is piecewise constant in position and jumps where the nearest target point
    changes. This is the mechanism of D1b, which the ratio form bounded in size but did not remove.
  - Below that, the changes are evaluation noise of 1–8e-6 relative (render 1e-9, transport 4e-10 absolute). This is
    above the line search's noise floor of 1e-7, so the backtracking keeps halving steps whose decrease it cannot
    resolve. The commit replay's differences in R5 (2e-7 to 6e-6 relative) are of the same size.
- Literature pass (8 primary sources: Tseng & Yun 2009; Wright 2015; Nesterov 2012; Beck & Tetruashvili 2013;
  Richtárik & Takáč 2016; Kerbl et al. 2023; Nicolet et al. 2021; Kingma & Ba 2015):
  - With one adjoint giving both block gradients, the best-supported scheme is a step length per block, each found by
    its own Armijo search from the same point with the same gradient, followed by one Armijo check of the combined
    step. With two coupled blocks, halving the combined step descends (the ω = 2 bound of Richtárik & Takáč, γ ≤ 2 in
    Nesterov). Tseng & Yun give convergence for nonconvex objectives under box constraints.
  - A cyclic update would need a second adjoint per iteration, because the second block's gradient is stale after
    the first block moves.
  - Nicolet et al. use one second-moment scalar per tensor for the preconditioned parameter (their "uniform Adam").
    Per-coordinate normalisation distorts the smooth direction, and R5's `--u_precond` used it.
- Open defects, not fixed yet: the floor's discontinuity (a continuous floor must keep "the target itself never
  pays" at thin tips), and a noise floor set below the measured evaluation noise.

**R5, 2026-09-30 10:12 CDT — the relaxation moved onto u as a preconditioner: both arms fail (pre-registered 00:23,
restarted 02:33).** 40k gallery, seed 97, against C_R.
- R5a (C_R + `--u_precond`): silhouette IoU median +0.0059, higher on 18 of 19 (worst C −0.0012): passes. Thin share
  median 13.6 → 11.8 %, lower on 13 of 19 (median −1.9 points): passes (it is not above C_R + 0.5). Gallery 182
  against 50 minutes: fails (needed ≤ 100). Collapsed windows 139 against D3b's 216: fails (needed ≤ 108).
- R5b (+ `--diag_coverage 1`): thin lower on 12 of 19 (median −0.9 points): fails. Also 141 collapsed windows (needed
  ≤ 81), 175 minutes (≤ 120), and beast frozen at window 10 (0.7313). Fails.
- Predictions: R5a kept D3b's silhouette gain, as predicted, but the collapses did not fall much (139 against 216).
  R5b did not keep D3c's thin gain.
- Beast's freeze now has a record. In windows 10–14 all ten trials failed the state check and none failed the
  merit. The commit rollout of the unchanged start was invalid too (`commit_invalid`, E 0.0955 each time). So the state
  that window 9 committed becomes invalid under any nearby control (inverted, J ≤ 1e-4 within 2T steps, or out of the
  domain). It is not a merit trap. Acceptance checks only the committed window's own 2T steps.
- Post-hoc (`tmp/r5_post.py`, not a verdict):
  - The cost is run length, not the preconditioner. Every arm takes about 5 s per window, but PC runs a median of
    112 windows against C_R's 28.
  - The long runs make steady progress, not churn. Cheburashka under PC lowers its merit by about 1 % per window from
    window 20 to 240. Its d_sil falls from 4.5e-4 to 1.1e-4 at 64 px, and its thin share from 17.2 to 10.9 % by window
    140, then to 8.7 % after the switch to 96 px at window 150. C_R stops at window 39 with d_sil 7.8e-4 and thin
    12.4 %. PC reaches C_R's end silhouette loss in a median 15 windows, and C_R's end thin share in a median 54 (16 of
    19 reach it).
  - The collapsed windows sit late (median at 90 % of the run) and fail on merit only (1799 merit and 0 state
    failures). The largest per-particle support penalty is lower in them than in other windows, so this is not D1b's
    single-particle trap.
  - End jitter (`jitter_rel`) is 25× lower without the forward relaxation (median 2.0e-7 under PC and 1.6e-7 under
    NOREL, against 5.0e-6 under C_R): the relaxation keeps moving the surface at the end. Chamfer 0.1142 against
    0.1157.
  - The coarse-to-fine switch comes at a fixed window, 150 (half the 300-window budget). It never fires under C_R,
    whose runs stop at 15–50 windows. It fired in 6 PC runs.
- Reading: without the forward relaxation, sub-cell shape (thin coverage, outline) keeps improving, but only by about
  1 % per window. The open question is now why the sub-cell modes converge so slowly, not whether the controls can
  reach them (D5).

**D4, 2026-09-30 00:23 CDT — beast's freeze under target-side coverage is intermittent (pre-registered 00:04).**
- Run 2 froze again in both variants: coverage without relaxation at window 11 (0.8719), two-sided support at window
  11 (0.8174). Their probed failed trials before the freeze are ordinary. Small steps change the objective by about
  +1e-7 relative: the kinetic term rises by 1e-5 while transport and render fall.
- The null windows themselves carried no record, and the window log is muted in the runner. They now record
  `null_reason`, the line-search counts, and the commit rollout's E against the accepted E.
- Run 3 (with that record) froze in neither: 0.9688 (96 windows) and 0.9679 (40 windows). So beast freezes in about
  four of six target-coverage runs and in neither C_R run. It is trap timing, as in S3b.
- One null window in run 3 was a commit whose replay differed from the accepted candidate by less than the float32
  rounding of E (both 9.74e-4). It was rejected because the window-start replay noise measured exactly zero, which
  sets the tolerance to 1e-7 relative. That acceptance rule is fragile.

**D3c, 2026-09-30 00:02 CDT — asked for and not smoothed away, u closes thin gaps (pre-registered 2026-09-29 22:26).**
C_R + `--diag_coverage 1 --no_layer_relax` against C_R, 40k gallery. `thin_uncovered` is lower on 14 of 19 meshes
(p = 0.032), median 13.6 → 11.3 %, median change −3.0 points. The largest falls: teapot −6.4, C −6.1, dragon −6.1,
fandisk −5.2, A −5.0, maxplanck −4.5. Silhouette IoU median +0.0065. This meets the registered reading "u can close
thin gaps, and the relaxation is what blocks a particle-scale objective". My prediction (a drop ≤ 2 points) is
refuted. Each change alone gave about −1 point (D3a −1.1, D3b −1.3); the two together give −3.0. Costs: the gallery
takes 170 against 50 minutes (dragon 210 windows, C 195), with up to 22 collapsed windows per run. Beast collapsed
again (null windows from window 13, silIoU 0.8965), as under R4's two-sided support; with coverage alone (D3a) it
did not. Not a recipe: both switches are diagnostics. The design questions are how a particle-scale target coverage
belongs in the objective, and what should replace the relaxation's smoothing so that it stops undoing the sub-cell
shape the objective asks for.

**D3b, 2026-09-29 22:30 CDT — the layer relaxation is not what keeps thin features uncovered, but it costs
silhouette (pre-registered 21:26).** C_R + `--no_layer_relax` against C_R, 40k gallery. `thin_uncovered` is lower on
12 of 19 meshes (p = 0.18; the reading needed 14), median 13.6 → 13.2 %, median change −1.3 points. So the relaxation
is not the limit, as predicted. Side finding: silhouette IoU rises on 18 of 19 meshes (median +0.0069, worst C
−0.0037). The runs go much longer (44–138 windows against 15–50; the gallery takes 155 against 50 minutes) and have
many collapsed windows (2–39 per run; homer 39). The relaxation holds back the render's outline fit and keeps the
line search smooth.

**D3a, 2026-09-29 22:25 CDT — an explicit particle-scale thin-coverage signal helps only a little (pre-registered
21:26).** C_R + `--diag_coverage 1` against C_R (R4 run), 40k gallery. `thin_uncovered` is lower on 16 of 19 meshes
(sign test p = 0.002), and the median falls from 13.6 to 12.0 %. The median per-mesh change is −1.1 points (largest:
teapot −4.2, A −3.4, C −2.8); beast +2.0 and bob +1.7 rise. Silhouette IoU median change 0.0000 (worst −0.0023,
teapot +0.0035); no collapse; the gallery takes 65 against 50 minutes. The result lies between the two registered
readings: consistent (≥ 14 of 19), but the median drop is 1.1 points, not ≥ 3. Late in a run the coverage term is
about twenty times the transport energy, yet about nine tenths of the thin gap remains. So the controls or the
dynamics, not only the objective, limit thin features: dFc acts through the grid (one MPM cell is 4.4 spacings
here), and u's per-particle normal offsets are what the layer relaxation removes (D3b).

**R4 stage 1, 2026-09-29 21:15 CDT — the two-sided support averaged over all target points: fails.** 40k gallery,
C_R against C_R + `--support_two_sided` (target side averaged over every target point).
- Primary: `thin_uncovered` lower on 11 of 19 meshes (sign test p = 0.32; the pass needed 14). Median 13.6 % in both
  arms (A, the log-form baseline: 14.3 %); median difference −1.0 point.
- Secondary: silhouette IoU median +0.0004, but beast collapsed. Its step fell to 6e-6 at window 6, then five null
  windows froze it at window 13 (0.8192, thin 54.6 %). C_R's beast: 0.9643. Collapsed windows elsewhere: 0–2 per run
  in both arms. Wall time 74 against 50 minutes for the gallery.
- Why it does little (from the recorded windows): the support is not switched off late (w_eff median 2.1 under C_R
  and 2.8 two-sided, range 1.2–7.1). The target side's share is diluted: at the end its B is 5.5e-6 against the body
  side's 1.9e-5. The mean runs over all target points (40k, mostly interior), and the uncovered thin surface points
  are under 1 % of them, so they barely move the mean. Averaging over the volume re-introduces the mass weighting that
  hides thin features. Surface measures (Chamfer on surface samples, varifolds) average over the surface.
- Beast's collapse is one case; it was not probed.

**T2, 2026-09-29 20:05 CDT — the training render does not see the thin gaps (pre-registered 19:58).**
`scripts/probes/settled/thin_visibility.py`; uncovered outer target points of the thin bins, classified per point.

| uncovered thin points | seen at 64/96 px | resolution-limited | occluded (front-visible) | silhouette-ambiguous |
|---|---|---|---|---|
| 300k dragon A_R, < 1 / 1–2 cells | 0.2 / 0.0 % | 32 / 11 % | 18 / 24 % (all) | 50 / 65 % |
| 300k dragon settled (S2), < 1 / 1–2 | 6.6 / 3.5 % | 26 / 18 % | 18 / 20 % (all) | 49 / 59 % |
| 40k dragon A, < 1 / 1–2 | 18 / 3.5 % | 28 / 44 % | 1.2 / 1.4 % | 53 / 51 % |
| 40k bunny A, < 1 / 1–2 | 0 / 9 % | 47 / 33 % | 0 / 1.5 % | 53 / 57 % |

- At the training resolutions the silhouette deficit at these points is nil: its median is 0.00 at 64 and 96 px on
  the 300k dragon (a 64 px pixel is 5.3 target spacings there, 2.2–2.6 at 40k), against 0.27 / 0.03 at 256 px on the
  dragon and 0.58–0.63 at 256 px at 40k. The render term does not see the gaps it would have to fill.
- About a third become visible at 256 px (resolution-limited). A render resolution that follows the particle spacing
  would reach these.
- Half or more stay invisible to any silhouette. Either another depth of the body fills the pixel in every view
  (silhouette-ambiguous: the outline is right, the 3D surface is not), or the point is hidden behind the target's
  own material in every view (occluded, 18–24 % at 300k and about 1 % at 40k; all of them are front-visible, so the
  shading term could see them).
- Prediction half right: the resolution case is a third, not most. The largest class everywhere is
  silhouette-ambiguous, and occlusion matters at 300k, not at 40k.
- How far the gaps are (20:10 CDT): the uncovered outer points lie a median 1.6–2.0 target spacings from the nearest
  body particle, just beyond the 1.5 threshold. p90 is 2.0–2.6 spacings at 40k and 3.1–3.5 on the 300k A_R dragon
  (4.5–5.0 on the settled baseline); p99 is 3.3–5.6. The same holds in the thick bin. The support estimator's kernel
  (h = r8 / 2, r8 = 1.97 spacings) weighs the median gap at 0.12–0.26 and the 300k p90 at 0.002–0.008. The missing
  signal covers the last one to three spacings, the scale of the u channel's per-window step (one spacing).
- Reading: no term sees an empty thin region in 3D at the particle scale. Transport sees it only blurred and
  mass-weighted, and the silhouette sees it only where the outline changes. A target-to-body surface term at the
  particle scale is indicated, designed after a literature pass. A render resolution following N covers about a
  third.

**V1, 2026-09-29 19:45 CDT — C_R's late reversals are not visible (pre-registered 19:35).** C_R run 3 (archives
kept) against A's first runs, 40k.
- `window_reversal.py`: the second-half reversing windows move the outer layer by a median 0.037–0.053 target
  spacing per window. That is 0.03–0.21× the first half's median step, in both arms: A dragon 0.21×, A cow 0.08×,
  C_R dragon 0.15×, spot 0.07×, cow 0.03×. The reversal share itself varies strongly between runs of one arm: C_R
  armadilo 72 %, 32 %, 0 %; C_R run 3 spot 79 %, dragon 48 %, cow 33 %, teapot and armadilo 0 %.
- `video_flicker.py` on the quick two-view splat videos, last 20 %: ALT 0.0003–0.0005 and ALT/DRIFT 1.07–1.20 in
  every run. A spot, with no reversals, has 1.20, so about 1.1 is the renderer's floor. C_R's tail ALT is 0.0001 above
  A's on spot and dragon and equal on cow. The first-difference D1 of 0.0003–0.0006 matches B1's settled tail (0.0003)
  and lies below v3's (0.0008).
- Verdict as registered: not visible. The reversing windows are far below half the earlier step. The prediction
  "well under a tenth" holds for three of the five runs (0.15× and 0.21× on the two dragons).
- Consequence: the S2 reversal gate counts sign flips with no size threshold. At this scale it measures near-converged
  jitter of about 1/20 spacing per window and does not separate the arms. Any future gate needs a size criterion
  derived before it is used.

**R3d (2), 2026-09-29 19:30 CDT — render influence under C_R (pre-registered 18:50): the render keeps its effect.**
The 300k dragon with C_R and `--render_weight_scale 0`, against the two C_R runs.

| 300k dragon | silIoU | chamfer | own threshold | fixed world distance | end E | windows / wall |
|---|---|---|---|---|---|---|
| C_R (render on, two runs) | 0.9833–0.9838 | 0.0595 | 34–35 / 21–22 / 15 / 7.4–8.2 % | 2.5 / 0.3–0.5 / 0.2–0.3 / 0.0 % | 5.6–6.2e-4 | 167–173 / 80 min |
| C_R, render off | 0.9733 | 0.0596 | 48 / 31 / 26 / 11 % | 4.8 / 1.9 / 1.0 / 0.2 % | 2.8e-4 | 74 / 34 min |

- The render adds +0.010 silhouette IoU (20× the C_R spread of 0.0005) and covers 13–14 / 10 / 11 / 3–4 more points
  of the bins at the own threshold, the most in the thin bins. It leaves the chamfer unchanged and makes the end
  transport less finished (E 5.6–6.2e-4 against 2.8e-4 without it): it trades transport for the silhouette.
- λ = 0.396 (calibrated at window 1); the render's share of the update is 0.33 at window 1 and 0.91–0.97 from mid-run
  on. The render-off run stops at 74 windows, so the second half of a C_R run (windows 80–175, about 45 minutes) is
  render-driven polishing. This is also the phase in which C_R's late reversals occur (R3d (1)).
- det F min 0.73 with the render off against 0.83–0.86 with it; anisotropy p90 1.16–1.24 against 1.13–1.17.

**R3d (1), 2026-09-29 19:25 CDT — run-to-run spread at 40k (follow-up pre-registered 18:50): passes.** Second runs of
A (defaults) and C_R on the 19-mesh gallery, seed 97.
- Median surface gap: A 7.1 / 7.6 % (its own run-to-run difference 0.5 point), C_R 7.2 / 7.1 %. The two-run means
  are C_R 7.15 % against A 7.35 %, so the stage-2 miss of 0.1 point was noise. Mean gap: C_R 7.92 / 7.99 %, A 8.15 /
  8.62 %.
- No mesh has both C_R runs below both A runs by more than 0.002. Dragon is the only mesh with both below, by at
  most 0.0006.
- Both C_R runs lie above both A runs on 12 meshes: A, armadilo, beast, bimba, bob, C, cow, fandisk, heart,
  maxplanck, spot, teapot. Of the four above the band, beast, C and teapot are confirmed; cheburashka is not (its
  second C_R run is 0.0001 below A's second).
- Not in the criteria, but an S2 gate: second-half reversals fail in both arms. A run 2: cow 89 %, spot 64 %,
  dragon 47 %. C_R run 2: dragon 83 %, cow 67 %, teapot 67 %, spot 58 %, armadilo 32 %, beast 23 %. C_R fails on
  more meshes. The gate counts sign flips of the outer layer's window-to-window motion with no size threshold, so
  the longer, near-converged tails of C_R's runs add flips (armadilo: reversal cos within ±0.17 while the merit still
  falls). Open: whether these late flips are a visible oscillation or motion below noise.

**R3 stage 2, 300k dragon, 2026-09-29 18:43 CDT — C_R passes P1 and P2.** C_R = ratio form + target floor +
`--loss_follows_n` (85³ loss grid), twice. No collapsed window; 179 and 175 windows (82 and 79 minutes). The largest
per-particle penalty stays at radius² (4.63e-3) throughout. E is 1.4–1.5e-2 at window 20, 3.7–3.8e-3 at 50,
1.3–1.4e-3 at 100, and 5.5–6.1e-4 at the end.

| 300k dragon | own threshold | fixed world distance | aniso p90 | silIoU / chamfer / det F min |
|---|---|---|---|---|
| A | 51–54 / 40–42 / 38–42 / 24–27 % | 19–24 / 14–18 / 12–20 / 8.6–12.5 % | 1.11–1.15 | 0.9686–0.9775 / 0.066–0.074 / 0.82–0.88 |
| A_R | 41–44 / 29–30 / 24 / 16 % | 8.4–8.5 / 3.4–3.9 / 3.2–3.8 / 1.9 % | 1.15–1.21 | 0.9814–0.9834 / 0.062 / 0.79 |
| C_R | 34–35 / 21–22 / 15 / 7.4–8.2 % | 2.5–2.6 / 0.3–0.5 / 0.2–0.3 / 0.0 % | 1.13–1.17 | 0.9833–0.9838 / 0.0595 / 0.83–0.86 |
| v3 (S2) | 33 / 16 / 7.7 / 1.5 % | 3.8 / 0.1 / 0 / 0 % | 2.3–4.1 | 0.9685 / 0.0569 / 0.16 |

At the fixed world distance C_R now covers the dragon as well as v3 does, with anisotropy near 1.15 instead of
2.3–4.1. At its own threshold v3 still covers the thicker bins better (by stretching). The finer loss grid adds to
the ratio form on the dragon (A_R → C_R: −7 to −9 points per bin at the own threshold). Cost: 80 minutes per run,
against 7–9 for A (which stopped at 30 windows) and 41.5 for v3.
Stage 2 verdict as registered: the dragon criteria pass; the 40k gallery misses its median-gap criterion by 0.1
point, and four meshes lie above the ±0.002 band. Not adopted yet. Render influence under C_R: twin not yet run.

**R3c, 2026-09-29 18:33 CDT — the ratio form alone at 40k (pre-registered 17:56): prediction confirmed, the target
floor is needed.** A_R (ratio form, global floor) on the 40k gallery against C_R (ratio form, target floor):
silhouette IoU below C_R on 13 meshes and above on 6 (largest: C −0.0024, dragon +0.0023). Median surface gap
7.6 % against C_R's 7.2 % and A's 7.1 % (mean 8.35 against 7.92 and 8.15). The thin-feature letters lose most:
C 15.5 % (C_R 12.0, A 12.4) and V 11.4 % (C_R 9.6, A 7.6). Reversals are lower under A_R (armadilo 16 %,
maxplanck 0 %; dragon 46 % as under A). With the support gradient 1.7–3× stronger through the bound, the global floor
tightens the features the target itself samples below it; the target floor avoids that.

**R3 stage 2, 40k gallery, 2026-09-29 17:55 CDT (pre-registered 16:23).** C_R (ratio form, target floor; at 40k the
loss grid is the MPM grid) against A, 19 meshes, seed 97.
- Silhouette IoU up on 16 meshes. The three down stay within −0.0007 (dragon −0.0007, V −0.0004, nefertiti −0.0002).
  Four lie above the ±0.002 band: beast +0.0067, C +0.0027, cheburashka +0.0021, teapot +0.0021. No mesh lies
  below it.
- Nefertiti does not collapse (0.9754, 40 windows; C in R2b: 0.803 at window 9).
- Surface gap: median 7.1 → 7.2 %. The pre-registered "≤ 7.1 %" **fails by 0.1 point**. The mean falls from 8.15 to
  7.92 %: 12 meshes better (beast 15.6 → 12.1, homer 10.0 → 8.9, armadilo 10.0 → 9.0) and 6 worse (V 7.6 → 9.6,
  nefertiti 5.6 → 6.3, A 6.4 → 7.2).
- S2 gates (not in the R3 criteria): second-half reversals fail on armadilo (72 %, streak 16, 52 windows) and
  maxplanck (33 %). Cow (52 → 16 %) and dragon (48 → 11 %), which failed under A, now pass. COM drift on C is 0.12
  spacing (A 0.05).
- Stage 2 as registered is not passed. The mesh gains and the collapse fix hold; the median gap misses by 0.1 point,
  and two meshes oscillate late.

**R3 stage 1, 2026-09-29 17:19 CDT — the missing-mass support removes the trap (pre-registered 16:23): passes.**
A_R (A + `--support_form ratio`) on the 300k dragon, twice. Neither run has a collapsed window: no accepted step
below 1e-6 after window 3, and no stop from collapsed rejections. The runs go on for 214 and 203 windows (about 50
minutes each) and end by the merit plateau and by three rejections of ordinary steps. The largest per-particle
penalty is exactly radius² (4.63e-3).
- P1: in the failed trials the objective change shrinks with the step. Run 2: 2.5e-2 at steps of 1e-3–1e-2, 2.7e-3
  at 1e-4–1e-3, 1.7e-5 at 1e-5–1e-4, 9e-7 below 1e-5, with B changing by 1e-8. Run 1 had no failed trial below 1e-4.
- P2: the transport keeps falling after window 20: E 3.4–3.6e-2 at window 20, 1.0–1.2e-2 at 50, 3.9–4.4e-3 at 100,
  and 5.4–8.3e-4 at the end (A: 1.2–2.7e-2 at its stop).

| 300k dragon | own threshold | fixed world distance | end E | aniso p90 | silIoU / chamfer |
|---|---|---|---|---|---|
| A (runs 1, 2) | 51–54 / 40–42 / 38–42 / 24–27 % | 19–24 / 14–18 / 12–20 / 8.6–12.5 % | 1.6–2.7e-2 | 1.11–1.15 | 0.9686–0.9775 / 0.066–0.074 |
| C (R2) | 43 / 28–29 / 25–27 / 14–16 % | 9.7 / 4.6–4.7 / 5.0–5.8 / 2.9–3.0 % | 5–6e-3 | 1.11–1.14 | 0.9794–0.9814 / 0.062 |
| A_R | 41–44 / 29–30 / 24 / 16 % | 8.4–8.5 / 3.4–3.9 / 3.2–3.8 / 1.9 % | 5.4–8.3e-4 | 1.15–1.21 | 0.9814–0.9834 / 0.062 |

The ≥ 4-cell bin at the fixed distance (1.9 %) is now below the 40k dragon's (7.2 %). The loss grid is unchanged, so
most of the 300k dragon's gap was the trap. Anisotropy in the thinnest bin rises to 1.20–1.21; det F min 0.785–0.791
(A 0.824–0.884). C_R on the 300k bunny: silIoU 0.9858, chamfer 0.0580, 32 windows, own-threshold 25.5 / 19 / 11 /
3.3 % (A 34 / 25.5 / 16 / 4.4 %, C 28 / 21 / 11 / 3.1 %). Cost: 200 windows at 300k take about 50 minutes, and
the merit was still falling slowly at the end (about halving every 40 windows).

**SF, 2026-09-29 15:50 CDT — the log-form support against the bounded ratio form at end states (read-only,
`scripts/probes/settled/support_forms.py`).** Per particle r = s / f; log form [−log r]₊², ratio form [1 − r]₊²,
both times the kernel radius squared.
- Tail: on the 300k dragon (A, C, D1b), particles with r < 0.1 hold 44–65 % of the log-form B, and the top particle
  alone holds 16–42 %. They lie 1.5–13.5 target spacings from the target, and their W1 isolation-gate weight is 1.00.
  The 300k bunny and the 40k bunny and dragon have no tail (r < 0.1 holds 0–3 %).
- Bulk (0.3 ≤ r < 1): the ratio form's penalty is 0.55–0.64 of the log form's, and its dL/dr 0.70–0.83. The two
  agree only as r → 1.
- Outer coupling: B shrinks 2–5× under the ratio form, so the support-gradient weight w(E/(E+wB))² that every
  particle feels rises: 300k dragon 7.3 → 7.8 (A) and 6.2 → 7.6 (C); 300k bunny 3.8 → 5.1; 40k dragon 0.07 → 0.25;
  40k bunny 0.30 → 0.68 (global floor). Net at 40k: a deficient particle's support gradient becomes about 1.7–3×
  stronger. The change is not confined to the tail.

**D1b, 2026-09-29 15:35 CDT — what carries the two-state value (pre-registered 15:06): the support penalty of one
isolated particle.** In the collapsing windows of the 300k dragon (29, 31, 33 of run 2), the transport without the
support bound changes smoothly with the step and decreases (−7e-5 → −2e-9 as the step falls from 2e-3 to 6e-6).
The support penalty B carries the step-independent offset: its mean rises by +3.5e-5 to +6.5e-5 at every step size.
B is dominated by one particle: the largest per-particle penalty is 94–161 against a mean of 8–11e-4 over 300k
particles, so that one particle holds about half of B. With the dFc step its penalty jumps from 113 to about 143,
at any step size; with the u step alone it stays near 113. The decoupled count at the last step is 0–2 and does
not follow the offset, so H_frag as posed (the final decoupling flag) is not supported. The particle's end position
still reacts discontinuously to the control. The penalty is quadratic in its log-density deficit, which grows as the
particle's distance to its neighbours squared, so a tiny position difference of one isolated particle becomes about
1 % of the objective late in the run. Windows 1–3 fail ordinarily (B's largest value 0.02–0.13, smooth).
Replicate (run 1, 15:38 CDT; silIoU 0.9772, 28 windows): the last three windows (26–28) collapse to steps of 5e-9 to
2e-8 and are rejected. In 40 small-step failed trials the offset is in B (median +9.0e-5, +1.3–1.5 % of the
objective), while the transport without support moves by 3e-8. The largest particle holds a median 63 % of B. One
decoupled particle is present throughout. The u-only step lowers the objective by 5e-6 relative in the same trials,
so the discontinuity enters with dFc.

**T1, 2026-09-29 15:30 CDT — thin features over time (`scripts/probes/settled/thin_time.py`, no new runs).** Per
window, uncovered outer target by thickness bin at the driven end and at the released end.
- Of the points uncovered at the end, 80–95 % (thin bins) were covered at some earlier released end.
- At 300k each release uncovers 9.5–12 % of the thinnest bin's points covered at the driven end (40k: 2.8–4.5 %),
  and covers about as many others. The window averages at the two ends agree within a point. The thin surface
  reshuffles at the particle scale from window to window without a net loss, so the "reached earlier" share is
  inflated by this churn.
- The mid-run setback (300k dragon A, windows 8–12) hits every bin together, the thick one included (20 → 37 %). It
  is a global redistribution, not a thin-feature loss.
- The thin bins then plateau: the 300k dragon from about window 20 (A 51 %, C 42 %), when its windows start to
  collapse (D1b), and the 40k dragon from about window 20 at 28–30 % over 52 windows.

**R2b, 2026-09-29 15:33 CDT — arm C before adoption (pre-registered 14:55): not adopted.**
- (1) 40k gallery (only the support floor changes; loss_res = grid confirmed): 15 meshes within ±0.002 of A,
  three above the band (beast +0.0038, C +0.0045, homer +0.0024), and nefertiti collapsed. Its step fell to 6e-6 at
  window 5, then five null windows froze it mid-morph at window 9 (silIoU 0.803, surface gap 42.5 %): the same trap.
  The median surface gap rose from 7.1 to 7.8 % (worse on cow, V and A; better on armadilo, beast, C, homer,
  maxplanck, spot and teapot). Fails as registered.
- (2) Render twin of C on the 300k dragon: silIoU 0.9814 → 0.9632 and surface gap 25.2 → 32.7 % with the render
  off, against a C spread of 0.002 and 1 point. The render effect stands under C.
- (3) 300k: nefertiti improves under C (gap 20.0 → 14.7 %, silIoU +0.0040); V does not (gap 14.2 → 15.6 %,
  silIoU +0.0013). Both V runs fail the reversal gate (36 %, 33 %). Fails as registered.

**D1, 2026-09-29 15:00 CDT — why the line search collapses late (pre-registered 14:50): H_u refuted; a
discontinuous transport value found.** Two 300k dragon runs with `--ls_probe`; the state check never failed.
- In 87–96 % of failed trials the dFc-only step raises the objective. The u-only step raises it in 2–16 % and
  mostly lowers it. u is not the cause.
- Ordinary failed trials (one to three per window) raise the end kinetic term (+0.03 to +0.16). This is normal
  backtracking.
- The collapsing windows (22 and 26–28 of run 2, 22 of run 1; 13–31 failed trials each) show something else. As the
  step shrinks to 1e-8 the objective change does not shrink: it settles at a constant +0.97 % / +1.2 % of the
  objective, all of it in the transport term (+8.0e-5 / +9.5e-5). The kinetic and render changes shrink with the
  step. The u-only variant usually reproduces the current value (±1e-5 relative), but in a few trials it shows the
  same +9.5e-5 offset with no change to dFc. So the transport value has two states near the current point,
  independent of the control.
- Candidate mechanism (D2): each Sinkhorn blur level stops at the first 4-sweep block whose marginal error is below
  tol. A tiny input change can move that stop by one block and shift the value by the truncation error. Late in the
  run that shift is about 1 % of the objective, far above the Armijo demand, so the search halves until the step
  means nothing, and the run stops.

**R2, 2026-09-29 14:52 CDT — loss resolution and support floor at 300k (pre-registered 14:05).** New code, seed 97.
A = both off (the S3/S3b runs), B = `--support_target_ref`, C0 = `--loss_follows_n` (loss grid 43³ → 85³ on the
dragon), C = both. Uncovered outer target at the run's own threshold, bins < 1 / 1–2 / 2–4 / ≥ 4 MPM cells:

| | dragon (two runs per arm) | bunny |
|---|---|---|
| A | 51–54 / 40–42 / 38–42 / 24–27 % | 34 / 26 / 16 / 4.4 % |
| B | 49–52 / 36–41 / 34–42 / 21–28 % | 28 / 20 / 13 / 3.7 % |
| C0 | 42–44 / 29–32 / 25–29 / 14–16 % | 28 / 22 / 13 / 3.3 % |
| C | 43 / 28–29 / 25–27 / 14–16 % | 28 / 21 / 11 / 3.1 % |

| dragon | silIoU | chamfer | det F min | windows | 2nd-half reversals | COM drift | surface gap (S2 metric) |
|---|---|---|---|---|---|---|---|
| A (5 runs; census 2) | 0.9686–0.9775 | 0.0663–0.0744 | 0.824–0.884 | 26–36 | 0–7 % | 0.02–0.03 | 36–39 % |
| B | 0.9705, 0.9777 | 0.0717, 0.0653 | 0.841, 0.790 | 21, 35 | 0 %, **29 %** | 0.03–0.04 | 33–39 % |
| C0 | 0.9770, 0.9777 | 0.0620, 0.0624 | 0.881, 0.888 | 29, 23 | **43 %**, 0 % | 0.03 | 25–28 % |
| C | 0.9794, 0.9814 | 0.0616, 0.0619 | 0.884, 0.894 | 27, 32 | 0 %, 0 % | 0.03 | 25–26 % |

Bunny silIoU / chamfer / gap: A 0.9840 / 0.0582 / 12.9 %, B 0.9852 / 0.0581 / 10.3 %, C0 0.9852 / 0.0579 / 10.5 %,
C 0.9845 / 0.0581 / 9.9 %; reversals 0 % and COM drift ≤ 0.01 in all four. Anisotropy p90 1.05–1.16 in every run and
bin. At one world distance the dragon's C runs leave 9.7 / 4.7 / 5.0–5.8 / 2.9–3.0 % uncovered, against A's 19–24 /
14–18 / 12–20 / 8.6–12.5 %.
- C0 lowers the dragon's uncovered share in all four bins beyond A's spread (prediction was "little change unless the
  run finishes"; refuted: the finer transport helps even though the runs still stop at 23–34 windows). Bunny: −6 /
  −4 / −3.4 / −1.1 points (the new code's bunny spread is one run; the old code's seed spread is ≤ 2.5 points).
  But one C0 dragon run reverses 43 % of its second-half windows: C0 alone fails the reversal gate.
- B alone does not beat A's spread on the dragon (fails); on the bunny it matches C0.
- C passes every pre-registered 300k criterion on both meshes: four of four bins beyond A's spread on the dragon
  (and beyond the old code's seed spread on the bunny), anisotropy ≤ 1.2, det F min at or above A's, no reversals,
  COM drift 0.03, silhouette IoU above A's range. It closes about a third of the dragon's gap to v3 (surface gap 38
  → 25 %, v3 11.7 %) and part of the bunny's (12.9 → 9.9 %, v3 7.8 %).
- Cost: the finer transport grid doubles the dragon's time per window (about 7.5 → 15 min per run).
- λ is recalibrated on the first window's physics gradient: 0.289 → 0.396 on the dragon, 0.165 → 0.248 on the
  bunny under the finer grid (render share of the last update 0.48–0.95, A 0.36–0.76). Render-off twin of C pending.
- Not adopted yet: C changes the support floor at every N, so the 40k gallery must hold (R2b), and the render
  influence under C needs its twin.
- Stopping control L (the dragon at D/26 with `--reject_stop 20 --patience 20`): no gain. From window 20 on, every
  window's accepted step collapses to 1e-8–1e-7 and the window is rejected, twenty times in a row, and the run ends at
  window 19's state (silIoU 0.9731, chamfer 0.0730). The dragon's unfinished transport is therefore not a budget
  matter: the run reaches states from which the line search cannot descend (D1, D1b).

**S3 / S3b, 2026-09-29 14:16 CDT — GPU-only refactor equivalence: FAILED at 300k.** 40k gallery (19 meshes, one new
run each against S2): new − old silhouette IoU median −0.0007, seven meshes up and twelve down. Sixteen meshes lie
within ±0.002 of S2. Beast lies within the old code's nine-run spread (0.7852–0.965; one old run froze mid-morph
after five null windows). Homer's first run is 0.0014 below the old range and V's 0.0008 above it; their second runs
are inside. 300k dragon (S3b): old six runs 0.9763–0.9787 (median 0.9775), new five runs 0.9686–0.9775 (median
0.9761). New lies below old in 26 of 30 pairs, one-sided p = 0.026. Chamfer: 4 of 5 new runs above every old run
(0.0663–0.0744 against 0.0647–0.0665). The pre-registered pass (new median inside the old range, p ≥ 0.05) fails.
The first window agrees to 8e-7 and the line-search step logic is identical. The difference is late: accepted steps
below 1e-5 make up 8.6 % of new windows 16+ against 3.1 % of old ones, while windows 1–15 are alike. The late-only
logic (rollback, anneal, null and reject handling, bonds, layer data) is being compared line by line. The refactor
is not committed until the cause is found. Correction to the paired 15-window beast runs: `--animations 15` was
overridden by the recipe's `--animations 300` in the old runs (they ran to their own stop), and in the new runs it
moved the render coarse-to-fine switch to window 7. Only their windows 1–7 are comparable, and those overlap.

**R1d, 2026-09-29 14:02 CDT — the 300k "inversion" at one world distance (pre-registered 13:58).** Uncovered outer
target by thickness bin (< 1, 1–2, 2–4, ≥ 4 cells), at the run's own threshold (1.5 of its target spacings) and at
one world distance for every N (1.5 spacings of a 40k sampling):

| | own threshold | fixed world distance | E at the end | aniso p90 |
|---|---|---|---|---|
| bunny 40k, settled | 19 / 12 / 8.5 / 2.9 % | 19 / 12 / 8.5 / 2.9 % | 1.2e-4 | 1.05–1.09 |
| bunny 300k, settled | 31 / 23 / 15 / 4.0 % | 4.5 / 1.2 / 0.3 / 0.1 % | 1.1e-4 | 1.06–1.10 |
| bunny 300k, v3 | 33 / 13 / 10 / 1.4 % | 0.9 / 0.1 / 0 / 0 % | 2.1e-3 | 1.29–1.42 |
| dragon 40k, settled | 27 / 15 / 12 / 7.2 % | 27 / 15 / 12 / 7.2 % | 2.8e-4 | 1.08–1.11 |
| dragon 300k, settled / new ×2 | 51–54 / 37–42 / 36–42 / 23–27 % | 18–24 / 12–18 / 11–20 / 7–13 % | 1.2–2.7e-2 | 1.11–1.16 |
| dragon 300k, v3 | 33 / 16 / 7.7 / 1.5 % | 3.8 / 0.1 / 0 / 0 % | 8.6e-3 | 2.3–4.1 |

- (a) Confirmed: the bunny has no inversion. At one world distance the 300k body covers 4–30× more than the 40k
  one. The "300k leaves more uncovered" of S2 and R1b is the threshold, which halves in world units.
- (b) Confirmed: the dragon at 300k is no better than at 40k at one world distance, and its transport ends 50–100×
  less finished (E). The dragon's 300k gap is a stopping problem first.
- What remains real: at its own threshold settled does not reach sub-spacing precision at 300k (bunny 1–2, 2–4, ≥ 4
  bins 23 / 15 / 4.0 % against v3's 13 / 10 / 1.4 %). v3 gets there on the same loss cell, by stretching
  (anisotropy p90 up to 4.1).

**R1c, 2026-09-29 13:55 CDT — render on/off by thickness (pre-registered 13:13; spread pairs 13:45).** New code, seed
97; render-off = `--render_weight_scale 0`. Uncovered outer target by thickness bin (< 1, 1–2, 2–4, ≥ 4 loss cells):

| | render on | render off | spread (same code, second run) |
|---|---|---|---|
| 300k dragon | 51 / 40 / 38 / 24 % | 74 / 57 / 45 / 27 % | 54 / 41 / 42 / 27 % (same seed) |
| 300k bunny | 34 / 26 / 16 / 4.4 % | 62 / 43 / 32 / 9.4 % | — |
| 300k bunny, old code | 31 / 23 / 15 / 4.0 % | 61 / 42 / 31 / 9.4 % | 31 / 23 / 17 / 3.8 % (seed 98) |
| 40k bunny | 18 / 12.5 / 9.3 / 3.2 % | 36 / 21 / 17 / 5.2 % | |
| 40k dragon | 28 / 14 / 9.1 / 6.9 % | 60 / 30 / 18 / 12 % | |
| 40k teapot | 27 / 25 / 6.2 / 4.3 % | 49 / 33 / 13 / 3.3 % | |
| 40k fandisk | 29 / 8.7 / 7.7 / 3.3 % | 59 / 24 / 9.6 / 5.4 % | |

- Prediction confirmed: without the render term the uncovered share grows most in the thinnest bin (+18 to +32
  points) and least in the thickest (−1 to +5 points; the 300k dragon's +3 is inside its run-to-run spread). The
  render effect exceeds both spread pairs (≤ 4 points) in every bin below 4 cells.
- Whole shape: silhouette IoU +0.007 to +0.040 with the render (dragon 300k 0.936 → 0.976, bunny 300k 0.968 → 0.984),
  chamfer unchanged (±0.001), the census's per-bin silhouette holes 2–4× lower. The body particles off the target (`out_nn_frac`) rise with the
  render on six of seven pairs (300k dragon 0.58 → 1.30 %, 40k fandisk 0.02 → 0.15 %); all stay below 1.3 %.
- Reading: the render term is what covers thin features at the end state; the transport term alone leaves 60–74 % of
  the thinnest bin uncovered. The render reaches only what the 64–96 px views resolve, so the remaining thin-bin gap
  (31–51 % at 300k) is the part that neither term resolves.

**R1b, 2026-09-29 13:20 CDT — end-state defects by the target's local thickness (pre-registered 13:02).** The first
census run capped the thickness at a fixed number of voxels (the jittered sampling left enclosed empty voxels; fixed
with a 3-D hole fill, the median dragon thickness is now 1.40 wu at 40k and 1.45 wu at 300k); the numbers below are
from the corrected run. Uncovered outer target (farther than 1.5 target spacings), by thickness in loss cells
(< 1, 1–2, 2–4, ≥ 4):

| | settled | v3 |
|---|---|---|
| 300k dragon | 51 / 37 / 36 / 23 % | 33 / 16 / 8 / 1.5 % |
| 300k bunny | 31 / 23 / 15 / 4.0 % | 33 / 13 / 10 / 1.4 % |
| 40k bunny | 19 / 12 / 8.5 / 2.9 % | 38 / 20 / 10 / 5.0 % |
| 40k dragon | 27 / 15 / 12 / 7.2 % | 30 / 21 / 14 / 5.9 % |

- (P1) Partly: thin features are the worst everywhere, but at 300k settled also leaves the thickest regions
  uncovered (dragon 23 % against 1.5 %): the imprecision at 300k is sub-cell and general, not only thin.
- (P2) Refuted: the support term is not switched off at 300k. The dragon ends with a large transport residual
  (E 1.2e-2, the bunny 1e-4), so the support weight w(E/(E+wB))² is 6.7 of 8, and its thin-bin gradient is about
  twice the transport's. At 40k it is about a tenth of the transport's. In the thin bins the render gradient is the
  largest at both N (4–20× the transport's).
- (P3) Confirmed: the target itself violates the support floor on 62–85 % of its thinnest-bin points at 40k and on
  40 % at 300k (1–2 % in the thickest bin): the global floor asks thin features to be denser than the target.
- v3 reaches its coverage by stretching material: the anisotropy of F (p90 of the singular-value ratio) reaches
  2.3–4.1 in its thin bins; settled stays at 1.05–1.16.
Next: a target-referenced support floor (each particle against the target's own density at its nearest target
point; no new constant) for the thin-feature part, and a loss resolution that follows the particle spacing for the
general 300k imprecision (literature: the blur is the sampling resolution; Feydy 2019, RobOT 2021). The surface
tangent neighbourhood addresses opposite-sheet counting, which the kernel scale (1–2 spacings) does not reach here.

**S2, 2026-09-29 12:40 CDT — adoption gates: settled transport against the v3 forms, seed 97.** All 19 meshes at 40k
and the 300k dragon pass the pre-registered gates (silhouette IoU, second-half reversals, centre-of-mass drift).

| 40k (19 meshes) | settled | v3 |
|---|---|---|
| silhouette IoU, median (range) | 0.9742 (0.9648–0.9805) | 0.9732 (0.9550–0.9801) |
| IoU difference settled − v3 | median +0.0024; below zero on A, heart, maxplanck, V (worst −0.0033, maxplanck) | |
| det F min, median | 0.941 | 0.804 |
| wall, median | 1.6 min | 3.8 min |
| target surface beyond 1.5 spacings, median | 7.2 % (lower on every mesh) | 11.0 % |
| second-half reversals | 0 % on 16 meshes, at most 11 % | up to 62 % (A, maxplanck) |
| centre-of-mass drift | ≤ 0.04 spacing | 0.09–2.03 spacing |

300k dragon: silhouette IoU 0.9769 against 0.9685, det F min 0.829 against 0.160, 7.7 min against 41.5 min,
centre-of-mass drift 0.03 against 0.99 spacing. **But** its end surface is coarser: 34.5 % of the target surface lies
farther than 1.5 spacings from the body (v3 11.7 %) and the chamfer is 0.0664 against 0.0569; the run stopped after
29 windows. At 40k settled transport leaves less of the surface uncovered on every mesh; at 300k it leaves more (the
bunny 11.7 % against 7.8 %, the dragon 34.5 % against 11.7 %): the relief gap grows with N. This is the head-and-body
relief item at its worst, and the gates as written do not catch it. (The
probe's stray count, particles outside the largest component at 1.5 end spacings, reads 7k–38k of 40k for both forms:
the threshold is below the sampling's own gaps, so that column is not a measurement and is not reported.)

**B1, 2026-09-29 — settled transport against the earlier pipelines (300k bunny, seed 97).**

| | settled | earlier (v3) | release legacy |
|---|---|---|---|
| silhouette IoU / chamfer | 0.9851 / 0.0581 | 0.9769 / 0.0556 | 0.9646 / 0.0594 |
| wall / windows | 7 min / 31 | 12 min / 47 | 28 min / 183 |
| det F min / stray particles | 0.934 / 0 | 0.608 / 30 | 0.581 / 40 |
| ear tip (reference particles; gate 13) | 11.1 | 8.8 | 4.3 |
| target surface beyond 1.5 spacings | 11.7 % | 7.8 % | 23.5 % |
| late surface motion (spacings per frame) | 0.007 / 0.009 | 0.128 / 0.113 | 0.049 / 0.080 |
| window reversals, whole / second half | 12 % / 23 % | 0 % / 0 % (pinned) | 70 % / 100 % |
| splat video flicker (tail) | 0.0003 | 0.0008 | — |

With a longer budget (`--reject_stop 20 --patience 20`, 68 windows, 13 min): ear tip 12.1, surface gap 10.4 %.

**B2, 2026-09-29 — gradients, render influence, momentum (settled, 300k bunny).**
- Tests pass; gradient path and line-search path agree to seven digits; finite differences agree with autograd within
  3 % for the physics, render and cleanup terms at windows 1 and 20.
- Render share of the control update: 0.33 at window 1, 0.90 at window 20; on `u` 0.96–0.97 (the physics gradient on
  `u` is nearly zero). The render and physics gradients never conflict (cos +0.14 to +0.26).
- Render-off twin: silhouette IoU 0.9851 → 0.9672 and surface gap 11.7 % → 23.5 %, against a seed spread of 0.002.
  For the earlier pipeline the same twin costs 0.005.
- Centre-of-mass drift over the morph: ≤ 0.02 spacing in every settled run; 0.6–0.8 for the earlier pipeline.

**S1, 2026-09-29 11:40 CDT — the branch.** `settled-base` = Michael Jin's settled transport (835af64, on the release
680622e) plus the earlier pipeline's 4K and quick splat renderers, their modules and tests, the measurement probes and
the server environment. Added: the renderers accept archives without pins; `--render_weight_scale` (render-off twin)
and `--reject_stop`. Tests: 207 passed, 2 skipped.
