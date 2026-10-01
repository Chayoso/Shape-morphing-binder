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
- **R12f, the two W1 rulers measured (pre-registered 2026-10-01 12:40 CDT, launched 12:43 after the suite passed on
  repo_r26 (269 passed on the second run; the first run failed the known unstable
  test_line_search_probe_is_diagnostic_only, which passed three reruns on this code and on repo_r25); repo_r26 =
  repo_r25 with records only, and the far-bound plumbing swept out).** The objective's W1 runs over the particles the isolation
  gate marked at the window start; the selection merit's common form runs over every particle, so that windows
  with different gates are compared with one ruler. No behaviour changes. Recorded per committed window:
  `merit_w1_gap` = the merit's W1 (all particles) minus the objective's (gated), at the committed state, and the
  verdicts of a shadow selection that follows the same trajectory but reads the merit with the gated W1
  (`shadow_reject`, `shadow_improved`, `shadow_stop`, `shadow_merit`) beside the actual ones. Also swept, no
  effect on a run: nn_far_k leaves the config, the prepare stage and the probes; an empty near band (berth at or
  beyond the loss cell) raises. Runs: the 40k gallery twice (19 meshes, seed 97) and the 300k dragon at 35 windows
  twice (`output/gpu/r12f`, `tmp/r12f_eval.py`). Two numbers decide (the user's rule): (1) the gap as a share of
  the merit; (2) how often the two rulers disagree on a window's accept/reject, on "improved" (which drives the
  plateau count and the stop), on the stop itself and on the delivered best window. If the gap is sizable and no
  verdict flips, the merit's W1 can be aligned for consistency as the near band was; if verdicts flip often, why
  the selection needs the all-particle W1 is looked at before anything is changed. Check on the sweep: the gallery
  within spread of R12e's two runs (silhouette median within ±0.002, thin median within ±1). Predictions: the gap
  is a large share of the late merit (tens of per cent: the all-particle W1 does not go to zero at the end, d_dt
  stays at 8–170 in legacy units); accept/reject flips are rare (under 5 % of the windows) but "improved" flips
  are not, because a near-constant offset changes the relative tolerance; the gallery is unchanged.
- **R12e, the selection merit reads the same near band as the objective (pre-registered 2026-10-01 10:59 CDT, launched
  10:59 after the suite passed on repo_r25 (270 passed); repo_r25 = repo_r24 with the common form's near band bounded
  at one loss cell).** In R12d the objective's
  near band ends at one loss cell but the selection merit's common form (the ruler that ranks and accepts windows)
  still counted every particle beyond the berth. R12e gives the common form the same band, read at the current
  state; the objective is unchanged. This is not a new term: the objective and its ruler now measure one thing.
  Because the band is read at the current state, the merit steps when a particle crosses the outer edge; the
  record `merit_far` keeps what the old ruler would have added, per window. The common form's W1 stays ungated
  (the objective's is isolation-gated at the window start): the same kind of mismatch, noted and not changed
  here. Runs: the 40k gallery twice (19 meshes, seed 97) and the 300k dragon at 35 windows twice
  (`output/gpu/r12e`, `tmp/r12e_eval.py`), against R12d's two runs. Pass, on the two-run means: silhouette
  difference median within ±0.002 and no mesh below by more than 0.004; `thin_uncovered` difference median within
  ±1 point and no mesh worse by more than 3 beyond the arms' own run-to-run difference; `kin` median within R12d's
  spread; the stray measures not above R12d's larger run by more than its own difference; 300k: silhouette within
  ±0.002, world-thin within ±1 point, the transport energy at window 34 not above R12d's larger value (2.0e-3) by
  more than the two R12d runs differ (0.6e-3). Reported: C's window count and end `kin` (does the agreeing ruler
  let C run on), `merit_far` against the merit, the windows' accept/reject counts. Predictions: the gallery and the
  300k run are unchanged within spread (late in a run little mass lies beyond the loss cell, early every window
  improves under either ruler); C still stops near 20 windows (its alternation is in the transport energy, not in
  the near band). Render influence: unchanged channel; λ and g_share reported.
- **R12d-s, the spread on the three meshes that rose (pre-registered 2026-10-01 10:41 CDT, launched 10:41; no code
  change).** spot, bimba and teapot at 40k, seed 97, four more runs each of R12d (repo_r24) and of R11f (repo_r22),
  six per code and mesh with the two each has (`output/gpu/r12ds`, `tmp/r12ds_eval.py`). Reading, on the six-run
  means: a mesh's thin loss is the definition's if the difference of the means exceeds 3 points, or exceeds 2
  points with the two codes' six values not overlapping; otherwise it is spread. R12d passes the thin criterion if
  no mesh's loss is the definition's. Prediction: spot's difference falls below 2 (R12c's arm with the same band
  gave 8.0, 7.7); bimba and teapot stay between +1 and +2.5.
- **R12d, the cleanup as the W1 pull plus a near band that ends at one loss cell (pre-registered 2026-10-01 09:47
  CDT, launched 09:47 after the suite passed on repo_r24 (269 passed); repo_r24 = repo_r22 with the box leash removed and the near band's far bound set to the loss
  cell).** Two changes to R11f, each measured alone first: the box leash is removed (R12b: inert) and the near
  band's eligibility becomes berth < distance to the nearest target point < one loss cell (tgt.ldx, the Sinkhorn
  blur length; R12c tested 4.5 spacings, the loss cell is 4.24–4.48). w_box and the far bound's 1000 leave the
  objective; w_dt and w_nn are unchanged (their scale is a later step). The selection merit's common form keeps
  the near band without a far bound, as in the arm R12c tested; aligning it is a separate change. Runs: the 40k
  gallery twice (19 meshes, seed 97) and the 300k dragon at 35 windows twice (`output/gpu/r12d`,
  `tmp/r12d_eval.py`), against R11f's two runs. Pass, on the two-run means, as R12: silhouette difference median
  within ±0.002 and no mesh but beast below by more than 0.004; `thin_uncovered` difference median within ±1 point
  and no mesh worse by more than 3 beyond the arms' own run-to-run difference; the stray measures (`stray_max`,
  `stray_final`, `out_nn_far_frac`, `out_dt_frac`) not above R11f's larger run by more than R11f's own difference,
  medians and the largest value without beast, and `outside_max` 0; `kin` median within R11f's spread; the G4 gate
  passed by as many runs; 300k: silhouette not below R11f's lower run by more than 0.002, world-thin not above
  R11f's by more than 1 point, the transport energy at window 34 at or below 2e-3, `out_nn_far_frac` and
  `stray_final` not above R11f's. Known in advance and reported, not hidden in a median: C stops near 20 windows
  still moving (R12c), the parked merit alternation. Predictions: all pass but C's end velocity; heart and spot keep
  R11f's thin; the 300k dragon reaches R11f's window-34 energy by about window 12. Render influence: the render
  channel is unchanged; λ and g_share reported.
- **R12c, which part of the near band the 40k meshes need (diagnostic, pre-registered 2026-10-01 09:26 CDT, launched
  09:26; no code change: `repo_r22f` = repo_r22 with the default nn_far_k = 4.5 instead of 1000).** The near
  band's eligibility is berth < distance to the nearest target point < nn_far_k spacings, frozen per window. With
  the far bound at 4.5 spacings (the bound its unit test and the `out_nn_far_frac` metric use) only particles in
  a band next to the target are pulled; the far pull on the rest of the body is off. Arm NEAR between R11f (the
  whole pull) and NB (none). Runs: the 300k dragon at 35 windows; heart, C, spot, A and dragon at 40k, two runs,
  seed 97 (`output/gpu/r12c`, `tmp/r12c_eval.py`). Reading: the 40k benefit is the near part's if heart's two-run
  mean thin is at or below 8.5 (R11f 7.1, NB 10.4) and C runs at least 40 windows in both runs; the 300k cost is
  the far part's if the transport energy at window 34 is at or below 2e-3 (NB 1.2e-3, R11f 8e-3). Predictions:
  both hold (the thin points and C's settling are near-surface work; the fight with the transport is the pull on
  particles far from the target). If both hold, the term's defect is its missing far bound, and what remains to
  define without a tuned number is the band's width and the size of the pull against a transport that falls with
  N; if heart needs the far pull, the near band is doing transport's work at 40k and the question goes back to
  the geometry energy. This is a measurement, not an adoption candidate: 4.5 is a number.
- **R12b, which removed term does what (leave-one-out, pre-registered 2026-10-01 08:55 CDT, launched 08:55; no code
  change: two copies of repo_r22 with one default set to zero, `repo_r22n` w_nn = 0 and `repo_r22b` w_box = 0).**
  Arms NB (near band off, box kept) and BX (box off, near band kept) between R11f (both on) and R12 (both off).
  Runs: the 300k dragon at 35 windows, one per arm; heart, C, spot, A and dragon at 40k, two runs per arm, seed 97
  (`output/gpu/r12b`, `tmp/r12b_eval.py`). Reading, per effect: (a) the 300k speed-up belongs to an arm if its
  transport energy at window 34 is at or below 2e-3 (R12 0.8e-3–1.1e-3, R11f 8e-3); (b) heart's thin loss belongs
  to an arm if its two-run mean is at or above 10 (R12 11.6, R11f 7.1); (c) C's early stop belongs to an arm if
  both runs stop before 30 windows with `kin` above 5e-3. Predictions: all three belong to NB; BX equals R11f on
  everything (no delivered frame leaves the target extent). If the 300k gain and heart's loss are both the near
  band's, the term is a nearest-point pull that helps where the target is thin and close (40k) and fights the
  transport where its per-particle size outgrows the transport's (300k): the question becomes what the near band
  did for heart that the surface proximity does not, not which weight to give it. Render influence: unchanged
  channel; λ and g_share reported.
- **R12, the cleanup reduced to the W1 pull (pre-registered 2026-09-30 23:50 CDT, launched 23:52 after the suite passed on
  repo_r23 (269 passed); repo_r23 = repo_r22 with the near-band pull and the box leash removed).** The cleanup was three terms with three constants: the
  isolation-gated W1 (w_dt 0.2), the near-band pull to the nearest target point (w_nn 0.2, with its berth and
  far bound) and, in the physics objective, the box leash beyond the target extent (w_box 10). R12 keeps the W1
  alone, with its weight unchanged (its scale is a later step): the surface proximity places the surface, the
  transport moves the mass, the W1 pulls isolated strays, and the domain box stays a validity constraint of the
  rollout. The selection merit's common form becomes the ungated W1. Measured before the change, on R11f's 40
  runs: `outside_max` is 0 in every run (no delivered frame leaves the target extent, so the leash is zero on the
  delivered path); the near band's population at the end is 0.00–0.27 % of the particles at 40k and 1.2 % on the
  300k dragon. Runs: the 40k gallery twice (19 meshes, seed 97), the 300k dragon at 35 windows twice, and beast four
  more times (`output/gpu/r12`, `tmp/r12_eval.py`), against R11f's two runs (and its six beast runs). Pass, on the
  two-run means: silhouette difference median within ±0.002 and no mesh but beast below by more than 0.004;
  `thin_uncovered` difference median within ±1 point and no mesh worse by more than 3 beyond the arms' own
  run-to-run difference; the stray measures (`stray_max`, `stray_final`, `out_nn_far_frac`, `out_dt_frac`): the
  medians and the largest value over the meshes without beast not above R11f's larger run by more than R11f's own
  difference between its two runs, and `outside_max` still 0; `kin` median within R11f's spread; the G4 ejection
  gate passed by as many runs; beast frozen in at most 5 of 6 (R11f: 4 of 6); 300k: silhouette within ±0.002,
  world-thin within ±1 point, `out_nn_far_frac` and `stray_final` not above R11f's larger run by more than its own
  difference. Predictions: all pass; the box changes nothing; `out_nn_far_frac` on the 300k dragon is the one
  number that may rise (0.11–0.12 % under R11f); the window gets slightly cheaper (the near-band assignment is no
  longer built). Render influence: the render channel is unchanged; λ and g_share reported.
- **R11f-b, how often each code meets the beast freeze (pre-registered 2026-09-30 23:29 CDT, launched 23:29; no code
  change).** beast at 40k, seed 97, four more runs each of PX85 (repo_r17), R11d (repo_r21) and R11f (repo_r22),
  six per code with the two each has (`output/gpu/r11fb`, `tmp/r11fb_eval.py`). A run is frozen if it commits
  fewer than 15 windows with `kin` > 0.05. Reading: the drift exposes the freeze if R11d freezes in at most 1 of 6
  and each code that carries the drift in at least 3 of 6; R11f is worse than PX85 only if its count exceeds
  PX85's by 3 or more. Prediction: R11d 0–1 of 6, PX85 and R11f 3–5 of 6 each, no difference between those two
  beyond 2. The freeze itself stays parked (the ejected filament and the domain margin); this only says whether
  R11f changes its rate.
- **R11f, the released motion as the stability term with the residual drift back inside the geometry energy
  (pre-registered 2026-09-30 22:34 CDT, launched 22:35 after the suite passed on repo_r22 (269 passed); repo_r22 =
  repo_r21 with the drift restored).** Physics
  objective: ot_scale · (S_ε + proximity + (T dt)² mean_i |v_T,i|²) + (T dt)² mean over the released steps and
  particles of |v|² + the regularisers. It is R11c's arm R with the legacy 200 wu replaced by (T dt)² (equal within
  7 % at 40k, 2.2× at 300k); no weight is added, w_kin (5) and w_kin_var (200) stay removed; the outer merit reads
  the transport energy without the drift, as in R11c. Runs: the 40k gallery twice (19 meshes, seed 97) and the
  300k dragon at 35 windows twice (`output/gpu/r11f`, `tmp/r11f_eval.py`), against PX85's two runs. Pass, on the
  two-run means as in R11d-s: silhouette difference median within ±0.002 and no mesh below by more than 0.004;
  `thin_uncovered` difference median within ±1 point and no mesh worse by more than 3 beyond the arms' own
  run-to-run difference on that mesh; `kin` and `kin_var` medians not above PX85's by more than PX85's own
  run-to-run difference of medians; no run stopping before 15 windows with the body moving; no freeze PX85 did not
  show; 300k: silhouette within ±0.002 and world-thin within ±1 point of PX85's two runs, `kin` at window 34 not
  above PX85's higher value (8.6e-3). Predictions: all pass; the 300k `kin` at or below PX85's range (the drift as
  in PX85 and a released coefficient 2.2× the legacy one). Render influence: the render channel is unchanged; λ
  and g_share reported.
- **R11e, does the end drift account for what R11d lost (diagnostic, pre-registered 2026-09-30 22:13 CDT, launched
  22:13; no code change: repo_r20 with `--diag_w_kin_rel 200`, R11c's arm R).** R11c-R is the one staged code that
  differs from R11d by the end drift (inside ot_scale) alone, its released-motion coefficient equal within 7 % at
  40k (and 0.46× R11d's at 300k). Runs: bob, heart, teapot and A at 40k, two runs each, seed 97, and the 300k
  dragon at 35 windows (`output/gpu/r11e`, `tmp/r11e_eval.py`), against the two runs of PX85 and of R11d.
  Reading: the drift accounts for the loss if, on bob, heart and teapot, the two-run mean thin is within 3 points
  of PX85's mean (or within the arms' own spread), and the 300k `kin` at window 34 falls inside or below PX85's
  two values (4.9e-3 to 8.6e-3). If thin stays at R11d's level, the difference is in the terms PX85 has and R11c-R
  does not (the end kinetic energy, the driven fluctuation); if the 300k `kin` stays at R11d's level, it follows
  the release coefficient or those terms, not the drift. Predictions: the 300k `kin` returns to PX85's range; thin
  on the three returns to within 3 points on at least two of them. A (R11d +1.4, within spread) is the control.
  Render influence: unchanged channel; λ and g_share reported.
- **R11d-s, the run-to-run spread of both arms (pre-registered 2026-09-30 21:18 CDT, launched 21:18; no code
  change).** A second run of R11d (repo_r21) and of PX85 (repo_r17), seed 97 as before (the rollout's atomics make
  two runs of one code differ), on the 19-mesh 40k gallery, and of both on the 300k dragon at 35 windows; outputs
  `output/gpu/r11ds`, read with `tmp/r11ds_eval.py`. It decides the three criteria R11d missed. Reading, on the
  two-run means per mesh and arm: (thin) R11d passes if the paired difference of the means has a median within ±1
  point and no mesh is worse by more than 3; a mesh beyond 3 whose difference is smaller than the larger of the two
  arms' own run-to-run differences on that mesh counts as spread; (kin, kin_var) passes if the difference between
  the arms' medians is no larger than the difference between PX85's two runs' medians; (300k) `kin` at window 34
  passes if the two R11d values and the two PX85 values overlap, silhouette within ±0.002 and world-thin within ±1
  point as before. Predictions: the thin shift is spread (teapot's +3.4 does not repeat; the median of the mean
  differences within ±0.5); the 40k `kin` medians are within spread; at 300k `kin` is higher under R11d in both
  runs. If the end velocity is reproducibly higher, that is a property of the definition (the end step carries 1/T
  of the term) to be decided as such, not tuned. Render influence: the render channel is unchanged; λ and g_share
  reported for all four runs per mesh.
- **R11d, the stability term = the released motion outside the transport's scale (pre-registered 2026-09-30 18:35
  CDT, launched 18:36 after the suite passed on repo_r21 (269 passed); repo_r21 = repo_r20 with the term moved out of ot_scale, the drift a record `stab_end`, the
  R11c switches removed).** L_stab = (T dt)² · mean over the released steps and particles of |v|², added to the
  physics objective unscaled (R11 had it inside ot_scale, 3–6× weaker); the residual drift is no longer costed
  (R11c-R carried it inside ot_scale beside the legacy piece, 15–33 % of the piece's size; the released motion
  contains the end step). The coefficient equals the legacy 100 wu at 40k within 7 % and is 2.2× it at 300k. No
  weight. Runs: the 40k gallery (19 meshes, seed 97) and the 300k dragon at 35 windows, proximity and the grid
  following N as R10 adopted; against R10's PX85 and, on R11c's eight meshes, R11c-R (`tmp/r11d_eval.py`). Pass, as
  R11: `thin_uncovered` median within ±1 point of PX85 and no mesh worse by more than 3 (homer's ±4 spread noted);
  silhouette median within ±0.002 and none below by more than 0.004; `kin` and `kin_var` medians not higher; no run
  stopping before 15 windows with the body still moving (`kin` > 0.05); wall ≤ 1.2×; no freeze PX85 did not show;
  the 300k dragon at 35 windows within ±0.002 silhouette and ±1 point world-thin of PX85's, `kin` not higher.
  Predictions: the six R11b meshes sound (R11c-R at the same magnitude was); the gallery within spread of PX85;
  `kin` at the median not higher; at 300k the stronger coefficient lowers `kin` and costs at most 1 point of
  world-thin at the budget. Render influence: the render channel is unchanged; g_share reported.
- **R11c, which removed velocity piece keeps a run sound (diagnostic, pre-registered 2026-09-30 17:58 CDT, before
  launch; repo_r20 = repo_r19 + the three pieces as switches, off by default).** R11b's code (proximity, the end
  drift as the stability term) with one removed piece put back at its legacy weight: K = the end kinetic energy
  (5), D = the driven phase's velocity fluctuation about its mean (200, the driven part of the old variance term),
  R = the released motion (200, its released part). On the six meshes R11b lost (bimba, cheburashka, cow, homer,
  nefertiti, spot) and two it kept (bunny, dragon), 40k, seed 97. Reading: a piece "keeps a run sound" if none of
  the six stops before 15 windows with the body still moving (`kin` > 0.05 at the end) and their thin and
  silhouette are within R11b's sound meshes' spread of PX85 (thin ±3, silhouette ±0.004). Predictions: D keeps
  them sound (it is the only term that sees the driven phase's velocity field; the transport alone pushes as hard
  as the control clip allows), R partly (it damps the release but not the push), K not (the end kinetic energy
  duplicates the drift). The surviving piece is then given a definition without its legacy weight.
- **R11b, the stability term as the end drift alone (pre-registered 2026-09-30 17:27 CDT, launched 17:29; repo_r19 =
  repo_r18 with the term changed).** After R11's failure the stability term is the residual drift of the released
  end, (T dt)² mean_i |v_T,i|², length², inside the geometry's scale, no weight: MJ's drift, now the only velocity
  term, with w_kin (5) and w_kin_var (200) gone and the drift moved out of the transport energy. A release that
  oscillates and ends at rest costs nothing under it; the released integral is recorded (`stab_release`) to see how
  often that happens. Runs and pass criteria as R11 (against R10's PX85: thin median within ±1 point, no mesh
  worse by more than 3; silhouette median within ±0.002, none below by more than 0.004; `kin` and `kin_var`
  medians not higher; wall ≤ 1.2×; no freeze PX85 did not show; the 300k dragon at 35 windows within ±0.002
  silhouette and ±1 point world-thin, `kin` not higher). Prediction: geometry and the end velocities within spread
  of PX85, since the two removed terms were small (w_kin |v_T|² ≈ 3e-5 and the variance term of that order against
  a transport energy of 1e-4 to 1e-3 late) and the drift they duplicated stays.
- **R11, the stability term: three velocity terms replaced by the released motion (pre-registered 2026-09-30 16:51
  CDT, before launch; repo_r18 = repo_r17 with the change).** The objective penalised the released end's velocity
  three times with three constants: the residual drift |T dt v_T|² inside the transport energy, the end kinetic
  energy w_kin |v_T|² (w_kin = 5) and w_kin_var (200) times the driven fluctuation plus the released motion. All
  three are removed. The stability term is L_stab = (T dt)² · mean over the released steps and particles of |v|²,
  length² like the geometry, inside the same scale (ot_scale), so that a body at rest after the release costs
  nothing, a constant released velocity costs what the drift did, and a release that oscillates and comes to rest
  only at its end costs as much as a constant one of the same speed (the drift alone charged nothing). No weight:
  the horizon converts velocity to length as before. The driven phase is not costed by it (the transport and the
  control terms already are). The window record keeps `kin` and `kin_var` as measurements and adds `stab`.
  Runs: the 40k gallery (19 meshes, seed 97) and the 300k dragon at 35 windows, proximity and the grid following
  N as R10 adopted; against R10's PX85 arm (the same code with the three old terms). Pass: `thin_uncovered`
  median within ±1 point of PX85 and no mesh worse by more than 3 points; silhouette median within ±0.002 and no
  mesh below by more than 0.004; the end kinetic energy `kin` and `kin_var` (both measured in both arms) not
  higher at the median; wall ≤ 1.2×; no freeze PX85 did not show; the 300k dragon at 35 windows within ±0.002
  silhouette and ±1 point world-thin of PX85's, `kin` not higher. Predictions: geometry within spread; the
  released motion lower (the term sees the whole release, the drift saw its end); wall unchanged. Render influence:
  the render channel is unchanged; g_share reported.
- **R10, the geometry ablation: coarse transport + surface proximity against the fine transport (pre-registered
  2026-09-30 15:55 CDT, launched 15:55; repo_r17).** Three arms: CR = the loss grid following N (85³ at 300k) with
  the ratio support (C_R); PX85 = the same grid with the surface proximity; PX43 = the MPM-cell grid (43³ at 300k)
  with the surface proximity. At 40k every arm's grid is the MPM cell, so the 40k gallery tests only the proximity
  against the support, on the 15 meshes R9 did not run (A, armadilo, beast, bimba, bob, cheburashka, cow, fandisk,
  heart, homer, maxplanck, nefertiti, ogre, spot, V; the R9 four join them), CR and PX85 on each. The 300k dragon
  and bunny run all three arms. Timing is read from the phase timers.
  Pass, 40k gallery (19 meshes with R9's four): `thin_uncovered` lower under the proximity on ≥ 14 of 19 with a
  median drop ≥ 2 points; silhouette IoU median within ±0.002 and no mesh below C_R by more than 0.004; wall ≤
  1.2×; no freeze C_R did not show. Pass, 300k: PX43's silhouette IoU and `thin_uncovered_world` within the two C_R
  runs' spread of PX85 on both meshes (the coarse grid loses nothing once the proximity places the surface) and
  its window at least 20 % cheaper; PX85 against CR as at 40k (thin lower by ≥ 2 points, silhouette within ±0.003).
  The gradient ratio at 300k is reported (the scale question: radius² shrinks with N). If PX43 loses silhouette or
  the thick body's coverage against PX85 (R2's fine-grid gain), the fine grid stays. Predictions: the gallery passes
  (R9's four gave −2.3 to −5.9); at 300k PX43 ≈ PX85 within spread and 25 % cheaper per window; the proximity's
  gradient ratio at 300k smaller than at 40k. Render influence: the render channel is unchanged; g_share reported.
  Amended 16:03 CDT, launched 16:04 (before any 300k result was read): the six converged 300k runs would have taken about 2.5
  hours, against the user's target of a 300k run in 15 minutes. They were killed at 7 minutes and replaced by the
  dragon's three arms under a 35-window budget (about 15 minutes at the measured window cost), one per GPU beside
  the gallery streams (`tmp/r10b.sh`). The 300k question becomes: at equal budget, which geometry gets furthest.
  Pass at 300k, restated: PX43 within PX85's silhouette by 0.002 and its `thin_uncovered_world` by 1 point at the
  budget, and at least 20 % cheaper per window; PX85 against CR as at 40k. The converged comparison waits.
- **R9, surface proximity in place of the density coverage (pre-registered 2026-09-30 15:42 CDT, before launch;
  repo_r17 = repo_r16 with the fine term's definition replaced).** D8 showed the density ratio blind to the 1.6–2
  spacing gaps. The fine part of the geometry objective is now the target-to-body surface proximity: at every outer
  target point y the kernel of the body's NEAREST particle, K(d_min(y)) = exp(−d_min²/2h²), against half the kernel
  at one sampling pitch, K(sp)/2, with the support's h = r8/2 and sp the target's median nearest-neighbour spacing;
  penalty radius² mean_y relu(1 − K(d_min)/(K(sp)/2))², length² like the transport, no bound, no weight. Its
  threshold sqrt(sp² + 2h² ln 2) = 1.53 sp is the hard metric's 1.5 spacings without a constant of its own; d_min
  is continuous in x whichever particle is nearest. The transport supplies mass; this term places the surface.
  Static check on the same-code C_R end states (d8CR): it charges 87–93 % of the thin points the metric calls
  uncovered (the 1.50–1.53 band holds the rest) and 0.0 % of the covered ones, the penalty at charged points 0.03–
  0.10 R² (most gaps sit just past the threshold).
  Runs: bunny, dragon, C and teapot at 40k, seed 97, C_R's flags otherwise, against the same-code C_R runs (d8CR);
  and the synthetic plate `assets/plate.obj` (about 11 × 7.5 × 0.56 wu after normalisation, 1.8 cells thick, every
  outer point thin), with its own C_R run, reported and not in the pass. Archives kept.
  Pass: (A) at the end states the term charges ≥ 85 % of the metric-uncovered thin points and ≤ 1 % of the
  covered ones, and its charged count is within 15 % of the metric's (the loss and the metric ask one question);
  (B) `thin_uncovered` at or below C_R − 2 points on at least 3 of the 4 gallery meshes; silhouette within ±0.003;
  wall ≤ 1.5×; no freeze C_R did not show; churn in the second half with closed ≥ opened (filling, not
  relocation). Predictions: thin −2 to −4 points on the four; the plate's uncovered share falls by more; the
  gradient ratio of order one mid-run as in D8. Render influence: the render channel is unchanged; g_share reported.
- **D8, the target-surface coverage in place of the support (pre-registered 2026-09-30 15:25 CDT, before launch;
  repo_r16 = repo_r15 + `TargetCoverage`, the phase timers and the gradient-ratio telemetry).** The first step of
  the loss reformulation (four terms: transport, rendering, stability, cleanup). The support asks each body
  particle for enough body around it and cannot see a target patch with no particle at it. `--support_form
  coverage` replaces it: at every outer target point y (the census outer set of the target's volume sample, a
  shell about one spacing thick), the body's kernel sum over its 32 nearest particles against half the target's
  own leave-one-out kernel sum there, the estimator, h and k of the support; penalty radius² mean_y relu(1 −
  s_b/f_t)², length² like the transport, no bound to E and no weight (a target-side deficit is not the uniform
  pressure the bound guarded against). The hard uncovered share stays the metric. Runs: bunny, dragon, C and
  teapot at 40k, seed 97, C_R's flags otherwise (the 85³ loss grid stays; the grid is the next ablation's
  question), against the two C_R runs' band (r7 primary, crv). Archives kept for the mechanism probe.
  Readings. (a) The question is whether the coverage gives the controls a more accurate sub-cell signal than the
  support, not whether it closes the gaps (D3a/c: a strong fine objective left nine tenths of them): prediction
  `thin_uncovered` −1 to −3 points against the band on at least three of four meshes; less than one point on all
  four means the observer alone changes nothing under these controls. (b) Silhouette within ±0.003 of the band;
  no freeze C_R did not show; wall ≤ 1.5× (the extra neighbour query). (c) `sup_grad_ratio`, the position-gradient
  norm of the coverage against the transport's at every committed state, recorded for the scale question (both
  length²; expected of order one early and falling as the surface is covered). (d) Mechanism, on the archives: at
  uncovered outer points the coverage gradient on the body particles within 2h points into the gap (median cosine
  with the gap direction above 0.5), the transport gradient on the bulk behind a thin region points toward it, and
  the churn (gaps closed against gaps opened per window, `thin_time.py`) is not above C_R's. Filling and not
  relocation is the pass on (d).
- **R8, coarse-to-fine at the coarse resolution's stop event (pre-registered 2026-09-30 14:27 CDT, before launch;
  repo_r15 = repo_r14 + the trigger).** Fix 3 of the agreed order. The switch to 96 px render targets fired at
  window 150, half the window budget: never reached by C_R's 40k runs (15–50 windows), reached by the 300k dragon
  only through its run length, and missed by R7's 300k run by eleven windows, which cost it the last 20 windows at
  96 px and 0.002 of silhouette. Now (`c2f_event`) the switch fires when the run at 64 px would stop, by the plateau,
  the patience or the rejection streak, and the run goes on at 96 px to its own stop; the fine epoch starts with a
  fresh render weight, convergence count and rejection streak, and the delivered slice is the best window of the
  last epoch, as before. No new constant: the existing tol, patience and reject_stop define the event.
  Runs: the 40k gallery (19 meshes, seed 97) and the 300k dragon, R7's code otherwise. Against R7. Pass: every run
  not frozen by null windows has a 96 px epoch; silhouette IoU median ≥ R7 + 0.001 and no mesh below R7 by more
  than 0.004; `thin_uncovered` median ≤ R7 + 1 point; null windows summed ≤ 1.5× R7's; wall ≤ 2.5× R7's (two
  epochs); the 300k dragon converged with silhouette ≥ 0.983 and chamfer ≤ 0.060.
  Predictions: silhouette +0.002 to +0.005 at the median (in R5's long runs the 96 px epoch cut d_sil threefold);
  thin −1 point or unchanged (T2: 96 px sees few of the thin gaps); wall 1.5–2×; the 300k dragon at or above 0.983
  once the switch is reached. Render influence: this change is the render channel's schedule alone; the physics
  objective is unchanged; each run records the coarse epoch's end and the fine epoch's end, and g_share at 96 px.
- **R7, the support floor continued to the particle's position (pre-registered 2026-09-30 12:03 CDT, before
  launch; repo_r14 = repo_r13 + the floor).** Fix 1 of the agreed order. Under `--support_target_ref` the floor of a
  particle at x was the leave-one-out density of its nearest target point, piecewise constant in x with steps of about
  13 % at the Voronoi boundaries (D6 pre-check: 59–61 % of particles sit next to such a step; D5: step-independent
  objective jumps). It is now f(x) = 1/2 (t(x) − K(0)), t the target's kernel sum at x over its 33 nearest target
  points, with the same kernel, h and k as the body density and the gradient of f kept (Kelsall & Diggle 1995;
  Monaghan 2005). It equals the old floor at every target point (1e-15), is continuous, and is at or below zero
  (no floor, no penalty, no gradient) only where the target's kernel sum is below one unit, about 1.4 spacings outside
  the target, where the old floor carried the nearest point's density outward and charged spray that the W1 cleanup
  handles. No new constant (K(0) = 1). The global floor and every other term are unchanged.
  Runs: the 40k gallery (19 meshes, seed 97) with C_R's flags, and the 300k dragon. Against the two C_R gallery runs
  (r4, crv). Pass, no regression plus the property: silhouette IoU median at or above the lower C_R median − 0.0005 and
  no mesh below the lower of its two C_R values by more than 0.004; `thin_uncovered` median within one point of the
  C_R band; null windows summed at or below the larger C_R sum; wall at or below 1.2× the larger C_R wall; at the end
  `sup_B` below C_R's (crv) on at least 15 of 19 meshes; 300k dragon converged with silhouette IoU ≥ 0.983 and chamfer
  ≤ 0.060, its thin metrics recorded as the comparator for what follows.
  Predictions: geometry within the run-to-run spread (thin ±1 point, silhouette ±0.002), `sup_B` 3–5× lower at the
  end, wall and null counts unchanged (the noise nulls of D6 (b) remain). Render influence: the change is in the
  physics objective's support term; g_share is reported from the runs; no render-off twin in this experiment.
  Relaunched 12:10 CDT: the 12:04 launch froze every run at window 1. Where the floor is at or below zero the
  deficit log f − log s was about −100, whose ratio-form gradient exp(−t) overflows float32 (0 × inf = NaN in the
  backward pass; the unit tests ran in float64). The deficit is now exactly zero on safe inputs wherever f ≤ s, the
  penalty there being zero anyway. The definition is unchanged; the suite passes with a float32 case added.
- **D6, three measurements before any fix (pre-registered 2026-09-30 11:43 CDT, before launch; repo_r13 = C_R +
  telemetry, no algorithm change).** Working order from now on (user, 2026-09-30): measure which part of the current
  code misbehaves, find the root problem, and only then change that definition; never an extra algorithm to block a
  problem. Runs: C_R + `--ls_probe` at 40k, seed 97, on bunny, cheburashka and homer (D5's tails) and beast three
  times (its C_R freeze is intermittent: one of three C_R runs so far). Archives kept for the floor pre-check.
  Recorded (new telemetry): per iteration the Adam direction's cosine with sign(g) and with g, its per-coordinate rms
  (1 = a sign step), the share of the gradient's energy in its largest 1 % of coordinates, the accepted step, the
  dFc step's rms and max and the u step in spacings; at every failed trial the spread of three repeated evaluations
  of the current point (objective, transport without support, B, render); every window's start-state validity with
  its reason, the commit rollout's reason and the state-check failure reasons; at a dead start state the free
  rollout as it is and with the last plastic assimilation undone.
  Readings. (a) H_sign, Adam acts as sign descent on dFc: supported if the median cos(d, sign g) ≥ 0.9 and the rms
  ≥ 0.8 at every iteration index while the top 1 % of coordinates carry ≥ 50 % of the gradient energy (a sign step
  then moves the 99 % that carry little); refuted if cos < 0.7 or the top 1 % carry < 20 %. The reset pattern is
  present if the iteration-1 accepted step is at most half the iteration-8 median. (b) Noise against jumps: at
  tiny-step failed trials (a < 1e-6), the step's change above 10× the repeat spread in ≥ 70 % of them means
  deterministic non-smoothness, which no tolerance may cover; within 2× the spread in ≥ 70 % means evaluation noise,
  and the line search's floor must then be measured where it is used; a mixed result is reported by term. (c) Dead
  state: at a beast freeze, the free rollout valid with the assimilation undone and invalid as it is means the
  assimilation makes the dead state and the acceptance must check the post-assimilation state; both invalid means
  the committed end state is unviable beyond the horizon (reason and first bad step recorded); both valid means the
  search's own trials fail (reasons recorded).
  Predictions: (a) supported, cos ≥ 0.95 and the top 1 % above 60 %, with a weak reset pattern (Adam's direction is
  sign-like at every iteration of a window, not only the first); (b) non-smoothness dominates, the repeat spread at
  or below 1e-7 in most trials and exactly zero in a good share; (c) the assimilation (low confidence).
  Side note (11:42 CDT): the suite on repo_r13 failed `test_a_nonfinite_window_cost_cannot_commit` once and passed it
  four times after: the first window of the tiny test cloud was a `commit_replay` null (commit E 1.8e-6 relative above
  the accepted value, tolerance 10 × replay noise = 1.4e-6). The same fragility as D4's and R5's null windows.
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

## Parked, 2026-09-30 15:05 CDT (user directive: only the loss reformulation from here)

The objective is to be rebuilt as four terms (transport: where to go; rendering: how it should look; stability:
settled after the release; cleanup: pathological strays), the current-particle support replaced by a target-surface
coverage, and a 300k run brought to about 15 minutes. Everything else measured today waits, with its evidence:
- Evaluation nondeterminism (D6 b, R7): two rollouts of one control differ in the positions in every window, max
  5e-6 spacings, never zero, from the transfers' atomics; the line search's 1e-7 floor and the commit tolerance
  assume finer resolution (one `commit_replay` null per run). Not a tolerance fix.
- Beast's freeze (D6 c): the head of an ejected filament coasts past the target's tip into the two-cell margin at
  windows 10–12; the global trajectory check vetoes every candidate; the fragment detectors are right (the stream is
  grid-coupled); the root is ejection plus mass-averaged terms that cannot see a flung handful.
- The late merit alternation at 300k (±1–2 % per window) and `reject_stop = 3` (R7's 300k stopped at 139).
- The u channel and the forward relaxation (D3b/c, R5, R6): revisit only once the objective sees sub-cell.
- Adam per window (D6 a): not sign descent; the direction's cosine with the gradient 0.3–0.4 after the first
  iteration; not established as a defect.
- R8 (event c2f): unjudged (killed at 6 of 19; the JSONs are in the tarball); the trigger is in repo_r15 and worked
  on the bunny (the switch at the rejection streak, six windows at 96 px).
- The loss's redundancy (three end-velocity terms, four pull-to-target terms, a dead `w_ctrl`): superseded by the
  reformulation.
- The surface roughness metric measures the outer band's depth scatter (baseline 1.2 spacings); redefine on the
  outermost layer before it gates anything.

## Results so far

**R12f addendum, 2026-10-01 14:10 CDT — offline replay: the flips are the dense distance's variation, not a dilution
of the tolerances.** The selection rule replayed on the recorded windows (`tmp/r12f_replay.py`) reproduces every
recorded verdict of the actual and of the shadow selection (0 mismatches in 1977 windows). Hypothesis tested: a
large near-constant term makes the relative tolerances lenient (brake at 5 % of the merit, "improved" at 0.3 %),
so the flips would vanish if the isolated merit carried a constant of the same size. They do not: with the run's
median gap added as a constant the disagreements with the actual verdicts are 108 accept/reject (5.5 %), 176
"improved" (8.9 %) and 58 stops, against 95, 177 and 45 without it. The hypothesis is refuted; the decisions turn
on how the dense distance changes from window to window. The 17 delivered-window flips are choices between near
ties: the two windows are 1 apart at the median (5 at most) and differ by 0.5 % of the merit at the median (1.4 %
and 1.7 % at most, under either ruler), against the 0.3 % tolerance of "improved".

**R12f, 2026-10-01 13:34 CDT — the all-particle W1 is two fifths of the selection merit and it decides windows: not
to be aligned with the objective's W1 (pre-registered 12:40).** 38 gallery runs, 1977 judged windows, and two 300k
dragon runs. The gap (the merit's W1 over all particles minus the objective's over the isolated ones) is 41 % of
the merit at the per-run median window (8–85 % across runs; heart 85 %, V 64 %, armadilo 21 %) and 40 % at the last
window; on the 300k dragon 40 % in the first window and 8–9 % from the median window on. A shadow selection that
follows the same trajectory with the isolated W1 disagrees on accept/reject in 95 windows (4.8 %), on "improved"
in 177 (9.0 %), on the stop in 45, and delivers a different best window in 17 of the 38 runs (1–5 windows apart);
on the 300k dragon in none. By the user's rule this is the third case: the two are not unified. The prediction
(accept/reject flips under 5 % and rare, "improved" more often) held in numbers and was wrong in weight: 17
delivered windows differ. What the all-particle W1 measures: across the second half of a run its rank correlation
with the Sinkhorn energy is +0.82 and with the density D_vol +0.81 (surface proximity +0.65), with the silhouette
loss −0.29 and with thin +0.24; window to window its changes follow the silhouette loss (+0.71) and the Sinkhorn
energy (+0.59). It is a dense body-to-target distance, largely the transport's information again and not the
silhouette's or the thin measure's. What it does to the choice: in the 17 runs the window the isolated ruler
would deliver has the lower silhouette loss in 13 and the lower thin in 10 (higher in 3, equal in 4), by 0.1–1.2
points; the all-particle ruler's window has the lower W1 in 17 and the lower transport energy in 8. Of the 46
windows only the all-particle ruler rejects, 41 raised the W1 while 25 raised the silhouette loss and 20 thin; of
the 49 only it accepts, 45 raised the silhouette loss and 26 thin while 9 raised the transport energy. So in the
merit the dense distance outweighs the render term: it carries windows that trade silhouette for distance and
stops windows that trade distance for silhouette. Per-window Chamfer and IoU are not recorded (the silhouette loss
stands in). Not measured: what a run does when it is actually steered by the isolated ruler (the shadow follows
the real trajectory). Check on the sweep: against R12e the gallery's silhouette difference median is +0.0000 and
thin +0.19 (beast froze in one run: 10 windows, silhouette 0.8212, the parked ejection); 300k dragon silhouette
0.9807, 0.9827, world-thin 1.6, 1.5, E at window 34 1.8e-3, 1.9e-3 (R12e 1.5e-3): within what two runs show.
Render influence: unchanged code path; the finding is itself about the render's weight in the selection. Naming:
the objective's term is a cleanup of isolated particles; the merit's is a state-quality distance; they should not
share the name W1.

**R12e adopted, 2026-10-01 12:40 CDT (the user's decision).** The cleanup is the isolation-gated W1 and the near
band between the sampling berth and one loss cell, in the objective and in the selection merit; the box leash and
the far bound of 1000 spacings are gone. Roles by distance to the target: inside the berth nothing acts; between
the berth and one loss cell, where the transport's blur cannot tell positions apart, the near band; beyond it the
transport and, for isolated particles, the W1. Recorded as a real trade, not as "geometry preserved": against R11f
spot loses about 2.8 thin points (six runs each), bimba and teapot 1–2; the 300k dragon is four to five times
further at window 34 with fewer strays. C's early stop is confirmed not to be a near-band mismatch (R12d and R12e
both stop at 18 windows) and is parked with the merit alternation. The near band is frozen here. Next, in this
order: the selection merit's W1 is measured before any change (R12f); then the scale of the W1 and the near band
against N (both are sums with a constant pull per particle).

**R12e, 2026-10-01 11:57 CDT — the selection merit reading the same band: nothing moves beyond spread; every
criterion passes (pre-registered 10:59).** Two runs against R12d's two. 40k gallery: silhouette difference median
−0.0002, worst −0.0021 (beast); `thin_uncovered` median −0.27, higher on 7 of 19, worst bob +2.6 inside its own
run-to-run difference (8.3, 13.5 against 9.2, 7.4); spot 9.4, 9.7 (R12d's six runs 8.5–11.6; R11f's mean 7.2);
`kin` medians 8.4e-5, 8.0e-5 against 8.1e-5, 9.2e-5; strays not above R12d's (`stray_final` and
`out_nn_far_frac` largest 0.013–0.015 against 0.025); `outside_max` 0; no run stops before 15 windows moving; the
G4 gate passed by 16 + 17 runs against 17 + 17; windows median 47, 47 against 46, 55; rejected windows 130 + 131
against 129 + 134 over the gallery. C stops at 18 and 18 windows with `kin` 1.3e-2 and 1.4e-2 (R12d 18 and 20):
the agreeing ruler does not let it run on, as predicted; its alternation is not in the near band. beast sound in
both runs (87 and 111 windows). 300k dragon at 35 windows: silhouette 0.9835, 0.9825 against 0.9820, 0.9831;
world-thin 1.0, 1.0 against 1.7, 0.8; transport energy at window 34 1.55e-3, 1.45e-3 against 2.03e-3, 1.43e-3;
`out_nn_far_frac` 0.007, 0.011 against 0.030, 0.014; `kin` at window 34 8.6e-4, 3.9e-3 against 1.4e-4, 5.9e-4
(the second run higher; not a criterion, noted); 14–16 minutes. What the old ruler counted and this one does not
(`merit_far` over the merit): at 40k 3–35 % in the first window and zero from the median window on (C 4 % at the
median, beast 11–14 %); on the 300k dragon 42 % in the first window, 10–13 % at the median, 2 % at the end. So the
two rulers differed by a tenth of the merit through most of a 300k run and the outcome is the same within spread:
the alignment is a consistency fix without a measurable effect on these runs. Render influence: λ at the first
window identical on every mesh (median 0.248; 0.396 at 300k); g_share at the end 0.88 against 0.89, 0.94–0.96
against 0.92 at 300k. The cleanup candidate is therefore: the isolation-gated W1 and the near band between the
sampling berth and one loss cell, in the objective and in the selection merit; no box leash. Against R11f it buys
a 300k dragon four to five times further at window 34 with fewer strays, and costs a real 2–3 thin points on spot
(1–2 on bimba and teapot) and C's settling (parked). Adoption is the user's decision.

**R12d-s, 2026-10-01 10:56 CDT — six runs per code: the three meshes shift by +0.6 to +2.8, none past the limit
(pre-registered 10:41).** spot: R11f 8.2, 6.8, 6.5, 6.2, 6.0, 9.4 (mean 7.20) against R12d 11.6, 9.9, 11.4, 8.8,
9.9, 8.5 (mean 10.04), +2.84, the two sets overlapping in one value; bimba 10.02 against 11.38, +1.36; teapot
10.10 against 10.73, +0.64 by the means and +1.9 by the medians (9.5 against 11.4; one outlier in each set).
Silhouette means equal within 0.0006. By the pre-registered reading all three are spread and R12d passes the thin
criterion. Read plainly, spot's shift is real, not spread (five of R11f's six values lie below all six of R12d's;
rank test one-sided p ≈ 0.01), and stays under the 3-point limit; the prediction that it would fall below 2 was
wrong. The far part of the near band was worth about 1.4 to 2.8 thin points on these three meshes, against a 300k
dragon that is four to five times further at window 34. R12d stands as the adoption candidate; before the final
adoption the selection merit is given the same band (R12e, the user's condition).

**R12d, 2026-10-01 10:39 CDT — the near band ending at one loss cell, the box removed: the 300k dragon is four to
five times further at window 34 with fewer strays, the 40k gallery holds except spot's thin and C's early stop
(pre-registered 09:47).** Two runs against R11f's two. 300k dragon at 35 windows: silhouette 0.9820, 0.9831
against 0.9802, 0.9820; world-thin 1.7, 0.8 against 3.1, 4.0; transport energy at window 34 2.03e-3, 1.43e-3
against 8.0e-3, 8.3e-3 (the first run misses the 2e-3 line by 1.5 %); `kin` 1.4e-4, 5.9e-4 against 7.9e-3,
6.8e-3; `out_nn_frac` 0.84, 0.60 against 1.21, 1.25 and `out_nn_far_frac` 0.030, 0.014 against 0.117, 0.111;
`stray_final` 0.002, 0.004 against 0.013, 0.021; window 23 s and 15 minutes, as every arm with the near band on.
40k gallery: silhouette difference median +0.0000, worst −0.0023 (C); `thin_uncovered` median +0.24, higher on 11
of 19; spot +3.3 (11.6, 9.9 against 8.2, 6.8) fails the limit of 3 beyond its spread (1.7), although R12c's arm
with the bound at 4.5 spacings gave spot 8.0, 7.7 and the loss cell is 4.39 spacings on spot; bimba +2.2 (12.0,
11.5 against 9.5, 9.7) and teapot +2.1 (11.4, 11.4 against 9.3, 9.3) are higher in both runs, A −2.1 (5.6, 7.9
against 9.0, 8.6: the watch-list item is gone) and homer −1.4 lower; heart 7.1, 8.3 against 6.5, 7.7 (kept).
`kin` medians 8.1e-5, 9.2e-5 against 8.7e-5, 9.0e-5 (passes). C stops at 18 and 20 windows with `kin` 1.4e-2 and
8.9e-3, as announced; it alone carries the stray maxima over the limits (`stray_final` and `out_nn_far_frac`
0.025 %, `out_dt_frac` 0.026 %, ten particles, against 0–0.005 %); without C the largest two-run `stray_max` is 0.17
against 0.34 (homer). `outside_max` 0 in every run; the G4 gate passed by 17 + 17 runs against 16 + 16; beast sound in both
runs (R11f frozen in 4 of 6). Window time at 40k unchanged (2.16–2.20 s). Render influence: λ at the first window
identical on every mesh; g_share at the end 0.89 against 0.89 at 40k, 0.92 against 0.92–0.93 at 300k. By the
user's rule for this experiment (heart and spot keep their thin, the 300k run speeds up; C's early stop is the
parked merit alternation and does not count) heart and the 300k run hold and spot does not: spot has ranged
6.8–9.9 across codes with the whole near band (PX85 9.9, 9.4), so whether +3.3 is the definition or the spread is
measured before the verdict (R12d-s).

**R12c, 2026-10-01 09:43 CDT — the near part of the near band covers the thin points, the far part costs the 300k
run and is what keeps C running (diagnostic, pre-registered 09:26).** With the far bound at 4.5 spacings: heart's
two-run mean thin 8.0 (7.7, 8.3; R11f 7.1, near band off 10.4) and spot's 7.8 (R11f 7.5, off 10.5): the thin
points are the near part's work, as predicted. 300k dragon at 35 windows: transport energy 3.1e-2 at window 8 and
1.6e-3 at window 34 (R11f 2e-1 and 8e-3, off 7.5e-3 and 1.2e-3), world-thin 1.0, silhouette 0.9821, `kin` 1.2e-3,
`out_nn_frac` 0.68 % and `out_nn_far_frac` 0.011 % (the lowest of the arms: R11f 1.2 % and 0.11 %, off 2.5 % and
0.20 %): the 300k cost is the far part's, as predicted. C stops at 19 and 19 windows with `kin` 1.1e-2–1.6e-2,
as with the near band off: the prediction that C's settling is near-surface work was wrong; C needs the far pull
to keep running (its silhouette at the stop is 0.9746–0.9766 against R11f's 0.9767–0.9770 after 52–63 windows,
its thin 9.7 against 9.2: the same shape, not at rest; the stop is three rejected candidates with the transport
energy rising, the parked merit alternation). A and the 40k dragon do not separate the arms. Timing: every arm
with w_nn > 0 costs 23 s per 300k window (0.73–0.76 s per line-search trial) against 15.5 s with it at zero (0.44
s), at equal trial counts, although the term is computed in both; the cause is not identified (noted for the
runtime phase). The far bound that was tested is a derived length: one loss cell (the transport's blur length) is
4.24–4.40 target spacings on the 40k gallery and 4.48 on the 300k dragon (the loss grid follows N), so "4.5
spacings" is one loss cell within 5 %. That gives the near band a definition without the number: it acts between
the sampling berth and one loss cell from the target, where the transport's blur no longer resolves a position;
farther out the transport owns the particle (R12d).

**R12b, 2026-10-01 09:24 CDT — all three effects belong to the near band; the box changes nothing (pre-registered
08:55).** (a) 300k dragon at 35 windows: with the near band off the transport energy is 7.5e-3 at window 8 and
1.2e-3 at window 34 (world-thin 0.6, silhouette 0.9808, window 15.5 s, 10 minutes), as R12; with the box off it is
1.9e-1 and 7.1e-3 (world-thin 2.7, 22.4 s, 15 minutes), as R11f. (b) heart's thin, two-run means: 10.4 with the
near band off (10.7, 10.1) against 7.1 with the box off (6.5, 7.7, R11f's two values to the digit) and 7.1 under
R11f; spot goes the same way (10.5 against 7.8 and 7.5). (c) C stops at 20 and 19 windows with `kin` 1e-2 when
the near band is off and runs 49 and 65 windows with the box off. A and the 40k dragon do not separate the arms
(A 7.4 / 7.6 against 8.8; dragon 14.0 / 14.0 against 12.7, within the 12–14 the dragon shows across codes). The
three predictions hold. Reading: the box leash is inert and can go. The near band is one term doing two things:
late in a 40k run it pulls body mass that sits outside the surface onto it (heart, spot: thin points covered; C:
the run settles), and early in a 300k run it pulls the whole body to its nearest target points against the
transport's assignment. Its definition has no band: every particle farther than one berth from its nearest target
point is eligible up to 1000 spacings, and the pull per particle does not fall with N while the transport's does.
What part of it the 40k meshes need is measured next (R12c).

**R12, 2026-10-01 08:52 CDT (the runs ended 00:44) — the W1 alone: the 300k dragon gets about four times further
in the same 35 windows and each window is a third cheaper; at 40k the geometry holds but heart loses thin and C
stops early (pre-registered 2026-09-30 23:50).**
300k dragon at 35 windows, two runs against R11f's two: silhouette 0.9824, 0.9827 against 0.9802, 0.9820;
world-thin 0.3, 0.9 against 3.1, 4.0; transport energy at window 34 8.0e-4, 1.1e-3 against 8.0e-3, 8.3e-3 (R12
is at 8e-3 by window 8); `kin` at window 34 1.5e-4, 1.0e-4 against 7.9e-3, 6.8e-3; `stray_final` 0.001, 0.002
against 0.013, 0.021; `out_nn_far_frac` 0.106, 0.189 against 0.117, 0.111 (the second run exceeds the limit, as
the prediction allowed); window 15.4 s against 22.9 s (gradient 8 s against 11, line search 3.3 s against 6.5),
10 minutes against 15. None of this was predicted ("the box changes nothing, the near band is 1.2 % of the
particles"): the prediction read the end state, where few particles are beyond the berth, and missed the early
windows, where almost every particle is. The near band is SUM_p m_p relu(|x_p − nearest target point| − berth)
over every particle beyond one berth (far bound 1000 spacings), assigned once per window: a nearest-point pull on
the whole body, of constant size per particle whatever N, against a transport whose gradient per particle falls
as 1/N. It competes with the transport's assignment, and its share grows with N. Which of the two removed terms
does it is measured next (R12b).
40k gallery, two runs against R11f's two: silhouette difference median −0.0004, worst −0.0012 (passes);
`thin_uncovered` median +0.37, higher on 12 of 19; heart +4.5 (11.9, 11.3 against 6.5, 7.7) fails the limit of 3
beyond its spread; spot +2.1, dragon +1.5, homer +1.5. `kin` medians 1.04e-4, 1.04e-4 against 8.7e-5, 9.0e-5:
outside R11f's own spread, fails; C stops at 20 and 17 windows on three rejected candidates with `kin` 9e-3 and
2e-2 and the transport energy rising (R11f: 63 and 52 windows, `kin` 1e-4), silhouette unchanged (0.9763, 0.9761
against 0.9770, 0.9767). Strays: `stray_max` median 0.005 against 0.0075 and its largest value without beast 0.33–
0.42 against 1.38–1.41 (lower); `stray_final` and `out_nn_far_frac` largest 0.005 against 0.0025 (at the limit);
`out_dt_frac` largest 0.0125 against 0.005 (5 particles of 40k, over the limit by 0.005); `outside_max` 0 in every
run; the G4 gate passed by 17 + 17 runs against 16 + 16. beast frozen in 2 of 6 (R11f 4 of 6), the sound runs
79–113 windows. Window time unchanged at 40k (2.14–2.18 s). Render influence: λ at the first window identical on
every mesh; g_share at the end 0.88 against 0.89 at 40k and 0.95–0.96 against 0.92–0.93 at 300k (the render's
share is larger once the transport has settled). Verdict: not adopted as it stands (heart, C, `kin`); the 300k
result is the largest effect of the whole reformulation and decides the next measurement.

**R11f adopted, 2026-09-30 23:50 CDT (the user's decision).** Settling is two complementary conditions, both
without a constant: the released motion (T dt)² mean over the released steps and particles of |v|² (trajectory
relaxation: without it six meshes stop still moving) and the residual drift (T dt)² mean_i |v_T,i|² inside the
geometry energy (terminal rest: without it the end is faster and small meshes lose thin). w_kin (5) and w_kin_var
(200) are deleted. Accepted with it: the 300k dragon is 4–5 windows (about 2 minutes) behind PX85 at the 35-window
budget, because the 40k balance is kept where the legacy unit conversion weakened the velocity terms 2.2×; the
runtime is taken back elsewhere (line search, the second adjoint, the tail). Parked, not a stability question:
the beast freeze (the `domain` ejection; 2 of 6 under PX85, 4 of 6 under R11f, 0 of 6 without the drift), a
forward/viability defect. Watch list: A's thin is +2.7 points in both runs (9.0, 8.6 against 5.8, 6.4), inside
the limit and reproducible; looked at again after the cleanup pruning. Stability is frozen here.

**R11f-b, 2026-09-30 23:45 CDT — beast freezes in 2 of 6 runs under PX85, 0 of 6 under R11d, 4 of 6 under R11f
(pre-registered 23:29).** Every frozen run stops at 11–12 windows with reason `domain` and |v|max 2.8–3.8 in its
last windows (0.06–0.8 in the sound ones); the sound runs of the three codes agree (silhouette 0.9678–0.9709,
thin 11.2–15.0). By the pre-registered reading R11f is not worse than PX85 (4 against 2, the threshold was a
difference of 3), and "the drift exposes the freeze" is not established as written (PX85 has 2 of 6, the reading
asked for at least 3); the prediction for PX85 (3–5) was wrong, the others held. Pooled, the two codes that carry
the drift freeze in 6 of 12 runs and the one without it in 0 of 6 (one-sided exact p = 0.05): the drift is
associated with the ejection, which stays parked; this is evidence for that dossier, not a reason to drop the
drift (without it the released end is faster and small meshes lose thin, R11d-s). With R11f and R11f-b the
stability question stands as: R11f reproduces PX85 on the gallery (geometry, end velocity, the beast freeze within
counts) without w_kin and w_kin_var, and is 4–5 windows behind PX85 on the 300k dragon at 35 windows because it
keeps the 40k balance where the legacy unit conversion weakened the velocity terms 2.2×. Adoption is the user's
decision.

**R11f, 2026-09-30 23:27 CDT — the released motion plus the drift: PX85's geometry and end velocity on 18 of 19
meshes; beast freezes in both runs; the 300k run is 4–5 windows behind at the budget (pre-registered 22:34).**
Two runs against PX85's two. Silhouette: difference of the means, median +0.0000; without beast the worst is
−0.0009. `thin_uncovered`: median +0.07, higher on 11 of 19, own run-to-run difference median 1.1 in both arms;
without beast no mesh is worse by more than 3 (A +2.7, both runs: 9.0, 8.6 against 5.8, 6.4); bob 9.6, 7.4, heart
6.5, 7.7 and teapot 9.3, 9.3 are back at PX85's level (R11d: 11.8, 15.7; 9.5, 13.1; 12.7, 12.7). `kin` medians
8.7e-5, 9.0e-5 against 9.9e-5, 7.3e-5 and `kin_var` 6.3e-5, 5.8e-5 against 6.7e-5, 6.2e-5: within PX85's own
spread, passes. beast stops at 12 windows still moving in both runs (silhouette 0.8505, 0.8983; reason `domain`,
the parked ejection: a frame outside the two-cell margin, every later window null); PX85 froze the same way in one
of its two runs, R11d in neither. The criterion "no run stopping before 15 windows with the body moving" fails on
beast. 300k dragon at 35 windows: silhouette 0.9802, 0.9820 against 0.9813, 0.9806 (passes); `kin` at window 34
7.9e-3, 6.8e-3 against 8.6e-3, 4.9e-3 (passes; R11d 1.4e-2); world-thin 3.1, 4.0 against 2.4, 2.9 (means +0.9; the
second run is 1.1 above PX85's higher value, a miss by 0.1); the transport energy at window 34 is 8.0e-3, 8.3e-3
against 6.4e-3 to 6.9e-3 in the five runs of the other codes (PX85, R11d, R11e), where PX85 stood at windows
29–30: reproducibly 4–5 windows behind. 15 minutes per run. ot_scale from the records is 0.21 on the 40k dragon
and 0.19–0.20 on the 300k dragon, so the transport's scale does not change with N: the legacy stability weights
(× wu = 1 / unit_ratio) were 2.2× weaker against the transport at 300k than at 40k, and R11f keeps the 40k balance
at 300k. At 40k R11f is behind R11d in the same phase too (E at window 10 8.3e-3–8.5e-3 against 6.0e-3) and
converges to the same end. Render influence: λ at the first window identical in the four runs of every mesh
(median 0.248; 0.396 at 300k), g_share at the end 0.89 against 0.88 (300k 0.92–0.93 against 0.93–0.94); no
render-off twin. Reading: with the drift restored the definition matches PX85 where PX85 is sound; open are the
beast freeze (2 of 2 against 1 of 2 and 0 of 2: counts too small to rank the codes, measured next) and whether
the 300k budget should pay 4–5 windows for a stability balance that does not weaken with N (a decision, not a
defect).

**R11e, 2026-09-30 22:32 CDT — the end drift accounts for what R11d lost (diagnostic, pre-registered 22:13).**
R11c's arm R (the released motion plus the end drift inside ot_scale), two runs: thin on bob 12.7, 8.3 (mean 10.5
against PX85's 9.15 and R11d's 13.75), heart 7.7, 7.7 (against 6.85 and 11.3), teapot 11.9, 8.5 (10.2 against 9.75
and 12.7), A 7.7, 5.8 (6.75 against 6.1 and 7.45): all three within 3 points of PX85's mean (+1.35, +0.85, +0.45),
though one run each of bob and teapot sits at R11d's level (the spread on these meshes is 3–4 points). End `kin`
1.8e-5 to 8.8e-5 on the eight runs, at PX85's level (1.1e-5 to 1.4e-4; R11d 1.5e-5 to 3.1e-4). 300k dragon at 35
windows: `kin` at window 34 5.7e-3, inside PX85's 4.9e-3 to 8.6e-3 (R11d 1.38e-2, 1.42e-2), silhouette 0.9805,
world-thin 2.6, E 6.4e-3, λ 0.396, g_share 0.94, 15 minutes. Both predictions hold. At 300k this arm's released
coefficient is 0.46× R11d's and its end velocity is still lower, so the drift, not the released coefficient, sets
the end velocity. Reading: stability is two measured functions, the released motion (a run stays sound) and the
residual drift of the released end (the end is at rest); neither replaces the other. The drift goes back where MJ
had it, inside the geometry energy (R11f).

**R11d-s, 2026-09-30 22:11 CDT — the spread of both arms: R11d's geometry is PX85's, its end velocity is higher, and
two small meshes lose thin beyond their spread (pre-registered 21:18).**
Two runs per arm, seed 97. Silhouette: difference of the two-run means, median +0.0000, worst −0.0014. `thin_uncovered`:
difference of the means, median +0.13 (higher on 11 of 19; own run-to-run difference median 1.1 under PX85 and 1.4
under R11d, up to 4.4): the +0.6 shift of the first run was spread. Per mesh: bob +4.6 (R11d 11.8, 15.7 against
9.6, 8.7; own spread 3.9) and heart +4.5 (9.5, 13.1 against 7.7, 6.0; own spread 3.6) fail the limit of 3 beyond
their spread; teapot is +3.0 and repeats exactly (12.7, 12.7 against 9.3, 10.2), so the prediction that it would
not repeat was wrong. The three have 168–236 thin points (a point is 0.4–0.6 %), their uncovered gaps have a
median of 1.53–1.54 spacings against the threshold of 1.5, and none is wider than two spacings. No R11d run stops
while moving in either round; PX85's second run froze on beast (11 windows, silhouette 0.8898, the parked
ejection). End velocity: `kin` medians 1.06e-4 and 1.33e-4 under R11d against 9.9e-5 and 7.3e-5 (between arms
3.3e-5, PX85's own 2.6e-5) and `kin_var` 6.7e-5 and 9.3e-5 against 6.7e-5 and 6.2e-5 (1.6e-5 against 4.8e-6): both
fail, the prediction (within spread at 40k) was wrong. 300k dragon at 35 windows: silhouette 0.9812, 0.9805 against
0.9813, 0.9806; world-thin 2.5, 2.3 against 2.4, 2.9; E at window 34 6.6e-3, 6.9e-3 against 6.5e-3, 6.9e-3; `kin` at
window 34 1.42e-2, 1.38e-2 against 8.6e-3, 4.9e-3: no overlap, fails as predicted. A 300k run of 35 windows takes
15 minutes alone on a GPU (both arms; 23 when shared). Render influence: λ at the first window identical in the
four runs of every mesh (median 0.248; 0.396 at 300k), g_share at the end 0.88 against 0.90 (300k 0.94 against
0.92). Reading: the released motion alone gives PX85's silhouette, transport and 300k geometry and keeps every run
sound, but leaves the released end 1.4× faster at 40k and 1.6–2.9× at 300k. On R11c's eight meshes the arm R (the
same released motion plus the end drift inside ot_scale) had `kin` at PX85's level (median 6.2e-5 against 6.5e-5;
R11d 1.1e-4 and 1.6e-4), so the drift, the one difference, is the candidate; whether it also accounts for bob,
heart and teapot is measured next (R11e). R11d is not adopted as it stands.

**R11d, 2026-09-30 21:17 CDT — the released motion outside ot_scale: the runs are sound and the geometry is kept,
three criteria are missed narrowly (pre-registered 18:35; read 21:17, the runs ended 19:07).**
Against PX85 on the 40k gallery: no run stops before 15 windows with the body moving (0 of 19; the six meshes R11
and R11b lost run 38–73 windows: bimba 11.0 / 0.9774, cheburashka 10.4 / 0.9772, cow 11.3 / 0.9740, homer 12.9 /
0.9779, nefertiti 8.2 / 0.9787, spot 9.4 / 0.9767), no freeze, windows median 46 against 47. Silhouette: median
−0.0002, worst −0.0017 (V); passes. `thin_uncovered`: paired difference median +0.6 (medians 10.4 against 9.3),
14 of 19 meshes higher, worst teapot +3.4 (limit 3; maxplanck +2.6, bob +2.2, heart +1.8; homer −3.3, ogre −2.5):
the per-mesh limit is missed by 0.4 on one mesh. `kin` median 1.06e-4 against 9.91e-5 and `kin_var` 6.72e-5
against 6.66e-5: "not higher" is missed by 7 % and 1 %. Wall is not comparable (both rounds shared GPUs
differently; 0.83× as measured). 300k dragon at 35 windows: silhouette 0.9812 against 0.9813, chamfer 0.0614
against 0.0613, world-thin 2.5 % against 2.4 % (34 committed windows against 35, one null commit), 23 minutes
both; the transport energy runs level (8.9e-2 / 1.6e-2 / 6.6e-3 at windows 10 / 20 / 34 against 9.6e-2 / 1.5e-2 /
6.3e-3 at 10 / 20 / 35): the 2.2× coefficient does not hold the geometry back. But `kin` is not lower as
predicted, it is higher from window 20 on (4.2e-2 / 2.0e-2 / 1.4e-2 at 20 / 25 / 34 against 3.0e-2 / 1.4e-2 /
7.9e-3): that criterion fails, and the prediction with it. Render influence: unchanged. λ is calibrated at a
window start at rest, where the stability term has no gradient: the first-window λ is identical on every mesh
(median 0.248 in both arms; 0.396 at 300k in both), the last-window λ within 0.004, g_share at the end 0.89
against 0.89 (300k 0.92 against 0.93); no render-off twin in this round. Reading: the definition restores what
R11 and R11b lost and matches PX85's silhouette and 300k geometry; whether the thin shift (+0.6) and teapot's
+3.4 are the definition or the run-to-run spread (homer: ±4 between two runs of one code) is not known, and the
end velocity may really be higher (a hypothesis, not measured: the term charges the end step with 1/T of its
weight, where the drift charged it alone). Not adopted on this reading; the spread is measured next (R11d-s).

**R11c, 2026-09-30 18:26 CDT — the released-motion piece is what keeps a run sound (pre-registered 17:58).**
On the six meshes R11b lost, the released motion at its legacy weight (R) keeps every one sound: bimba 10.2 /
0.9770 / 50 windows, cheburashka 10.1 / 0.9774 / 85, cow 9.9 / 0.9708 / 70, homer 12.6 / 0.9767 / 59, nefertiti
8.0 / 0.9783 / 60, spot 7.7 / 0.9771 / 40 (thin / silhouette / windows; PX85: 10.7 / 0.9769 / 69, 10.0 / 0.9776 /
52, 10.7 / 0.9719 / 65, 16.1 / 0.9754 / 65, 8.0 / 0.9785 / 47, 9.9 / 0.9775 / 44), end `kin` 4e-5 to 1.3e-3, every
thin within 3 points and every silhouette within 0.004 of PX85 (homer −3.5 thin, inside its ±4 run-to-run spread).
The end kinetic energy (K) and the driven fluctuation (D) each leave five of the six stopping at 8–14 windows with
the body still moving (`kin` 0.14–0.57); cow alone survives under both (K 9.6 / 0.9720 / 61, D 9.1 / 0.9725 / 27).
bunny and dragon are sound under all three (R: 9.1 / 0.9756 / 41 and 13.6 / 0.9734 / 94; the dragon's silhouette
0.0017 below PX85, thin +0.4). The prediction (D keeps the runs sound, R partly) was wrong: the driven phase's
fluctuation about its mean is not what the push needs, and R11b's reading (the end at rest plus a quasi-static
push) is corrected: what a settled morph needs is little motion over the whole relaxation after the control is
removed, so the term must charge every released step, at a magnitude the drift and the end kinetic energy did not
reach. Magnitude: the legacy
piece is 200 wu |v|² / (2 T N) = 100 wu · mean_release |v|², 6.5e-3 to 7.0e-3 per unit of mean |v|² at 40k
(unit_ratio 1.43e4–1.55e4); R11's released motion sat inside ot_scale (0.15–0.33, measured on bunny, cow and spot)
at (T dt)² ot_scale = 1.0e-3 to 2.3e-3, three to six times weaker: R11 failed on magnitude, not on form. Outside
ot_scale the same expression has the coefficient (T dt)² = 6.96e-3 (dt = 0.00417 at every N): the legacy magnitude
at 40k without a constant, 2.2× it at 300k (unit_ratio 3.13e4). That definition is R11d.

**R11b, 2026-09-30 18:00 CDT — the end drift alone: fails on six meshes the same way (pre-registered 17:27).**
Against PX85: bimba, cheburashka, cow, homer, nefertiti and spot stop at 8–11 windows with the body still moving
(`kin` 0.29–0.54; thin +7 to +10, silhouette −0.006 to −0.025); the other thirteen are within spread (thin
−2.9 to +3.0, silhouette within ±0.0015). Medians: thin +0.8, silhouette −0.0007, `kin` 15× higher, windows 28
against 47; 300k dragon 0.9799 against 0.9813 with `kin` 4× higher. The prediction (the removed terms were small)
was wrong: their values are 0.1–0.3 % of the merit early in a run, but they are the only terms that see the
driven phase's velocity field (the variance about the mean flow) and the release's motion before its end, so
without them the transport pushes as hard as the control clip allows, the body carries momentum into the
release, the released end is not at rest, and the outer brake stops the run when a window tries to calm it. The
stability of a settled morph is therefore two things: the end at rest (the drift) and a push that stays
quasi-static (the driven-phase regularity). Which of the two removed pieces does the work is the next
measurement (R11c, leave-one-out on the six failing meshes and two sound ones).

**R11, 2026-09-30 17:27 CDT — the released-motion integral as the stability term: fails (pre-registered 16:51).**
Against R10's PX85 on the 40k gallery: `thin_uncovered` median 10.9 against 9.3 (+1.8; cow +14.2, spot +12.8,
bimba +11.8, cheburashka +9.3, A +5.6), silhouette median −0.0014 (cow −0.026, spot −0.016), the end kinetic
energy 100× higher (median 1.1e-2 against 9.9e-5), runs of 8–11 windows on six meshes, wall 0.69×; the 300k dragon
0.9789 against 0.9813 with `kin` 5× higher. Every criterion fails. Mechanism (cow, spot): from window 9 a window
that lowered the release's motion (stab 6.3e-3 → 2.6e-3) raised the transport energy (1.6e-2 → 2.3e-2) and the
merit by 18 %; the outer brake (a rise above 5 %) rejected it three times and the run stopped at window 8 with the
body still moving at 0.7 wu/s. The mean over the release charges the elastic settling after a push, motion that
must happen early in a morph, so the term fights the transport where the body has to move fast; the drift charged
only what remained at the release's end. The released integral stays as a record (`stab_release`).

**R10, 2026-09-30 16:51 CDT — the geometry ablation: the proximity replaces the support on the whole gallery; the
coarse grid does not replace the fine one (pre-registered 15:55, the 300k part restated 16:03).** `tmp/r10_eval.py`.
- 40k gallery, proximity (PX) against the ratio support (CR), same code, 19 meshes (R9's four included):
  `thin_uncovered` lower on 17 of 19 (sign test p < 0.001), median −2.8 points (pass: ≥ 14 and ≥ 2). Largest
  gains maxplanck −7.3, teapot −5.9, bob −3.9, armadilo / bunny / V −3.7, fandisk −3.6; unchanged cheburashka
  (0.0), nefertiti −0.7; one regression, homer +4.6 (11.5 → 16.1), to be looked at. Silhouette IoU median 0.9769
  against 0.9758 (+0.0005 at the median, worst −0.0012 fandisk; pass). Wall 0.89× (pass). No freeze in either arm
  (beast ran 62 and 66 windows). The proximity runs go longer on most meshes (median 47 against 37 windows).
- 300k dragon at a 35-window budget, on GPUs shared with the gallery streams: CR (85³ + support) silhouette 0.9808,
  chamfer 0.0614, thin 30.4 % own / 4.5 % world, 45 s per window, 33 min; PX85 (85³ + proximity) 0.9813, 0.0613,
  22.6 / 2.4 %, 34 s, 23 min; PX43 (43³ + proximity) 0.9797, 0.0643, 27.3 / 4.9 %, 21 s, 15 min. At equal wall
  time of 15 minutes PX85 (24 windows, d_sil 1.02e-3, thin 26.6 %) matches PX43 (35 windows, 1.06e-3, 27.3 %) and
  both are far ahead of CR (15 windows, 3.7e-3, 41.6 %); at 20 minutes PX85 (33 windows, 8.7e-4, 23.2 %) leads
  and PX43 has stopped. PX85 against CR passes (thin −7.8 / −2.1, silhouette +0.0005, 25 % cheaper per window).
  PX43 against PX85 fails the restated criterion (`thin_uncovered_world` +2.5 against ≤ 1; chamfer worse; no
  gain at equal time): the fine transport grid still places the surface beyond the thin set, as R2 found. The
  gradient ratio ‖∇L_surf‖/‖∇S_ε‖ at 300k: 0.56–0.66 mid-run, about 1.0 at the end (40k: 1.5–4): radius² falls
  with N and the two parts stay within one order of magnitude without a weight; the N dependence is recorded.
- Homer's +4.6 (17:20 CDT, `output/gpu/homer`, `tmp/gap_hist.py`): a rerun of both arms with the same code and
  seed gave CR 15.9 % and PX 12.0 % (−3.9), the opposite sign of the gallery's 11.5 / 16.1. Homer's thin share
  swings about ±4 points between runs of one arm (the parked nondeterminism of the rollout), so single-run
  per-mesh differences of that size are noise; the 19-mesh sign test is the statement, not any one mesh. The gap
  distribution at the rerun's end states shows what the proximity does: gaps wider than 2 spacings 1.7 % → 0.0 %,
  wider than 1.75 spacings 5.9 → 1.3 %, wider than 1.6 spacings 11.1 → 5.9 %, wider than 1.5 spacings 15.9 → 12.0
  %, while the band 1.5–1.6 spacings holds 6.1 % under the proximity against 4.8 % under the support: the term
  removes the real gaps and parks the residue just past its threshold. `thin_uncovered_2sp` (gaps wider than two
  spacings) joins the record from the next deployment, as the measure a threshold near 1.5 cannot park points at.
- Verdict: the geometry objective is S_ε(ρ_b, ρ_t) on the grid following N plus the surface proximity, no support,
  no bound, no support weight. "300k in 15 minutes" with today's window cost is PX85 at about 24 windows (silhouette
  near 0.981, thin 26.6 %); more needs fewer rollouts per window and fewer windows, the runtime items after the loss.

**R9, 2026-09-30 15:55 CDT — surface proximity in place of the density coverage: the geometry passes, the
detection criterion at the optimised end states was mis-specified (pre-registered 15:42).** Against the same-code
C_R runs (d8CR); `tmp/r9_eval.py`, `tmp/r9_detect.py`, `tmp/r9_churn.py`.
- (B) `thin_uncovered`: bunny 11.2 → 7.5 (−3.7), dragon 15.5 → 13.2 (−2.3), C 11.9 → 9.0 (−2.9), teapot 15.3 →
  9.3 (−5.9): at or below C_R − 2 on 4 of 4 (pass; the prediction was −2 to −4). Silhouette +0.0011 / +0.0013 /
  +0.0001 / +0.0008 (pass). Wall 0.65× (pass; the term's neighbour query runs over the outer target points, not
  the body). Nulls 0–3, the c2f switch fired in every run, no freeze (pass). The plate: 6.1 → 7.0 (+0.8), reported;
  an already easy thin slab gains nothing from the term.
- (A) False positives 0.0 % at every end state (pass). Recall at the optimised end states 60 / 86 / 84 / 86 % on
  the thin points (registered ≥ 85 %: bunny and C below) and the charged count 18–31 % below the metric's
  (registered ±15 %: fail). The cause is the two thresholds: the term's 1.53 spacings against the metric's 1.50.
  The optimiser closes gaps until they sit just under its own threshold, so the residual gaps pile up in the
  0.03-spacing band the metric still counts (penalty at charged points 0.00–0.03 R², i.e. gaps barely past the
  threshold). At C_R's states, where the criterion was meant to test whether the loss sees what the metric sees,
  recall was 87–93 %. The literal criterion fails; the loss and the metric ask one question up to a 2 % threshold
  difference. The metric stays at 1.5 spacings for comparability with every earlier number.
- Churn in the second half: closed / opened per window 12/13 (bunny), 26/28 (dragon), 49/34 (C), 10/11 (teapot),
  34/26 (plate): the registered closed ≥ opened holds on 2 of 5 strictly; on the other three the two are equal
  within two per window and the charged count is flat or falling (118 → 104, 321 → 297; teapot 58 → 66). An
  equilibrium at the threshold, not a relocation loop.
- The gradient ratio ‖∇L_surf‖/‖∇S_ε‖: 0.06–0.39 in the first window, 1.5–3.8 mid-run, 1.9–4.1 at the end: the
  proximity's gradient is a few times the transport's late, larger than D8's coverage (about 1). Still one order of
  magnitude with no weight; to be read at 300k in the next ablation before it is called scale-free.
- Verdict: the reformulation's fine term is adopted for the geometry ablation (R10). The support (ratio, target
  floor, bound, weight 8) is out of the geometry objective.

**D8, 2026-09-30 15:40 CDT — the density coverage in place of the support: it gives a surface-side signal, and it
is blind to the gaps that matter (pre-registered 15:25).** Against the same-code C_R runs (d8CR, repo_r16 with the
ratio support), `tmp/d8_eval.py`, `tmp/d8_mech.py`.
- (a) `thin_uncovered` bunny 11.2 → 10.8 (−0.5), dragon 15.5 → 13.5 (−2.1), C 11.9 → 10.7 (−1.3), teapot 15.3 →
  14.0 (−1.3): lower on 4 of 4, at least a point on 3 of 4, inside the registered −1 to −3. (b) Silhouette within
  ±0.001 (median +0.0002); nulls 0–3; wall 0.92× (the coverage's neighbour query is cheaper than the support's);
  the c2f switch fired in every run; no freeze. (c) The gradient ratio ‖∇L_cov‖/‖∇S_ε‖ is 0.04–0.22 in the first
  window, 0.9–1.6 mid-run and 0.5–1.1 at the end: the two parts of the geometry objective are of one size without
  a weight. (d) On the rim particles (within 2h of an uncovered thin point) the coverage descent points into the
  gap (cosine median +1.00, by the kernel's construction) and the transport descent partly so (+0.35 to +0.52);
  behind the rim the transport points toward the gap weakly (+0.34 to +0.40); the coverage gradient reaches 1–6 %
  of the particles; the churn per window is net closing early (C: 313 closed against 106 opened) and near
  equilibrium late (3–26 against 2–14): filling, with no relocation loop.
- Why only 1–2 points: at the thin points the metric calls uncovered (no body particle within 1.5 spacings; gaps
  1.64–1.69 spacings at the median, 1.89 at p90) the body's kernel sum is 3.3–4.4× the target's own density
  (p10 1.3–1.4), so the floor of half the target density charges 0 % of them. The target's density at a thin tip is
  a fraction of a unit (few target neighbours) and the tails of body material 2–3 spacings away exceed half of it.
  Halving the kernel width charges them (recall 100 %) but also 72–78 % of the covered ones; a floor from the
  target's local nearest-sample distance puts the threshold at 2.1 spacings (thin tips have sparser neighbours) and
  charges 10–15 %. Every floor taken from the target's LOCAL value collapses where the feature is thin. The metric
  does not: it uses the sampling pitch. The definition changes accordingly (R9).

**D7, 2026-09-30 15:16 CDT — two measurements for the loss reformulation (no algorithm change).**
- PCGrad's actual effect, from the records of finished runs (`tmp/pcgrad_stats.py` on the tarball; `g_raw_cos`
  is cos(g_phys, g_render) before the projection, and the removed fraction of the render gradient is |cos| when
  negative). 40k, 39 C_R and R7 runs, 1201 windows: the raw render gradient conflicts with the physics gradient in
  27 % of the early windows, 38 % of the middle and 78 % of the late ones (raw cos median +0.19 → +0.08 → −0.23),
  and when it conflicts the projection removes a median 19 / 18 / 32 % of it (p90 43 / 42 / 63 %). 300k dragon,
  3 runs, 478 windows: conflicts 7 / 33 / 53 %, removed 3 / 3 / 4 % (p90 6 / 8 / 10 %). So PCGrad is nearly inert
  at 300k and material at 40k late; its removal (one adjoint per iteration instead of two) has to be tested at
  40k, not assumed.
- Where a 300k window's time goes: phase timers (`t_start`, `t_grad`, `t_ls`, `t_commit`, wall seconds per window)
  added to the telemetry; the 300k dragon for 5 windows with the 85³ loss grid (`--loss_follows_n`, the C_R form)
  and with the 43³ grid (the MPM cell), launched 15:16 on repo_r15 (`output/gpu/timing`, `tmp/timing_eval.py`).
  Result (15:22 CDT, the first five windows, the two runs side by side on GPUs 1 and 2): 85³ window 47.3 s = start
  5.4 (warm start and two replays) + gradients 15.5 (8 iterations: tape rollout and two adjoints each) + line
  search 23.5 (15 candidate rollouts) + commit 3.5; 43³ window 35.4 s = 5.0 + 14.8 + 12.7 (11 candidates) + 2.7.
  So the gradient phase does not depend on the loss grid (the adjoint is the cost), a candidate evaluation costs
  about 1.15 s at 43³ and 1.55 s at 85³, and a window is 20–25 rollouts of 40 MPM steps at 300k. The finer grid
  costs about 25 % per window here, not the 2× R2 recorded per run. A 300k run of 15 minutes therefore needs about
  30–40 windows at either grid, or fewer rollouts per window; the grid choice alone cannot deliver it.

**Stopped and cleared, 2026-09-30 14:59 CDT (at the user's request).** R8 (the event c2f; 6 of 19 gallery runs and
the 300k dragon in progress) and R7's second 300k dragon were killed unread, and every result folder on hyde06 was
deleted (`output/gpu`, `before`, `corrected`, `settled`, `mj`, `scratch`). The small files (JSON, logs, videos, texts)
of everything up to that point are in `/data/relcfd/chayo/results_before_clean_2026-09-30.tgz` (723 MB); the
archives are gone. R8 has no verdict. What runs now: sphere → bunny at 40k and 300k on repo_r15 (C_R + the continued
floor + the event c2f), with the two-view splat render (`output/gpu/show/`).

**R7, 2026-09-30 14:25 CDT — the continued floor: the 40k gallery passes every criterion; the 300k dragon fails the
silhouette threshold by stopping early (pre-registered 12:03, launched 12:10).** `tmp/r7_eval.py`, `tmp/r7_300k.py`.
- 40k gallery against the two C_R runs (crv, r4): silhouette IoU median 0.9758 against 0.9746 / 0.9755 (pass), worst
  mesh −0.0001 (A; pass); thin median 13.5 % against the band 13.6–15.5 (pass); null windows 20 against 26 / 15
  (pass); wall 52 against 51 / 50 min (pass); `sup_B` at the end lower on 19 of 19, median 5.0e-6 against 1.7e-5
  (3.4×, as predicted 3–5×), so the support's effective weight rose from 2.1 to 6.1; roughness 1.207 against 1.199.
  Beast froze again (window 12, 0.8647), as it did in one C_R run: the domain trap of D6 (c) is independent of the
  floor. Geometry moved within the run-to-run spread, as predicted; thin did not move.
- 300k dragon: silhouette IoU 0.9811 (fails ≥ 0.983; C_R 0.9833–0.9838), chamfer 0.0598 (pass), converged, 138
  windows in 83 min, thin 28.2 % at the own threshold and 2.0 % at the world distance (the comparator from here on),
  `sup_B` 7.5e-6 against 1.2e-5. The run tracked the two C_R runs window for window: d_sil at window 100 2.65e-4
  against 2.64e-4 / 2.19e-4, at window 138 2.48e-4 against 2.41e-4 / 1.92e-4, E alike. It then stopped at window 139
  on three consecutive outer-merit rejections (the late window-to-window merit alternation of ±1–2 %, which the C_R
  runs show as well, in their case never three in a row), and so never reached the coarse-to-fine switch at window
  150 that gave the C_R runs their last 20 windows at 96 px and their final silhouette. The threshold is therefore
  confounded by the window-150 schedule (fix 3's subject), but the criterion stands as registered: fail until a second
  300k run shows whether the early stop recurs (running, 14:22).
- Replay-difference telemetry (D6 (b)'s open question, 512 windows): the two replays of one control differ in the
  positions in every window, max 5.3e-6 spacings (p90 7.5e-6, never zero), rms 3.7e-7 spacings; relative term
  differences median 2.2e-7 (transport), 1.3e-7 (kinetic), 1.0e-7 (render), p90 4.7e-6 / 9e-7 / 3.3e-7. The noise
  enters the rollout (the transfers' atomics), and the transport term amplifies it most.

**D6, 2026-09-30 12:00 CDT — the three measurements (pre-registered 11:43).** C_R + `--ls_probe`, 40k: bunny (25
windows), cheburashka (35), homer (40), beast ×3 (38, 43, 28); `tmp/d6_eval.py`.
- (a) H_sign refuted. The Adam direction is a sign step only at the first iteration of a window (cos(d, sign g)
  0.80–0.91, rms 0.36–0.65). From the second iteration on, cos(d, sign g) is 0.34–0.54 and the rms per coordinate
  falls to 0.09–0.16 by the eighth; cos(d, g) is 0.27–0.43. The gradient's energy is concentrated: its largest 1 % of
  coordinates carry 50–80 %. There is no reset pattern: the first iteration's accepted step is 0.65–0.95× the
  eighth's, and the rise is the 1.1× growth rule. The line search is not binding under C_R: 0–4 % of iterations fail,
  and the accepted step is 4e-4–1.5e-3 (D5's 1.5e-4 was the preconditioner arm). What the measurement shows instead:
  after one accepted step most coordinates' gradient signs have changed, so the momentum built inside a window is
  weakly aligned with the current gradient. Not a defect established here; recorded.
- (b) Evaluation noise, not jumps, under C_R. Three repeated evaluations of the same point differ by a median
  2e-8–7e-7 relative (p90 2e-7–1.8e-6), exactly zero in 6–36 % of the trials; per term, the transport without support
  by ~1e-9, B by ~5e-12, the render by ~1e-10 (absolute). Tiny-step failed trials (a < 1e-6) were rare (11, all on the
  bunny) and their changes sat at the noise level (64 % within 2× the spread, none above 10×). So the reading is
  evaluation noise: the objective is not a deterministic function of the control at the 1e-6 level (the transfers'
  atomics), while the line search's floor (1e-7) and the commit tolerance (10× a two-evaluation estimate that reads
  zero in a third of the windows) assume a finer resolution. Each run lost one window to a `commit_replay` null
  (windows 10, 26, 31, 32). The D5 non-smoothness (step-independent support jumps) was measured on the preconditioner
  arm; under C_R at 40k it is not the dominant term. Root: nondeterministic evaluation. A tolerance is not the fix;
  the source (where the two replays differ: positions or the loss side) is now recorded at every window start
  (`replay_dx_max`, `replay_dx_rms` in spacings, `replay_dlv/dlk/dlr`).
- (c) Beast did not freeze in three full runs (start state valid in 100 % of windows; no state-check failure in any
  of the ~420 failed trials of the five runs). Of six further runs with a 20-window budget (a diagnostic control),
  one froze (run 7, window 11), and the probe fired: the start state fails the check for the reason **domain**, not
  inversion. The free rollout from the committed state leaves the two-cell safety margin (det F stays 0.95, no
  particle near inversion), with the last plastic assimilation undone as well. All ten trials, the warm start and
  the commit rollout fail for the same reason, so no control can be accepted: the exit happens in the first steps,
  before a control acts. Window 10 had committed normally (its own 2T frames inside, `clamped` 0, v_max 2.8 wu/s).
  So the committed end state carries momentum that crosses the margin box within the next window even at zero
  control, and the check is global: one particle suffices. The margin box is the leash, 1.25× the larger cloud's
  extent (7.78 wu for beast; the target reaches 6.2), and beast's ejection gate fails (0.4 % stray at the end), so
  ejected particles are the likely carriers. Which particles, and their velocity, is the next measurement (an
  archive kept at a freeze); the fix follows from that, not from a larger box or a softer check.
  **Particle level (14:50 CDT, run 15 of twelve short runs; `tmp/beast_dead_probe.py`, `beast_clump.py`,
  `beast_frag.py`).** The six particles nearest the box are the head of a stream: a clump of about 17 (each with 5
  neighbours within a cell), 3.3–3.6 cells from the dense body, all moving outward along one axis at 0.010–0.011
  wu per step, 0.8–1.1 cells from the box, 22–33 steps from crossing it (the next window has 40). They accelerated
  inside the body from window 1 to 5 (0.002 → 0.015 wu per step, a filament pulled toward the target's far
  extremity at |x| = 6.2), left the dense body at window 6, passed the target's tip at window 9 and coasted with
  the drag's slow decay (0.021 → 0.012) to the box at window 11; every C_R freeze of beast so far sits at windows
  10–12, the travel time. Behind the head a continuous stream of 113 particles trails from the target's tip
  outward (49, 37, 21 per cell, then one empty cell, then the head). Neither fragment detector flags any of them
  in any window, and both are right by their definitions: no particle is alone in its 3³ cells, and the stream's
  occupancy, dilated by one cell, is one component with the body. So the bond pull-back never applies, and it
  should not: the stream is grid-coupled. What is wrong is upstream: the objective cannot see it. The end kinetic
  term, the velocity variance and the box term are means over 40 000 particles, and the transport energy is
  mass-weighted, so 17 particles at ten times the body's speed change none of them measurably; the support is
  per-particle but one-sided; nothing asks the optimiser to bring the head back, and once it is within a window of
  the box the global trajectory check (`positions_in_domain`: one particle in the two-cell margin at any frame
  vetoes the candidate) leaves no admissible control. The freeze is therefore the ejection defect (the
  thin-feature droplet family: a filament overshooting a far extremity) meeting a hard constraint, and its root is
  the same blindness as the thin problem's: mass-averaged terms do not see small sets, in space (sub-cell gaps) or
  in count (a flung handful). Not a fix for the veto or the box, which are downstream; folded into the objective
  question.
- Floor pre-check for fix 1 (`scripts/probes/settled/floor_probe.py` on the four D6 end states): A'' equals the
  current floor at every target point (max difference 1e-15); it is at or below zero for 0.4–0.7 % of particles, all
  at least 1.4 spacings off the target (0–1 of ~29 700 particles within one spacing), mostly outer; within one
  spacing its ratio to the current floor is 0.95 (p10 0.78, p90 1.10), between one and two spacings 0.77; the end
  state's support penalty falls 3–5× (bunny mean 5.8e-4 → 1.1e-4, paying particles 1.5 → 0.5 %); under the current
  floor 59–61 % of particles have a second-nearest target point whose floor differs by more than 10 % (median 13 %),
  the step a Voronoi crossing takes. The pre-check passes: fix 1 goes ahead as R7.

**Rollback to C_R, 2026-09-30 10:52 CDT (at the user's request).** R4 to R6 stacked layers onto the u channel, each
covering the side effect of the one before: the two-sided support, the thin-coverage term, the preconditioner in place
of the relaxation, per-block step lengths and the uniform second moment. The code is back at C_R (1c6218d: ratio
support, target floor, loss grid following N). The switches are removed: `--support_two_sided`, `--diag_coverage`
(`ThinCoverage`), `--no_layer_relax`, `--u_precond`, `--block_steps` and `--u_uniform_adam`. What stays is
measurement: the thin-set metrics (`physmorph/thin.py`), the surface roughness (`physmorph/surface.py`), the
null-window reasons and the line-search probe. R6 was stopped after 7 of 57 runs and is not judged. Still open, as
defects of existing definitions rather than new layers:
- The target floor is read at the nearest target point, so it jumps (D5). A literature pass (Kelsall & Diggle 1995;
  Davies et al. 2018; Monaghan 2005) supports evaluating the body and target kernel sums at the same point, with the
  self term treated alike on both sides.
- The line search's noise floor is below the measured evaluation noise.
- The coarse-to-fine switch is tied to half the window budget, so it never fires under C_R.
- The dFc step fails at about 1.5e-4 in the tail. A candidate cause is that Adam restarts every window, so its 8
  iterations are near-sign steps. Not yet measured.

V2 (visual check, 10:47 CDT): C_R on the 40k gallery and on the 300k dragon and bunny, seed 97, with quick two-view
splat renders (`output/gpu/crv/`, `tmp/crv.sh`). Each archive is removed after its render.

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
