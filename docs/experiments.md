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
- **D10, where a 300k dragon window's time goes, with the near band on and off (diagnostic, pre-registered 2026-10-01
  23:18 CDT, before launch; repo_r31 = repo_r30 with a profile switched on by `--profile`, off by default; the
  near-band-off arm is a copy with the default w_nn = 0).** The user's target is a 300k dragon in about 15 minutes;
  the frozen code's run to its own stop takes 42 minutes (125 window attempts of 20 s: gradients 9.3 s, line search
  4.9 s, start 2.0 s, commit 2.0 s, records 2.0 s), and at window 40 (13 minutes) it is at 0.94 % world-thin against
  0.70 % at the end. From the phase timers: with the near band on a line-search trial costs 0.75–0.86 s in windows
  11–35 against 0.42 s with it off (the same trial count), a gradient iteration 1.5 s against 1.07 s; late in the
  run (windows 61–125) the near-band-on run is at 0.40 s and 1.04 s too. The term's own computation is in both
  arms. Recorded per window: seconds and calls of the MPM rollouts (evaluation and tape), the Sinkhorn solves and
  their sweep count, the surface term, the render, the cleanup, and the three adjoints (physics, cleanup, render).
  Runs: the 300k dragon at 35 windows in each arm (`output/gpu/prof`, `tmp/prof_eval.py`). No criterion: the
  result is which part holds the 0.3–0.45 s per evaluation. Prediction: the Sinkhorn solves (more sweeps to
  converge in the near-band-on states; every solve starts from zero duals).
  **Result (2026-10-01 23:37 CDT): the Sinkhorn solves, as predicted.** 35 windows took 14.9 min with the near band
  on and 10.4 min with it off (silhouette IoU 0.9824 and 0.9818). Per window, windows 11–20: Sinkhorn 13.5 s on
  against 5.0 s off, at the same number of solves (44 a window): 92 blocks of four sweeps per solve against 39;
  windows 21–35: 12.2 s against 4.1 s (84 blocks against 33). Every other part is the same in both arms, per
  window: the three adjoints 1.7 s each (0.21 s a call), the surface term 1.0 s (5.5–5.8 s in windows 1–5), the MPM
  rollouts 1.9–2.1 s (tape and evaluation together), the render 0.3 s, the cleanup 0.02 s. With the near band on
  the Sinkhorn solves are 55–60 % of a window's timed work; they are 35–45 % with it off. The near band's own
  computation costs nothing; the states it produces take 2.4 times the sweeps to converge from zero duals.
- **D11, which term holds a floating particle where it is (diagnostic, pre-registered 2026-10-01 23:47 CDT, before
  launch; repo_r32 = repo_r31 with `--term_dump`, off by default: at every committed state, each term's position
  gradient per particle: transport, surface, near band, spray cleanup, weighted render; the suite passes, 269).**
  The 4K frames of the frozen code's 300k bunny were read frame by frame (`scripts/probes/settled/frame_forensics.py`,
  `frame_sheets.py`, `web_probe.py`). Three groups of particles make the visible defects at the ears: a web
  between the ears in raw frames 80–320 (1526 particles more than 3 target spacings from the target at frame 240;
  from the top cap of the sphere, 8.8 spacings a window upward and away from the target in windows 1–2, thinned to
  1.58 coverage radii with 78 % of them partly supported, which the 4K renderer draws enlarged and translucent); a
  tuft at the notch in frames 400–700 (90 particles, 61 of them from the web, still 5 spacings out when the accepted
  step has fallen from 0.02 to 0.001); and a fringe on the ears at the end (374 particles 1.5–2.1 spacings outside
  the target, 40 % of them from the sphere's outer two spacings against 6 % of all particles, moving outward from
  1.0 to 1.66 spacings after window 8). 38 more particles sit 3.2 spacings outside the body near the feet from
  window 5 to the end. No oscillation: path over net motion 1.00–1.2 in every window, largest excursion from the
  chord 0.04 spacings (half a 4K pixel) after window 24. Run: the 300k bunny again with the dump, its archive and 4K
  frames replacing the first run's (`output/gpu/render/terms_bunny`). No criterion. Predictions: (a) on the end
  fringe both local terms are exactly zero on at least 60 % of the particles (they lie inside the near band's berth
  of 1.97 spacings and below the spray gate's isolation of 1.2), and the remaining gradient on them is no larger
  than on surface particles within one spacing; (b) on the particles more than 3 spacings out at the end the near
  band's gradient points at the target (cosine above 0.9) and the sum of the other terms does not oppose it (cosine
  above −0.5): they stay because the accepted step has collapsed (2e-4–4e-4 from window 16, anneal 0.05), not
  because the terms balance.
  **Result (2026-10-02 00:10 CDT; run: silhouette IoU 0.9877, 60 committed windows, 12.2 min; render: g_share 0.33 →
  0.84 by window 6, λ 0.249; `scripts/probes/settled/term_probe.py`, `flux_probe.py`, `web_terms.py`, `web_probe.py`,
  `compare_ref.py`, outputs in `output/gpu/render/forensics_bunny_t`).**
  (a) First part confirmed, second refuted. At the end 3237 particles (1.1 %) lie 1.5–3 spacings from the target
  sample; 84.2 % of them are inside the near band's berth (near band on for 0.1 % of those, spray cleanup for
  13.3 %), and both local terms are zero on 73.6 % of the set (74.8 % of the 444 above the ear base). The summed
  gradient on the set is still 3.0e-6 per particle against 3.2e-7 on the surface particles: the 513 beyond the
  berth carry the near band's pull of constant size (6.6e-6, 13 times the all-particle rms of the sum), and the
  render term is 1.4e-6 on the set with no inward direction (cosine +0.02). The set is fixed late in the run: 2439
  of the 3322 particles there at window 40 are still there at window 59; the rest cross the 1.5 edge both ways by a
  jitter of ±0.01 spacings a window.
  (b) Confirmed. On the 33 particles more than 3 spacings out at the end the near band is on for all, its descent
  direction points at the nearest target point (cosine +0.92), and the other four terms do not oppose it (+0.32).
  The inward motion delivered to them is 0.6 spacings a window at window 6, 0.05 at 12, 0.02 at 20, 0.005 at 45
  and 0.0003 at 58; they are 3–4 spacings out. The objective asks for the right motion; the step that is accepted
  for the whole body no longer delivers it.
  The web. In windows 0–3 the particles above the ear base and more than 3 spacings out (994–4176 of them) move
  9–10.6 spacings a window along the transport's descent direction (cosine 0.94, 0.92, 0.89, 0.78), which points
  upward (0.66–0.74): the cap of the sphere rises as one ridge and the notch between the ears is cut afterwards.
  The render term is three times larger per particle there (1.2–1.5e-5 against 3.7–4.9e-6) but the motion does not
  follow it (cosine 0.09–0.14). From window 3 the motion follows the spray cleanup (0.58–0.63), 6 → 2.2 spacings a
  window by window 7, then 1.4, 1.0, 0.5 as the step falls: what is left at windows 10–18 is the tuft at the notch.
  The picture. The run's own target sample drawn by the 4K renderer has the same soft, feathered ear edges as the
  morph's last frame (10–90 % width of the ear's silhouette edge: 28.5 px median for the target sample, 22 px for
  the morph): at the end the softness of the 4K picture is the splat rule's on a 300k volume sample, not the
  particle arrangement's. No oscillation in either run (first run: path over net motion 1.00–1.2 in every window,
  largest excursion from the chord 0.04 spacings after window 24).
  The 300k dragon, the same reading (frozen code, to its own stop: silhouette IoU 0.9840, world-thin 0.6 %, 147
  committed windows, 54.1 min; `output/gpu/render/forensics_dragon`). The target sample drawn by the 4K renderer
  has the same blur as the morph's last frame: horns as feathered, banded blobs, no teeth or scales, soft jaw
  edges (`ref_vs_end.jpg`, `pair_end.jpg`). The dragon's blur at the end is therefore the renderer's and the
  sampling's: 300k particles fill the volume, about 87k of them are front-most, each is drawn as a disc of one
  spacing with normals averaged twice over 32 neighbours, and a horn is a few spacings thick. What the morph adds
  is transient: particles more than 3 spacings out 46 862 at raw frame 240, 2905 at 480, 652 at 960, 267 at 1920,
  77 at the end; they sit in the concave gaps (the mouth, behind the neck, between the neck and the body, the tail
  folds), as the bunny's web sits between the ears. Outer target points with no particle within 2 spacings: 21.1 %,
  11.1 %, 5.9 %, 5.1 %, 2.75 % at the same frames (the crown between the horns, the mouth, the tail fold); partly
  supported particles 18.7 % → 6.7 %.
- **D12, which part of the 4K display renderer blurs the picture (diagnostic, no simulation; pre-registered
  2026-10-02 11:55 CDT, before the run; `scripts/probes/settled/render_axes.py`).** D11 found the run's own target
  sample as blurred as the morph's last frame. The display rule (`scripts/render_splat_photoreal.py`): a disc of
  radius = the target's nearest-neighbour spacing × clamp(8th-neighbour distance / coverage radius, 1, 4), a quarter
  as thick; normals = the gradient of a density field blurred over 3 spacings, replaced by a neighbour's where
  weak, then averaged twice over 32 neighbours; a 3-pixel filter on the normal buffer. The reference is the dragon
  mesh itself: three million surface samples with the mesh's face normals, the same rasteriser, camera and
  material. Drawn against it: the 300k target sample with one part of the rule changed at a time (disc × 0.7 and
  × 0.5; density blur 1.5 and 0.75 spacings; no neighbour averaging; no image filter; a combination); the morph
  frames at raw 480 and at the end with the same changes; volume samples of the same mesh at 1.2M and 2.4M with the
  rule unchanged. Measured per picture: the angle between its pixel normals and the reference's, the width of
  the silhouette edge, the interior shading detail as a share of the reference's, the silhouette's IoU with the
  reference, and the covered pixels lost against the unchanged rule on the same particles. No criterion.
  Predictions: (1) the density blur is the largest part: blur 1.5 spacings without averaging cuts the mean normal
  error on the 300k target by at least a third and raises the detail by at least half; dropping the averaging or
  the image filter alone changes less. (2) The edge width follows the disc (× 0.7 gives about × 0.7); at × 0.5 the
  morph frame at raw 480 loses more than 1 % of its covered pixels (the stretched sheets open), at × 0.7 the target
  loses less than 1 %. (3) At 2.4M with the rule unchanged the normal error and the edge width fall about with the
  spacing: the rule is written in spacings, so at a fixed N its widths, not N, are the limit. (4) The best 300k
  variant still stays above the 2.4M sample in normal error: features below a lattice step are not in the sample.
  **Result (2026-10-02 11:59 CDT; `output/gpu/render/d12_dragon`, sheets of the head crop per object).** At 4K one
  pixel is 0.0045 wu: the 300k sample's nearest-neighbour spacing is 7.7 px, its lattice step 13.8 px, the
  normals' density blur 23 px. The mesh reference shows horns, teeth, brow ridges and scales; the 300k sample shows
  none of the teeth or scales under any variant. Mean normal error against the reference (degrees) / silhouette
  edge width (px) / shading detail (share of the reference's):
  the number of particles, rule unchanged, all resampled from the fitted mesh: 300k 18.5 / 9.3 / 0.40; 1.2M 13.3 /
  5.0 / 0.50; 2.4M 10.9 / 3.7 / 0.56 (with blur 1.5 and no averaging: 14.3, 10.6, 9.2);
  the rule's parts on the run's 300k target: as it is 19.3 / 9.5 / 0.42; disc × 0.7 18.8 / 8.0 / 0.47; disc × 0.5
  19.5 / 6.8 / 0.73 (the lattice rows show); blur 1.5 18.0 / 8.8 / 0.47; no averaging 16.9 / 8.0 / 0.43; blur 1.5
  without averaging 15.4 / 7.8 / 0.59; blur 0.75 without averaging 28.7 / 10.4 / 1.61 (lattice noise); no image
  filter: no change;
  the same on the morph's last frame: as it is 21.3 / 7.0 / 0.42; no averaging 19.8 / 5.9 / 0.45; blur 1.5 without
  averaging 23.6 / 7.3 / 0.84; on raw frame 480: 27.2 → 29.1 with blur 1.5 without averaging.
  (1) Refuted in size: blur 1.5 without averaging lowers the 300k target's normal error by 20 %, not a third, and
  raises the detail by 40 %; on the morph frames it raises the error (the morphed particles are rougher below
  the blur than the target's lattice, and the blur hides that); the image filter does nothing. (2) Partly: the
  edge narrows less than the disc (× 0.84 at × 0.7, × 0.72 at × 0.5); × 0.5 loses 4.8 % of the covered pixels on
  raw frame 480 as predicted; × 0.7 loses 2.0 % on the target, more than predicted, but that loss is the outline
  moving in toward the mesh's (IoU with the reference 0.951 → 0.957). (3) Confirmed: from 300k to 2.4M (spacing
  × 0.5) the normal error falls × 0.59 and the edge × 0.40. (4) Confirmed: the best 300k variant (14.3–15.4) stays
  above 2.4M with the rule unchanged (10.9).
  Reading: the largest part of the blur is not a filter but the sample. 300k particles fill the volume, the
  surface gets one particle per 13.8 px at 4K, and the dragon's teeth and scales are smaller than that step, so
  they are absent from the target sample and from any morph toward it. The rule's own widths add a smaller part,
  and narrowing them trades blur for lattice and arrangement noise on the morph frames. The morph itself adds 2
  degrees at the end (21.3 against 19.3). The mesh was fitted to the sample by bounding box (per-axis ratios
  within 1.5 %); the run's target and its resample from the fitted mesh agree (19.3 and 18.5).
  Addendum (12:10 CDT, `scripts/probes/settled/surface_origin.py`, the 300k dragon run): where the particle budget
  is and who makes the final surface. Within one lattice step of the surface: 4.5 % of the particles in the source
  and 6.0 % in the last frame (within two steps 14.9 % and 20.4 %): nine tenths of the particles are interior. Of
  the last frame's outer layer, 65 % was within two steps of the source's surface, 78 % within four, and 11 % came
  from deeper than eight; of the source's outer layer, 80 % ends within two steps of the final surface and 99 %
  within four. The surface material mostly stays surface material.
- **D13, the source's surface carried by the archived motion (display-only diagnostic, no simulation;
  pre-registered 2026-10-02 12:12 CDT, before the run [the stamp first read 12:14, two minutes ahead of the
  server clock at the launch: corrected]; `scripts/probes/settled/surface_tracers.py`).** D12: nine
  tenths of the particles are interior and the 300k sample puts one particle per 13.8 px on the surface. The source
  mesh (the 80-face sphere), fitted to the source sample and subdivided to 164k and 655k vertices, is moved through
  the delivered frames of the frozen 300k dragon run as massless tracers: over each stretch of frames (4 frames
  while the body moves fast, 8, then 20) a tracer takes the displacement of the particles around it (an affine
  least-squares fit over its 24 nearest particles). The archive holds no grid velocities; the particles'
  displacements are the simulation's motion sampled at 300k points, and they include the outer layer's own
  offset (the u control), which a grid velocity would not. The carried mesh is drawn as D12's reference and
  measured against the target mesh. Nothing in the physics or the objective changes; a tracer picture does not
  show stray particles, so it is read beside the particle picture, not instead of it. Predictions: (1) the tracers
  stay on the body: at the end the median tracer is within one lattice step of its nearest particle and 99 %
  within two. (2) The picture: silhouette edge at most 3 px (particles 7.0), silhouette IoU with the reference
  above the particles' 0.947, mean normal error 13–17 degrees (particles 21.3; the 2.4M sample 10.9); no teeth or
  scales, because the motion has none below a lattice step. (3) 164k and 655k tracers agree within 1 degree and
  0.5 px: past about 100k the tracer count is not the limit, the motion's own resolution is. (4) The triangles'
  area grows by a median of 1.2–2 and by more than 10 on at least 1 % of the faces (horns, limbs).
  **Result (2026-10-02 12:19 CDT; the first attempt at 12:12 failed at raw frame 1200: the source mesh was not
  welded, and the weights of a tracer more than about 14 lattice steps from every particle underflowed; fixed by
  welding, a weight width never narrower than the neighbourhood, and no extrapolation of the fit beyond the
  neighbourhood; faces with a vertex more than two lattice steps from every particle are not drawn;
  `output/gpu/render/d13_dragon`).** The source's surface carried through the whole run is not the body's surface
  at the end. At the last frame (164k / 655k tracers): mean normal error 33.7 / 34.5 degrees against the particles'
  21.3; silhouette IoU with the reference 0.935 / 0.937 against 0.947; edge 1.7 / 1.8 px; "detail" 2.7 / 3.5 times
  the reference's, which is crumpling, not relief. The carried mesh: total area 3.7 / 4.6 times the start; 5.1 /
  5.9 % of the neighbouring faces meet at more than 90 degrees (p99 158–160); triangle area median × 1.16–1.20, p99
  × 42, largest × 2354 / 5999; 18 / 25 % of the carried area spans gaps the material has left (the open mouth is
  crossed by stretched faces whose ends still sit on material); and 12.8 / 9.6 % of the particles' own outer layer
  is more than two lattice steps from any tracer: that surface was made from interior material (D12's addendum:
  11 % of the final outer layer came from deeper than eight steps). At raw frame 480 it is already so (32–33
  degrees, 2.3 % folds, 5–6 % of the area over gaps).
  (1) Half: the median tracer is 0.61 lattice steps from its nearest particle, but the 99th percentile is 2.4,
  not 2. (2) Refuted, except the edge width: the picture is worse than the particles' in normal error and in
  silhouette. (3) Confirmed: 164k and 655k agree within 1 degree and 0.1 px. (4) Confirmed (median 1.16–1.20; p99
  42).
  Reading: over a whole run the particle motion is not a smooth deformation of the source's boundary. The body
  opens gaps and brings interior material to the surface, so a surface fixed at the start folds, stretches over
  the gaps and misses a tenth of the final surface. A dense surface has to be taken again from the current body
  (per window, or per frame), and then it holds no more than the particles do at that moment. The tracers follow
  an interpolation of archived particle displacements over 4–20 frames, not the simulation's grid velocity at
  every step; that the surface changes its material is measured on the particles alone (D12's addendum), not only
  through this interpolation.
- **D14, what surface relief a window's rollout can hold and make, by wavelength (open loop: no objective, no
  optimiser; pre-registered 2026-10-02 12:28 CDT, before the run; `scripts/probes/settled/subcell_response.py`).**
  Whether a feature below a cell is missing because the physics cannot make it or because the objective cannot
  see it is settled in two steps: this one asks the rollout alone. A slab of 8 × 3 × 4 cells at rest (dx 0.3024,
  the run's material, time step, 20 controlled + 20 released steps, outer layer, relaxation, bonds and
  assimilation, built as `window/setup.py` builds a window), sampled as the pipeline samples (one jittered
  particle per voxel) at the particle spacing of a 300k run (0.054 wu = 0.18 cells; 17k particles) and of a 2.4M
  run (0.027 wu; 133k). Relief y = A sin(2πx/λ) with λ = 4, 2, 1, 0.5, 0.25 cells, measured as the amplitude of the
  least-squares sinusoid at λ through the top layer's heights. A0 hold: the relief (A = λ/8) is in the sample, no
  control, four windows. A1: a flat slab, u = 0.5 spacings × sin on the outer layer for one window, then two free
  windows. A2: a flat slab, dFc_yy = 0.02 sin on every particle for the controlled half of one window. Three limits
  are in play: the grid (cubic B-splines, nodes dx apart), the particles (no wavelength below about two spacings:
  0.36 cells at 300k, 0.18 at 2.4M), and the layer relaxation (every step each outer-layer particle moves 1/20 of
  the way toward what its 24 layer neighbours within 2 spacings share). Already on record (kernels.py, 2026-09-19,
  40k bunny): the control stress acts at the grid's correlation length and a force on one particle moves it 2.4 %
  in a window. Predictions, from the relaxation read as a Gaussian filter of width 2 spacings applied 40 times at
  1/20: A0, relief left after one window: at 300k spacing 0.8, 0.45, 0.17 for 4, 2, 1 cells (± 0.15) and the 0.5
  and 0.25-cell relief not in the sample at the start (under half of what was asked); at 2.4M spacing 0.94, 0.8,
  0.45, 0.17 for 4, 2, 1, 0.5 cells. A1: u makes at least 0.8 of its command at 4 cells and at most 0.5 at 1 cell
  (300k), and what it makes then decays as in A0. A2: relief per wavelength, against the 4-cell row: at least 0.5
  at 2 cells, at most 0.2 at 1 cell, at most 0.05 at 0.5 and 0.25 cells, the same at both spacings (the grid, not
  the particles, sets it). The closed-loop step (the frozen recipe on a target with the same ridges) follows for
  the wavelengths that pass here.
  **Result (2026-10-02 12:30 CDT; `output/gpu/d14/d14.log`, and `d14_norelax.log`: the same with the layer
  relaxation switched off, a control arm added after the first reading).** The limit is counted in particle
  spacings, not in cells, and the part that sets it is the layer relaxation.
  A0, relief left after windows 1 and 4, by wavelength in spacings (300k spacing / 2.4M spacing agree at equal
  spacings): 45 spacings 1.00, 1.00; 22 spacings 0.99, 0.95 and 0.98, 0.95; 11 spacings 0.84, 0.55 and 0.84, 0.56;
  5.6 spacings 0.33, 0.11 and 0.45, 0.20; 2.8 spacings 0.31, 0.20 and 0.23, 0.05. The sample itself carries 0.97–0.99
  of the asked relief at 22 spacings and more, 0.93 at 11, 0.75 at 5.6, 0.40–0.56 at 2.8. In cells: at 300k
  spacing 11 spacings are 2 cells and 5.6 are 1 cell; at 2.4M they are 1 cell and half a cell. With the
  relaxation off, 1.00 at every wavelength and window: at rest the rollout moves nothing else.
  A1, the u channel, share of the commanded 0.5 spacings (end of the controlled half / window's end / after two
  free windows): 22 spacings 0.97 / 0.96 / 0.87–0.89; 11 spacings 0.92 / 0.79 / 0.52; 5.6 spacings 0.65 / 0.28 /
  0.04; 2.8 spacings 0.60–0.63 / 0.22–0.25 / 0.05–0.09. With the relaxation off: 0.97–0.98 at every wavelength,
  kept through the release and the two free windows.
  A2, the dFc channel at its clip for one window, relief at the window's end with the relaxation off (wu; 300k /
  2.4M spacing): 4 cells 0.0165 / 0.0150; 2 cells 0.0051 / 0.0020; 1 cell 0.0040 / 0.0026; half a cell 0.0007 /
  0.0006; a quarter cell 0.0001 / 0.0004. A ridge of A = λ/8 is 0.15, 0.076, 0.038, 0.019 and 0.009 wu: the stress
  channel makes about a tenth of it per window at one cell and above, a thirtieth at half a cell, and nothing
  measurable at a quarter. With the relaxation on, the rows below one cell are at the fit's noise (0.001 wu).
  Predictions: A0 refuted in size at long wavelengths (0.99 and 0.84 at 22 and 11 spacings against 0.8 and 0.45:
  the relaxation removes only what the neighbourhood does not share, so it is milder than a plain filter there)
  and at the edge of the band at 5.6 spacings (0.33 against 0.17 ± 0.15); the sample lacking the 2.8-spacing
  relief: confirmed (0.40). A1: 0.97 at 4 cells confirmed; at 1 cell 0.65 in the controlled half (predicted at
  most 0.5) and 0.28 at the window's end; the decay follows A0. A2: mostly refuted: per wavelength the 1-cell
  relief is 0.7–0.97 of the 4-cell relief at the window's end, not under 0.2; the fall comes at half a cell.
  Reading: the rollout can carry relief down to what the particles can sample, and the u channel can make it; the
  stress channel works down to about one cell. What removes relief of 6 spacings and less within a window or two,
  and half of the 11-spacing relief in four windows, is the outer-layer relaxation, which counts anything
  narrower than its neighbourhood (24 layer neighbours, 2 spacings) as sampling roughness. At 300k that is 1–2
  cells (0.3–0.6 wu); the dragon's teeth (about 0.06 wu, one spacing) are below even the sample. So the dense
  surface samples of the resampling proposal would not add relief: the relief is taken out in the forward model,
  at a scale set by the particle spacing. Measured on a slab at rest; a moving body adds stress and the bonds.
- **D16, is the layer relaxation needed for a smooth surface: the 40k gallery with it on and off (the last
  diagnostic on this question, pre-registered 2026-10-02 15:10 CDT, before launch; `layer_roughness.py`,
  `tmp/d16.sh`, `tmp/d16_eval.py`).** D15: the relaxation is what removes relief of 11 spacings and less. What it
  was put in for, particle-scale roughness, was never measured with it off (D3b on 2026-09-29 read the silhouette,
  the thin share and the run length, on the earlier objective). 19 meshes, 40k, seed 97, the frozen recipe: ON =
  repo_r34, OFF = repo_r34x (the relaxation's rate zero, nothing else; a diagnostic arm, not a candidate). Both
  arms in this batch, because no earlier run kept its final state. Measured: (1) silhouette IoU, thin share,
  chamfer; (2) the outermost layer only: the plane residual the relaxation removes (d − d̄ over 24 layer
  neighbours), for the source sample, the target sample and the end state; (3) the end state's signed offset from
  the target MESH, split by scale over the layer: at most about 4 spacings, 4–11 spacings, larger (the target's own
  relief is in the mesh, so it is not counted as roughness; the target sample's own value is the level of a
  perfect sample); (4) committed, rejected and null windows, minutes, the accepted step of the last ten windows,
  the end jitter, the window at which OFF reaches ON's final merit. One bunny ON run for the probe's check gave:
  the removable residual 0.285 spacings in the source sample, 0.308 in the target sample, 0.055 at ON's end; the
  ≤ 4-spacing offset 0.223 in the target sample, 0.137 at ON's end.
  Readings fixed before the run. A, the relaxation is not needed for smoothness: OFF's ≤ 4-spacing offset is at
  most 1.1 times the target sample's own on at least 14 of 19 meshes. B, it is needed but its definition is wrong:
  OFF's ≤ 4-spacing offset is above 1.5 times the target sample's on at least 10 of 19. C, the cost is the
  optimiser's: OFF's silhouette IoU is higher on at least 14 of 19 and its median window count is at least twice
  ON's. Origin: OFF's removable residual at or below the source sample's on at least 14 of 19 means the roughness
  left is the sampling's, not added by the morph. Predictions: A and C hold (OFF's ≤ 4-spacing offset about the
  target sample's, 0.2–0.3 spacings, against ON's 0.14; OFF's removable residual 0.25–0.40, near the source
  sample's; silhouette higher on at least 15 meshes; three to five times the windows); OFF's 4–11-spacing offset
  lower than ON's on at least 12 of 19 (the features the relaxation erases); beast or one other mesh freezes in
  one arm.
  **Result (2026-10-02 16:30 CDT; 38 runs, no guard; `output/gpu/d16`, `python3 tmp/d16_eval.py`).** Readings A and
  C hold; B does not.
  Geometry, OFF against ON: silhouette IoU higher on 17 of 19 (median +0.0052; bunny 0.9757 → 0.9849, dragon
  0.9752 → 0.9804, A 0.9787 → 0.9871, maxplanck 0.9769 → 0.9857), thin share lower on 15 of 19 (median −1.9 points;
  dragon 14.2 → 9.9, maxplanck 9.2 → 3.8, spot 9.7 → 4.8), chamfer median −0.0016. The two exceptions: beast froze
  under OFF at window 9 (0.7900, ten null windows, the parked ejection), and C stops at 19–20 windows in both arms
  (0.9760, 0.9750).
  Roughness of the outermost layer, in spacings (medians over the 19 meshes). The ≤ 4-spacing offset from the
  target mesh: target sample 0.20, ON 0.12, OFF 0.19; OFF over the target sample's 0.96, at most 1.1 times it on
  17 of 19 (the other two are beast, frozen, and C, 1.24), above 1.5 times on none; OFF over ON 1.63. The plane
  residual the relaxation removes: source sample 0.286, target sample 0.292, ON 0.061, OFF 0.316; OFF at or below
  the source sample's on 2 of 19 (it sits about a tenth above it). The 4–11-spacing offset: target sample about
  0.19, ON about the same, OFF over ON 1.14, lower than ON on none.
  The run, OFF against ON: committed windows median 289 against 46 (OFF runs to the 300-window budget on 15
  meshes); minutes summed 333 against 89; null windows 156 against 30; rejected windows 7 against 97; the accepted
  step of the last ten windows about 1e-4 against 6e-4 (maxplanck and nefertiti end at a zero step); end jitter
  1e-7–5e-7 against 3e-6–9e-6. OFF reaches ON's final merit at window 6–26 (bunny 10 against ON's 46 windows,
  dragon 26 against 65, fandisk 8 against 25): to ON's own quality OFF is the faster arm, and then it goes on
  improving and does not stop. Render: λ of the first window 0.248 and 0.249, median g_share 0.88 and 0.81.
  Predictions: A and C confirmed (the ≤ 4-spacing offset 0.17–0.24 under OFF, the removable residual 0.28–0.34,
  silhouette higher on 17); the windows rose by 6.2, more than the predicted three to five, bounded by the budget;
  beast froze under OFF as predicted. Refuted: OFF's 4–11-spacing offset is not lower than ON's on any mesh (in
  this band both arms sit at the target sample's own level, so at 40k the gallery's lost relief does not show
  there; the silhouette and the thin share show it). The origin reading is not met by the letter: OFF's
  removable residual is a tenth above the source sample's on 17 of 19.
  Reading: with the relaxation off the surface is as rough as a perfect sample of the target, no rougher (0.96 of
  it at 4 spacings and less); with it on the surface is smoother than a sample of the target can be (0.6 of it),
  which is the over-smoothing D14 and D15 measured as lost relief. The relaxation is not needed for smoothness.
  What it does provide is the stop: with it the run ends after 20–80 windows; without it the same recipe reaches
  that quality sooner and then keeps finding about 1 % a window until the budget ends (D5's slow tail). The
  open part is therefore the stopping and step-length behaviour without the relaxation, and beast's ejection, not
  the surface.
- **D99, the drag scales the whole velocity field (pre-registered 2026-10-05 16:04 CDT, queued behind D97's dragon
  readings; code: `mpm/kernels.py` k_p2g; the record's angular momentum budget `L_start`, `L_grid`, `L_jump`
  (`window/telemetry.py`); server `repo_r85` = HEAD with this; `tmp/d99.sh`, `output/gpu/d99`).** After D98 the
  rotation left is the grid's (bunny L 0.0075°, all of it). The budget per step (APIC's angular momentum, x × v
  plus the affine part with D = dx²/3) on a 40k bunny run (12 windows, `tmp/ang_budget.py`): the grid's steps
  changed it by 23.8 over the run (path 75), the position updates below the grid by 3.8; with `--drag 0` the
  grid's change is 0.15 (path 0.29), 150 times less. P2G damps the particle's velocity v by the drag
  (m v (1 − dt drag)) and not its affine part m C: the drag took the translation and left the spin, which turns
  a body whose local spins and bulk motion balance. The definition: the drag scales the particle's whole
  velocity field v + C (x − x_p), so zero momentum and zero angular momentum stay zero and the dissipation
  stays. One factor. Test: a ball in rigid rotation with APIC's affine term and no elasticity keeps its angular
  momentum at the drag's factor per step to 1e-4 (without drag: conserved); it fails on the code before by
  3 %. Tests: 284 passed, 2 skipped (one contract test with a null first window on the tiny cloud failed once
  in the full run and passed six of six alone, as before the change).
  Runs: the defaults, L (GPU 3) and P (GPU 2), bunny then dragon 300k, read with `runeval3.sh`, against D98's
  runs. Then the 40k gallery (render arm, 19 meshes) of this code against D97's code (`repo_r82`), for the two
  definition changes together.
  Criteria: the body's net rotation over the run falls by at least half in both arms on both meshes and the
  record's L_grid to the size of L_jump or less; the render arm at or ahead of its twin on every momentum
  measure; the display and yardstick at least 10 % ahead of the twin; neither arm's display behind D98's by
  more than the repeat spread (0.0013 IoU); kinetic energy at the end not above D98's beyond the spread (the
  spin the drag now damps was dissipated before, a little more kinetic energy may remain).
  Expectation: the rotation falls to the jumps' share (the position updates still move r and leave v),
  about a sixth of now; nothing else moves.
- **D98, the relaxation moves the body neither along nor about any axis (pre-registered 2026-10-05 14:26 CDT, queued
  behind D97's bunny readings; code: `mpm/kernels.py` k_layer_relax / k_layer_project, `mpm/traj.py`; server
  `repo_r83` = HEAD with this; `tmp/d98.sh`, `output/gpu/d98`).** D97's cause (bunny, both arms): the layer's
  relaxation carries 94 % of the body's net translation (L 2.82e-4 of 3.00e-4 wu, P 3.21e-4 of 3.42e-4) and a
  quarter of its net rotation (24–27 %; the grid's advection carries the rest of the rotation, 73–76 %, and 6 % of
  the translation; the minimum spacing and u nothing). The definition changes as u's did in D94: at every step
  the relaxation's normal displacements s = −frac (d − d̄ − ref) lose their part along the six rigid modes of the
  layer (n and r × n, r about the window's starting centre of mass; the inverse Gram matrix of the modes frozen
  per window): on the tape, one kernel sums s onto the modes (atomic adds), the projection kernel subtracts it.
  A position constraint below the grid then moves no mass as a whole and turns none. No constant. Tests: the
  relaxation alone leaves a rough slab's centre of mass and orientation (net/gross below 1e-4 and 1e-3); the
  adjoint against finite differences with the layer on; 282 passed, 2 skipped.
  Runs: the defaults, L and P, bunny then dragon 300k on GPUs 0 and 1, read with `runeval3.sh`, against D97's
  runs (the code before) and their record.
  Criteria: the body's net translation over the run (the record's body vectors summed) falls at least five times
  in both arms on both meshes and the relaxation's part to the record's noise; the net rotation falls by about
  its relaxation share; the render arm at or ahead of its twin on every momentum measure (centre of mass
  largest and final, its velocity, net rotation, what still moves at the end, kinetic energy at matched
  windows); the display and yardstick at least 10 % ahead of the twin over the common range after window 10;
  neither arm's display behind D97's by more than the repeat spread (0.0013 IoU) and the field's roughness not
  above D97's.
  Expectation: the translation falls to the grid's 6 %; the rotation by a quarter, the grid's part remaining
  (D99 would measure where the grid's rotation comes from: the support gate on the APIC term, the walls and
  the velocity clamp, or the position updates' change of r × v); the surface is as smooth (the rigid part of the
  relaxation's displacement is its mean over the layer, not its rough part).
- **D97, where the remaining drift comes from (a measurement; pre-registered 2026-10-05 13:52 CDT at launch, the bunny
  relaunched 13:55 with the vectors in the record (`*_vcom`, `*_vrot`: summed over windows they attribute the net drift); code:
  the record `grid_*`, `spacing_*`, `relax_*`, `body_*` beside `u_*` (`window/telemetry.py` below_grid_record);
  server `repo_r82` = HEAD with the exact exterior speed-up and this record; `tmp/d97.sh`, `output/gpu/d97`).**
  After D94 the render arm is still behind its twin on the dragon's centre of mass (0.0096 against 0.0074
  pitches) and rotation (0.0178° against 0.0168°), and the bunny's render arm drifts late while the body barely
  moves. Every committed window now records the body's net translation (world units) and rotation (radians)
  over the window by each position update: the grid's advection dt v, the minimum spacing's push (recomputed
  from each step's positions with the frozen lists), u (zero since D94), and the rest of what the steps move
  outside the grid (the layer's relaxation, and the bonds' re-coupling where a particle is decoupled); body =
  their sum, checked against the centre of mass between consecutive windows (test). Runs: the defaults, L and
  P, bunny 300k now, dragon 300k once D95's dragon archive is gone; read with `runeval3.sh` (their display and
  momentum are also a second repeat of D94's code, for the spread). Read: per window and summed over the run,
  each part's translation and rotation against the body's, their correlation over windows, both arms.
  Expectation: the grid's part is near zero in both (it conserves momentum but for the boundary and the
  velocity clamp); the relaxation's part carries most of the body's drift (it acts on every layer particle in
  every step, along normals that do not balance where the layer is uneven), the spacing's part less (pairs
  push each other symmetrically but for the pairs in one particle's frozen list only); the render arm's
  relaxation part is larger where its layer is rougher. What would follow: the same definition as D94's for
  the part that carries the drift (free of the six rigid modes over the layer, or pairwise-symmetric lists for
  the spacing).
  **Result (bunny runs 13:55–14:19, dragon 14:43–15:28; read 14:20 and 15:40; `tmp/drift_parts.py`): the
  relaxation is the drift.** Net translation over the run, each part's share of the body's (L / P): bunny
  relaxation +0.94 / +0.94, grid +0.06 / +0.06, spacing 0.00 / +0.01, u 0; dragon relaxation +0.98 / +0.98,
  grid +0.02 / +0.02. Net rotation: bunny grid +0.73 / +0.76, relaxation +0.27 / +0.24; dragon grid +0.86 /
  +0.75, relaxation +0.14 / +0.26; spacing and u nothing. The parts sum to the body within 3e-10 wu. The
  render arm's relaxation part is the twin's (bunny 2.8e-4 against 3.2e-4 wu, dragon 3.4e-4 against 3.3e-4):
  the drift is the definition's, not the render's. Second repeat of D94's code (bunny, common range after
  window 10, L against P): display 1 − IoU −28 / −30 / −29 %, difference −17 / −28 / −18 %, yardstick
  silhouette −48 %, shading −24 % (particle cloud −4 %); momentum: centre of mass 0.0061 against 0.0070
  pitches, net rotation 0.0092° against 0.0088° (D94's pair: 0.0056° against 0.0113°: the arms' order on the
  rotation flips between repeats), what still moves 0.0032 against 0.0030, kinetic energy last ten 6.1e-6
  against 5.3e-6. Wall: the dragon 39.4 min against D94's 44–47 (the exact exterior speed-up). D98 follows.
- **D96, where the wall time of a 300k run goes on the defaults (a measurement; pre-registered 2026-10-05 12:56
  CDT; the user: "왜 속도가 느린거야? … 30-40분 쯤 걸리는 거 같은데"; server `repo_r80`, `tmp/d96.sh`,
  `tmp/time_rows.py`, `output/gpu/d96`).** From the records: the 300k bunny took 8.7 min on 2026-10-02 (D19,
  96 px: 44 windows, 10.7 s a window: start 0.9, gradients 7.0, line search 2.3, commit 0.5) and takes 22–28 min
  now (D93, D94: 67–89 windows, 17–20 s a window: start 3.0–3.2, gradients 6.9–7.4, line search 5.5–6.9, commit
  0.4); the dragon 39–48 min (95–132 windows, 21–23 s a window). One gradient costs what it did (0.94 s against
  0.90); one line-search trial costs three times as much (0.60–0.75 s against 0.24) and the start of a window
  three and a half times; the runs are 1.5–2 times longer. Run: the bunny 300k defaults with `--profile` (the
  timed sections of every evaluation and gradient per window: MPM tape and evaluation, transport geometry,
  render, exterior, cleanup, the three adjoints) on GPU 1 after D95's effect map. Read: the seconds per window
  of each section and their calls, set against what each adopted change since D19 adds (the render picture
  following N, 96 → 188 px, D89; the exterior render terms, D62; the transport grid following N, R2; the
  minimum spacing, D70; the render weight at every window, D81, and the window count).
  Expectation: the trial's extra half second is the render (four times the pixels) and the exterior (a surface
  per evaluation); the start's extra two seconds the finer transport grid; the extra windows come from the
  stopping (the merit keeps finding a little more each window), not from the cost of a window.
  **Result (run 13:15–13:39, 22.6 min, 72 windows; read 13:42; `output/gpu/d96/prof_rows.txt`).** Per window
  (windows 5 on, 15.8 s by the record's clocks): the exterior's search for its discs (Zhu–Bridson field on the
  fixed lattice, `Tracked`, timed inside the render) 5.19 s, 2.3 searches a window at 2.29 s each; the render
  pictures themselves 0.59 s (21.9 evaluations, 27 ms each); the three adjoints 5.54 s (render 2.03, physics
  1.77, cleanup 1.74; 8 each, 0.22–0.25 s); the transport geometry 2.18 s (its Sinkhorn solves 1.57, the
  surface proximity 0.56); the MPM 2.06 s (evaluations 1.29, tape 0.77). The expectation is refuted on its
  main part: the render's four times the pixels costs half a second a window; the window's extra five seconds
  against D19 are the exterior's searches, one at the window's start and one or two in the line search when a
  tenth of the discs has moved more than half a lattice pitch. A run is 22.6 min = 72 windows × 18.8 s: against
  D19's 8.7 min (44 × 10.7 s), 1.6 times the windows and 1.75 times the window, the exterior making most of the
  latter. One gradient (0.94 s) is unchanged.
  Inside one search (`tmp/ext_time.py`, the bunny's kept frames, alone on a GPU, 1.7 s): the field at the 3.2M
  lattice nodes 1.22–1.29 s, of which its gradient (computed and thrown away: the corners need the sign alone)
  half (the value alone 0.64–0.68 s, bitwise the same); the projection of the 200k crossed cells' centres
  0.38–0.40 s; the rest 0.08 s. The field now reads the value alone at the corners (`ZhuBridson(q, grad=False)`,
  test: the same values under no_grad): the search 1.77 → 1.16 s with bitwise the same discs (points, normals,
  slopes, particle lists; `tmp/ext_same.py`), about 1.4 s of a window's 16–19 s. Tests: 280 passed, 2 skipped
  (server `repo_r81`).
- **D95, where the render gradient acts (a visualisation on the defaults; pre-registered 2026-10-05 12:16 CDT,
  chains launched 12:29; the user: "랜더 Gradient가 어떻게 영향을 끼치는지도 그 값들 visualization 한 번 해
  줄레?", then "300K 로 돌려서 … gradient 시각화 랜더를 하란 말이었어. 마치 loss_viz.mp4처럼"; probes
  `loss_video.py` (now a fourth panel: the render push along the normal), `render_influence.py`; server
  `repo_r80`, `tmp/d95.sh`, `output/gpu/d95`).** The render arm of the D94 code with `--grad_dump`, bunny 300k
  on GPU 0 and dragon 300k on GPU 2, each launched when its D94 chain has finished (the dragon also after the
  bunny's full archive is gone: one at a time on the disk). The splat video of the morph (every third frame):
  shape error, render pull |λ g_render|, physics pull |g_physics|, and the render push −λ g_render · n (red out,
  blue in), each frame coloured by its window's first gradient, the losses below. At every window's first
  gradient: the weighted render term's position gradient λ(∇silhouette + ∇shading) and the physics objective's
  (everything but the cleanup) on every particle; each one's push along the outward normal; the particle's
  signed offset from the target surface; the linear-response rollouts (each channel's control gradient alone,
  scaled to the window's accepted change, for dFc and for u). Read per window: the render's share of the step,
  the two gradients' cosine, whether each push points to the target (Σ push·(−offset) / Σ |push||offset| on the
  outer layer), the outer layer's move under each channel alone, the gradients' share on the outer layer and in
  their top 5 % particles.
  Expectation (B2, D81): the render gradient sits on the outer layer (over 90 % of its squared norm) and on few
  particles (top 5 % over 60 %), at the silhouette's edges and the shading's creases; it points to the target
  there (corr > 0); it moves the layer more through u than through dFc; the physics gradient is spread through
  the body and makes the early windows' moves. Caveat: the u rollouts take the raw gradient, not projected free
  of the rigid modes as D94's step is, so their moves carry a rigid part the run removes.
  **Bunny (run 12:55–13:29, 84 windows of which 79 committed; read 13:40).** The first video coloured every
  window on window 1's scale; λ falls tenfold (0.255 → 0.0157 at window 12 → 0.001 from window 40), so from
  window 20 on both pull panels were uniformly green and the push grey: the video is redrawn with each window
  on its own 99th percentile (`loss_video.py` now so; from the kept frames, every 12th, and every window's dump),
  `output/video_2026-10-05/gradient/gradviz_bunny300k_w.mp4`. Read per window (`influence_bunny/rows.txt`):
  the render gradient is 85–95 % on the outer layer (physics 9 % at window 0, 55–76 % from window 8, 48–50 %
  at the end) and 98–99.6 % of it in the top 5 % particles in every window (physics 29 % at window 0, 86–93 %
  later): it acts on few particles, in patches over the surface, not along the silhouette only. It points to
  the target: toward +0.51 at window 0 (physics −0.02: on the sphere the transport moves material along the
  surface), +0.73–0.82 at windows 4–10 (physics +0.75–0.90), +0.36–0.44 from window 20 to the end, level with
  the physics (+0.30–0.45). The two gradients are nearly orthogonal: cosine +0.05–0.21 in the first ten
  windows, +0.01–0.03 from window 20. At an equal step, the render direction moves the outer layer as far as
  the physics direction through dFc (6.8 against 4.3 pitches at window 0, 0.02–0.05 each late), and through u
  further in the late windows (0.03–0.05 against 0.02–0.04). The render's share of the step: 0.33 at window 0,
  0.68 at window 3, 0.29–0.39 from window 20. The effect map (D94's arms, independent sample) is at the
  sampling floor: the outer layer's mean |offset| 0.265 against 0.272 pitches at the last frame, beyond one
  pitch 0.61 % against 0.67 %; per particle the two arms look alike, the render's gain is in the silhouette
  and shading, not in the mean offset.
  **Dragon (run 13:44–14:36, 110 windows; video 14:43, read 14:47).** `gradviz_dragon300k.mp4` (every third
  step, each window on its own scale). The same picture as the bunny: the render gradient 85–95 % on the outer
  layer and 98.8–99.5 % in its top 5 % particles (physics 7 % → 47–79 % and 15 % → 63–93 %); toward the target
  +0.11 at window 0, +0.71 at window 10, +0.48–0.62 later (physics +0.09, +0.35, +0.18–0.36); cosine with the
  physics +0.06–0.10 (the outer layer +0.09–0.30); the render's share 0.33 → 0.56 at window 10 → 0.31–0.37.
  **Do the two gradients conflict? (the user's question, read 14:55; `tmp/conflict.py`).** On the control, the
  cosine of the two gradients before PCGrad: bunny median +0.05, below zero in 8–19 of 62–81 windows (at most
  −0.07); dragon median +0.23, below zero in 2–3 of 110–115 (at most −0.10); after PCGrad never below zero, so
  no accepted step works against the physics objective. Per particle it is otherwise: the render pushes a layer
  particle along its normal against the physics push on 10–26 % of its weight in windows 0–10 and on 40–56 %
  from window 20 on (dot products below zero 12–38 % of their total size): once the shape has arrived the two
  objectives place the surface differently below a pitch (the transport after one target sample's points, the
  render after the mean picture of eight), and the end state is their balance. PCGrad removes the global
  conflict only. Open: whether this local balance is the render arm's 10 % more motion at the end (D94).
  It is not (read 15:00; `tmp/end_motion.py`, the last 20 kept-frame pairs of D94's and D97's arms): 82–89 % of
  the squared motion at the end is on the outer layer (6–9 % of the particles), and there along the normals
  (0.019–0.028 pitches a pair against 0.004–0.010 tangential), in both arms alike: bunny layer 0.0232 against
  0.0223 (D97), 0.0204 against 0.0203 (D94), dragon 0.0292 against 0.0293; the interior 0.002–0.0045. What still
  moves at the end is the layer breathing along its normals, the relaxation flattening it every window and
  the material restoring it, in the twin as much as in the render arm.
- **D94, u moves the body neither along nor about any axis (pre-registered 2026-10-05 12:04 CDT at launch; code:
  `window/solve.py` (WindowOptimizer.free_of_rigid); server `repo_r80` = HEAD with this; `output/gpu/d94`).**
  D93's cause: u's displacement of the layer, g u n per window, is a position update outside the grid's momentum
  balance, and it is where the body's drift comes from. The definition of u changes: after every step u loses
  the part of its displacement along the six rigid modes of the layer (Σ g u n = 0, Σ g u (r × n) = 0, r about
  the window's starting centre of mass), a projection with the 6 × 6 Gram matrix of those modes, so that the
  control below the grid conserves linear and angular momentum as the grid does. Twenty lines; no constant.
  The target needs no net translation or rotation (source and target share their centre of mass and frame),
  so nothing the morph needs is in the removed part. Test: u_com and u_rot under 1e-6 in every window of a
  small run (it fails on the previous code).
  Runs: the defaults, render arm (L) and physics-only twin (P), bunny and dragon 300k, read with
  `runeval3.sh`, against D93's runs of the code before.
  Criteria: the render arm at or ahead of its twin on every momentum measure on both meshes (centre of mass
  largest and final, its velocity over the last ten windows, net rotation, what still moves at the end,
  kinetic energy over the last ten windows), and every display and yardstick measure still at least 10 %
  ahead of the twin; neither arm's display behind D93's by more than the repeat pair's spread (0.0006 IoU,
  0.0004 difference).
  Expectation: the centre of mass's drift falls by an order of magnitude in both arms; the net rotation falls
  by a half to two thirds; the bunny's render arm turns no more than its twin; the display is unchanged.
  Risk: the projection takes from u in the windows where the transport moves material as a whole across the
  body, and the early transit is slower.
  **Result (runs 12:04–12:54, readings done 13:43, read 13:46; `output/gpu/d94`, `tmp/allframes.py`,
  `tmp/matched_rest.py`): u is free of the rigid modes (u_com, u_rot ≤ 1e-12 in every window), the drift of both
  arms falls 2–6 times and the display holds; the criterion "the render arm at or ahead of its twin on every
  momentum measure" is not met, on small remainders.** Windows: bunny L 58 commits (D93 73), P 80; dragon L
  113, P 107 (44 and 47 min). Momentum, L against P (centre of mass largest / end in pitches; net rotation;
  what still moves at the end, pitches a pair; kinetic energy last / last ten): bunny 0.0044 / 0.0044 against
  0.0071 / 0.0071, 0.0056° against 0.0113°, 0.0032 against 0.0029, 3.4e-6 / 7.2e-6 against 4.0e-6 / 5.9e-6;
  dragon 0.0096 / 0.0096 against 0.0074 / 0.0074, 0.0178° against 0.0168°, 0.0063 against 0.0068, 2.2e-5 /
  2.4e-5 against 2.4e-5 / 2.7e-5. The centre-of-mass velocity's last ten windows and the net-over-gross ratios
  of the last 20 pairs are L's on both meshes; its largest and the run's linear ratio are P's on the dragon.
  Against D93 (the code before): bunny L's path 0.0251 → 0.0052 pitches and rotation 0.0125° → 0.0056°, its end
  0.0043 → 0.0044 (it drifts late, 0.0008 → 0.0042 between frames 720 and 2160, while the body moves 0.003–0.005
  pitches a pair: not u); dragon L's end 0.019 → 0.0096, rotation 0.072° → 0.018°, P's 0.022 → 0.0074,
  0.105° → 0.017°. The expected order of magnitude is met by neither; the remainder is the layer's
  relaxation and the minimum spacing, the other two position updates below the grid (D93). The bunny's rest
  is L's at matched windows by 10 % (kinetic energy over commits 49–58 7.2e-6 against 6.5e-6) while its move
  per window is smaller (3.9e-4 against 4.3e-4); the "last ten" gap of 21 % is partly where each run stopped.
  Display, the common range after window 10 against the independent sample, L against P (1 − IoU / mean
  difference): bunny front −25 % / −17 %, crop −27 % / −25 %, far side −28 % / −17 %; dragon front −28 % /
  −14 %, crop −38 % / −24 %, far side −30 % / −14 %. Yardstick, exterior: bunny silhouette −47 %, shading
  −22 %; dragon −58 %, −32 %; on the particle cloud the bunny's shading is −8 % (dragon −21 %). Detached sets
  in the dragon's last 60 kept frames: L 6.8 (most 22) against P 9.0 (25). The last 20 frames: bunny front
  difference −10 %, far side −9 %; the rest at least −18 %. Against D93's display (last 20): bunny L crop
  0.9795 against 0.9806, dragon L front 0.9899 against 0.9908 and crop 0.9774 against 0.9792; but the twin moved
  as much (dragon P crop 0.9708 against 0.9721), so the repeat spread at 300k is about 0.0013, not 0.0006. Kept:
  the u control conserves what the grid conserves, and both arms drift less. Next (D97): the relaxation's and
  the minimum spacing's net translation and rotation per window, measured as D93 measured u's.
- **D93, where the render arm's small excess of drift on the bunny comes from (a measurement on the defaults;
  pre-registered 2026-10-05 09:33 CDT at launch; the user: "렌더러가 물리 only 를 이기는 방향으로 수정 하고, …
  (모멘텀, metric, loss 전부)"; code: the record `u_com`, `u_rot` (`window/solve.py`, 67f900a); server `repo_r79`;
  `output/gpu/d93`; `tmp/d93.sh`).** The last measures on which the render arm is behind its physics-only twin
  are the bunny's net rotation over the run (LK 0.0130°, LG 0.0127°, LF 0.0139° against PN 0.0101°, PA 0.0112°)
  and its centre of mass at the end (LK 0.0041 pitches against PN 0.0024, PA 0.0033); every other momentum
  measure is the twin's or better, and on the dragon all are. The grid conserves momentum; three position
  updates do not by construction: the u channel (the layer moved by u n over a window), the layer's relaxation
  (by −(d − d̄) n per step) and the minimum spacing where a pair is in one particle's frozen list and not in the
  other's. The render term acts mostly through u. The record now carries u's net translation and rotation of
  the body per window. On a three-window 40k run u's net translation per window (2e-5 to 5e-5 wu) is the size
  of the body's centre-of-mass drift per window.
  Runs: the defaults, render arm (L) and physics-only twin (P), bunny and dragon 300k; read with
  `runeval3.sh`. Measured: per window u's net translation and rotation against the body's (from the record's
  centre of mass and the kept frames' rotation), summed over the run, both arms.
  What would follow: if u carries the excess, u's displacement field is made free of net translation and
  rotation (projected onto the complement of the six rigid modes of the layer), a definition of the control
  that respects the conservation the grid already has.
  **Result (runs and readings done by 11:20 CDT, read 12:02; `tmp/u_drift.py`): u is the source of the
  drift, in both arms.** Summed over the run, u's net translation against the body's centre-of-mass path:
  bunny L 1.19e-3 against 1.08e-3 wu (ratio 1.10, correlation over windows +0.98), P 1.85e-3 against 1.65e-3
  (1.13, +0.99); dragon L 2.43e-3 against 2.35e-3 (1.03, +0.99), P 2.86e-3 against 2.83e-3 (1.01, +1.00). u's net
  rotation against the body's: bunny L 0.48, P 0.62; dragon L 0.65, P 0.66 of it (correlation +0.37 to +0.91);
  the rest of the rotation is in the other position updates and the grid. The repeat of the render arm's
  excess on the bunny: net rotation 0.0125° against 0.0087°, centre of mass at the end 0.0043 against 0.0023
  pitches; on the dragon the render arm is ahead on both (0.072° against 0.105°, 0.019 against 0.022). The
  display, against the independent sample, repeats D91 (bunny L front 0.9912 / 0.0074, crop 0.9806 / 0.0075,
  far side 0.9908 / 0.0067 against P 0.9883 / 0.0083, 0.9736 / 0.0102, 0.9884 / 0.0078; dragon L 0.9908 /
  0.0089, 0.9792 / 0.0128, 0.9911 / 0.0093 against P 0.9871 / 0.0105, 0.9721 / 0.0162, 0.9878 / 0.0110). D94
  follows.
- **D92, D91 on the 40k gallery, read against an independent sample (pre-registered 2026-10-05 01:25 CDT at
  launch; `tmp/d92.sh`, `tmp/d92.queue`, `tmp/gallery_ind.py`; `repo_r77`; `output/gpu/d92`).** Three arms on
  the 19 meshes, all with the minimum spacing and D81's weight and D89's one resolution: SK (eight target
  samples), SG (one), SP (the physics-only twin). Each end state is read against its own target sample and
  against an independent 40k sample of the mesh in the same frame (seed 99; the run's silhouette IoU over 24
  views at 128 px and chamfer), beside the floor (the target sample against the independent one).
  Criteria: SK's silhouette IoU against the independent sample at or above SG's on most meshes, and above SP's
  on all; against its own sample it may fall (it no longer fits that one sample).
  Expectation: at 40k the samples' noise is a smaller part of the picture at 96 px (a pixel is 1.3 pitches) and
  SK's gain over SG is small (within ±0.001) on most meshes; SK above SP on all 19.
  **Result (all 57 runs done by 04:00 CDT, read 2026-10-05 08:59; `tmp/d92_rows.py`): as expected.** Against
  the independent sample SK is above its physics-only twin on 19 of 19 (1 − IoU −9 to −27 %, median −16 %) and
  level with SG (at or above it on 9 of 19, median difference −0.0001). Eight target samples cost nothing at
  40k and change nothing there.
- **D91, the render's target pictures are the mean over independent samples of the target (pre-registered
  2026-10-05 00:38 CDT at launch; code: `pipeline/target.py` (build_target(draws=)), `prepare.py` (draws),
  `sampling/mesh.py` (stratified_draws, draws_in_frame), `run/runner.py`, `scripts/pipeline_run.py
  --render_target_draws K`, default 1 = as before; server `repo_r77` = HEAD 922b4de with this; `output/gpu/d91`;
  `tmp/d91.sh`, `tmp/runeval3.sh`).** D90's cause: the render term's target is the one sample whose pictures
  the operator draws, and half the render arm's measured lead was its fit of that sample. What the render
  term should match is what the operator draws of the mesh, the expectation over samples; the definition of
  the target pictures becomes their mean over K samples of the target: the pipeline's own and K − 1 further
  independent stratified draws (same fill, other seeds: 198, 298, …; seed 99 is kept out, it is the
  evaluation's reference) in the target's exact frame (checked: a draw with the target's seed reproduces it
  to 2e-7 wu). Every picture the render term reads is averaged (the exterior's silhouettes and shading, and
  the particle cloud's for runs without the exterior). K = 8 lowers the pictures' sampling noise √8 times; the
  transport and the other terms still read the one sample. Tests: a draw equal to the sample changes nothing,
  two samples give the mean picture (`tests/test_settled_contracts.py`).
  Runs: bunny and dragon 300k, HEAD's recipe (D81's weight, D89's one resolution, D70's rule, the exterior)
  with `--render_target_draws 8` (LK), against LG and LN (the same with K = 1) and the physics-only twins PA,
  PN. From here every run is read against the independent sample (`runeval3.sh`: yardstick and display with
  `ref=`), on every kept frame, with momentum, arrangement, offsets from the mesh and relief bands.
  Criteria: against the independent sample, LK ahead of LG/LN on the display's differences and on the
  yardstick, and at least 10 % ahead of the twin on every display measure (the bunny's front and far side
  were 4–8 % under K = 1, D90); physics measures and momentum not behind the twin by more than PA–PN's spread.
  Expectation: the gap between the own-target and the independent reading closes (the render arm no longer
  fits one sample); against the independent sample the differences fall 5–10 % below LG's, the IoUs stay.
  Risk: the mean picture is softer at its edges than any sample's, and the silhouette term's balance between
  holes and spray (w_hole 2, w_spray 1) reads a soft edge as partly covered; the body may settle a little
  outside.
  **First result, the end frames (2026-10-05 01:24 CDT; the last five kept frames of each run against the
  independent sample, `tmp/d90.sh`; the all-frame readings are running): with eight target samples the render
  arm is 11 to 34 % ahead of its physics-only twin on every display measure of both meshes, and ahead of the
  one-sample runs.**

  | against the independent sample | bunny front | thin crop | far side | dragon front | horn crop | far side |
  |---|---|---|---|---|---|---|
  | floor (the target sample itself) | 0.9907 / 0.0062 | 0.9794 / 0.0069 | 0.9904 / 0.0059 | 0.9909 / 0.0075 | 0.9791 / 0.0114 | 0.9918 / 0.0072 |
  | PA (twin, D70) | 0.9887 / 0.0083 | 0.9728 / 0.0102 | 0.9895 / 0.0075 | 0.9866 / 0.0112 | 0.9724 / 0.0163 | 0.9876 / 0.0116 |
  | PN (twin, D89's code) | 0.9884 / 0.0083 | 0.9739 / 0.0102 | 0.9890 / 0.0078 | 0.9869 / 0.0108 | 0.9714 / 0.0158 | 0.9868 / 0.0114 |
  | LG (K = 1) | 0.9907 / 0.0078 | 0.9800 / 0.0079 | 0.9899 / 0.0069 | 0.9897 / 0.0096 | 0.9765 / 0.0141 | 0.9899 / 0.0101 |
  | LN (K = 1, repeat) | 0.9901 / 0.0078 | 0.9791 / 0.0079 | 0.9896 / 0.0070 | 0.9895 / 0.0100 | 0.9780 / 0.0139 | 0.9897 / 0.0101 |
  | LK (K = 8) | 0.9916 / 0.0072 | 0.9809 / 0.0071 | 0.9911 / 0.0067 | 0.9899 / 0.0093 | 0.9796 / 0.0126 | 0.9913 / 0.0098 |

  LK against PN, the twin of the same code, as relative changes of 1 − IoU and of the pictures' difference:
  bunny front −28 %, −13 %; crop −27 %, −30 %; far side −19 %, −14 %; dragon front −23 %, −14 %; crop −29 %,
  −20 %; far side −34 %, −14 %. (Against PA: bunny −26/−13, −30/−30, −15/−11; dragon −25/−17, −26/−23,
  −30/−16 %.) Against the one-sample runs (the pair LG, LN differs by 0.0002–0.0015 in IoU and 0–0.0004 in
  difference) LK's differences are lower on all six measures (bunny −8, −10, −3 %; dragon −5, −10, −3 %) and its
  IoUs higher on five. LK passes the floor in IoU on the bunny (0.9916, 0.9809, 0.9911 against 0.9907,
  0.9794, 0.9904) and in the dragon's crop (0.9796 against 0.9791): it matches an independent draw of the mesh
  better than another draw does. Run measures: bunny LK 76 commits, silhouette IoU 0.9865, thin 3.5 %,
  transport energy at the end 2.4e-5 (PN 2.2e-5), kinetic energy over the last ten windows 6.1e-6 (PN 7.9e-6);
  dragon LK 88 commits, silhouette IoU 0.9854 (PN 0.9835), thin 3.8 % (3.7 %), holes 0.035 % (0.067 %),
  transport 6.3e-5 (7.1e-5), kinetic energy over the last ten 3.0e-5 (5.0e-5). The soft-edge risk did not
  show: no measure moved outward.
  **All frames (finished by 04:06 CDT, read 08:59; `runeval3.sh`, the last 20 kept frames, against the
  independent sample): the end-frame reading holds.** LK against PN: bunny front 1 − IoU −28 %, difference
  −13 %; crop −29 %, −31 %; far side −19 %, −15 %; dragon front −23 %, −15 %; crop −26 %, −20 %; far side −34 %,
  −14 %. The yardstick against the independent sample: exterior silhouette −46 % (bunny) and −51 % (dragon),
  shading −22 % and −25 %; the particle cloud's silhouette −32 % and −25 %, shading −7 % and −18 %. The one-sample
  runs LG, LN are behind LK on all twelve display numbers but one (dragon front IoU, level). Criteria met on
  both meshes: every display measure and both yardstick terms at least 13 % ahead of the twin; physics and
  momentum measures not behind it (D86's table and D89's rest rows).
- **D90, the display against an independent sample of the mesh: the render arm's lead is half what its own
  target sample showed (a measurement; 2026-10-05 00:28 CDT; `tmp/dense_ref.py`, `tmp/collect_ends.py`,
  `tmp/d90.sh`, `tmp/d90_rows.py`; `surface_layer_probe.py refpitch=run`; `output/gpu/d90`).** Every display
  number since D62 reads a run against its own target sample, the 300k sample whose exterior pictures are also
  the render term's targets. A render arm can fit that one sample, its sampling noise included, beyond what
  the mesh would allow: LG's front IoU against it, 0.9940, is above the floor that an independent sample of
  the mesh reaches against it (D65: 0.9906). So the end frames (the last five kept, averaged) are read here
  against a second 300k sample of the mesh drawn in the exact frame of the pipeline's target (seed 99, the
  prepare stage's frame from `load_normalized(frame=)`; the pipeline's own target is reproduced to 0 wu), drawn
  by the same operator at the same density. A 2.4M sample drawn with the run's pitch was tried first and is not
  usable: its outer particles sit nearer the surface (a finer fill), its exterior is larger, and every 300k
  state, its own target included, reads 0.97 against it.

  | against an independent sample | bunny front | thin crop | far side | dragon front | horn crop | far side |
  |---|---|---|---|---|---|---|
  | the pipeline's target sample (the floor) | 0.9907 / 0.0062 | 0.9794 / 0.0069 | 0.9904 / 0.0059 | 0.9909 / 0.0075 | 0.9791 / 0.0114 | 0.9918 / 0.0072 |
  | P (no spacing rule, physics-only) | 0.9858 / 0.0099 | 0.9669 / 0.0122 | 0.9876 / 0.0092 | | | |
  | PA (physics-only, D70) | 0.9887 / 0.0083 | 0.9728 / 0.0102 | 0.9895 / 0.0075 | 0.9866 / 0.0112 | 0.9724 / 0.0163 | 0.9876 / 0.0116 |
  | PN (physics-only, D89's code) | | | | 0.9869 / 0.0108 | 0.9714 / 0.0158 | 0.9868 / 0.0114 |
  | LF (D81) | 0.9907 / 0.0079 | 0.9801 / 0.0082 | 0.9899 / 0.0072 | 0.9894 / 0.0098 | 0.9767 / 0.0144 | 0.9900 / 0.0101 |
  | LG (D81, one resolution) | 0.9907 / 0.0078 | 0.9800 / 0.0079 | 0.9899 / 0.0069 | 0.9897 / 0.0096 | 0.9765 / 0.0141 | 0.9899 / 0.0101 |
  | PT (physics-only, relief reference) | 0.9867 / 0.0090 | 0.9721 / 0.0115 | 0.9874 / 0.0082 | 0.9830 / 0.0123 | 0.9635 / 0.0180 | 0.9828 / 0.0133 |
  | LR (D81, relief reference) | 0.9902 / 0.0082 | 0.9791 / 0.0095 | 0.9895 / 0.0073 | 0.9877 / 0.0110 | 0.9719 / 0.0160 | 0.9878 / 0.0111 |

  LG against its physics-only twin, as relative changes of 1 − IoU and of the pictures' difference: bunny
  (against PA) front −18 %, −6 %; crop −26 %, −23 %; far side −4 %, −8 %. Dragon (against PN, the twin of the
  same code) front −21 %, −11 %; crop −18 %, −11 %; far side −23 %, −11 %. Against its own target sample the
  same runs read −40/−13, −38/−31, −33/−19 and −43/−22, −40/−25, −35/−18 % (D83): about half of that lead was
  the fit of the one sample. What stays is real and on the dragon above the 10 % bar on every display measure;
  on the bunny it is above it in the crop and in the front's IoU, and under it in the front's and far side's
  difference and the far side's IoU. The render arm reaches the floor in IoU on the bunny (front 0.9907, crop
  0.9800 against 0.9907, 0.9794); what is left there is the pictures' difference (0.0078 against 0.0062),
  which the runs, smoother than any sample (3.2° against 9.7°), cannot make against a noisy reference.
  The relief reference (D88) is behind on every measure here: PT behind PA, LR behind LF, by 0.002–0.009 in
  IoU and 4–16 % in difference.
  Consequences: (1) from now on the display and the yardstick are read against the independent sample (the
  probes take `ref=`; `d90/{bunny,dragon}_ind300k.npz`); (2) the render term's own target is one sample of the
  mesh, and the render arm spends part of its effort fitting that sample's noise. Its target pictures should be
  what the operator draws of the mesh in expectation, not one draw: that is a definition of the render target,
  D91.
- **D89, one render resolution, the fine one, from the first window (adopted from D83; pre-registered
  2026-10-04 23:04 CDT at launch; the user: "그 다음은 Opus로 진행하는데, 계속 goal 까지 작업 지속 해 줘"; code:
  `scripts/pipeline_run.py`, `render_res = render_res_hi = the fine resolution following N`; server `repo_r76`
  = HEAD with this; `output/gpu/d89`).** D83's diagnostic made the definition: the coarse stage fitted a
  picture whose pixel was wider than the detail (3.2–3.8 pitches at 300k, D73) and left the fine stage three
  windows at the end under D81's weight. Started at the fine resolution the 300k render arm was 13–43 % ahead
  of its physics-only twin on every display measure and 24–68 % on the yardstick (D83), and the 40k runs
  equal D81's (D85's SG). The coarse-to-fine machinery stays in the runner (direct callers and tests) and is
  inactive from the command line, where both resolutions are now the same. One consequence for the
  physics-only twins: they too went through the switch, which reset their selection epoch and gave them 5–10
  more windows after their plateau (PA: plateau at 93, stop at 100); without it they stop at the plateau.
  Runs: (a) the physics-only twins on this code, bunny and dragon 300k with D70's rule (PN), so that every
  render arm from now on is read against a twin of the same code; (b) a second render arm on the same code
  (LN), so that the render arm's own run-to-run spread is known (D83's LG is the first; a single
  run swings); (c) the 40k gallery's remaining 13 meshes with the one resolution (SG), beside D72's arms. Read
  as D81, momentum included (D86).
  Criteria: the user's bar on both meshes: LG and LN each at least 10 % ahead of PN on 1 − IoU and the
  pictures' difference in front, crop and far side and on the yardstick's two terms; not behind PN on the
  other measures by more than PA–PN's own difference (the twin's run-to-run spread, read from these two
  twins); at 40k SG within the pair's difference of SF on most meshes.
  Expectation: LN within 0.0005 IoU and 5 % difference of LG; PN stops 5–10 windows before PA with the
  display within 0.001 IoU of PA's; the kinetic energy at the end of the render arms within the twins'
  spread (PA and PB differed by 20 % on the bunny, D71).
- **D88, the relaxation's reference is the target's own relief (pre-registered 2026-10-04 21:31 CDT, launched
  again 21:36; code: `kernels.k_layer_project`, `mpm/traj.py`, `window/layer.py` (TargetRelief),
  `window/setup.py`, `pipeline/target.py`, `prepare.py`, `sampling/mesh.py`, `--layer_relief`, off by default;
  server `repo_r75` = D81's code with D74's pictures and this; `output/gpu/d88`; `tmp/d88.sh`).** The cause is
  D82's: the relaxation takes out of the layer whatever its neighbourhood's mean does not explain, the target's
  relief below a cell with the sampling noise, because it relaxes the rough residual d − d̄ towards zero. The
  change gives it the reference it lacks: the same quantity measured on the target mesh's own surface (n / 4
  points of the mesh in the target's frame, the layer's weights cut at the distance of the layer's 24th
  neighbour in the target sample's layer, 3.0 spacings), read at the layer particle's nearest surface point;
  zero for a particle farther than one spacing from the surface. The mesh's surface has the relief and none
  of the sampling noise, so the noise still goes. One term in the kernel; the rest is carrying the mesh's
  surface from the prepare stage to the window. D87 is the check made before it.
  Runs: bunny and dragon 300k with D70's rule, each as the physics-only twin (PT) and with the render on the
  exterior, 126 then 192 px (LR); against PA and LF (the same two without the reference). Read as D81, with
  the octave bands of the end frames (D76's) and the momentum (D86's).
  Criteria: the carried share at 5.4 pitches at least 0.10 above PA's and LF's (0.18, 0.20 bunny; 0.14 dragon)
  and at 10.8 not under theirs; the roughness at most 1.3 times theirs; against the mesh, the bands and the
  field normal's error not above theirs; the display's IoUs within a run's spread of theirs (the target
  picture is drawn from a random sample, which does not carry the mesh's relief either, so the pictures'
  difference is read but does not decide); thin share, holes, transport and kinetic energy, momentum not
  behind. The render's effect as before: LR against PT on every measure.
  Expectation: 0.30–0.40 at 5.4 pitches on the bunny (D87's 0.39–0.41 after one window on a finished run),
  0.25–0.35 on the dragon; bands −5 %, normal error −5 %; roughness 3.5–4°. Risks: the reference is read at
  the nearest surface point, which jumps across a thin part's two faces; the minimum spacing presses the
  layer back where the relief asks it inward (the dragon's 0.44 → 0.14 came from that rule, not from the
  relaxation).
  First launch (21:31) withdrawn after three minutes: the reference was wrong. A check of each face normal
  against the sample (does it point away from the eight nearest particles) turned 12.6 % (bunny) and 15.4 %
  (dragon) of a sound mesh's normals, the turned points lost their same-side neighbours, and a point without
  weight read the world's origin as its centroid (D39's fault again): values up to 38 and 82 pitches, rms
  0.21 and 0.59 where the mesh's own is 0.088 and 0.137. Found from the size printed at the target's build,
  measured (`tmp/relief_check.py`: the surface is registered to the sample, median distance 0.6 pitches; the
  fault is the turning), and corrected: the normals are the mesh's, turned all at once only if as a whole
  they point inward, and a point without weight has a zero value. Largest values now 0.93 and 1.48 pitches (the
  meshes' sharp creases). Tests on hyde06 in `repo_r75`: 277 passed, 2 skipped, exit 0, before the
  correction, and again after it (277 passed, 2 skipped, exit 0).
- **D87, whether a relaxation towards the target's relief keeps the relief and the smoothness (a feasibility
  check on kept end frames, no simulation; 2026-10-04 21:00 CDT; `tmp/relief_relax_test.py`, `tmp/d87.sh`;
  `output/gpu/d87`).** The run's own projection (the pipeline's layer data, 1/20 a step over 40 steps) is
  applied once to the end frames of PR (no relaxation), PA and LF, towards zero as the run does and towards
  r*, the rough residual of the mesh's surface at the particle's nearest surface point. The mesh's r* is 0.088
  pitches rms; PA's layer has 0.046 with a correlation of +0.13 to it, PR's 0.36 with +0.12: what a finished
  run's layer has at this scale is not the target's.

  | bunny 300k end frame | carried share at 2.7 / 5.4 / 10.8 / 21.6 pitches | bands below 4 / 4–11 | field normal's error | roughness | front IoU / difference |
  |---|---|---|---|---|---|
  | 300k sample | 0.36 / 0.66 / 0.86 / 0.95 | 0.103 / 0.132 | 11.5° | 9.8° | |
  | PA as it is | 0.03 / 0.18 / 0.73 / 0.92 | 0.085 / 0.142 | 9.2° | 3.2° | 0.9899 / 0.0077 |
  | PA, one window towards zero | 0.02 / 0.15 / 0.67 / 0.91 | 0.086 / 0.145 | 9.4° | 3.5° | 0.9894 / 0.0083 |
  | PA, one window towards the mesh's relief | 0.19 / 0.39 / 0.73 / 0.91 | 0.080 / 0.132 | 8.6° | 3.7° | 0.9894 / 0.0081 |
  | LF as it is | 0.05 / 0.20 / 0.72 / 0.93 | 0.084 / 0.140 | 9.2° | 3.3° | 0.9934 / 0.0069 |
  | LF, towards zero | 0.05 / 0.16 / 0.67 / 0.92 | 0.086 / 0.143 | 9.4° | 3.7° | 0.9925 / 0.0076 |
  | LF, towards the mesh's relief | 0.23 / 0.41 / 0.73 / 0.92 | 0.080 / 0.129 | 8.5° | 3.8° | 0.9926 / 0.0074 |
  | PR as it is | 0.24 / 0.53 / 0.97 / 0.95 | 0.132 / 0.190 | 14.9° | 9.2° | 0.9743 / 0.0136 |
  | PR, towards zero | 0.12 / 0.36 / 0.86 / 0.93 | 0.101 / 0.156 | 11.4° | 6.3° | 0.9838 / 0.0105 |
  | PR, towards the mesh's relief | 0.30 / 0.61 / 0.92 / 0.94 | 0.098 / 0.150 | 10.9° | 6.6° | 0.9817 / 0.0112 |

  One window of the projection towards the mesh's relief doubles the share carried at 5.4 pitches on the
  finished runs (0.18 → 0.39, 0.20 → 0.41) and brings 2.7 pitches from nothing to 0.19–0.23, lowers the offset
  from the mesh in both bands by 6–8 % (the 4–11 band to a sample's 0.132) and the field normal's error by
  7 %, where one more window of the present relaxation lowers the carried share and leaves the offsets. On PR
  it takes the noise down as the plain relaxation does (roughness 9.2° → 6.6°, 6.3° towards zero) and keeps
  the relief (0.61 at 5.4 pitches against 0.36). Against the target picture, which is a random sample's, both
  projections cost the same 0.0004–0.0007 in the front difference (the layer is moved, nothing else adapts).
  So the reference does what D82 asked for on a still frame; in a run the objective and the spacing rule act
  on the same layer, which is D88.
- **D86, the momentum of every 300k run of D70–D84 (a record, at the user's request "momentum 쪽도 잘 기록 해
  줘"; 2026-10-04 20:46 CDT; `scripts/probes/settled/momentum_probe.py`, now with a row per pair of kept
  frames; `tmp/momentum_rows.py`, `tmp/momentum_plot.py`; `output/gpu/d86`: `m_*.log`,
  `momentum_{bunny,dragon}300k.png`, local `output/video_2026-10-04/d81_weight_every_window/`).** No external
  force acts and every run starts at rest, so conserved linear momentum keeps the centre of mass where it is
  and conserved angular momentum keeps the body from turning. From the kept frames (every 12 steps, uniform
  mass): the centre of mass's displacement from the first frame, the net over the gross linear and angular
  move per pair of frames, the net rotation summed over the run, the mean particle move per pair over the
  last 20 pairs. From the run's record (every committed window): the centre of mass's velocity and the
  kinetic energy. Lengths in display pitches (0.048 world units; a body is 100–130 pitches across).

  | bunny 300k | centre of mass, largest / end | its velocity, largest / last ten windows | net / gross linear, run / last 20 | net / gross angular, run / last 20 | net rotation | still moving at the end | kinetic energy, last / last ten |
  |---|---|---|---|---|---|---|---|
  | PA, physics-only twin | 0.0077 / 0.0033 | 9.1e-6 / 3.5e-8 | 1.08e-2 / 1.20e-2 | 1.01e-2 / 9.1e-3 | 0.011° | 0.0030 | 3.9e-6 / 4.5e-6 |
  | LA, weight held (D70) | 0.0047 / 0.0041 | 7.6e-6 / 7.6e-8 | 2.2e-3 / 2.8e-3 | 5.6e-3 / 4.0e-3 | 0.008° | 0.0074 | 1.9e-4 / 3.1e-4 |
  | LC (D74) | 0.0031 / 0.0026 | 7.7e-6 / 9.3e-8 | 2.3e-3 / 3.4e-3 | 6.3e-3 / 8.5e-3 | 0.004° | 0.0046 | 3.1e-5 / 7.8e-5 |
  | LD (D77) | 0.0045 / 0.0039 | 9.0e-6 / 3.1e-8 | 8.8e-3 / 9.4e-3 | 1.18e-2 / 5.6e-3 | 0.012° | 0.0033 | 7.6e-6 / 6.6e-6 |
  | LE (D78) | 0.0047 / 0.0040 | 8.0e-6 / 6.9e-8 | 9.5e-3 / 9.4e-3 | 1.22e-2 / 5.4e-3 | 0.011° | 0.0037 | 7.0e-6 / 8.5e-6 |
  | LF (D81) | 0.0049 / 0.0046 | 7.1e-6 / 1.6e-8 | 9.9e-3 / 1.40e-2 | 8.6e-3 / 6.9e-3 | 0.014° | 0.0029 | 3.2e-6 / 4.8e-6 |
  | LH (D84) | 0.0058 / 0.0057 | 6.2e-6 / 3.7e-8 | 4.1e-3 / 4.8e-3 | 6.3e-3 / 3.3e-3 | 0.009° | 0.0029 | 1.1e-5 / 7.7e-6 |
  | PR (D82, physics-only, no relaxation) | 0.0083 / 0.0013 | 9.0e-6 / 1.2e-8 | 5.6e-3 / 3.8e-3 | 8.9e-3 / 7.8e-3 | 0.008° | 0.0004 | 3.3e-7 / 5.4e-7 |

  | dragon 300k | centre of mass, largest / end | its velocity, largest / last ten windows | net / gross linear, run / last 20 | net / gross angular, run / last 20 | net rotation | still moving at the end | kinetic energy, last / last ten |
  |---|---|---|---|---|---|---|---|
  | PA, physics-only twin | 0.0257 / 0.0230 | 7.6e-6 / 6.2e-8 | 6.2e-3 / 8.0e-3 | 9.6e-3 / 7.5e-3 | 0.108° | 0.0065 | 1.7e-5 / 3.4e-5 |
  | LA, weight held (D70) | 0.0094 / 0.0094 | 8.1e-6 / 1.5e-7 | 2.3e-3 / 3.6e-3 | 3.9e-3 / 5.5e-3 | 0.024° | 0.0132 | 1.7e-3 / 8.3e-4 |
  | LC (D74) | 0.0116 / 0.0115 | 7.5e-6 / 1.2e-7 | 1.8e-3 / 1.8e-3 | 3.1e-3 / 2.6e-3 | 0.026° | 0.0265 | 1.8e-4 / 1.6e-3 |
  | LW (D75) | 0.0189 / 0.0182 | 8.9e-6 / 4.3e-8 | 3.6e-3 / 4.2e-3 | 6.9e-3 / 5.0e-3 | 0.072° | 0.0067 | 4.2e-5 / 4.8e-5 |
  | LD (D77) | 0.0167 / 0.0162 | 7.6e-6 / 1.4e-7 | 4.1e-3 / 6.3e-3 | 8.5e-3 / 8.5e-3 | 0.083° | 0.0104 | 5.8e-5 / 8.2e-5 |
  | LE (D78) | 0.0170 / 0.0169 | 8.2e-6 / 8.9e-8 | 4.0e-3 / 4.9e-3 | 9.5e-3 / 1.01e-2 | 0.088° | 0.0069 | 3.0e-5 / 3.6e-5 |
  | LF (D81) | 0.0194 / 0.0184 | 8.5e-6 / 5.7e-8 | 6.9e-3 / 9.5e-3 | 9.0e-3 / 7.3e-3 | 0.079° | 0.0056 | 1.8e-5 / 2.1e-5 |
  | LG (D83) | 0.0210 / 0.0189 | 7.9e-6 / 5.9e-8 | 6.9e-3 / 1.06e-2 | 8.7e-3 / 8.7e-3 | 0.071° | 0.0077 | 4.5e-5 / 4.3e-5 |

  Reading. In every run the centre of mass moves by less than 0.03 pitches (3e-4 of the body's size) and the
  body turns by at most 0.11°; the centre of mass's velocity peaks at 6–9e-6 during the transit and is 1e-8 to
  1.5e-7 over the last ten windows. With the weight held (LA, LC) the drift is smaller than the twin's and the
  rest is worse: the kinetic energy stays 10 to 100 times the twin's from window 30 on and the particles
  still move 2 to 4 times as far per pair at the end. With the weight calibrated at every window (LF) the
  render arm's kinetic energy follows the twin's curve through the whole run (the plot), ends at 3.2e-6 against
  3.9e-6 (bunny) and 1.8e-5 against 1.7e-5 (dragon), over the last ten windows 4.8e-6 against 4.5e-6 and 2.1e-5
  against 3.4e-5; what still moves at the end is 0.0029 against 0.0030 and 0.0056 against 0.0065 pitches a
  pair; the centre of mass's displacement is below the twin's on both (0.0049 against 0.0077, 0.019 against
  0.026); the net rotation is below it on the dragon (0.079° against 0.108°) and above it on the bunny (0.014°
  against 0.011°). The net-over-gross ratios are the twin's (about 1e-2): of a pair's motion a hundredth is a
  common translation or rotation, with or without the render term. D83's dragon run rests a little less well
  (kinetic energy over the last ten windows 4.3e-5 against the twin's 3.4e-5, 0.0077 against 0.0065 pitches a
  pair). Without the relaxation (PR) the body ends ten times more at rest than any other run.
- **D85, D81's and D84's weights at 40k (pre-registered 2026-10-04 18:49 CDT at launch; `tmp/d85.sh`,
  `tmp/d85.queue`; `repo_r71` (arm SF) and `repo_r74` (arm SH); outputs beside the gallery's in
  `output/gpu/d72`).** D80's first six meshes (below) show that D77's form fails at 40k. The two weights that
  replaced it are read on the same six meshes (bunny, dragon, A, armadilo, beast, bimba), each with the
  minimum spacing, against S (the weight held from the first window), the pair B1, B2 and the twin SP.
  Criteria per mesh: the run's silhouette IoU not under S's by more than the pair's difference, the thin share
  uncovered not over S's by more than the pair's difference, still ahead of the twin on both and on the
  yardstick; commits within a quarter of S's.
  Expectation: SF (the gradient-norm rule at every window) loses part of S's lead at 40k, where the heavy
  weight is what carries the thin parts (silhouette IoU −0.002 to −0.004); SH (a third of the merit by value)
  keeps it within the pair's difference.
  **Result (2026-10-04 20:57 CDT; a third arm SG, D83's fine picture from the first window with D81's weight,
  was added at 19:25; SG on four meshes so far): at 40k every one of the new weights gives up part of the
  held weight's silhouette, and all of them keep the render arm ahead of its twin with the physics measures
  two to ten times nearer the twin's.** Run's silhouette IoU (S is the held weight, SP the twin):

  | mesh, 40k | SP (twin) | S (held) | SF (D81) | SG (D81 + D83) | SH (D84) | thin uncovered: SP / S / SF / SG / SH | transport energy at the end against S: SF / SG / SH |
  |---|---|---|---|---|---|---|---|
  | bunny | 0.9662 | 0.9777 | 0.9719 | 0.9721 | 0.9760 | 9.2 / 5.3 / 7.3 / 7.7 / 5.9 % | 0.44 / 0.41 / 0.58 |
  | dragon | 0.9650 | 0.9767 | 0.9723 | 0.9720 | 0.9753 | 12.8 / 8.2 / 9.3 / 9.5 / 8.7 % | 0.42 / 0.37 / 0.52 |
  | A | 0.9681 | 0.9807 | 0.9720 | 0.9743 | 0.9788 | 15.0 / 4.3 / 10.5 / 6.9 / 6.2 % | 0.41 / 0.42 / 0.51 |
  | armadilo | 0.9614 | 0.9727 | 0.9659 | 0.9668 | 0.9675 | 11.0 / 5.7 / 8.3 / 7.3 / 5.7 % | 0.26 / 0.32 / 0.24 |
  | beast | 0.9621 | 0.9711 | 0.9663 | | 0.9680 | 8.8 / 4.6 / 7.3 / – / 6.1 % | 0.10 / – / 0.09 |
  | bimba | 0.9657 | 0.9790 | 0.9721 | | 0.9759 | 19.2 / 6.1 / 13.8 / – / 7.9 % | 0.41 / – / 0.45 |

  Against S: SF −0.0044 to −0.0087, SG −0.0047 to −0.0065, SH −0.0013 to −0.0052 (the pair B1, B2 differs by
  0.0000–0.0006; beast by 0.0063). Against the twin: SF +0.0039 to +0.0072 (silhouette error −11 to −21 %),
  SH +0.0059 to +0.0106 (−16 to −34 %); S's own lead is +0.009 to +0.013 (−24 to −39 %), bought with a
  transport energy 2.4 to 12 times the twin's and a kinetic energy at the end 3.6 to 40 times. SF's expectation
  was too mild (the loss is −0.004 to −0.009, not −0.002 to −0.004) and SH's too good (it is outside the
  pair's difference on five of six). The fine picture from the start does not bring the silhouette back at
  40k. So at 40k the held weight is what makes the run's silhouette IoU, at the physics measures' cost, and
  no weight read here has both; which of the two the 40k gallery is to show is a choice, not a measurement.
  The whole gallery for D81's weight (2026-10-04 23:00 CDT; SF on 18 of the 19 meshes, with S, the pair and
  the twin beside it): against S the run's silhouette IoU is lower on all 18 (−0.0017 to −0.0087; inside the
  pair's own difference on two), the thin share uncovered higher on all 18 (+0.5 to +7.7 points), the
  transport energy at the end 0.10 to 0.74 times S's on 17 (1.30 on one); against its physics-only twin SF
  is ahead in silhouette IoU on all 18 (+0.0018 to +0.0100) and in the thin share on 17 (−1.5 to −9.7 points;
  +0.9 on one). The six meshes were not a special case.
- **D84, the render weight calibrated by the terms' values (a diagnostic on a server copy, no committed code;
  pre-registered 2026-10-04 18:39 CDT, queued behind an evaluation; `repo_r74` = D81's code with the
  balancer fed the objective without its render term and the render term in place of the two gradients'
  norms; `output/gpu/d84`).** Three weights have now been run, read as the render term's share of the merit's
  value and of the step late in the run: held from the first window (D70, D74, D77's merit) 63–88 % and 0.94;
  a tenth of that (D75) 14 % in the coarse stage, 33 % in the fine one, 0.76; the gradient-norm rule at every
  window (D81) 5–7 % and 0.33. D75's is the best run so far and D81's first run has a fine stage of three
  commits in which neither part falls: with the physics gradient at its floor a third of the step is a third
  of very little, and a merit that is 95 % physics accepts no step that buys render at any cost in physics.
  The norm of a gradient at its floor says little about what the term still has to give; the terms' values
  do not have that fault. The diagnostic keeps the rule's form and constant and changes what it balances: λ ×
  (render term) = 0.5 × (the rest of the objective) at each window's first evaluation, through the same moving
  average, so the render term is a third of the merit's value at every window. Bunny 300k first (LH), against
  LF, LE and the twin.
  Criteria: the user's bar; the render term's share of the merit 0.30–0.36 at every window from 10 on; a fine
  stage of more than five commits with the render terms falling; commits within a fifth of the twin's;
  transport and kinetic energy at the end within twice the twin's.
  Expectation: a share of the step of 0.6–0.8 late; the coarse stage as long as LF's; the display between
  LE's and better. Risk: at a third of the merit the render term's 3–7 % movement is 1–2 % of the merit, the
  size of the physics gain of a late window, and the early stops of D77 come back in a milder form.
  **Result, bunny 300k (2026-10-04 20:35 CDT): not better than D81's weight at 300k.** LH: 79 commits, the
  render term 0.31–0.32 of the merit as designed, its share of the step 0.80–0.89, a fine stage of about 20
  commits (animations 62–82). Display: front 0.9940 / 0.0072, crop 0.9864 / 0.0073, far side 0.9940 / 0.0059,
  roughness 3.7°, against LF's 0.9935 / 0.0066, 0.9873 / 0.0058, 0.9938 / 0.0056, 3.3°: the IoUs level, the
  pictures' differences 5 to 26 % worse than LF's (still under the twin's 0.0075, 0.0086, 0.0068). The
  yardstick's exterior silhouette is the lowest of all runs (0.000261; LF 0.000374), its shading LF's
  (0.000455), the particle cloud's shading above the twin's (0.000222 against 0.000158). The physics objective
  is behind again: transport energy 2.7 to 3.0 times the twin's at windows 20–60 and 5.7e-5 against 2.1e-5 at
  the end, kinetic energy over the last ten windows 7.7e-6 against 4.5e-6. The fine stage's length was bought
  with the physics part. At 40k the same weight keeps the held weight's silhouette (D85).
- **D83, the fine render picture from the first window (a diagnostic on a server copy, no committed code;
  pre-registered 2026-10-04 18:28 CDT, queued behind the evaluations then running; `repo_r73` = D81's code
  with the starting resolution set to the fine one; `output/gpu/d83`).** What the runs so far show: the render
  arm's lead over its twin is made in the fine stage. In D75's dragon run the horn crop is 0.002–0.004 ahead
  of the twin through the 64-px stage (0.9784–0.9817 against 0.9743–0.9787 at windows 50–90) and 0.9876 after
  ten windows at 192 px, the front difference 0.0098 before them and 0.0084 after. Under D81's weight the fine
  stage comes after the body has converged and lasts three commits (bunny, animations 76–78). The coarse
  stage was made for the cost of the particle-cloud picture; on the exterior a 192-px window costs 30 % more
  than a 96-px one and a 64-px window no less (D76), and 96 px from the start was the better schedule once
  before (D18, D19). The diagnostic: one resolution, 192 px, from the first window, no switch; everything else
  D81's. Bunny and dragon 300k (LG), against LF (D81), LE, the twins.
  Criteria: the user's bar (10–20 % or more ahead of the twin on each error measure, not behind on the rest);
  for this change, ahead of LF on the display's crop IoU and difference and on the yardstick's silhouette, a
  run no longer than 1.5 times LF's.
  Expectation: the yardstick's silhouette below LF's from window 20 on; the crop's difference 5–10 % below
  LF's at the end; the first ten windows slower to arrive than with a coarse picture. Risk: a fine picture
  while the body is far from its target gives a poor early guide (the silhouette's reach is a pixel).
  **Result, dragon 300k (2026-10-04 20:35 CDT; the bunny's run started 20:32): ahead of D81's run on the
  display and the yardstick, a little behind it in rest.** LG: 101 commits (LF 102, the twin 97), 2435 s
  against LF's 2905 s, both on shared GPUs. Display: front 0.9932 / 0.0080, horn crop 0.9870 / 0.0104, far
  side 0.9929 / 0.0089, roughness 4.8°; against the twin 1 − IoU −43 %, −40 %, −35 % and the pictures'
  difference −22 %, −25 %, −18 % (LF: −36 %, −31 %, −33 % and −18 %, −21 %, −18 %). Yardstick: exterior
  silhouette 0.000338 and shading 0.000439 (LF 0.000450, 0.000474; the twin 0.001051, 0.000685). The run's own
  measures: silhouette IoU 0.9857 (LF 0.9843, twin 0.9819), thin share uncovered 3.0 % (4.3 %, 5.3 %), chamfer
  0.0544 (0.0545, 0.0545), transport energy at the end 5.4e-5 (5.7e-5, 6.8e-5), bands 0.107, 0.159, the field
  normal's error 10.8°, the density's spread 0.191, discs apart 6.8 %. Behind LF and the twin: holes 0.042 %
  (0.028 %, 0.037 %), and rest (D86: kinetic energy over the last ten windows 4.3e-5 against LF's 2.1e-5 and
  the twin's 3.4e-5; 0.0077 pitches a pair still moving against 0.0056 and 0.0065). The criteria of this change
  are met on the dragon; the early windows were not slower.
  **Result, bunny 300k (2026-10-04 22:40 CDT): level with D81's run on the display, ahead of it on the
  yardstick and on the run's length, and level with the twin on the physics measures.** LG: 86 commits (LF 71,
  the twin 87), 2864 s on a GPU shared with gallery runs. Display: front 0.9940 / 0.0065, thin crop 0.9867 /
  0.0059, far side 0.9938 / 0.0055, roughness 3.2° (LF 0.9935 / 0.0066, 0.9873 / 0.0058, 0.9938 / 0.0056, 3.3°);
  against the twin 1 − IoU −40 %, −38 %, −33 % and the pictures' difference −13 %, −31 %, −19 %. Yardstick:
  exterior silhouette 0.000289 and shading 0.000432 (LF 0.000374, 0.000446; the twin 0.000787, 0.000568), the
  particle cloud's shading under the twin's (0.000151 against 0.000158). The run's own measures: silhouette
  IoU 0.9863, thin share uncovered 3.5 % (LF 4.0 %, the twin 4.8 %), chamfer 0.0550, transport energy at the
  end 2.15e-5 (the twin 2.10e-5; LF 2.6e-5, the difference having been LF's shorter run), kinetic energy over
  the last ten windows 5.2e-6 (the twin 4.5e-6), bands 0.084, 0.140, field normal's error 9.2°, density's
  spread 0.166, discs apart 1.6 %, the centre of mass's largest displacement 0.0042 pitches (0.0077), net
  rotation 0.013° (0.011°). So with one resolution from the first window the bunny's render arm is ahead of
  its twin by 13 to 40 % on the display and by 24 to 63 % on the yardstick, 27 % on the thin share, and is
  the twin's on transport energy; what is left behind it is 16 % in kinetic energy over the last ten windows
  and 0.002° of rotation. At 40k the same change with D81's weight equals D81's alone (D85's SG against SF:
  0.9721 against 0.9719, 0.9720 against 0.9723, 0.9743 against 0.9720, 0.9668 against 0.9659) in fewer
  windows. The diagnostic is a candidate for the code: the coarse stage has no use left on the exterior.
- **D82, the outer layer's relaxation switched off under the minimum spacing (a diagnostic on a server copy, no
  committed code; pre-registered 2026-10-04 18:12 CDT; `repo_r72` = D81's code with the relaxation's fraction
  set to zero, u kept; `output/gpu/d82`).** The question is the user's "detail": D76 measured that a run carries
  0.14–0.22 of the mesh's relief at 5.4 pitches and 0.73–0.77 at 10.8, a 300k sample 0.60–0.66 and 0.86–0.87;
  the runs with the bounded share carry what their twins carry (bunny LD, LE 0.22, 0.21 against PA's 0.18 at
  5.4 pitches and 0.75, 0.74 against 0.73 at 10.8; dragon LW 0.19 against 0.14 and 0.77 against 0.76; the
  unbounded arms lost it, 0.56 at 10.8), and D71's form of the spacing rule does not differ from D70's (0.17,
  0.72). So the render term does not write relief below a cell, and what erases it is on the physics side:
  the relaxation (D14: it halves relief at 11 spacings and removes it below 6) and the spacing rule (the
  dragon's 0.44 → 0.14 at 5.4 pitches). The relaxation was kept after D16 because without it the surface was
  as rough as a sample and the run did not stop; the spacing rule now gives the evenness. Bunny 300k,
  physics-only, the rule on, the relaxation off (PR), against PA; the dragon if the bunny answers.
  Criteria: the carried share at 5.4 pitches at least 0.10 above PA's and at 10.8 not under it; the field's
  roughness at most 1.3 times PA's (3.2° → 4.2°; a sample's is 9.8°); the pictures' difference and the IoUs
  not behind PA's; where the run stops.
  Expectation: 0.30–0.40 at 5.4 pitches, 0.80 at 10.8; roughness 4–6°; the run longer than PA's.
  First launch (18:12) failed: it was put on a GPU whose evaluation then grew to 34 GB, and the run ran out of
  memory in its first window; queued again behind that evaluation.
  **Result (2026-10-04 20:57 CDT): the relaxation is what erases the relief, and it is still what makes the
  surface smooth.** PR, 114 commits (it stops by itself, at animation 115, unlike D16's 40k runs), against PA:

  | bunny 300k, physics-only with the rule | carried share at 2.7 / 5.4 / 10.8 / 21.6 pitches | roughness | bands below 4 / 4–11 | field normal's error | front IoU / difference | thin crop | far side | yardstick: exterior silhouette / shading |
  |---|---|---|---|---|---|---|---|---|
  | 300k sample | 0.36 / 0.66 / 0.86 / 0.95 | 9.8° | 0.103 / 0.132 | 11.5° | 0.9906 / 0.0063 (the floor) | 0.9795 / 0.0071 | 0.9904 / 0.0059 | |
  | PA, relaxation on | 0.03 / 0.18 / 0.73 / 0.92 | 3.2° | 0.085 / 0.142 | 9.3° | 0.9900 / 0.0075 | 0.9784 / 0.0086 | 0.9907 / 0.0068 | 0.000787 / 0.000568 |
  | PR, relaxation off | 0.24 / 0.53 / 0.97 / 0.95 | 9.2° | 0.132 / 0.190 | 15.0° | 0.9742 / 0.0136 | 0.9497 / 0.0179 | 0.9724 / 0.0128 | 0.003118 / 0.001605 |

  Without the relaxation the surface carries the relief nearly as a sample does (0.53 of the sample's 0.66 at
  5.4 pitches, all of it at 10.8), and is as rough as a sample (9.2°), stands 0.17 pitches further out (mean
  offset +0.476 against +0.308), has 4.3 % of its discs apart from the mesh (1.8 %), 7.3 % of the thin target
  uncovered (4.8 %), and a display behind PA's on every measure by a factor of 1.8 to 2.6 in the error. It
  ends ten times more at rest (kinetic energy 3.3e-7). The first criterion is met (5.4 pitches +0.35, 10.8
  +0.24), the second and third fail (roughness 2.9 times PA's; the pictures behind). The expectation had the
  relief right and the roughness too low: the minimum spacing does not give the smoothness by itself, the
  two rules together do. What this settles for the detail: the relief below a cell is there to be had at this
  N without more particles, and the rule that removes it is the relaxation, which takes out of the layer
  whatever its neighbourhood's mean does not explain, sampling noise and the target's own relief alike. It
  relaxes towards zero because it does not know what the target has there.
- **D81, the render weight is calibrated at every window, and the selection rescores its references
  (pre-registered 2026-10-04 17:51 CDT at launch; the user: "계속 진행 해 줘 지금 10 ~ 20% 이상 나와야 의미 있는
  결과인 거니까"; code: `window/solve.py`, `run/selection.py`, `run/runner.py`, `target.py`; server `repo_r71`,
  with D74's pictures; `output/gpu/d81`; `tmp/d81.sh`).** What D77 left: the render arm stops at about 50
  commits where its twin runs to 87–97. Measured on the records of D77's and D78's runs (`tmp/rescore.py`: the
  merit is the physics part plus λ times the render term, so each window's merit can be scored again with
  another weight): with λ held from the first window the render term is 63–88 % of the merit's value; the
  weight the calibration rule gives at those windows is 0.6–1.6 % of λ; the render term moves by ±3–7 % from
  one window to the next at its floor, so the merit moves by ±2–6 % with it. The windows whose rejection ends
  the coarse stage have a positive physics gain (+3.4, +1.6, +2.8, +0.8 %) and a merit that is worse as
  recorded (−1.3, −2.1, −1.6, −6.0 %) and better with the rule's weight (+3.2, +1.5, +2.6, +0.6 %). D77 bounded
  the step and left the merit with the first window's weight, so the line search and the acceptance still
  judged by a function that is mostly the render term.
  The change is to the weight's definition, and takes D77's bound back out: the calibration is made at every
  window's first gradient (the balancer's own rule and moving average, which were there and were held after
  the first window), and held for that window. The step is again the gradient of the function the line search
  and the acceptance read. Because the merit is linear in the weight, the selection rescores its two
  references (the last accepted window and the best one) with the judged window's weight from their own
  render terms, in place of opening a new cost epoch when the weight moves; the delivered slice is chosen with
  every window scored at the last weight. An epoch remains what it was at a change of the render pictures. The
  solver loses six lines net; the selection gains about ten.
  Runs: bunny and dragon 300k, D70's rule, render on the exterior, 126 then 192 px (LF), against the twins
  PA, D77's and D78's LD and LE, and D75's LW. Read as D74.
  Criteria: the user's bar, the render arm ahead of its physics-only twin by 10–20 % or more on each measure
  (as a relative change of the error: 1 − IoU, the pictures' difference, the yardstick's terms), and not
  behind on the rest (transport energy, kinetic energy at the end, momentum, bands, arrangement, thin share,
  chamfer, holes). What the cause predicts: the render term's share of the merit's value falls from 63–88 % to
  under a fifth by window 30; the coarse stage does not end before the twin's would (commits within a fifth of
  the twin's); the render's share of the step is a third at each window's first gradient once the average has
  caught up, above it during the first 20 windows (the average lags the falling rule).
  Expectation: display at LD's and LE's level or better with 30–40 more windows; transport energy and kinetic
  energy at the end within 1.5 times the twin's. Risks: the weight keeps moving and the rescored references
  never let the gate latch, so the run goes to the window budget; a weight that small lets the render terms
  drift up late in the run.
  **Result, the bunny (2026-10-04 18:50 CDT; the dragon's run is finished and being read): the render arm is
  12 to 41 % ahead of its twin on every display measure and level with it or ahead on the physics measures.**
  LF, 71 commits (the twin 87; LE 49), 1494 s on a shared GPU:

  | bunny 300k | front IoU / difference | thin crop | far side | roughness | yardstick: exterior silhouette / shading | thin uncovered | transport, kinetic energy at the end |
  |---|---|---|---|---|---|---|---|
  | PA (twin) | 0.9900 / 0.0075 | 0.9784 / 0.0086 | 0.9907 / 0.0068 | 3.2° | 0.000787 / 0.000568 | 4.8 % | 2.1e-5, 3.9e-6 |
  | LE (D78) | 0.9938 / 0.0068 | 0.9881 / 0.0062 | 0.9936 / 0.0057 | 3.4° | 0.000315 / 0.000449 | 3.9 % | 3.7e-5, 6.9e-6 |
  | LF (D81) | 0.9935 / 0.0066 | 0.9873 / 0.0058 | 0.9938 / 0.0056 | 3.3° | 0.000374 / 0.000446 | 4.0 % | 2.6e-5, 3.2e-6 |

  Against the twin, as relative changes of the error: front 1 − IoU −35 %, difference −12 %; crop −41 %, −33 %;
  far side −33 %, −18 %; the yardstick's silhouette 2.1 times below and its shading 21 % below; thin share
  uncovered −17 %; kinetic energy at the end −18 %; the centre of mass's displacement 0.0049 against 0.0077
  pitches; still moving at the end 0.0029 against 0.0030 pitches a pair. Level with the twin: the bands
  (0.084, 0.140 against 0.085, 0.142), the field normal's error (9.2° against 9.3°), the density's spread
  (0.170 against 0.168), chamfer (0.0551 against 0.0550), the roughness. Behind it: the transport energy at
  the end, 2.6e-5 against 2.1e-5 (1.1 times the twin's at equal windows: 9.6e-5, 3.8e-5, 2.75e-5 at windows
  20, 40, 60 against 8.5e-5, 3.4e-5, 2.5e-5; the twin runs 16 windows longer), and the net rotation, 0.014°
  against 0.011°.
  What the cause predicted: the weight falls from 0.257 to 0.033 by window 8, 0.0086 by 15, 0.0034 by 22 and
  stays at 0.0018–0.0021 from window 35 on; the render's share of the step is 0.52–0.64 in the first ten
  windows and 0.32–0.39 from window 20; the render term is 5–7 % of the merit's value late in the run; the
  coarse stage runs to animation 76 (D78's ended at 43) and ends on patience, not on a streak of rejections.
  The first risk did not come true (the gate latches, the run stops). The fine stage is weak: three commits,
  in which the physics part rises by 2.5 % and 1.7 % and the 192-px silhouette term goes from 6.0e-4 to 5.5e-4
  (D77's fine stage took it from 7.3e-4 to 4.9e-4 in seven commits): with the physics gradient at its floor a
  third of the step is a third of very little, and a merit that is 94 % physics accepts no step that buys
  render at a cost in physics. The display is nonetheless at LE's level or ahead of it in the differences;
  D83 and D84 are the two readings of that weakness.
  **Result, the dragon (2026-10-04 19:31 CDT): 18 to 36 % ahead of the twin on every display measure, and
  ahead of or level with it on every other measure read.** LF, 102 commits (the twin 97), the coarse stage to
  animation 108, the fine stage three commits; 2905 s on a shared GPU:

  | dragon 300k | front IoU / difference | horn crop | far side | roughness | yardstick: exterior silhouette / shading | thin uncovered | transport, kinetic energy at the end |
  |---|---|---|---|---|---|---|---|
  | floor | 0.9912 / 0.0074 | 0.9793 / 0.0112 | 0.9918 / 0.0072 | 13.6° | | | |
  | PA (twin) | 0.9881 / 0.0102 | 0.9784 / 0.0139 | 0.9890 / 0.0108 | 5.0° | 0.001051 / 0.000685 | 5.3 % | 6.8e-5, 1.7e-5 |
  | LW (D75) | 0.9928 / 0.0086 | 0.9863 / 0.0106 | 0.9923 / 0.0089 | 4.9° | 0.000390 / 0.000494 | 4.4 % | 1.5e-4, 4.2e-5 |
  | LD (D77) | 0.9918 / 0.0092 | 0.9847 / 0.0115 | 0.9919 / 0.0094 | 4.9° | 0.000426 / 0.000502 | 3.6 % | 1.4e-4, 5.8e-5 |
  | LF (D81) | 0.9924 / 0.0084 | 0.9852 / 0.0110 | 0.9926 / 0.0089 | 4.9° | 0.000450 / 0.000474 | 4.3 % | 5.7e-5, 1.8e-5 |

  Against the twin, as relative changes: front 1 − IoU −36 %, difference −18 %; horn crop −31 %, −21 %; far
  side −33 %, −18 %; the yardstick's exterior silhouette −57 % and shading −31 %, and on the particle cloud
  −41 % and −23 %; the run's silhouette error −13 %; thin share uncovered −19 %; holes 0.028 against 0.037 %;
  transport energy at the end −16 %; discs apart from the mesh 6.8 against 7.6 %; the centre of mass's
  displacement 0.019 against 0.026 pitches; net rotation 0.079° against 0.108°; still moving at the end 0.0056
  against 0.0065 pitches a pair. Level: chamfer (0.0545 in both), the kinetic energy at the end (1.8e-5
  against 1.7e-5), the bands (0.107, 0.160 against 0.109, 0.166), the field normal's error (11.0° against
  11.3°), the density's spread (0.194 against 0.198), the roughness. Behind on nothing. By window the arm is
  ahead from the start: horn crop 0.9662, 0.9779, 0.9796, 0.9817, 0.9842, 0.9841 at windows 20 to 70 against
  0.9562, 0.9701, 0.9700, 0.9743, 0.9788, 0.9787; front difference 0.0119, 0.0101, 0.0096, 0.0089, 0.0088,
  0.0086 against 0.0136, 0.0120, 0.0114, 0.0109, 0.0103, 0.0099. It passes the floor in every IoU and in the
  crop's difference. It equals D75's hand-set weight in the display with the physics measures at the twin's
  level, which D75 did not have.
  Windows discarded at the commit check: six of 83 (animations 5, 6, 12, 18, 30, 75). That is the base rate:
  the twin has seven, D70's render run two; the accepted candidate and its replay differ by 1e-7 to 1e-6 of
  the objective against a tolerance of 1e-7, with the replay noise measured at the start control at zero to
  5e-8. A guess that the moving weight caused them was checked on the records before any change and is wrong.
- **D80, the bounded share on the 40k gallery (pre-registered 2026-10-04 17:38 CDT at launch; `tmp/d80.sh`,
  `tmp/d80.queue`; server `repo_r70`; outputs beside D72's in `output/gpu/d72`).** D77's stated risk: at 40k
  the unbounded share (0.82–0.92) is what gives the render arm its lead over the physics-only twin
  (silhouette IoU +0.009 to +0.013 with the rule). One arm is added to D72's five on the 19 meshes: SA, the
  minimum spacing with the bounded share (D74's change does nothing at 40k). D72's 43 runs paused at 16:17 for
  the 300k experiments are queued behind it.
  Criteria per mesh, SA against S (the same recipe with the unbounded share) and against the pair B1, B2: the
  run's silhouette IoU not under S's by more than the pair's own difference; the thin share uncovered not over
  S's by more than the pair's difference; no new flag; SA still ahead of its twin SP in silhouette IoU and on
  the yardstick; the transport energy at the end not over S's.
  Expectation: the transport energy and the kinetic energy at the end fall towards the twin's; the yardstick's
  silhouette is at S's level or lower (as at 300k); the run's silhouette IoU within the pair's spread of S's
  on most meshes. Risk: thin targets (bob, V, fandisk), where the render term carries the thin parts at 40k.
  **Result on the first six meshes (2026-10-04 18:48 CDT; the arm was stopped there, the definition having
  been replaced by D81's): D77's form fails at 40k.** SA against S (`tmp/d80_table.py`): the run's silhouette
  IoU is lower on all six (−0.0059 bunny, −0.0057 dragon, −0.0081 A, −0.0074 armadilo, −0.0039 beast, −0.0082
  bimba; the pair's own difference is 0.0000–0.0006, beast's 0.0063); the thin share uncovered is higher on
  all six (+0.8 to +8.4 points); the runs stop at 17–34 commits where S's run 37–124; the transport energy at
  the end is 1.2 to 3.4 times S's; on the yardstick the exterior's silhouette is 1.03 to 1.74 times S's. SA is
  still ahead of its physics-only twin (silhouette IoU +0.004 to +0.006, thin −1.6 to −4.6 points). The
  expectation was wrong: the step is bounded, the merit keeps the first window's weight, and at 40k the runs
  end three times earlier, the early stop that D81's entry explains. D77's form is not a candidate at any N.
- **D79, whether the exterior's render terms carry the disc lattice's phase (a measurement, no run;
  2026-10-04 16:56 CDT; `tmp/ext_noise.py`; `output/gpu/d79/noise.log`).** A candidate for the short fine
  stage: the run reads its render terms on discs found on a lattice fixed in space, and the set of crossed
  cells changes as the body moves. The end frame of D70's LA runs is read 12 times with the lattice shifted by
  a random part of one cell. The spread of the terms over the shifts, relative: at 192 px silhouette 0.4 %
  (bunny and dragon), shading 0.1 %; at 126 px (the dragon) 0.4 % and 0.1 %; at 64 px 2.1 % and 1.3 %, shading 0.3 % and
  0.2 %. The same terms on the last four kept frames (12 steps apart) differ by ±9 % (silhouette) and ±3 %
  (shading). So the lattice is not what moves the terms from window to window: the state is. Refuted as the
  cause of the fine stage's end.
- **D76, the relation between N and the render resolution, and how fine the 300k surface is (one agent, at the
  user's request "N 과 px 간의 관계 조사하고, 현재 surface가 어디까지 detail 하게 갔는지도 조사"; received 2026-10-04
  17:37 CDT; `output/gpu/d76`: `twins.log`, `cost.log`, `bands_*.log`, `calib_*.log`; scripts `tmp/ag2_*`).**
  N and px. A pixel is 2 × extent / res; in pitches, at 64 / 96 / 126 / 192 px: 40k 1.6–1.9 / 1.1–1.3 / 0.8–1.0
  / 0.5–0.6; 100k 2.2–2.6 / 1.5–1.7 / 1.1–1.3 / 0.7–0.9; 300k 3.2–3.9 / 2.1–2.6 / 1.6–2.0 / 1.1–1.3; 1.5M
  5.4–6.6 / 3.6–4.4 / 2.8–3.4 / 1.8–2.2 (bunny–dragon; the cube law checked on samples of 1.2M and 2.4M). A
  4K display pixel is 0.09 pitches at 300k. Over every pair of a render run and its physics-only twin in the
  records (exterior path, 30k to 300k, 20 pairs): at the end of the 64-px stage the render arm is ahead on
  the yardstick in all 15 pairs with a pixel of 3.2 pitches or less and in none of the five at 3.9 (the
  dragon at 300k, so the bracket has one mesh on each side). Cost on the bunny: a window at 192 px is 3–4 s
  (30 %) dearer than at 96 px, the 64-px stage is not cheaper than 96 px (early windows find the discs 3–5
  times). Literature (ar5iv text): Mip-Splatting (Yu 2024) bounds a primitive from below by the sampling
  rate, ν̂ ≥ 2ν with a 3D filter Σ + (0.2 / ν̂) I; 3DGS (Kerbl 2023) uses a quarter resolution only as a warm-up of
  the first 500 iterations; none runs its main stage with a pixel of 3–4 primitive spacings.
  Detail. The share of the mesh's own relief the displayed surface carries, per octave band (end frames; the
  high-pass applied twice, so that smooth curvature does not leak into the fine bands; calibrated on the mesh
  blurred by a known Gaussian; nominal wavelength 2.7 / 5.4 / 10.8 / 21.6 pitches = 30 / 60 / 120 / 240 4K
  pixels = 2 / 4 / 9 / 17 % of the bunny):

  | carried share | 2.7 | 5.4 | 10.8 | 21.6 pitches |
  |---|---|---|---|---|
  | bunny: 300k sample (the ceiling at this N) | 0.36 | 0.66 | 0.86 | 0.95 |
  | bunny: P / L192 (no spacing rule) | 0.13 / 0.13 | 0.24 / 0.23 | 0.73 / 0.62 | 0.94 / 0.92 |
  | bunny: PA / LA (D70) | 0.03 / 0.04 | 0.18 / 0.18 | 0.73 / 0.61 | 0.92 / 0.90 |
  | bunny: samples of 1.2M / 2.4M | 0.58 / 0.66 | 0.83 / 0.87 | 0.95 / 0.96 | 0.98 / 0.98 |
  | dragon: 300k sample | 0.27 | 0.60 | 0.87 | 0.95 |
  | dragon: P / L192 | 0.21 / 0.22 | 0.44 / 0.32 | 0.83 / 0.74 | 0.95 / 0.95 |
  | dragon: PA / LA (D70) | −0.06 / −0.05 | 0.14 / 0.12 | 0.76 / 0.70 | 0.94 / 0.94 |
  | dragon: samples of 1.2M / 2.4M | 0.54 / 0.63 | 0.80 / 0.86 | 0.94 / 0.96 | 0.97 / 0.98 |

  The mesh's own relief in these bands is 0.024–0.036, 0.08–0.11, 0.24–0.31 and 0.7–0.9 pitches rms. A run
  carries more than half of the relief only above a wavelength of 9 to 13 pitches (7–10 % of the bunny, 95–145
  display pixels), a 300k sample above 6.5 to 7.5. Read against the calibration a sample's displayed surface
  is the mesh blurred by 1.2–1.4 pitches (the field's kernel alone is 0.9), a run by 1.6 (dragon P) to 2.5
  (LA). Each finer band costs eight times the particles: the 5.4-pitch band needs 2.4M at today's distance of
  a run from its sample, or no more particles if the run reached its own sample. The minimum spacing (D70)
  costs relief at 5.4 pitches (dragon 0.44 → 0.14, bunny 0.24 → 0.18) and below; the unbounded render arms
  carry 0.06–0.12 less than their twins at 10.8 pitches. D66's "not erased relief" is corrected by this: its
  slopes are reproduced, and against a known blur they mean a blur of 2 pitches against a sample's 1.2. The
  "2.4M" of D12 (300k × 8, half the spacing) agrees with the samples' cube law. Not done: the two faces of a
  thin part are averaged together above 2 pitches; one end frame per run; nothing at 256 or 384 px.
- **D77 and D78, the render's share of the step is bounded by the rule that calibrates it (pre-registered
  2026-10-04 16:17 CDT at launch of the first run; the user: "A, B 둘다 봐야", "Render gradient가 유의미한 결과를
  낼 때까지 계속 진행"; code: `window/solve.py`; servers `repo_r69` (D77, this change alone) and `repo_r70`
  (D78, with D74's pictures); `output/gpu/d77`, `output/gpu/d78`; `tmp/d77.sh`).** D73's first cause: the
  weight λ is set once so that λ|g_render| = 0.5 |g_phys|, and held; the physics gradient then decays 30-fold
  and the render gradient 1.4-fold, and from window 10 on the step is 91–96 % render. The change: the same
  rule with the same constant holds at every gradient as a bound on the step, the render gradient enters at
  min(λ, 0.5 |g_phys| / |g_render|), so its share of the step is at most a third. The merit does not change: E
  = L_phys + λ L_render with the one λ is what the line search and the outer acceptance compare, and the line
  search's slope is taken on E's own gradient. Five lines; no new constant. This is not the earlier balancer
  (which moved the merit's own weight every iteration and was a source of oscillation): the function being
  minimised stays one function.
  Primary sources (one agent, received 16:10 CDT; method and theory sections read from the arXiv sources, not
  cover to cover; Désidéri 2012 only through Sener and Koltun): PCGrad (Yu 2020) acts only on a negative
  cosine, "the original gradient remains unaltered" otherwise, and names the regime here, a dominating
  gradient whose improvement "may be significantly overestimated". Rules driven by loss ratios (GradNorm,
  Chen 2018) put "more weight on tasks whose losses are dropping more slowly", which is the term at its floor.
  MGDA's min-norm point (Sener and Koltun 2018) takes the physics gradient alone whenever the cosine exceeds
  the ratio of the norms (0.05 here), dropping the render from the step. The rules that bound an auxiliary
  gradient by the main one's norm are MetaBalance (He 2022: the auxiliary gradient is scaled to the target's
  norm when it exceeds it), MTAdam (Malkiel and Wolf 2020, anchored on the first term) and the gradient-
  statistics weights of physics-informed networks (Wang, Teng and Perdikaris 2021), none with a line search
  on a fixed merit. For that part: a direction d = −(g_p + β g_r) is a descent direction of E for any β ≥ 0
  when the cosine of the two gradients is not negative, which the projection guarantees, and backtracking on
  a descent direction keeps Zoutendijk's condition (Nocedal and Wright, Thm 3.2); through Adam's diagonal
  scaling that is not automatic, so the slope is computed on E's gradient as the code already does. Reported
  cautions: equalising norms lifts a noisy gradient to parity (MetaBalance: exact equality "might not be
  optimal for the target task"); Kurin 2022 and Xin 2022 find such methods land on the scalarisation front,
  the question being which weight, not which method.
  Runs: bunny and dragon 300k with D70's rule and the render on the exterior, 64 then 192 px (LD, D77) and
  126 then 192 px (LE, D78), against PA (the physics-only twin), LA (D70), LC (D74) and LW (D75). Read as D74,
  every kept frame, momentum included.
  Criteria: D74's list (the render arm at least as good as its twin on every measure, at the end and at
  every window from 20 on). What the cause predicts, for D77 alone: the render's share at most 0.34 in every
  window; the physics gain per window and the transport energy within a factor of two of the twin's from
  window 20 on; the kinetic energy at the end within a factor of ten of the twin's (it was 50–100 times).
  Expectation: the transport energy and the rest at the end come to the twin's level; the early lead of the
  render arm (windows 2–10, share 0.5–0.9 until now) shrinks; with 64 px (D77) the display ends level with
  the twin, not ahead, because the coarse picture has little to add; with 126 px (D78) the yardstick
  silhouette is below the twin's throughout and the display ahead at the end. Risk: at 40k, where the large
  share helps, the bound takes that help away (the gallery has to be read before this is a default).
  **Result, the bunny (2026-10-04 17:40 CDT; the dragon's two runs are under way): with the share bounded the
  render arm is ahead of its physics-only twin on every display measure, at every window from 20 on and at
  the end.** Last 20 kept frames (LD: the bound alone, 64 then 192 px; LE: the bound with D74's 126 px; PA the
  twin; LA and LC the unbounded runs of D70 and D74; the floor is D65's second sample):

  | run | front IoU / difference | thin crop | far side | field roughness | yardstick: exterior silhouette / shading | commits, seconds |
  |---|---|---|---|---|---|---|
  | floor | 0.9906 / 0.0063 | 0.9795 / 0.0071 | 0.9904 / 0.0059 | 9.8° | | |
  | PA (twin) | 0.9900 / 0.0075 | 0.9784 / 0.0086 | 0.9907 / 0.0068 | 3.2° | 0.000787 / 0.000568 | 87, 1156 |
  | LA | 0.9923 / 0.0084 | 0.9796 / 0.0094 | 0.9926 / 0.0068 | 4.5° | 0.000438 / 0.000537 | 47, 821 |
  | LC (D74) | 0.9927 / 0.0087 | 0.9822 / 0.0092 | 0.9933 / 0.0074 | 4.5° | 0.000414 / 0.000573 | 68, 1829 |
  | LD (D77) | 0.9937 / 0.0070 | 0.9858 / 0.0069 | 0.9936 / 0.0056 | 3.4° | 0.000326 / 0.000460 | 52, 1959 |
  | LE (D78) | 0.9938 / 0.0068 | 0.9881 / 0.0062 | 0.9936 / 0.0057 | 3.4° | 0.000315 / 0.000449 | 49, 823 |

  (LC's, LD's seconds were taken on GPUs shared with the gallery.) Against the twin, LD and LE: front IoU
  +0.0037 and +0.0038, difference −7 % and −9 %; thin crop IoU +0.0074 and +0.0097, difference −20 % and −28 %;
  far side IoU +0.0029, difference −18 % and −16 %. Both pass the floor in every IoU and in the crop's and the
  far side's difference. By window, LD against PA: crop IoU 0.9796 against 0.9708 at window 20 and 0.9814
  against 0.9751 at 30; front difference 0.0082 against 0.0099 and 0.0080 against 0.0092. At window 10 the
  bounded arm is behind (crop 0.9095 against 0.9257): the early lead of the unbounded arm is gone, as expected.
  What the cause predicted holds: the share is 0.33 in every window; the transport energy is 1.2 to 1.4 times
  the twin's at equal windows (1.03e-4 against 8.5e-5 at window 20, 4.8e-5 against 3.4e-5 at 40; it was 4 to 5
  times); the kinetic energy at the end 7.6e-6 and 6.9e-6 against the twin's 3.9e-6 (LA 1.9e-4). And the render
  terms themselves are lower than with the unbounded share: the yardstick's exterior silhouette 2.4 and 2.5
  times below the twin's (LA 1.8), below it at every window from 20 on (6.9e-4, 5.4e-4, 5.0e-4, 3.2e-4 at
  windows 20–50 against 1.26e-3, 9.3e-4, 8.7e-4, 7.9e-4), and the shading term 19 % and 21 % below the twin's,
  the first time a render arm lowers it on the bunny. The run's own 64-px silhouette term reaches 9e-5 where
  the unbounded arm stayed at 2.8e-4: with the physics objective not starved the render term is fitted
  better, not worse.
  The other measures against the twin (LD / LE / PA): run's silhouette IoU 0.9863 / 0.9864 / 0.9849; thin
  share uncovered 4.2 / 3.9 / 4.8 %; chamfer 0.0551 / 0.0551 / 0.0550; holes none; bands 0.086, 0.141 / 0.085,
  0.141 / 0.085, 0.142; the field normal's median error 9.5° / 9.5° / 9.3°; density's spread 0.175 / 0.176 /
  0.168; centre of mass's largest displacement 0.0045 / 0.0047 / 0.0077 pitches; net rotation 0.012° / 0.011° /
  0.011°; still moving at the end 0.0033 / 0.0037 / 0.0030 pitches a pair. Not yet at the twin's level:
  transport energy at the end 3.7e-5 against 2.1e-5 and the kinetic energy at the end twice the twin's, the
  roughness and the density's spread by a few per cent. Part of that is run length: the render arm stops at 52
  and 49 commits (the coarse stage ends at animation 47 and 43 with three rejections, LD's fine stage after 7
  commits), the twin at 87; the rejected windows have a positive physics gain (+2 to +3 %) and a merit
  raised by the render terms (the silhouette term up 9–28 % in the candidate). The stop is the outer merit's,
  not the lattice's (D79).
  **Result, the dragon (2026-10-04 18:40 CDT): ahead of the twin on every display measure, by less than the
  hand-set weight of D75.** Last 20 kept frames:

  | dragon 300k | front IoU / difference | horn crop | far side | roughness | yardstick: exterior silhouette / shading | thin uncovered | commits |
  |---|---|---|---|---|---|---|---|
  | PA (twin) | 0.9881 / 0.0102 | 0.9784 / 0.0139 | 0.9890 / 0.0108 | 5.0° | 0.001051 / 0.000685 | 5.3 % | 97 |
  | LD (D77) | 0.9918 / 0.0092 | 0.9847 / 0.0115 | 0.9919 / 0.0094 | 4.9° | 0.000426 / 0.000502 | 3.6 % | 65 |
  | LE (D78) | 0.9917 / 0.0094 | 0.9827 / 0.0124 | 0.9915 / 0.0099 | 4.9° | 0.000468 / 0.000550 | 4.6 % | 66 |
  | LW (D75) | 0.9928 / 0.0086 | 0.9863 / 0.0106 | 0.9923 / 0.0089 | 4.9° | 0.000390 / 0.000494 | 4.4 % | 103 |

  As relative changes of the error against the twin (1 − IoU, then the pictures' difference): LD front −31 %,
  −10 %; crop −29 %, −17 %; far side −26 %, −13 %. LE front −30 %, −8 %; crop −20 %, −11 %; far side −23 %,
  −8 %. LW −39 %, −16 %; −37 %, −24 %; −30 %, −18 %. The bunny's, for comparison: LD −37 %, −7 %; −34 %, −20 %;
  −31 %, −18 %; LE −38 %, −9 %; −45 %, −28 %; −31 %, −16 %. The yardstick's silhouette is 2.5 and 2.2 times
  below the twin's and its shading 27 % and 20 % below. The other measures (LD / LE / PA): run's silhouette
  IoU 0.9862 / 0.9840 / 0.9819; chamfer 0.0548 / 0.0548 / 0.0545; holes 0.021 / 0.044 / 0.037 %; bands 0.107,
  0.159 / 0.109, 0.162 / 0.109, 0.166; the field normal's error 11.0° / 11.1° / 11.3°; discs apart 8.1 / 8.0 /
  7.6 %; density's spread 0.213 / 0.217 / 0.198; centre of mass 0.017 / 0.017 / 0.026 pitches; net rotation
  0.083° / 0.088° / 0.108°; still moving at the end 0.010 / 0.007 / 0.0065 pitches a pair; transport energy at
  the end 1.4e-4 / 1.5e-4 / 6.8e-5; kinetic energy at the end 5.8e-5 / 3.0e-5 / 1.7e-5. The 126-px coarse
  picture adds nothing here once the share is bounded (LE is not ahead of LD). What separates LW from both is
  its length and its fine stage: 103 commits against 65, and ten commits at 192 px in which the horn crop
  goes from 0.9816 to 0.9876 and the front difference from 0.0098 to 0.0084; LD and LE end their coarse stage
  at about animation 60–65 and their fine stage after five to eight commits. The relief the surface carries is the
  twin's in these runs (D82's entry).
- **D75, the render weight at a tenth (a probe with an existing flag, not a candidate; pre-registered
  2026-10-04 15:59 CDT at launch; `--render_weight_scale 0.1`, server `repo_r66`; `output/gpu/d75`).** D73's
  first cause: the weight is set once, and the render's share of the step grows to 0.91–0.96 while the physics
  objective's gain falls 2 to 100 times below the twin's. The probe asks only whether that share is what
  starves it: with a tenth of the weight the share is about 0.05 in the first windows and 0.6–0.7 late. Dragon
  300k, D70's rule, render on the exterior (64 then 192 px), one run (LW), against LA and PA.
  Expectation: from window 20 on the transport energy and the physics gain lie between the twin's and LA's,
  nearer the twin's; the kinetic energy at the end is an order below LA's; the render arm's early lead (crop
  0.93 against 0.83 at window 10) is smaller; the coarse stage's yardstick silhouette is not above the twin's.
  If the transport energy does not come down, the share is not what starves the physics objective and A is
  not the weight's definition.
  **Result (2026-10-04 17:40 CDT): the share is the cause; with a tenth of the weight the render arm passes
  its twin on the dragon in every display measure.** LW: 103 commits (the twin 97; the coarse stage runs to
  animation 98, not cut short), share 0.48–0.83.

  | dragon 300k | front IoU / difference | horn crop | far side | roughness | yardstick: exterior silhouette / shading | thin uncovered | transport energy, kinetic energy at the end |
  |---|---|---|---|---|---|---|---|
  | floor | 0.9912 / 0.0074 | 0.9793 / 0.0112 | 0.9918 / 0.0072 | 13.6° | | | |
  | PA (twin) | 0.9881 / 0.0102 | 0.9784 / 0.0139 | 0.9890 / 0.0108 | 5.0° | 0.001051 / 0.000685 | 5.3 % | 6.8e-5, 1.7e-5 |
  | LA | 0.9897 / 0.0106 | 0.9724 / 0.0162 | 0.9888 / 0.0115 | 6.9° | 0.000577 / 0.000678 | 8.1 % | 5.5e-4, 1.7e-3 |
  | LW | 0.9928 / 0.0086 | 0.9863 / 0.0106 | 0.9923 / 0.0089 | 4.9° | 0.000390 / 0.000494 | 4.4 % | 1.5e-4, 4.2e-5 |

  Against the twin: front IoU +0.0047, difference −16 %; horn crop IoU +0.0079, difference −24 %; far side IoU
  +0.0033, difference −18 %; the yardstick's exterior silhouette 2.7 times below and its shading 28 % below
  (the first fall of the shading term on the dragon); the run's silhouette IoU 0.9846 against 0.9819; bands
  0.108, 0.160 against 0.109, 0.166; the field normal's error 10.9° against 11.3°; discs apart 7.6 % in both;
  the centre of mass 0.019 against 0.026 pitches; still moving at the end 0.0067 against 0.0065 pitches a
  pair. By window the arm is level at 20 (crop 0.9544 against 0.9562) and ahead from 30 on (0.9761 against
  0.9701, front difference 0.0113 against 0.0120). On the yardstick its exterior silhouette is 1.3 to 1.7
  times below the twin's through the whole 64-px stage (9.5e-4, 7.8e-4, 7.0e-4, 6.8e-4 at windows 30, 40, 60,
  80 against 1.46e-3, 1.29e-3, 1.05e-3, 1.03e-3): with less weight the coarse picture's fit does carry over.
  Behind the twin: the transport energy at the end (2.2 times), the kinetic energy at the end (2.5 times), the
  hole share (0.049 against 0.037 %), the density's spread (0.213 against 0.198), chamfer by 0.0002. The
  expectations held except one: the early lead did not shrink to nothing (crop 0.917 against 0.831 at window
  10). A tenth is a number put in by hand; D77 is the definition that should give this without one.
- **D74, the render pictures follow N (pre-registered 2026-10-04 15:59 CDT at launch; the user: "300K + 에서
  무조건 'render gradient'를 먹인 게 'physics only'보다 좋아야 해. 모든 metric과 momentum 보존까지 전부. A, B 둘다
  봐야 할 거 같네"; code: `scripts/pipeline_run.py`; server `repo_r68`; `output/gpu/d74`).** D73's second
  cause: at 300k the coarse picture's pixel is 3.2–3.8 pitches, where at 40k, the N at which the two
  resolutions were chosen, it is 1.9 (64 px) and 1.3 (96 px). Under `--loss_follows_n` the transport grid
  already follows the particle spacing above 40k particles; the render pictures did not. The change: they
  follow by the same factor, (N / 40000)^(1/3): at 300k 64 → 126 px and 96 → 188 px (a fine resolution given
  on the command line is taken as given: 192 here, as in D70). Three lines; nothing changes at 40k. On the
  exterior the discs' lattice is bounded by 0.92 pitches, so the coarse stage holds the same 89 460 discs on
  the dragon and 75 347 against 58 134 on the bunny; the pictures have four times the pixels.
  Runs: bunny and dragon 300k with D70's rule, render on the exterior, 126 then 192 px (LC), against D70's LA
  (64 then 192) and the physics-only twin PA (kept; the twin does not depend on the picture). Read as D70 on
  every kept frame, with the run's momentum.
  Criteria. The requirement is the user's: the render arm at least as good as its physics-only twin on every
  measure, at the end and at every window from 20 on: the display's IoU and difference (front, crop, far
  side), the yardstick's exterior silhouette and shading, the bands and normals, the density's spread, the
  thin share uncovered, chamfer, holes, the transport energy, the kinetic energy at the end, the centre of
  mass's displacement and the net angular move. D74 alone is judged on what its cause predicts: during the
  coarse stage the yardstick's exterior silhouette at or below the twin's at every window from 20 on, and the
  horn crop not behind the twin's by more than a run's spread (0.003).
  Expectation: the coarse stage's yardstick silhouette falls below the twin's, as at 40k and on the bunny; the
  horn crop comes within 0.003 of the twin's during the coarse stage; the transport energy stays several times
  the twin's and the kinetic energy at the end 50–100 times (D73's first cause, which this does not touch).
  Momentum as it is read (`scripts/probes/settled/momentum_probe.py`, every kept frame, uniform mass, a start
  at rest): the centre of mass's displacement; per pair of kept frames the net over the gross linear move and
  the net over the gross angular move about the centre of mass; the net rotation summed over the run; the mean
  particle move per pair over the last 20 pairs. The runs so far:

  | run | centre of mass, largest / end (pitches) | net / gross linear, mean | net / gross angular, mean | net rotation | still moving at the end (pitches a pair) | kinetic energy at the end |
  |---|---|---|---|---|---|---|
  | dragon P | 0.011 / 0.007 | 4.7e-3 | 7.3e-3 | 0.059° | 0.0065 | 1.4e-5 |
  | dragon L192 | 0.007 / 0.007 | 1.8e-3 | 3.4e-3 | 0.021° | 0.0143 | 1.6e-3 |
  | dragon PA | 0.026 / 0.023 | 6.2e-3 | 9.6e-3 | 0.108° | 0.0065 | 1.7e-5 |
  | dragon LA | 0.009 / 0.009 | 2.3e-3 | 3.9e-3 | 0.024° | 0.0132 | 1.7e-3 |
  | bunny P | 0.006 / 0.003 | 9.2e-3 | 8.4e-3 | 0.010° | 0.0028 | 3.1e-6 |
  | bunny L192 | 0.004 / 0.004 | 2.1e-3 | 8.0e-3 | 0.008° | 0.0071 | 5.1e-4 |
  | bunny PA | 0.008 / 0.003 | 1.08e-2 | 1.01e-2 | 0.011° | 0.0030 | 3.9e-6 |
  | bunny LA | 0.005 / 0.004 | 2.2e-3 | 5.6e-3 | 0.008° | 0.0074 | 1.9e-4 |

  In drift the render arm is not behind: the centre of mass moves by at most 0.03 pitches and the body turns
  by at most 0.11° in any run, and the render arm's net-to-gross ratios are 2 to 4 times lower (in part
  because its gross motion is larger). Where it is behind is rest: at the end its particles still move twice
  as far per pair (0.013 against 0.0065 pitches on the dragon, 0.0074 against 0.0030 on the bunny) and its
  kinetic energy is 50 to 100 times the twin's. The end-at-rest terms are part of the physics objective that
  D73 found starved.
  **Result (2026-10-04 17:40 CDT): what the cause predicts holds, and by itself it is not enough.** LC, 126
  then 192 px, the share unbounded (0.93–0.95):

  | run | front IoU / difference | thin crop | far side | roughness | yardstick: exterior silhouette / shading | commits |
  |---|---|---|---|---|---|---|
  | bunny PA / LA / LC | 0.9900 / 0.0075, 0.9923 / 0.0084, 0.9927 / 0.0087 | 0.9784 / 0.0086, 0.9796 / 0.0094, 0.9822 / 0.0092 | 0.9907 / 0.0068, 0.9926 / 0.0068, 0.9933 / 0.0074 | 3.2°, 4.5°, 4.5° | 0.000787 / 0.000568, 0.000438 / 0.000537, 0.000414 / 0.000573 | 87, 47, 68 |
  | dragon PA / LA / LC | 0.9881 / 0.0102, 0.9897 / 0.0106, 0.9892 / 0.0109 | 0.9784 / 0.0139, 0.9724 / 0.0162, 0.9701 / 0.0177 | 0.9890 / 0.0108, 0.9888 / 0.0115, 0.9890 / 0.0117 | 5.0°, 6.9°, 7.6° | 0.001051 / 0.000685, 0.000577 / 0.000678, 0.000700 / 0.000671 | 97, 89, 60 |

  The coarse stage's yardstick silhouette is below the twin's at every window from 20 on, on both meshes
  (dragon 1.47e-3, 1.28e-3, 8.8e-4, 8.5e-4 at windows 20–50 against 2.47e-3, 1.46e-3, 1.29e-3, 1.26e-3; bunny
  5.6e-4, 5.5e-4, 5.7e-4, 5.4e-4 against 1.26e-3, 9.3e-4, 8.7e-4, 7.9e-4): the criterion of D74 alone is met.
  The bunny's crop IoU is ahead of the twin from window 20 to the end (0.9775, 0.9800, 0.9817, 0.9823 against
  0.9708, 0.9751, 0.9757, 0.9772). The dragon's horn crop is ahead at window 20 (0.9624 against 0.9562), level
  at 40 and behind at 30 and 50 (0.9646 against 0.9701, 0.9659 against 0.9743) and at the end (0.9701 against
  0.9784): outside the 0.003 allowed. The pictures' difference stays behind on both (the bunny's front stops
  at 0.0083–0.0088 from window 20 while the twin goes on to 0.0075). The physics objective is as starved as
  before, as expected: transport energy at the end 1.7e-4 (bunny) and 8.7e-4 (dragon) against the twins'
  2.1e-5 and 6.8e-5, particles still moving 0.005 and 0.027 pitches a pair at the end against 0.003 and 0.007.
  So the finer coarse picture removes the misfit of the coarse stage and leaves the starvation; it is kept
  as part of D78.
- **D73, why the render arm is behind its physics-only twin at 300k: what the step is made of (a reading of
  kept records and logs, no run; 2026-10-04 15:50 CDT; the user: "왜 physics가 render를 섞은 것 보다 좋은지 분석해서
  말해 줘"; `tmp/why_rows.py` on each run's record with its yardstick and display logs).** Per committed window:
  the sizes of the two gradients as the optimiser adds them, the physics objective's gain, the transport
  energy, the run's own 64-px silhouette term, and of the same window the yardstick's exterior silhouette at 96
  px and the display's horn crop. Dragon 300k, both arms with D70's rule (PA physics-only, LA render on the
  exterior, 64 px to animation 83):

  | window | \|g_phys\| PA / LA | λ\|g_rend\| LA | render's share | physics gain PA / LA | transport PA / LA | own silhouette, 64 px, PA / LA | yardstick, 96 px, PA / LA | crop IoU PA / LA |
  |---|---|---|---|---|---|---|---|---|
  | 2 | 9.3e-4 / 9.4e-4 | 9.9e-4 | 0.51 | +0.17 / +0.19 | 1.34 / 1.32 | 2.0e-1 / 1.6e-1 | 1.8e-1 / 1.2e-1 | 0.145 / 0.363 |
  | 10 | 1.8e-4 / 1.6e-4 | 1.9e-3 | 0.92 | +0.44 / +0.35 | 3.4e-2 / 1.2e-2 | 2.2e-2 / 3.1e-3 | 1.7e-2 / 4.4e-3 | 0.831 / 0.931 |
  | 20 | 3.4e-5 / 6.3e-5 | 6.4e-4 | 0.91 | +0.140 / +0.023 | 1.5e-3 / 3.4e-3 | 2.3e-3 / 8.2e-4 | 2.5e-3 / 2.5e-3 | 0.956 / 0.953 |
  | 30 | 1.1e-5 / 4.9e-5 | 7.3e-4 | 0.94 | +0.091 / +0.001 | 4.5e-4 / 2.3e-3 | 1.4e-3 / 8.8e-4 | 1.5e-3 / 1.9e-3 | 0.970 / 0.962 |
  | 40 | 1.0e-5 / 4.2e-5 | 7.7e-4 | 0.95 | +0.073 / +0.032 | 2.4e-4 / 1.7e-3 | 1.1e-3 / 3.7e-4 | 1.3e-3 / 1.4e-3 | 0.970 / 0.965 |
  | 60 | 7.7e-6 / 3.3e-5 | 8.2e-4 | 0.96 | +0.051 / +0.013 | 1.1e-4 / 8.4e-4 | 1.0e-3 / 3.8e-4 | 1.05e-3 / 1.23e-3 | 0.979 / 0.962 |
  | 80 | 7.0e-6 / 3.0e-5 | 5.1e-4 | 0.94 | +0.024 / +0.003 | 7.6e-5 / 6.4e-4 | 9.7e-4 / fine stage | 1.03e-3 / 5.7e-4 | 0.978 / 0.973 |

  1. The weight is set once, at the first window (at window 2 the two gradients are equal, share 0.51). The
  physics gradient then falls 30-fold by window 30, as the body arrives; the render gradient falls 1.4-fold.
  From window 10 on the render gradient is 10 to 25 times the physics gradient and the step is 91–96 % render.
  The cosine between the two is +0.1 to +0.25: they do not oppose each other, one is simply larger.
  2. The physics objective is starved of the step: its gain per window is 6 times smaller than the twin's at
  window 20, 100 times at window 30, 2 to 7 times after; the transport energy is 2 to 8 times the twin's at the
  same window and 8 times at the end. The same without the rule (D64's run against D60's: share 0.90–0.93,
  transport 3 to 5 times the twin's at windows 20–60), so this is not the rule's doing. Until window 10 the
  render arm is ahead on everything (crop 0.931 against 0.831): while the body is far from its target the
  coarse picture is a good guide.
  3. What that share buys after window 20: the run's own 64-px silhouette term 3 times below the twin's (3.7e-4
  against 1.1e-3 at window 40), and on the 96-px yardstick an exterior silhouette that is not lower but higher
  than the twin's from window 30 to the end of the stage (1.9e-3 against 1.5e-3, 1.4e-3 against 1.3e-3, 1.23e-3
  against 1.05e-3). A 64-px pixel is 3.8 pitches on the dragon: the term is fitted at its own resolution in a
  way that does not hold one resolution finer, and the horn crop is behind from window 20. The fine stage,
  whose picture does see the difference (5.7e-4 at window 80), comes at animation 79–84 and lasts 11–22
  windows.
  4. Bunny 300k, the same composition (share 0.95, transport 4 to 5 times the twin's), but there the 64-px fit
  carries over: the yardstick is 1.35 times below the twin's at window 20 (9.3e-4 against 1.26e-3) and twice
  below once the fine stage runs (4.3e-4 against 8.7e-4 at window 40); the IoU is ahead.
  5. 40k (D72's dragon, both arms with the rule), the same share (0.82–0.92) and two to three times the twin's
  transport energy; there the twin's own 64-px silhouette term stalls at 3.1e-3 to 4.5e-3 with a physics gain
  near zero from window 40 (+0.006, −0.010) and 13 % of the thin target uncovered, an error a 64-px picture
  sees (a pixel is 1.9 pitches at 40k), and the render arm removes it (9.8e-4 at window 20): silhouette IoU
  +0.009 to +0.013 on the six meshes read so far.
  Conclusion: at 300k the render arm is behind for two reasons that act together. Its weight is fixed at the
  first window while the physics gradient decays, so after window 10 the step is a render step and the physics
  objective gets a few per cent of it. And at 300k that step is taken on a picture whose pixel is 3 to 4
  pitches, in which the physics-only arm (which with the rule already stands at the floor on the dragon's
  crop) has no error left to see; the stage whose picture is fine enough comes late and is short. At 40k the
  second reason is reversed: the picture is finer against the particles and the physics-only arm leaves more.
  Not measured: whether a bounded share, or a fine picture from the window where the coarse term stops
  falling, would put the 300k render arm ahead. Those are two definitions (the weight's calibration, the
  stage's trigger), one experiment each.
- **D72, the minimum spacing on the 40k gallery (pre-registered 2026-10-04 14:32 CDT at launch; the user's rule
  of 2026-09-24: a change holds on the whole gallery before it is adopted; `tmp/d72.sh`, `tmp/d72.queue`; server
  `repo_r66`, D70's rule as it is, D71's change not in it; `output/gpu/d72`).** 19 meshes at 40k, the default
  recipe, five arms: two repeats without the rule (B1, B2; a single run's spread, 2026-10-03), with the rule
  (S), and the physics-only twins of both (BP, SP), 95 runs. Read: every window of each run's record; the
  render yardstick on every kept frame of all five; the offset probe on every kept frame and the arrangement
  at the end for B1, S and SP. The kept frames are deleted after the probes.
  Criteria, per mesh, S against the pair B1, B2: the silhouette IoU not under the pair's lower value by more
  than the pair's own difference; the thin share uncovered not over the pair's higher value; no guard, hole or
  ejection flag the pair does not have; the two finer bands and the density's spread not over B1's; commits
  and seconds within a quarter of the pair's range. The render's effect: S against SP on the yardstick, beside
  B1 against BP.
  Expectation: at 40k a cell holds 24 particles, not 184, so there is less unevenness under the surface to
  remove; the bands fall by a tenth to a fifth, the run's own metrics stay inside the pair's spread, the thin
  share uncovered falls on most meshes; beast (ejection) and C (early stop) stay inside their own spread.
  Risk: sheets a few particles thick at 40k (bob's ring, V, fandisk's edges) fatten.
  **Result (2026-10-05 01:05 CDT, all 19 meshes, every arm; `tmp/d72_table.py`): the minimum spacing passes
  the gallery.** S (the rule, the held weight of that time) against the pair B1, B2 without it: the run's
  silhouette IoU at or above the pair's lower value less the pair's difference on 19 of 19 (+0.0001 to +0.0044;
  C −0.0025 against a pair differing by 0.0024); the thin share uncovered lower on 18 (−1.2 to −9.8 points;
  teapot 0); the bands below 4 and 4–11 pitches and the density's spread under B1's on 19 of 19; no flag the
  pair does not have (beast's gate fails in all three, as before); length within a quarter of the pair's on 17
  (bunny 48 commits against 30–37, V). The thin sheets did not fatten in any measure read: bob's thin share
  −6.8 points, V's −5.1, fandisk's −2.2. Expectation partly wrong: the bands fell, and so did the thin share,
  by more than expected (the rule fills thin parts the pair leaves uncovered).
  D89's one resolution at 40k (arm SG, with D81's weight; the 13 meshes on `repo_r76`, six on `repo_r73`):
  against its physics-only twin SP the silhouette IoU is higher on 19 of 19 (+0.0026 to +0.0129) and the thin
  share lower on 19 (−0.4 to −12.7 points); against S (the held weight) lower on 18 (−0.0019 to −0.0075; C
  +0.0013) with the transport energy at the end 0.10 to 0.85 times S's — D85's reading of D81's weight, now on
  every mesh.
- **D71, the minimum spacing lets no material out through the free surface (pre-registered 2026-10-04 13:26
  CDT at launch; code: `kernels.k_update`; server `repo_r67`; `output/gpu/d71`, `tmp/d71.sh`).** The cause is
  D70's last paragraph: the rule's move has a mean of zero inside the body and an outward mean in the outermost
  layer (+0.26 and +0.46 pitches summed over a run), because a layer particle has no neighbour outside. The
  rule is meant to redistribute material inside the body, and its definition has no condition at the body's
  boundary. The change is that condition and nothing else: a particle of the outermost layer (the relaxation's
  mask and normal, frozen at the window's start) keeps only the part of its move along the surface. Its
  coordinate along the normal stays with the two rules that already own it, the relaxation and u. Two lines in
  the kernel; r, the neighbour rows and every particle under the layer are as in D70. Not done: the half a
  layer particle declines is not given to its partner (a pair across the layer's boundary closes at half the
  rate along the normal).
  Runs: D70's four again (bunny and dragon 300k; physics-only PB, render on the exterior at 192 px LB), read
  as D70 on every kept frame, with `spacing_flux_probe.py` on each.
  Criteria: the mean offset within 0.05 pitches of the runs without the rule on all four (D70 failed three);
  the outward move under the layer (to 4 pitches) summed over the run at most half of D70's; D70's gains kept:
  the band below 4 pitches at or under a sample's, the 4–11 band's excess at most half of that without the
  rule, the density's spread at or under 0.21 (dragon with render 0.24), the pictures' difference within 0.0005
  of D70's, no solid region's IoU lower than D70's by more than 0.0008; the thin share uncovered read against
  D70's and the runs without the rule, which separates fattening from the swelling. The render's effect is
  read as before (LB against PB on the yardstick and the display, every frame), and where the render arm
  stops.
  Expectation: the bunny's mean offset returns to within 0.03 of the runs without the rule; the dragon keeps
  part of it (its outward move is also under the layer), within 0.08; the bands hold; the thin share uncovered
  rises part of the way back (to 7–10 %). Risks: particles pile under the layer (the density just under the
  surface rises), particles from under the layer pass between the layer's particles and become the layer.
  **Result (2026-10-04 14:55 CDT): the premise is refuted, the change does nothing, and it is taken out of the
  code (the kernel, the trajectory and the test are D70's again).** PB and LB are the runs with the change, PA
  and LA D70's (last 20 kept frames; `output/gpu/d71`, `twin_{bunny,dragon}300k.png`):

  | run | mean offset | below 4 / 4–11 | front | thin crop | far side | commits |
  |---|---|---|---|---|---|---|
  | bunny PA | +0.308 | 0.085 / 0.142 | 0.9900 / 0.0075 | 0.9784 / 0.0086 | 0.9907 / 0.0068 | 87 |
  | bunny PB | +0.303 | 0.086 / 0.144 | 0.9899 / 0.0080 | 0.9777 / 0.0092 | 0.9897 / 0.0069 | 71 |
  | bunny LA | +0.382 | 0.087 / 0.145 | 0.9923 / 0.0084 | 0.9796 / 0.0094 | 0.9926 / 0.0068 | 47 |
  | bunny LB | +0.385 | 0.089 / 0.149 | 0.9926 / 0.0087 | 0.9826 / 0.0089 | 0.9932 / 0.0071 | 36 |
  | dragon PA | +0.701 | 0.109 / 0.166 | 0.9881 / 0.0102 | 0.9784 / 0.0139 | 0.9890 / 0.0108 | 97 |
  | dragon PB | +0.695 | 0.111 / 0.166 | 0.9888 / 0.0097 | 0.9793 / 0.0143 | 0.9896 / 0.0106 | 102 |
  | dragon LA | +0.741 | 0.115 / 0.168 | 0.9897 / 0.0106 | 0.9724 / 0.0162 | 0.9888 / 0.0115 | 89 |
  | dragon LB | +0.749 | 0.114 / 0.170 | 0.9886 / 0.0114 | 0.9675 / 0.0195 | 0.9870 / 0.0118 | 74 |

  The mean offset did not move (−0.005, +0.003, −0.006, +0.008 pitches against the 0.04–0.15 to be removed);
  the bands, the arrangement (spread 0.168, 0.204, 0.194, 0.245) and the thin share uncovered (4.7, 6.2, 5.5,
  7.9 %) are D70's; the physics-only pictures are D70's within a run's spread. Both expectations were wrong.
  What was wrong in the premise (`scripts/probes/settled/particle_depth_probe.py`, the particles' own signed
  distance to the mesh on the end frames; the mesh is fitted by its bounding box, so the values are read
  against each other, not against zero):

  | bunny | outermost layer's mean | particles outside the mesh | beyond +0.5 pitches | the display's mean offset |
  |---|---|---|---|---|
  | target sample | −0.155 | 2.73 % | 1.04 % | +0.202 |
  | P | −0.107 | 3.38 % | 1.37 % | +0.269 |
  | PA / PB | −0.160 / −0.163 | 3.09 / 3.11 % | 1.16 / 1.19 % | +0.308 / +0.303 |
  | L192 | −0.118 | 3.38 % | 1.39 % | +0.283 |
  | LA / LB | −0.091 / −0.102 | 3.33 / 3.39 % | 1.40 / 1.40 % | +0.382 / +0.385 |

  | dragon | outermost layer's mean | particles outside the mesh | beyond +0.5 pitches | the display's mean offset |
  |---|---|---|---|---|
  | target sample | +0.243 | 9.24 % | 4.62 % | +0.562 |
  | P | +0.374 | 11.01 % | 6.07 % | +0.568 |
  | PA / PB | +0.296 / +0.294 | 10.49 / 10.43 % | 5.59 / 5.54 % | +0.701 / +0.695 |
  | L192 | +0.291 | 10.74 % | 6.18 % | +0.594 |
  | LA / LB | +0.324 / +0.328 | 10.75 / 11.04 % | 6.08 / 6.29 % | +0.741 / +0.749 |

  With the rule the particles do not stand further out than without it: physics-only they stand further in
  (the layer by 0.05 and 0.08 pitches, fewer particles outside the mesh, the bunny's layer where a sample's
  is). The body did not swell. What is further out is the display's zero set over particles that are not: the
  field |q − x̄(q)| = 0.8 a reads an even arrangement 0.04–0.15 pitches further out than an uneven one with the
  same or a more inward standing (the layer holds 5.9 % of the bunny's particles with the rule and 4.1 %
  without; a rough surface dips into its gaps and its mean offset is lower for it). So D70's criterion "the
  body not further out" measured the display's reading of the arrangement, and its failure says nothing
  against the rule; D70's paragraph "why the body is further out" drew a conclusion its measurement did not
  carry: the rule's move on a layer particle does have an outward mean (that part stands), but the objective
  holds the layer where it is, and removing that move changes nothing that was read. The thin share uncovered
  (13–18 % → 5–8 %) is therefore the rule's and not a swelling's.
  Left open by this: the display draws a run with the rule 0.10–0.18 pitches further out than it draws the
  target sample (+0.31–0.38 against +0.20 on the bunny), because the target's picture is drawn from a random
  sample, rougher (9.8°, 13.6°) than the runs now are (3–7°). The reference picture and the floor of D65 are a
  random sample's; a run that is more even than a random sample is no longer measured against something
  better than itself. That is a matter of the yardstick, not of the run.
  The render's effect, D70 and D71 together (two runs per mesh with render, two without): on the bunny the
  render arm is ahead in IoU in both pairs (front +0.0023 and +0.0027, crop +0.0012 and +0.0049, far side
  +0.0019 and +0.0035) and not in the pictures' difference (−0.0003 to +0.0009); on the dragon it is not ahead:
  front +0.0016 and −0.0002, horn crop −0.0060 and −0.0118, difference worse by 0.0004–0.0052, and its
  yardstick silhouette is 1.8 times below the twin's in one run (0.000577 against 0.001051) and not below in
  the other (0.000978 against 0.000936). Where it loses, read on every frame: during the 64-px stage (to window
  79–84 on the dragon) the render arm's horn crop is behind the twin's at every window from 20 on (LA 0.9508,
  0.9592, 0.9662, 0.9694, 0.9598, 0.9717 at windows 20, 30, 40, 50, 60, 80 against PA's 0.9543, 0.9695, 0.9686,
  0.9752, 0.9765, 0.9792), having been ahead at window 10 (0.9278 against 0.7820); the fine stage then lasts
  11 or 22 windows before the outer merit ends it. A 64-px pixel is 3.8 pitches on the dragon at 300k (3.2 on
  the bunny; 1.3 and 1.1 at 192 px). With the rule in place the physics-only arm is the better picture on the
  dragon, and the render arm's gain is the bunny's IoU. Recorded; the stage's trigger and the merit's stop are
  not changed here.
- **D70, a minimum spacing in the position update (pre-registered 2026-10-04 11:55 CDT; the user: "A를 진행 해
  보자", "원인을 감추지만 말아 줘. overengineering도 금지 … 항상 '알고리즘' core를 수정", "'항상' 랜더 Gradient가
  영향을 끼치는지 확인"; code: `kernels.k_update`, `--min_spacing 0.9`; server `repo_r66`; `output/gpu/d70`).**
  The cause (D66–D69): the grid holds about 200 particles a cell and cannot see two of them pressed together;
  where the material is stretched and torn the density under the surface ends uneven, and the drawn surface
  with it. The position update already holds the two rules that act below the grid (the bonds' re-joining and
  the outer layer's relaxation), and its comment left compression to the grid. One rule is added there: a
  particle is moved away from each of its 16 nearest (frozen at the window's start) that is nearer than r, by
  half the overlap, over one window (the bonds' fraction per step); r is 0.9 of the pitch the rest volume
  gives, the spacing of a Poisson-disk sample of the body's density (random sequential packing stops at a
  packing fraction of 0.38). Twelve lines in the kernel, the neighbour rows in the window's setup; no surface
  rule, no kernel sums. On kept frames (D69) ten such projections brought the band below 4 pitches to a
  sample's level on the bunny and removed half of its excess on the dragon, a quarter to two fifths of the
  4–11 band's, with no material leaving the surface; the body's surface moved out by 0.006–0.01 pitches a
  projection, which in a run the objective has to hold.
  Runs, 300k, bunny and dragon, each with the minimum spacing: the physics-only twin (kept, as D60's) and the
  render on the exterior at 192 px (D64's setting); against D60's and D64's runs without it. Read on every kept
  frame: the surface's offset from the mesh by band and the normals (D66's probe), the density's spread and the
  nearest particle under the surface, the display against the target and against D65's floor, the render terms
  on the yardstick, and the render's effect under the new rule (with-render against its own physics-only twin,
  as D62).
  Criteria: under the surface the density's spread and the nearest particle's 5th percentile at a sample's
  level (0.21; 0.36 pitches) at the end; the band below 4 pitches within 5 % of a sample's and the 4–11 band's
  excess at most half of what it is without the rule (bunny 0.039–0.045, dragon 0.075–0.104); the display's
  difference against the target a third of the way from the present runs to the floor or better (bunny front
  0.0096 → 0.0085, dragon 0.0127 → 0.0109), with no solid region's IoU lower than without the rule by more than
  the runs' spread (0.0008); the body not further out (mean offset within 0.05 pitches of the run without the
  rule); the transport energy and the run's length within a quarter of the run without it; the render's effect
  on its own terms kept (silhouette at least twice below the twin's at the end).
  Expectation: the fine band reaches a sample's; the 4–11 band's excess falls by a third to a half; the
  pictures' difference by a fifth to a third of the gap; a window costs under 5 % more. Risks: thin sheets
  fatten (a sheet one or two particles thick is pushed to r), the body swells where clumps were, the transport
  slows because the projection works against a control that presses material together.
  Primary sources (one agent; received 12:06 CDT, after the runs were started; five of ten papers read in full,
  the others through their authors' theses or later papers, as marked in its report): the rule is of the kind
  hybrid particle-grid methods use, Ando and Tsuruno 2011 (Eq. 7) and Ando, Thuerey and Tsuruno 2012 (Eq. 18):
  every step a particle is pushed away from each neighbour within a distance d, along the unit vector, weighted
  1 − r²/d², with no rule for the free surface; they re-sample the velocity afterwards and name the fault of
  the isotropic form, it "can lead to a thickening of thin surfaces, or a smearing out of sharp features". In
  MPM, Baumgarten and Kamrin 2023 shift material points and leave "material and state properties unchanged",
  as here (F, C and the velocity are not touched); their grid-based correction is a node quantity and cannot
  order particles below a cell, and of the rules they compare the neighbour-based one of Xu, Stansby and
  Laurence 2009 (δr = C α Σ (r̄²/r²) n over a fixed list of neighbours) gave the evenest arrangement. The
  concentration-gradient shifting of SPH (Lind 2012, Khayyer 2017, Sun 2017–2019) needs a free-surface rule
  (tangential projection, eigenvalue thresholds, a step limit) to keep the surface in place, which is what
  D69's first candidate lacked. What differs here from Ando's form: the push is the overlap below r (zero
  beyond it), not a kernel of the distance, so a pair at r or farther is left alone.
  **Result (2026-10-04 13:27 CDT; the four runs to their own stop; every reading on all kept frames, the tables
  the mean of the last 20; `output/gpu/d70`: `t_`, `a_`, `o_`, `e_`, `f_` logs, `twin_{bunny,dragon}300k.png`,
  local `output/video_2026-10-04/d70_min_spacing/`).** P and L192 are D60's and D64's runs without the rule, PA
  and LA the same two with it.

  | run | seconds | commits | silhouette IoU | thin uncovered | chamfer | transport energy at the end |
  |---|---|---|---|---|---|---|
  | bunny P | 975 | 79 | 0.9846 | 13.2 % | 0.0577 | 3.8e-5 |
  | bunny L192 | 1258 | 63 | 0.9859 | 13.4 % | 0.0578 | 1.9e-4 |
  | bunny PA | 1156 | 87 | 0.9849 | 4.8 % | 0.0550 | 2.1e-5 |
  | bunny LA | 821 | 47 | 0.9862 | 6.7 % | 0.0553 | 1.6e-4 |
  | dragon P | 1684 | 102 | 0.9823 | 16.5 % | 0.0583 | 1.4e-4 |
  | dragon L192 | 4820 | 69 | 0.9836 | 18.3 % | 0.0588 | 1.0e-3 |
  | dragon PA | 1781 | 97 | 0.9819 | 5.3 % | 0.0545 | 6.8e-5 |
  | dragon LA | 2582 | 89 | 0.9832 | 8.1 % | 0.0552 | 5.5e-4 |

  Under the surface (end frame, 1.5–4 pitches under the outer layer): the density's spread is 0.168 (bunny PA),
  0.197 (LA), 0.198 (dragon PA), 0.238 (LA), against 0.29 and 0.43 without the rule and a sample's 0.21; every
  particle's nearest neighbour is at r (median 1.07–1.08 display pitches, 5th percentile 1.05–1.07; a sample's
  0.71 and 0.36). That is not a jammed packing (the packing fraction is about 0.41): a pair that was pressed is
  moved to r and left there, and in a random sample nearly every particle has such a neighbour.
  The surface against the mesh, in pitches (D66's probe; a sample: bunny 0.103 / 0.132 / +0.202 / 11.5°, dragon
  0.121 / 0.153 / +0.562 / 12.6°):

  | run | below 4 pitches | 4–11 | mean offset | field normal's error, median |
  |---|---|---|---|---|
  | bunny P | 0.118 | 0.176 | +0.269 | 13.1° |
  | bunny L192 | 0.118 | 0.169 | +0.283 | 12.9° |
  | bunny PA | 0.085 | 0.142 | +0.308 | 9.3° |
  | bunny LA | 0.087 | 0.145 | +0.382 | 9.7° |
  | dragon P | 0.164 | 0.257 | +0.568 | 16.9° |
  | dragon L192 | 0.161 | 0.228 | +0.594 | 16.6° |
  | dragon PA | 0.109 | 0.166 | +0.701 | 11.3° |
  | dragon LA | 0.115 | 0.168 | +0.741 | 11.1° |

  The display against the target (IoU / mean picture difference; the floor is D65's second sample):

  | run | front | thin crop | far side | field roughness |
  |---|---|---|---|---|
  | bunny floor | 0.9906 / 0.0063 | 0.9795 / 0.0071 | 0.9904 / 0.0059 | 9.8° |
  | bunny P | 0.9868 / 0.0099 | 0.9705 / 0.0116 | 0.9883 / 0.0086 | 11.2° |
  | bunny L192 | 0.9912 / 0.0096 | 0.9794 / 0.0100 | 0.9926 / 0.0079 | 11.2° |
  | bunny PA | 0.9900 / 0.0075 | 0.9784 / 0.0086 | 0.9907 / 0.0068 | 3.2° |
  | bunny LA | 0.9923 / 0.0084 | 0.9796 / 0.0094 | 0.9926 / 0.0068 | 4.5° |
  | dragon floor | 0.9912 / 0.0074 | 0.9793 / 0.0112 | 0.9918 / 0.0072 | 13.6° |
  | dragon P | 0.9875 / 0.0118 | 0.9717 / 0.0160 | 0.9871 / 0.0130 | 22.2° |
  | dragon L192 | 0.9898 / 0.0127 | 0.9726 / 0.0182 | 0.9901 / 0.0137 | 20.9° |
  | dragon PA | 0.9881 / 0.0102 | 0.9784 / 0.0139 | 0.9890 / 0.0108 | 5.0° |
  | dragon LA | 0.9897 / 0.0106 | 0.9724 / 0.0162 | 0.9888 / 0.0115 | 6.9° |

  Criteria. Met: no pressed pairs and the density's spread at or under a sample's (dragon LA 0.03 above it); the
  band below 4 pitches under a sample's on both meshes (0.085–0.087 against 0.103, 0.109–0.115 against 0.121);
  the 4–11 band's excess over a sample cut from 0.037–0.044 to 0.010–0.013 (bunny) and from 0.075–0.104 to
  0.013–0.015 (dragon), more than the half asked; the pictures' difference a third of the way to the floor or
  more (bunny front 0.0099 → 0.0075 physics-only, 0.0096 → 0.0084 with render; dragon 0.0118 → 0.0102, 0.0127 →
  0.0106); the transport energy lower, not higher (end value −45 % and −52 % physics-only, −12 % and −45 % with
  render); the physics-only runs 6–19 % longer. Not met: the body is further out, by 0.04 (bunny PA), 0.10
  (bunny LA), 0.13 (dragon PA) and 0.15 pitches (dragon LA) against the 0.05 allowed; the dragon's far side with
  render is 0.0013 lower in IoU than without the rule (allowed 0.0008); the render arm's silhouette on the
  yardstick is 1.8 times below its twin's on both meshes, not twice. (The first of these is withdrawn by D71:
  the particles are not further out, the display's zero set is; see there.) The expectation held for the bands and was
  low for the pictures (the physics-only bunny closed two thirds of the gap to the floor); the risk "the body
  swells" came true, "thin sheets fatten" is not separable from it here (the thin share uncovered falls from
  13–18 % to 5–8 %, part of which is the surface standing further out), "the transport slows" did not.
  The render's influence under the rule (first-window λ 0.206 bunny, 0.336 dragon; median g_share 0.94 on both;
  the λ = 0 twin is PA; path as D62, the exterior's silhouette term through u and dFc). On the yardstick the
  exterior's silhouette term ends at 0.000438 against the twin's 0.000787 (bunny) and 0.000577 against 0.001051
  (dragon), 1.8 times below; without the rule the ratio was 2.5 and 2.4 (0.000510 against 0.001259, 0.000542
  against 0.001291). The render arm's own level did not move; the twin came down, because the rule alone lowers
  the exterior's silhouette term by 37 % and 19 %. Shading: no effect (0.000537 against 0.000568, 0.000678
  against 0.000685). On the display the render arm is ahead in IoU on the bunny (+0.0023 front, +0.0012 crop,
  +0.0019 far side) and on the dragon's front (+0.0016), behind on the dragon's horn crop (−0.0060) and behind
  in the pictures' difference on both (0 to +0.0023). It does not change what the rule gives: the bands and
  the roughness are the same in both arms within 0.006 pitches and 2° (the density's spread is 0.03–0.04
  higher with render). Part of the
  difference is the stop: at window 40 the bunny's render arm is ahead in front IoU (0.9918 against 0.9891) and
  level in difference (0.0084 against 0.0086); its fine stage begins there and ends at window 55, three
  candidates rejected by the outer merit while the physics gain is positive (+0.003 to +0.005, reversal −0.45
  to −0.51), 47 commits against the twin's 87, and the twin lowers its difference from 0.0086 to 0.0075 in the
  windows the render arm does not run. That is the merit alternation parked on 2026-09-30, now the thing that
  ends the render arm; recorded, not changed here.
  Why the body is further out (measured, `scripts/probes/settled/spacing_flux_probe.py`, every kept frame of
  PA): the rule's own move is
  split along the outward normal by depth under the outermost layer. Deeper than 4 pitches its mean is zero on
  every frame (within 0.0001 pitches a window): there the rule only redistributes. In the outermost layer it is
  outward: +0.039 pitches a window on the first frame (+0.038 on the target sample itself), +0.004 (bunny) and
  +0.008 (dragon) at window 20, +0.0005 and +0.0011 at the end, where it is half of the layer's whole move;
  summed over the run +0.26 pitches (bunny) and +0.46 (dragon); under the layer to 1.5 pitches +0.04 and +0.21,
  from 1.5 to 4 pitches +0.03 and +0.13. A layer particle has neighbours under it and none outside, so half of
  every overlap is an outward move nothing balances: the rule as written carries material out through the free
  surface, and the objective holds only part of it back. The rule is missing its condition at the free surface
  (the same fault as D69's first candidate, at a smaller size; Ando's form shares it). D71 is that condition.
  (Corrected by D71, 14:55 CDT: the outward mean of the rule's move on the layer is measured and stands; "the
  objective holds only part of it back" was inferred, not measured, and is wrong: the particles end no further
  out than without the rule, and taking the layer's outward move away changes nothing.)
- **D69, what is uneven under the surface, and whether evening it evens the surface (measurements on kept
  frames, no code of the run changes; pre-registered 2026-10-04 11:43 CDT; the user: "A를 진행 해 보자 … 원인을
  감추지만 말아 줘. overengineering도 금지 … 항상 '알고리즘' core를 수정"; `tmp/arrangement.py`,
  `tmp/shift_test.py`; `output/gpu/d69`).** Before anything is put into the rollout, two things are read on
  the end frames of the physics-only twins.
  First, which property of the arrangement is uneven, on the particles between 1.5 and 4 pitches under the
  outermost layer, against the target sample and by D68's fifths: the spread of the number of particles
  within 1.5 pitches (density), the distance to the nearest particle (clumping), the ratio of the largest to
  the smallest principal spacing of the 16 nearest (anisotropy).
  Second, on those frames only (nothing is simulated): the particles under the outermost layer are shifted down
  the gradient of their own concentration, the particle shifting of incompressible SPH (Xu 2009, Lind 2012;
  the formulas are being checked against the papers by an agent), tangentially within two smoothing lengths of
  the layer, the layer itself untouched, 1, 3 and 10 times; the surface of each result is read against the
  mesh in D66's bands, and a target sample shifted 10 times is the control.
  The question: does an even arrangement under the surface give an even surface, and how far must particles
  move for it. If the bands fall to the sample's level with shifts of a fraction of a pitch, the rule belongs
  in the rollout's position update, next to the outer layer's relaxation; if the arrangement evens and the
  bands stay, the unevenness is in the body's shape and not in its arrangement, and shifting is not the change.
  Expectation: the density's spread and the clumping return to the sample's within 3 shifts; the band below 4
  pitches falls to the sample's and the 4–11 band by half of its excess; particles move 0.2–0.4 pitches rms.
  **Result (2026-10-04 11:45 CDT): the density under the surface is what is uneven, and evening it evens the
  surface, most of the fine band and a third to a half of the band above; shifted this crudely the body swells,
  so the free-surface rule is what the rollout's version has to get right.**
  The arrangement under the surface (1.5 to 4 pitches deep): the spread of the number of particles within 1.5
  pitches is 0.21 of its mean on a target sample and 0.29 on the bunny's end frame, 0.43 on the dragon's; the
  nearest particle's 5th percentile 0.36 pitches on a sample, 0.30 and 0.23 on the end frames (pairs pressed
  together); the anisotropy 1.43 (median) against 1.52 and 1.61. By D68's fifths the density's spread runs from
  0.37 (neighbours kept 0.38) to 0.23 (kept 1.00) on the bunny and 0.52 to 0.32 on the dragon. The density's
  unevenness is the large change (+36 %, +105 %), the anisotropy the small one (+6 %, +13 %).
  Shifting, on the end frames (bands in pitches: below 4 / 4–11; a sample: bunny 0.103 / 0.132, dragon 0.121 /
  0.153):

  | | as it is | 1 shift | 3 shifts | 10 shifts | the target sample after 10 |
  |---|---|---|---|---|---|
  | bunny: surface bands | 0.118 / 0.177 | 0.108 / 0.164 | 0.106 / 0.166 | 0.122 / 0.185 | 0.089 / 0.123 |
  | bunny: density's spread, nearest 5th percentile | 0.289, 0.30 | 0.234, 0.43 | 0.208, 0.71 | 0.220, 0.91 | 0.140, 0.92 |
  | bunny: mean offset from the mesh, discs farther than 2 pitches | +0.27, 1.8 % | +0.32, 2.5 % | +0.36, 3.1 % | +0.42, 6.3 % | +0.25, 2.5 % |
  | dragon: surface bands | 0.165 / 0.257 | 0.148 / 0.197 | 0.143 / 0.196 | 0.156 / 0.212 | 0.110 / 0.146 |
  | dragon: density's spread, nearest 5th percentile | 0.430, 0.23 | 0.303, 0.36 | 0.259, 0.64 | 0.271, 0.91 | 0.142, 0.91 |
  | dragon: mean offset, discs farther than 2 pitches | +0.57, 7.9 % | +0.70, 11.0 % | +0.75, 13.4 % | +0.80, 21.6 % | +0.61, 3.8 % |

  One to three shifts take the band below 4 pitches from an excess of 0.015 to 0.003–0.005 on the bunny (a
  sample's level) and from 0.044 to 0.022–0.027 on the dragon, and the 4–11 band from 0.045 to 0.032–0.034
  and from 0.104 to 0.043–0.044: the arrangement under the surface does make a good part of the unevenness,
  and the rest of the 4–11 band is in what this test did not move (the outer layer and the body's shape).
  What the crude rule gets wrong: one shift moves a particle 0.48–0.63 pitches rms (a few by 6–13), the body
  swells (the mean offset grows by 0.05–0.13 pitches a shift) and by ten shifts material stands outside the
  surface (6 % and 22 % of the discs farther than two pitches from the mesh) and the bands are back up. That is
  the shifting's known fault at a free surface; the papers' free-surface rule and step limit are being read
  before a line goes into the rollout. The expectation held for the bunny's fine band and the density, not for
  the size of the moves.
- **D67, how deep the runs' unevenness sits (a measurement, no code of the run changes; pre-registered
  2026-10-04 11:24 CDT; `exterior_offset_probe.py … peel=K`; `output/gpu/d67`).** D66: the runs add unevenness
  of their own at 4 to 11 pitches and below, and three causes are ruled out. Two kinds of cause are left and
  they differ in depth: the outer layer's own normal offsets (the u channel and its relaxation act on that
  layer alone), or the material under it (the stress control and MPM move the body, and the field reads every
  particle within three pitches). The outermost particle layer (the pipeline's layer rule) is taken off, once
  and twice, and the surface of what is left is read against the mesh in the same bands, on the two samples
  and on the end frames of the physics-only twins and of the 192 px runs of both meshes. A sample peeled the
  same way is the reference at each depth.
  Expectation: if the outer layer carries it, the runs' excess over the samples falls to a third or less after
  one layer is off; if the body carries it, the excess stays within a fifth of what it is with the layer on. I
  expect the first for the band below 4 pitches and the second for 4 to 11 pitches (an MPM cell is 6.25
  pitches, and the stress control works at about a cell).
  **Result (2026-10-04 11:25 CDT): the body under the outer layer carries it, in both bands; the outer layer
  is the evenest part of a run and hides about half.** Offsets in pitches with 0 / 1 / 2 layers off (the two
  samples of a mesh agree to 0.006):

  | | below 4 pitches | 4–11 pitches | field's normal against the mesh's, median |
  |---|---|---|---|
  | bunny, samples | 0.103 / 0.098 / 0.103 | 0.135 / 0.146 / 0.158 | 11.6° / 11.7° / 12.7° |
  | bunny P, end | 0.118 / 0.137 / 0.146 | 0.177 / 0.233 / 0.253 | 13.2° / 16.9° / 18.8° |
  | bunny L192, end | 0.120 / 0.139 / 0.141 | 0.171 / 0.237 / 0.247 | 13.0° / 17.5° / 18.4° |
  | dragon, samples | 0.121 / 0.122 / 0.128 | 0.155 / 0.176 / 0.195 | 12.7° / 13.1° / 14.2° |
  | dragon P, end | 0.165 / 0.185 / 0.192 | 0.257 / 0.289 / 0.313 | 17.1° / 20.9° / 23.4° |
  | dragon L192, end | 0.161 / 0.194 / 0.197 | 0.229 / 0.301 / 0.316 | 16.6° / 24.0° / 25.2° |

  With the layer on, the bunny's runs exceed a sample by 15 % below 4 pitches and 27–31 % between 4 and 11;
  with one layer off by 40 % and 60 %, with two by 40 % and 57–60 %. The dragon's: 33–36 % and 48–66 % with
  the layer, 52–59 % and 64–71 % without. The excess grows when the layer is taken off, where the first
  reading would have had it fall to a third: the outer layer, the one the relaxation keeps regular, is not
  where the unevenness is; the particles under it are, and the field, which reads three pitches deep, shows
  them through it. My expectation for the band below 4 pitches was wrong.
- **D68, whether the unevenness sits where the material was stretched (a measurement on kept frames;
  pre-registered 2026-10-04 11:26 CDT, after D67's reading that the body under the outer layer carries the
  unevenness; `tmp/stretch.py`).** A fixed set of particles carried through a large deformation ends with an
  uneven arrangement where the material was stretched or torn (a sphere's shell becomes ears and horns). Per
  particle, the deformation from the first frame to the end frame is fitted over its 16 nearest neighbours of
  the first frame (largest and smallest stretch), with the share of those neighbours still among its 32
  nearest at the end; the end frame's discs take the mean over the particles within 1.5 pitches and are put
  into five equal groups by each measure. Expectation: the offsets' rms in both finer bands rises with the
  stretch and falls with the neighbours kept, the most stretched fifth having at least 1.5 times the rms of
  the least stretched. If there is no such trend, the stretch is not the cause and what is left is the
  rollout's own handling of the particles under the surface (sub-cell arrangement, the stress control).
  **Result (2026-10-04 11:27 CDT; end frames of the physics-only twins and the 192 px runs): the unevenness
  sits where the material lost its neighbourhood; where it kept it, the surface is as even as a sample's.**
  Fifths of the end frame's discs by the share of the first frame's neighbours still near (the fifth's median),
  with the offsets' rms below 4 pitches and between 4 and 11:

  | | fewest kept | | | | most kept | a sample |
  |---|---|---|---|---|---|---|
  | bunny P | 0.28: 0.140, 0.221 | 0.54: 0.127, 0.190 | 0.72: 0.116, 0.173 | 0.85: 0.110, 0.156 | 0.97: 0.092, 0.129 | 0.103, 0.135 |
  | bunny L192 | 0.29: 0.136, 0.188 | 0.59: 0.135, 0.199 | 0.76: 0.121, 0.176 | 0.89: 0.108, 0.155 | 0.98: 0.093, 0.129 | 0.103, 0.135 |
  | dragon P | 0.17: 0.189, 0.267 | 0.28: 0.169, 0.280 | 0.38: 0.173, 0.288 | 0.50: 0.158, 0.250 | 0.69: 0.132, 0.188 | 0.121, 0.155 |
  | dragon L192 | 0.17: 0.184, 0.247 | 0.29: 0.169, 0.234 | 0.40: 0.164, 0.238 | 0.54: 0.152, 0.223 | 0.73: 0.132, 0.199 | 0.121, 0.155 |

  On the bunny the fifth of the surface whose material kept 0.97 of its neighbours is at a sample's level in
  both bands, and the fifth that kept 0.28 has 1.5 and 1.7 times as much. By the largest stretch the same
  order (1.4 → 6.8 times: 0.095 → 0.136 and 0.132 → 0.221). The dragon needs more of the material: the median
  largest stretch is 2.3–2.4 against the bunny's 1.6, the median share of neighbours kept 0.56–0.62 against
  0.88, and even its best fifth (0.69–0.73 kept) is above a sample. The expectation holds (1.4–1.7 between the
  outer fifths).
  So the cause is not a term of the objective and not the outer layer's rule: a fixed set of particles carried
  through the morph ends unevenly arranged where the sphere's material was stretched and torn into the
  target's parts, the field reads that arrangement three pitches deep, and neither the transport, the surface
  proximity nor the render term sees an arrangement at that scale. That is also why it is there by window 20
  and stays. What could move it, each a change of a definition and none decided: how the body's particles are
  arranged after being stretched (a redistribution of the near-surface particles; the relaxation regularises
  one layer only); how much the morph has to stretch the material (the transport's map); or how the display
  reads an uneven arrangement (a field less sensitive to it, which changes the picture and not the body).
- **D66, where the drawn surface departs from the target mesh, by scale (a measurement, no code of the run
  changes; pre-registered 2026-10-04 10:27 CDT; `scripts/probes/settled/exterior_offset_probe.py`;
  `output/gpu/d66`).** D65: at 300k what is left is the pictures' difference, and it is in the simulated
  surface with or without a render term. For the exterior's discs (a lattice of 0.92 pitches) of every kept
  frame of the physics-only twin, the base run and the runs with the render on the exterior at 96 and 192 px,
  and of the two target samples (seeds 97 and 98): the signed distance to the target mesh (three million
  surface samples with normals, point to plane), split by Gaussian means over the discs into the part below
  about 4 pitches, between 4 and 11, and larger; and the angle of the field's normal and of the displayed
  (averaged) normal to the mesh's.
  The three bands name three different parts of the code: below 4 pitches is the particles' arrangement at and
  under the surface (the sampler, MPM, the outer layer's offset and its relaxation); 4 to 11 pitches is the
  band the relaxation erases and the stress control works in; larger is the shape the transport places.
  Expectation: the runs exceed the samples mostly below 4 pitches (the field reading an interior that is less
  even than a fresh sample), on the dragon also between 4 and 11; the displayed normal's error of the runs
  exceeds the samples' by about the ratio of the pictures' differences (1.3–1.9). Whatever band holds the
  excess is the part to look into next; nothing is changed by this entry.
  **Result (2026-10-04 11:16 CDT; every kept frame of the eight runs, the two samples of each mesh; mean of the
  last 20 kept frames; offsets in pitches; an MPM cell is 6.25 pitches): the runs place the shape as well as a
  sample; what they add is unevenness of their own at one to two cells and below, which is neither erased
  relief nor the grid's imprint; the dragon also keeps material where the target has no surface.**

  | | below 4 pitches | 4–11 pitches | larger | normal against the mesh's, median: field / displayed | discs farther than 2 pitches from the mesh |
  |---|---|---|---|---|---|
  | bunny, samples (seeds 97, 98) | 0.103 | 0.132–0.138 | 0.50–0.52 | 11.5–11.7° / 7.7–7.8° | 2.9–3.1 % |
  | bunny P | 0.118 | 0.176 | 0.496 | 13.1° / 8.9° | 1.7 % |
  | bunny R | 0.142 | 0.198 | 0.514 | 14.4° / 9.9° | 3.4 % |
  | bunny La | 0.119 | 0.172 | 0.508 | 13.2° / 9.2° | 2.1 % |
  | bunny L192 | 0.118 | 0.169 | 0.515 | 12.9° / 9.2° | 2.1 % |
  | dragon, samples | 0.121–0.122 | 0.153–0.156 | 0.53–0.54 | 12.6–12.8° / 9.7–9.8° | 3.3–3.4 % |
  | dragon P | 0.164 | 0.257 | 0.538 | 16.9° / 11.4° | 7.7 % |
  | dragon R | 0.173 | 0.258 | 0.513 | 18.5° / 12.3° | 11.1 % |
  | dragon La | 0.164 | 0.230 | 0.522 | 17.0° / 11.5° | 9.8 % |
  | dragon L192 | 0.161 | 0.228 | 0.523 | 16.6° / 11.5° | 9.3 % |

  Above 11 pitches the runs are at the samples' level (the half pitch there is common to all: the field's own
  bias and the mesh fitted by its bounding box). The excess is in the two finer bands and largest between 4 and
  11 pitches: +28 to +33 % on the bunny (P, L; R +47 %), +47 to +68 % on the dragon; below 4 pitches +15 %
  and +33 to +43 %. It is there by window 20 and does not shrink afterwards (bunny P 0.179 → 0.176, the
  dragon's L runs 0.235–0.243 → 0.228–0.230): it is not a matter of running longer. The expectation put the
  excess mostly below 4 pitches; it is larger in the band above.
  It is not erased relief (`mesh_relief`: the least-squares slope of a disc's offset on the mesh's own relief
  of the band at that place; −1 would be a surface that lost it): on the dragon the samples have −0.16 below 4
  pitches and −0.06 between 4 and 11, the runs −0.15…−0.16 and −0.05…−0.07; on the bunny the samples −0.22
  and −0.08, the runs −0.38 and −0.16…−0.18, a further sixth and a tenth of a relief of 0.08 and 0.27
  pitches, that is 0.013 and 0.027 pitches against excesses of 0.06 and 0.11. The target's detail at these
  scales is drawn as a sample of the same N draws it.
  It is not the MPM grid's imprint (`tmp/grid_phase.py` on the end frames' discs): where in a cell a disc lies
  explains 0.4–1.1 % of the offsets' variance in either band; a rotated lattice of the same pitch 0.3–0.7 %.
  The render: on the exterior it lowers the 4–11 band a little against P (dragon 0.228–0.230 against 0.257,
  bunny 0.169–0.172 against 0.176); on the particle cloud it raises both bands on the bunny.
  The dragon: 8 to 11 % of a run's drawn surface is farther than two pitches from the mesh where a sample has
  3.4 %, and four fifths to nine tenths of it lies on the mesh's outer side: material in the mouth and between
  the coils, 5 to 8 % of the surface. The bunny has none of it (1.7–2.1 % against the samples' 3 %). The runs'
  surface also sits 0.07–0.19 pitches further out than a sample's on the bunny.
  So "complete at 300k" is smoothness, not detail: the runs' own unevenness at one to two cells and below, and
  on the dragon the material in the concavities. What makes the unevenness is not found yet; ruled out are
  the render term (the physics-only twin has it), the loss of the target's relief, and the grid's phase. Left
  to measure: the outer layer's own offsets (the u channel), the unevenness of the particles under the surface
  that the field reads, and the stress control.
- **D65, the floor at 300k: an independent sample of the target against the reference sample (a measurement,
  no run is judged; pre-registered 2026-10-04 10:23 CDT; the user: "M 단위로는 일단 뒤로 … 먼저 알고리즘 쪽을
  완벽하게 한 다음 최후의 수단으로 1.5M"; `output/gpu/d65`).** What the algorithm can still gain at a fixed N is
  the distance between a run's end and what a perfect run would show, which is another sample of the same
  target, not the reference sample itself. The bunny's and the dragon's targets are sampled again at 300k with
  seed 98 (a one-window run whose target sample alone is used) and read as a state against the seed-97
  reference by the display probe (`measure ref=… target own`): front, thin crop, far side.
  Expectation: IoU 0.994–0.997 and a pictures' difference of 0.005–0.008; the best runs so far (bunny 0.9912 /
  0.0096, crop 0.9794 / 0.0100; dragon 0.9898 / 0.0127, crop 0.9726 / 0.0182) would then be 0.004–0.007 in IoU
  and a third to a half in the difference above the floor on the bunny, and further on the dragon's crop. The
  number says what "complete at 300k" means for every later experiment; nothing is adopted by it.
  **Result (2026-10-04 10:25 CDT): the solid regions of the bunny are at the floor already; what is left at
  300k is the pictures' difference, on both meshes and with or without a render term, and the dragon's horn
  crop.**

  | against the seed-97 reference | front: IoU / difference | thin crop | far side | field's roughness |
  |---|---|---|---|---|
  | bunny, the seed-98 sample (the floor) | 0.9906 / 0.0063 | 0.9795 / 0.0071 | 0.9904 / 0.0059 | 9.8° |
  | bunny, render on the exterior at 192 px | 0.9912 / 0.0096 | 0.9794 / 0.0100 | 0.9926 / 0.0079 | 11.2° |
  | bunny, physics-only | 0.9868 / 0.0099 | 0.9705 / 0.0116 | 0.9883 / 0.0086 | 11.2° |
  | dragon, the seed-98 sample (the floor) | 0.9912 / 0.0074 | 0.9793 / 0.0112 | 0.9918 / 0.0072 | 13.6° |
  | dragon, render on the exterior at 192 px | 0.9898 / 0.0127 | 0.9726 / 0.0182 | 0.9901 / 0.0137 | 20.9° |
  | dragon, physics-only | 0.9875 / 0.0118 | 0.9717 / 0.0160 | 0.9871 / 0.0130 | 22.2° |

  The floor's IoU is lower than expected (0.991, 0.979 in the crops): two samples of one mesh differ that much
  at 300k. Against it: the bunny's run with the render on the exterior is at the floor in every solid region
  (the physics-only twin is 0.004 and 0.009 below); the dragon's is 0.0014 below in the front and 0.0067 in
  the horn crop. The pictures' difference is above the floor everywhere: by 34–52 % on the bunny and 63–90 % on
  the dragon with the render, and as much or more without it (bunny 0.0099 against 0.0063). So that gap is not
  the render's: it is in the simulated surface itself, which the field reads as rougher than a sample of the
  same N (bunny 11.2° against 9.8°, dragon 21–22° against 13.6°). "Complete at 300k" is therefore: the solid
  regions at 0.991 / 0.979 (reached on the bunny), and the difference down to 0.006–0.007 (bunny), 0.007 /
  0.011 (dragon), where the runs are at 0.008–0.010 and 0.013–0.018. Finer than that is the sample's own
  limit (D12), which only N moves.
- **D64, the fine render picture at the field's resolution (pre-registered 2026-10-04 02:15 CDT;
  `--render_exterior --render_res_hi 192`; `output/gpu/d64`).** Why: on the display (4K, against the target,
  every kept frame) the render on the exterior is ahead of the physics-only twin on the bunny but not on the
  dragon at 300k, whose horn crop it leaves behind the twin (IoU 0.966 against 0.973 at the end, 0.950 against
  0.970 at window 20), although its own terms at 96 px are 2 to 2.6 times lower. At 96 px a pixel of the
  dragon's picture is 2.5 pitches at 300k (0.122 wu) and a horn is one to two pixels thick: the term cannot
  place what it does not resolve, and it holds nine tenths of the gradient. The particle cloud bounded the
  picture's resolution (a pixel must hold particles); the exterior's bound is the field's lattice, 0.92
  pitches, which a pixel of 192 px reaches at 300k (1.27 pitches on the dragon, about one on the bunny). One
  value changes: the fine level of the coarse-to-fine event, 96 → 192 (the lattice follows as defined, half a
  pixel and no coarser than 0.92 pitches).
  Runs: dragon 300k and bunny 300k, one each first; read against D62's P, La and Lb on the yardstick at 96
  and at 192 px and on the display.
  Criterion: the dragon's horn crop on the display ends at the physics-only twin's or better (IoU 0.973,
  difference 0.0160) with the front picture no worse than La's; on the bunny nothing is lost against La.
  Expectation: the horn crop gains; a window costs more (about twice the discs, four times the pixels); the
  transport energy stays above the twin's.
  **Result (2026-10-04 04:26 CDT; one run each; every kept frame; `output/video_2026-10-04/d64_fine_picture`):
  the dragon's horn crop reaches the physics-only twin's in the solid region and the whole picture passes it;
  the pictures' difference in the crop stays above the twin's; the bunny loses nothing.**
  The display against the target, mean of the last 20 kept frames (IoU of the solid regions / the pictures'
  difference):

  | run | front | thin crop | far side |
  |---|---|---|---|
  | dragon 300k P | 0.9875 / 0.0118 | 0.9717 / 0.0160 | 0.9871 / 0.0130 |
  | dragon 300k R | 0.9852 / 0.0149 | 0.9702 / 0.0191 | 0.9850 / 0.0171 |
  | dragon 300k La, Lb (96 px) | 0.9876 / 0.0135, 0.9884 / 0.0124 | 0.9686 / 0.0194, 0.9691 / 0.0185 | 0.9865 / 0.0145, 0.9881 / 0.0137 |
  | dragon 300k L192 | 0.9898 / 0.0127 | 0.9726 / 0.0182 | 0.9901 / 0.0137 |
  | bunny 300k P | 0.9868 / 0.0099 | 0.9705 / 0.0116 | 0.9883 / 0.0086 |
  | bunny 300k La, Lb (96 px) | 0.9910 / 0.0092, 0.9909 / 0.0092 | 0.9779 / 0.0098, 0.9777 / 0.0106 | 0.9922 / 0.0081, 0.9921 / 0.0075 |
  | bunny 300k L192 | 0.9912 / 0.0096 | 0.9794 / 0.0100 | 0.9926 / 0.0079 |

  Dragon: the horn crop's IoU goes from 0.9686–0.9691 at 96 px to 0.9726, the twin's level (0.9717; its last
  frame 0.9730), seven times the two 96 px runs' difference; the front and the far side pass the twin by 0.002
  and 0.003, where the 96 px runs were level with it. The pictures' difference stays above the twin's: +8 % in
  the front, +14 % in the crop (the 96 px runs: +5 to +14 % and +16 to +21 %). The criterion is met in the
  solid regions and not in the crop's difference.
  Bunny: the IoUs are 0.0002–0.0017 above the 96 px runs' (the crop's by eight times their difference), the
  differences inside their range.
  On the yardstick the run moves its effort to the picture it is given: at 192 px the bunny's exterior
  silhouette ends at 0.00079 against 0.00117–0.00125 for the 96 px runs and 0.00213 for the twin; at 96 px it
  ends at 0.00051 against their 0.00025–0.00031 (the twin 0.00126). The transport energy ends where the 96 px
  runs' does (bunny 1.9e-4, dragon 1.0e-3).
  Cost: the bunny's run, alone on its GPU, 1 258 s for 84 window attempts, 15.0 s each against 13.6 at 96 px
  (173 000 discs against 57 000 in the fine stage, one search a window; the gradient time is unchanged at
  7.8 s); the dragon's run shared its GPU and its time (4 820 s) says nothing.
  Rendering influence: λ at the first window as at 96 px (0.203, 0.321), recalibrated at the switch; g_share
  median 0.91 and 0.93; the path is D62's.
  What this leaves: the difference of the pictures in the horn crop (shading and edge position at 4K) is still
  better without any render term; one run per mesh, so the spread at 192 px is not measured; the event rule
  still switches once, from 64 px, whatever the fine level.
- **D63, stage 3: fewer interior particles under the exterior (pre-registered 2026-10-04 00:28 CDT, while D62's
  dragon runs are still out; the user: "이게 증명되면 3단계 … 그 후 입자수를 줄이면서 확인 … 항상 '모든 프레임'에서
  검증"; `output/gpu/d63`).** The interior at N = 100 000, 50 000 and 30 000 on the bunny and the dragon (300 000
  is D62), each with one physics-only twin, run once and kept (`repo_r64`, `--render_weight_scale 0`), and two
  runs with the render on the exterior (`repo_r65`, `--render_exterior`). The display stays the exterior of
  300 000 discs.
  Read at each N on every kept frame: the render terms on the yardstick against that N's own target sample
  (physics-only against with-render, D62's criterion, so that the render's effect is shown at every N and not
  assumed from 300k); the display against one reference for every N, the exterior drawing of the 300k target
  sample from the same camera (front, thin crop, far side: IoU of the solid regions, the pictures'
  difference); the run's seconds.
  What is asked: down to which N the displayed result stays within the two 300k runs' own spread of the 300k
  result on the whole picture and on the thin crop, and what a run costs there.
  Expectation: the whole picture holds to 50k or lower; thin parts thicken with the field's pitch (its offset
  is 0.8 pitches, so the thinnest drawn sheet is 1.44 times as thick at 100k as at 300k, 1.8 at 50k, 2.15 at
  30k), so the thin crop is what gives way first, between 100k and 50k; sheets in concavities grow the same
  way; run time falls about with N (dragon 100k: 8 min against 54 at 300k).
  **Result (2026-10-04 03:32 CDT; every kept frame; `output/video_2026-10-04/d63_fewer_particles/twin_*.png`;
  the display is read with `surface_layer_probe.py … measure ref=<300k frames>`): the render keeps its effect
  on its own terms at every N; the displayed result does not hold below 300k, and what bounds it is the sample,
  not the run.**
  The render at each N (yardstick, each N's own target, last frame): the exterior's silhouette is 3.5–4.9
  times below the physics-only twin's on the bunny at 100k (0.00050, 0.00070 against 0.00245), 4.0–4.3 at 50k,
  3.3–3.7 at 30k, and 2.4–3.1 on the dragon at 50k, 3.7–4.2 at 30k; the shading 12–26 % below, beyond the
  two runs' difference at every N; the transport energy 3 to 4.5 times the twin's.
  The display against the 300k reference (mean of the last 20 kept frames; IoU of the solid regions / the
  pictures' difference), with, in the last column, the same reading of that N's own target sample, what a
  perfect run at that N would show:

  | mesh, N | run time (s): P, La, Lb | front: P | front: La | thin crop: P | thin crop: La | front and thin crop of the N's own target sample |
  |---|---|---|---|---|---|---|
  | bunny 300k | 975, 955, 798 | 0.9868 / 0.0099 | 0.9910 / 0.0092 (Lb 0.9909 / 0.0092) | 0.9705 / 0.0116 | 0.9779 / 0.0098 (Lb 0.9777 / 0.0106) | the reference |
  | bunny 100k | 232, 342, 273 | 0.9815 / 0.0155 | 0.9831 / 0.0153 | 0.9492 / 0.0185 | 0.9589 / 0.0163 | 0.9825 / 0.0111, 0.9607 / 0.0136 |
  | bunny 50k | 124, 168, 169 | 0.9723 / 0.0204 | 0.9671 / 0.0227 | 0.9390 / 0.0224 | 0.9355 / 0.0240 | 0.9698 / 0.0174, 0.9404 / 0.0208 |
  | bunny 30k | 124, 137, 108 | 0.9653 / 0.0248 | 0.9666 / 0.0256 | 0.9071 / 0.0307 | 0.9033 / 0.0299 | 0.9672 / 0.0214, 0.9059 / 0.0270 |
  | dragon 300k | 1 684, 2 259, 2 245 | 0.9875 / 0.0118 | 0.9876 / 0.0135 | 0.9717 / 0.0160 | 0.9686 / 0.0194 | the reference |
  | dragon 100k | 460, 464, 457 | 0.9750 / 0.0205 | 0.9752 / 0.0217 | 0.9495 / 0.0315 | 0.9341 / 0.0352 | 0.9791 / 0.0171, 0.9498 / 0.0267 |
  | dragon 50k | 218, 242, 221 | 0.9615 / 0.0258 | 0.9570 / 0.0271 | 0.9355 / 0.0375 | 0.9196 / 0.0429 | 0.9619 / 0.0221, 0.9387 / 0.0318 |
  | dragon 30k | 165, 166, 130 | 0.9515 / 0.0330 | 0.9505 / 0.0331 | 0.8676 / 0.0592 | 0.8723 / 0.0564 | 0.9560 / 0.0291, 0.8896 / 0.0509 |

  The two 300k runs of the bunny agree to 0.0002 in the IoUs and 0.0008 in the crop's difference; against
  that, every lower N is outside: at 100k the whole picture is 0.008–0.012 lower in IoU and the thin crop
  0.019–0.035, at 50k 0.024–0.031 and 0.042–0.049, at 30k 0.024–0.037 and 0.075–0.096. The question had no N
  below 300k for an answer.
  Why: each N's own target sample, drawn by the same exterior, is already that far from the 300k reference
  (bunny 100k: 0.9825 and 0.9607; the runs: 0.9815–0.9831 and 0.9492–0.9589). The runs sit at their sample's
  level, the render moves them by a few thousandths either way, and the physics-only twin is as close. The
  exterior takes the fuzz off at any N; it does not add resolution: where the drawn surface lies is set by the
  field's pitch, which is the interior's (kernel 3 pitches, offset 0.8). The expectation that the whole
  picture would hold to 50k was wrong; the thin crop does give way first, already at 100k.
  What a lower N buys: a run of 4.5–5.7 min at 100k, 2.8–4 at 50k, 1.8–2.8 at 30k against 13–16 (bunny) and
  37 (dragon) at 300k. Fewer interior particles under this exterior is therefore a trade of thin-part accuracy
  for time, not a free saving; a finer drawn surface from fewer particles would need the interior itself to
  be finer where the surface is (the variable-mass shell of the proposal's fourth stage), which is the user's
  decision.
- **D62, stage 2: the render terms read on the exterior (pre-registered 2026-10-03 23:46 CDT; the user: "2단계
  구현 후에는. 그래프를 그려가면서 Physics-only, w/ render 를 비교해가면서 실험 … 이게 증명되면 3단계"; code
  `--render_exterior`: `physmorph/render/exterior.py` (Tracked), `render_loss.py` (shaded_discs, d_exterior),
  `window/objective.py` (render_terms); server `repo_r65`; `output/gpu/d62`).** One definition changes: what the
  silhouette and the shading terms are read on. The weights, views, resolutions, the calibration of λ and the
  projection against the physics gradient stay.
  The discs: the zero set of the field (kernel 3 pitches, offset 0.8) of the released end state, one disc for
  each cell it crosses of a lattice of half a render pixel (no coarser than 0.92 pitches, the largest lattice
  that puts a node in the field's smallest body). They are found once per window, at the first state it
  evaluates; from then on a disc moves along its normal by one Newton step of the field of the current
  particles and takes that field's gradient as its normal. Both are functions of the particles within the
  kernel radius plus one pitch of the disc (lists frozen for the window), so the terms' gradients reach them.
  The silhouette: the same operator (the splat, 1 − exp(−k w), the asymmetric penalty) on the discs, against
  the target sample's own discs. The shading: a disc is drawn where it faces the camera, with its facing cosine
  times the existing front bias as weight and ambient + (1 − ambient) cosine as shade; matched target.
  Why it could matter: the gradient lands on the particles that make the drawn surface, with that surface's
  normals and without its back faces; on the particle cloud it lands on whichever particles fall in unsaturated
  pixels, and the shading reads a blurred density's normals on every particle.
  Runs: bunny 300k, dragon 100k, dragon 300k with `--render_exterior` (L), two each for the spread; against
  D60's physics-only twins (P, one each, not rerun) and D59's base runs (R: render on the particle cloud).
  Read on every frame and every window (`render_twin_plot.py`): the exterior display against the target (IoU of
  the solid regions and the pictures' difference: front, thin crop, far side), the crop's soft pixels, the
  field's roughness; each run's own silhouette and shading terms (P and R share the particle definition), the
  transport energy, g_share, λ; and both definitions of the render terms evaluated by one tool on every kept
  frame of every run (`render_terms_probe.py`), so that P, R and L are read on one yardstick.
  The render term "has its effect" if, on that yardstick, the exterior-defined silhouette + shading of L is
  below P's at the end and in the mean over the frames both have after window 10, by more than the two L runs
  differ, and the display's difference against the target likewise; the transport energy is recorded beside it
  (what the render costs). The same reading is recorded for R with the particle definition.
  Cost: the window time, against 1.3 of the base's.
  Expectation: L lowers its own terms and the display's difference against P; against R a similar silhouette
  and a lower shading difference on thin parts; 10–30 % more time; in early windows the discs move one to two
  lattice pitches from where they were found (the 40k smoke: 1.3–2.0), less later. If L shows no effect beyond
  the spread, the term does not act at 96 pixels, and the resolution is what to look at next (the exterior
  allows a finer picture than the particle cloud).
  **First run, failed (bunny 300k, 23:46–23:54 CDT, kept as `bunny300k_L0`): stopped at 9 windows, silhouette
  0.848.** The selection merit rose from window 5 (0.068 → 0.079; the base is at 0.011 there) and three
  candidates in a row were rejected, twice. Read from the run's own record: in the windows with long line
  searches the discs' displacement was 2.0 lattice pitches at the 90th percentile (1.86–2.65), which is the
  bound of the Newton step, offset over slope, about 1.8 pitches: a disc whose point the body has moved over
  reads a field that no longer changes there, stays behind and has no defined normal. The control moves the
  surface 1.0–1.4 pitches a window at 300k (`move` 0.05–0.07 wu), so discs found once per window, at its first
  (uncontrolled) state, are behind the surface for most of the search; the silhouette term stalled at 0.06
  (base: 0.008 at window 5). The definition above ("found once per window") was wrong; the rule that replaced
  it at 00:05 CDT: the discs are found again, at the state being read, once a tenth of them has moved more
  than half a lattice pitch from where they were found (they have left their cells; the lattice is fixed in
  space, so the same surface gives the same discs). Two more things from this run: the gradient time was
  14–28 s a window against 7 (the gather of the discs' particles had a sorting backward; `index_select` has
  not); a six-window check of the new rule at 300k follows the base (merit 0.0102 at window 5, gradient time
  7.1–9.0 s, 7–12 searches a window early).
  **Result on the yardstick (2026-10-04 01:25 CDT; every kept frame of every run, `render_terms_probe.py` at
  96 px; plots `output/video_2026-10-04/d62_render_twins/twin_*.png`; P = D60's twin, R = D59's base run, La and
  Lb = the two runs with the render on the exterior): read on the exterior, the silhouette of L is 2 to 5.6
  times below the physics-only twin's at the end on all three, by far more than the two L runs differ; the
  render on the particle cloud (R) leaves the exterior's silhouette where the physics-only twin has it.**

  | run | exterior silhouette, mean after window 10 / last | exterior shading | particle-cloud silhouette | particle-cloud shading | transport energy, last window | seconds, commits |
  |---|---|---|---|---|---|---|
  | bunny 300k P | 0.00150 / 0.00128 | 0.00093 / 0.00078 | 0.00131 / 0.00129 | 0.00054 / 0.00028 | 3.8e-5 | 975, 79 |
  | bunny 300k R | 0.00140 / 0.00118 | 0.00103 / 0.00092 | 0.00088 / 0.00031 | 0.00058 / 0.00033 | 1.37e-4 | 904, 73 |
  | bunny 300k La | 0.00108 / 0.00034 | 0.00082 / 0.00059 | 0.00147 / 0.00107 | 0.00080 / 0.00060 | 1.78e-4 | 955, 56 |
  | bunny 300k Lb | 0.00073 / 0.00023 | 0.00070 / 0.00057 | 0.00127 / 0.00111 | 0.00072 / 0.00060 | 1.82e-4 | 798, 48 |
  | dragon 100k P | 0.00261 / 0.00227 | 0.00162 / 0.00136 | 0.00279 / 0.00258 | 0.00064 / 0.00045 | 1.5e-4 | 460, 75 |
  | dragon 100k R | 0.00221 / 0.00137 | 0.00187 / 0.00134 | 0.00220 / 0.00063 | 0.00071 / 0.00037 | 4.1e-4 | 917, 107 |
  | dragon 100k La | 0.00147 / 0.00081 | 0.00154 / 0.00113 | 0.00257 / 0.00202 | 0.00085 / 0.00065 | 6.4e-4 | 464, 52 |
  | dragon 100k Lb | 0.00195 / 0.00115 | 0.00178 / 0.00128 | 0.00297 / 0.00258 | 0.00093 / 0.00073 | 8.2e-4 | 457, 50 |
  | dragon 300k P | 0.00317 / 0.00134 | 0.00164 / 0.00086 | 0.00310 / 0.00148 | 0.00085 / 0.00032 | 1.4e-4 | 1 684, 102 |
  | dragon 300k R | 0.00245 / 0.00123 | 0.00181 / 0.00110 | 0.00214 / 0.00049 | 0.00093 / 0.00045 | 6.9e-4 | 3 237, 151 |
  | dragon 300k La | 0.00162 / 0.00063 | 0.00128 / 0.00080 | 0.00252 / 0.00166 | 0.00098 / 0.00072 | 9.8e-4 | 2 259, 61 |
  | dragon 300k Lb | 0.00175 / 0.00051 | 0.00135 / 0.00071 | 0.00274 / 0.00157 | 0.00108 / 0.00071 | 8.4e-4 | 2 245, 69 |

  The render on the particle cloud (R against P, the reading the user asked for): it lowers its own silhouette
  term 2.6 to 4.1 times by the end and takes nine tenths of the gradient (g_share 0.87–0.91), but the
  silhouette of the drawn surface is 8 % below P's on the two 300k meshes (40 % at the dragon's 100k), the
  drawn surface's shading is worse than P's at 300k (+18 %, +28 %), and the transport energy ends 2.8 to 4.9
  times P's. At 300k the term's work goes into the cloud's own picture, which a few particles in a pixel can
  satisfy, and not into the surface that is displayed.
  The render on the exterior (L against P): the exterior's silhouette ends 3.8–5.6 times below P's on the
  bunny, 2.0–2.8 on the dragon at 100k and 2.1–2.6 at 300k; the two L runs differ by 0.0001–0.0003 where the
  effect is 0.0007–0.0015. The shading ends 25 % below P's on the bunny; on the dragon it is 6–18 % below, which
  is within the two runs' difference at the end (beyond it in the mean at 300k). Silhouette plus shading, the
  registered criterion: met at the end on all three; in the mean after window 10 on the bunny and on the dragon
  at 300k, and by one of the two runs at 100k. Most of the gain comes at the fine resolution: within a few
  windows of the switch to 96 px the exterior's silhouette falls three to four times (the plots).
  What L does not do: the particle cloud's own terms stay at P's (silhouette) or above (shading); the run's
  particle-silhouette IoU is 0.986 / 0.982 / 0.975–0.978 against R's 0.987 / 0.985 / 0.981.
  Cost: the transport energy ends 4.3 to 7 times P's (R: 2.8 to 4.9); a window attempt takes 1.2 times R's on
  the bunny, 1.0 on the dragon at 100k and 1.38 at 300k (28.2 s against 20.5; the criterion was 1.3), with 3 to
  7 searches a window early and one late; the runs stop after 48–69 commits where R takes 73–151, so a run is
  0.5 to 1.06 of R's time. Render influence: λ at the first window 0.203 / 0.323 / 0.321 (R: 0.249 / 0.42 /
  0.396), g_share median 0.91–0.94; the path is the gradient of the two terms through the discs' positions and
  normals to the particles within the kernel of each disc; the λ = 0 twin is P.
  **The display's own reading (2026-10-04 03:25 CDT; the exterior at 4K against the target sample's: front
  camera, its thin crop, a camera on the far side; every kept frame of P, R and La; the same plots, display
  panels; mean after window 10 over the frames all three have / last frame): met on the bunny, in the solid
  regions on the dragon at 100k, not on the dragon at 300k.**

  | run | front: IoU | front: pictures' difference | thin crop: IoU | thin crop: difference | far side: IoU | far side: difference | field's roughness, last |
  |---|---|---|---|---|---|---|---|
  | bunny 300k P | 0.9855 / 0.9870 | 0.0108 / 0.0099 | 0.9656 / 0.9703 | 0.0130 / 0.0116 | 0.9868 / 0.9887 | 0.0094 / 0.0087 | 11.4° |
  | bunny 300k R | 0.9847 / 0.9869 | 0.0125 / 0.0125 | 0.9609 / 0.9698 | 0.0144 / 0.0126 | 0.9873 / 0.9899 | 0.0110 / 0.0106 | 15.0° |
  | bunny 300k La | 0.9883 / 0.9909 | 0.0104 / 0.0092 | 0.9675 / 0.9771 | 0.0133 / 0.0102 | 0.9911 / 0.9922 | 0.0091 / 0.0080 | 11.4° |
  | dragon 100k P | 0.9772 / 0.9799 | 0.0143 / 0.0127 | 0.9555 / 0.9618 | 0.0184 / 0.0156 | 0.9799 / 0.9813 | 0.0157 / 0.0144 | 13.9° |
  | dragon 100k R | 0.9786 / 0.9809 | 0.0167 / 0.0140 | 0.9540 / 0.9643 | 0.0204 / 0.0176 | 0.9808 / 0.9838 | 0.0178 / 0.0147 | 15.8° |
  | dragon 100k La | 0.9846 / 0.9863 | 0.0149 / 0.0125 | 0.9630 / 0.9641 | 0.0192 / 0.0181 | 0.9833 / 0.9858 | 0.0160 / 0.0138 | 13.6° |
  | dragon 300k P | 0.9802 / 0.9874 | 0.0155 / 0.0119 | 0.9569 / 0.9730 | 0.0215 / 0.0160 | 0.9800 / 0.9874 | 0.0169 / 0.0132 | 22.6° |
  | dragon 300k R | 0.9811 / 0.9853 | 0.0178 / 0.0148 | 0.9571 / 0.9718 | 0.0242 / 0.0186 | 0.9800 / 0.9852 | 0.0196 / 0.0169 | 23.3° |
  | dragon 300k La | 0.9838 / 0.9871 | 0.0158 / 0.0137 | 0.9572 / 0.9664 | 0.0234 / 0.0201 | 0.9825 / 0.9862 | 0.0173 / 0.0147 | 21.3° |

  R, the render on the particle cloud, is behind the physics-only twin in the pictures' difference on all
  three, front and far side (10–27 % more at the end), and leaves a rougher surface (15.0° against 11.4° on the
  bunny): on the display it does not help, at any of the three.
  La on the bunny is ahead of P on every measure at the end (IoU +0.004 front, +0.007 thin crop, +0.003 far
  side; differences −7 %, −12 %, −8 %). On the dragon at 100k it is ahead in the solid regions (front +0.006,
  far side +0.005, crop +0.002) and level in the differences except the crop (+16 %). On the dragon at 300k
  it is level with P in the solid regions at the end (front 0.9871 against 0.9874) and behind in the horn crop
  (0.9664 against 0.9730) and in the differences (+15 % front, +26 % crop, +11 % far side). Earlier it is far
  ahead (window 10: front IoU 0.966 against 0.897; window 20: level, with the crop at 0.950 against 0.970): the
  render brings the body in sooner and leaves the horns behind the twin, which goes on to 102 commits where La
  stops at 61 with seven times the transport energy.
  So what is shown: the render on the exterior acts on the drawn surface at every mesh and N by its own terms
  (96 px, 18 views), reproducibly; on the 4K picture that gain shows where a 96 px pixel resolves the target's
  parts (the bunny; the dragon's body), and it is a loss on the dragon's horns at 300k, which are one to two
  such pixels thick (D64 takes the picture's resolution).
  Also on La's frames: the last frame of the dragon at 300k shows no set apart from the main one (R's: 3 733
  pixels, the blob under the tail), and the base display's own soft pixels in the horn crop are 10 680
  against R's 14 473.
  The spread of this reading (04:26 CDT; Lb's frames, mean of the last 20): the two L runs agree to 0.0001–0.0002
  in the front and crop IoU on the bunny (0.9910 / 0.9909, 0.9779 / 0.9777) and on the dragon at 100k (0.9869 /
  0.9868, 0.9674 / 0.9676), and to 0.0008 and 0.0005 on the dragon at 300k (0.9876 / 0.9884, 0.9686 / 0.9691); the
  pictures' differences vary more between the two (0.0123 against 0.0148 on the dragon at 100k). Against that
  spread: L is ahead of P in the solid regions on the bunny (+0.004, +0.007) and on the dragon at 100k (+0.007,
  +0.006), level in the front and behind in the horn crop (−0.003) on the dragon at 300k.
- **D61, the exterior's field: Solenthaler's factor on the offset (stage 1c; display only; pre-registered
  2026-10-03 23:33 CDT; the user: "OK. 들어가 줘" to the field first, then stages 2 and 3; code
  `physmorph/render/exterior.py`, `surface_layer_probe.py … FIELD`).** D59's webs between the ears, sheets in
  concavities and bridged sparse particles are one fault: between near but separate bodies the particles' mean
  x̄(q) moves faster than q and |q − x̄| stays below the offset where there is no body.
  Primary sources (one agent, the papers' own text; copies in the session's scratch directory): Zhu and Bridson
  2005 name the fault ("spurious blobs of surface can appear, since x̄ may erroneously end up outside the surface
  in concavities") and smooth it away on a grid; Solenthaler, Schläfli and Pajarola 2007 (Eq. 23–26) remove it at
  its cause, "in concave regions or between near but separated particles", by multiplying the offset with
  f = γ³ − 3γ² + 3γ, γ = (t_high − EV_max)/(t_high − t_low), EV_max the largest eigenvalue of ∂x̄/∂q, t_low 0.4,
  t_high 2.0 (f = 1 below t_low; 0 above t_high, implied); Yu and Turk 2013 (anisotropic kernels) address
  bumpiness, make no claim on gaps, and their centre smoothing itself pulls components within four spacings
  together (they add component labels against it); Adams 2007 needs distances carried from step to step;
  Bhattacharya 2015 has "difficulty generating very thin surfaces". So the change is Solenthaler's factor with
  the paper's constants on the field as it is (kernel radius 3 pitches); the eigenvalue is the largest real part,
  from the characteristic cubic in closed form; the offset is set once more so that the target samples keep the
  base display's solid area (the factor is about 0.947 on a flat surface, so 0.845 pitches first, then corrected
  by the measured ratio).
  Runs: the twelve key states of D59 stage 1a (Poisson-disk build) and every frame of D59's three runs (lattice
  build), each against its `zb` rows.
  Criteria: on the target samples the pixels seen of sets apart from the main one (8 362 at the dragon's 100k,
  2 933 at 300k) fall below a fifth; the web between the bunny's ears at windows 5–8 is gone on the crop
  sheets; thin parts kept (the crop's solid pixels at 0.95 of the base's or more on every key state, as now);
  the fringe still gone on every frame; the field's roughness within 1.2 of `zb`'s.
  Expectation: sheets and webs gone; sparse spray drawn as small separate blobs or not at all; the dragon's
  detached horn tips at 300k drawn apart instead of bridged, if their gap is wider than the lattice resolves;
  roughness slightly up, since the factor varies with the eigenvalue. If thin parts erode or the surface breaks
  up, that is recorded and `zb` stays.
  **Result (2026-10-03 23:38 CDT; lattice build; the three target samples, the bunny's windows 5, 8, 12 and end,
  the dragon's window 15 and end at 300k and end at 100k; `output/gpu/d61`): the factor removes the sheets on
  the target samples and roughens every simulated state; `zb` stays.**
  Target samples: sets apart from the main one 0 discs on both dragons (zb: 1 194 discs seen on 7 457 pixels at
  100k, 2 933 pixels at 300k), the mouth is open, roughness 10.5–14.0° (zb 9.7–14.3°); the offset of 0.845
  pitches keeps the solid area (0.993–1.007 of the base display's).
  Simulated states: the field's normal departs 24–28° from its neighbours' mean on the bunny (zb 12–15°) and
  37–38° on the dragon at 300k (zb 22–24°), heights 0.18–0.30 pitches (zb 0.10–0.22): 1.6–2 times zb's, against
  the criterion of 1.2. The surface there is a foam of lumps: where the outer particles sit in clumps the
  eigenvalue is above 1 between them and the offset drops. 10–40 % more discs (pockets: 8 000–18 000 discs in
  80–330 sets apart, 2 600–9 200 pixels of them seen), 9–19 s a frame against 2.5–4.
  The web between the bunny's ears at windows 5–8 is still there: it is material, a spray dense enough to be a
  body under either field, not a sheet of the field (the expectation was wrong). What the factor does draw
  apart are the detached pieces zb joins to the body: a horn tip and a blob at the snout of the dragon at
  100k, the tail tip and the blob under the tail at 300k.
  Criteria: sheets on the target samples, met; web gone, not met; thin parts kept, met (the crops' solid pixels
  0.99–1.02 of the base's); fringe gone, met (crop's soft pixels 5 100–6 100 against the base's 14 100–27 300);
  roughness, not met. `zb` stays as the exterior's field, and with it the two known faults: sheets in
  concavities (0.2 % of the picture at 300k, 0.6 % at 100k) and detached pieces drawn as joined. The `gaps`
  branch is in this commit only as the record and goes out with the next one.
- **D60, the physics-only twins of D59's three base runs, run once and kept (pre-registered 2026-10-03 23:23
  CDT; the user: "Physics는 계속 돌리지마. 한 번 돌리면 그냥 그 값가지고 재사용해 … '모든 프레임'에서 검증이 완료 되어야
  해"; `tmp/run12.sh`; `output/gpu/d60`).** The base (`repo_r64`) with `--render_weight_scale 0` on the bunny at
  300k and the dragon at 300k and 100k, D59's commands otherwise (seed 97, own stop). Kept of each run: its JSON
  (every window's record: transport, silhouette, shading, the render weight and its share, all read at scale 0
  as well), every 12th simulated frame, and the per-frame table of the display measures; the full archive is
  deleted. These three are the reference of every later render experiment at the same mesh and N and are not
  run again. Nothing is tested by them alone; what is compared with them is registered with each experiment.
- **D59, a displayed exterior apart from the simulated interior, stage 1a: the layer built on single states
  (display only, no simulation; pre-registered 2026-10-03 20:06 CDT, before the probe exists; the user approved
  the stage: "OK. 들어가 줘"; `scripts/probes/settled/surface_layer_probe.py`; D57's kept key frames and target
  samples of the bunny at 300k and the dragon at 300k and 100k; `output/gpu/d59`).** The interior stays what it
  is: the MPM particles. The exterior is a set of M = 300 000 surface discs with no mass, defined from the
  particles of one state:
  the surface is the zero set of Zhu and Bridson's field f(q) = |q − x̄(q)| − r̄, x̄ the kernel-weighted mean
  of the particles within R = 2 pitches of q (the pitch a = (V/N)^(1/3) of the volume sample), r̄ a particle
  radius: a thin sheet of particles keeps its thickness and an isolated particle is a sphere, where a density
  level would erode whatever is thinner than the kernel;
  the discs are a Poisson-disk set on that surface (a dense pool of points projected onto f = 0 by Newton
  steps, then the largest set with no two points nearer than r, r chosen so that the count is M; Bowers 2010,
  Corsini 2012), each drawn as a disc with the field's gradient as normal, a tangential sigma of 0.65 of the
  set's median neighbour distance (the ratio the base display has on a target sample) and full opacity; no
  enlargement rule, no density normals, no live support.
  Drawn with the base renderer's rasteriser and camera, beside the base display of the same particles: the
  target's own samples (the level no morph can beat today), each run's end frame, and its frames at windows 20
  and 39 (thin parts still partly detached sets).
  Criteria, on the target samples first: the ears and horns have a clean edge in the exterior drawing (the
  fringe the base drawing shows is gone, judged on the same crops; soft pixels of the crop lower than the
  base's); the exterior's solid region agrees with the base's (intersection over union 0.97 or more, after
  r̄ is set so that the two areas match); the thin parts keep their thickness (no ear or horn eroded). Then
  what the layer does with the dragon's detached horn tips at 300k: bridged, separate blobs, or dropped, to be
  recorded as found. Expectation: the target samples come out clean with the thin parts kept; mid-morph frames
  show detached sets as beads rather than fur; r̄ near 0.6–0.9 pitches. If the thin parts are eroded or
  spurious blobs appear in concavities (the field's known weakness), the field is the thing to change next
  (Adams 2007's carried distance, Yu and Turk's anisotropic kernels), not the sampling. Stage 1b, the same
  layer carried from frame to frame with insertion and deletion by neighbour count, follows only if 1a holds.
  **Result (2026-10-03 21:02 CDT; twelve states: target sample, end frame, windows 20 and 39 of the three runs;
  sheets in `output/video_2026-10-03/d59_exterior_layer`): the criteria hold on every state; the field's known
  weakness shows in the concavities, most at 100k.**
  The field settled first, on the bunny: R = 2 pitches is bumpy at the particles' scale; R = 3 with r̄ = 0.8
  pitches keeps the base display's solid area (0.995–1.008 of it; 1.015 and 1.034 at r̄ = 1.0 and 1.2);
  smoothing the particle centres (Yu and Turk's Eq. 6 alone, without their kernels) made worms and is dropped.
  Three faults of the probe itself were found by measurement and are not the field's:
  the dark pits of the first drawings were surface without discs (1.1 % of the zero set on the target sample
  and 4.4 % on the end frame farther than 1.5 r from a disc), 1.85–2.0 pitches from the nearest seed where the
  covered surface is 0.8–0.87 from one: the pool was seeded from the base display's surface mask (5 % of the
  particles), which does not cover the zero set; seeded from the grid nodes within a cell of the zero set, 0.00 %
  is uncovered;
  the fine dark specks were the far side seen through one layer of discs: at the pre-registered sigma (0.65 of
  the neighbour distance) 10–16 % of the surface lets more than a fifth through, at sigma = r (the set's
  covering radius) 0.0 %;
  one build never ended: two equal random priorities in the Poisson-disk selection; priorities are a permutation.

  | state | discs | IoU of the solid regions | crop's soft pixels, base → exterior | crop's solid pixels, exterior / base | field's roughness: normal against its 48 neighbours' mean, height above their plane | sets apart from the main one: discs, pixels seen (crop) |
  |---|---|---|---|---|---|---|
  | bunny 300k, target sample | 307 668 | 0.9918 | 20 114 → 2 853 | 0.988 | 9.7°, 0.084 a | 4 301, 78 (8) |
  | bunny 300k, end (window 55) | 294 919 | 0.9881 | 14 567 → 2 965 | 1.010 | 12.8°, 0.106 a | 424, 79 (0) |
  | bunny 300k, window 20 | 292 312 | 0.9879 | 16 076 → 2 863 | 1.005 | 11.3°, 0.101 a | 322, 0 |
  | bunny 300k, window 39 | 293 488 | 0.9880 | 14 117 → 3 001 | 1.011 | 12.4°, 0.106 a | 417, 299 (0) |
  | dragon 300k, target sample | 301 264 | 0.9878 | 17 701 → 3 257 | 0.970 | 14.3°, 0.132 a | 343, 2 933 (45) |
  | dragon 300k, end (window 160) | 304 109 | 0.9871 | 13 718 → 3 194 | 0.986 | 24.1°, 0.221 a | 6 634, 748 (101) |
  | dragon 300k, window 20 | 304 284 | 0.9866 | 16 477 → 2 888 | 0.989 | 22.4°, 0.197 a | 2 025, 689 (148) |
  | dragon 300k, window 39 | 304 209 | 0.9878 | 13 430 → 2 874 | 0.983 | 21.8°, 0.200 a | 2 501, 265 (0) |
  | dragon 100k, target sample | 292 263 | 0.9791 | 21 715 → 2 901 | 0.950 | 14.3°, 0.095 a | 1 309, 8 362 (2 270) |
  | dragon 100k, end (window 96) | 298 279 | 0.9822 | 17 425 → 2 078 | 0.981 | 16.1°, 0.111 a | 2 428, 1 564 (49) |
  | dragon 100k, window 20 | 297 671 | 0.9811 | 16 340 → 1 769 | 0.972 | 13.5°, 0.093 a | 197, 0 |
  | dragon 100k, window 39 | 297 880 | 0.9827 | 15 864 → 1 874 | 0.977 | 13.8°, 0.104 a | 2 249, 4 989 (0) |

  The edges: the fringe of the base drawing is gone on all twelve (the crop's soft pixels fall to 11–22 % of the
  base's; over the whole picture 41 069 → 6 384 on the bunny's target sample, 91 166 → 11 362 on the dragon's
  at 100k), the solid regions agree (IoU 0.979–0.992), ears and horns keep their width. The exterior's discs are
  a quarter to a third of a pitch apart with a sigma of 0.26–0.47 of the base's, so the edge is as sharp as
  that; an interior of 100 000 particles under 300 000 discs draws as clean an edge as one of 300 000.
  The dragon's horn tips at 300k, detached sets to window 160 under the base display (D57): bridged. The field
  joins what lies within about its radius, the horns are drawn as solid tongues at windows 20, 39 and 160, and
  the sets apart from the main one cover 101–148 pixels of the horn crop. What the layer therefore does not
  show is whether the particles there are connected: it draws the same horn over a detached set and over an
  attached one.
  The surface under its own normal is lumpy: the field's gradient departs 10–14° from its neighbours' mean on the
  target samples and 13–24° on the runs' frames, with heights of 0.08–0.22 pitches (the dragon's end frame at
  300k is 1.7 times as rough as its target sample, the bunny's 1.3 times: a reading of the particles that
  neither drawing shows once the normals are averaged). Drawn with the base display's own treatment of its
  normals (the mean over the neighbours within the reach of its 32 nearest particles, 2.7–2.9 pitches, twice;
  256 discs at most) the same discs give a smooth surface (0.9–3.1°); both drawings are on every sheet.
  The known weakness: in gaps narrower than about the kernel (between the horns, in the mouth, between the
  coils) the field puts sheets that belong to no surface, seen as sets apart from the main one: 8 362 pixels
  (0.58 % of the solid picture) on the dragon's target sample at 100k, 2 933 at 300k, and the mouth is partly
  closed. At R = 2 pitches they shrink (3 505 and 520 pixels) and the mouth opens, but the surface is twice as
  rough (height 0.29 pitches on the 300k target sample, 0.48 on its end frame), 4.5 % of the discs go to
  pockets inside the body (2 244 sets) and 60 836 discs face against their neighbours: the radius trades one
  fault for the other, so the remedy is the field's definition (as pre-registered), not R.
  Cost as built: 60–100 s a state (about 25 million field evaluations at 129 neighbours each), too slow for
  every frame. Rendering influence: none, no run was changed; the drawings are of archived states.
  **Stage 1b, the exterior on every frame of the video (pre-registered 2026-10-03 21:04 CDT, before the code).**
  Three fresh runs of the base (`repo_r64`, D57's commands: bunny 300k started 20:58, the dragon at 300k and
  100k 20:59), of which every 12th simulated frame is kept (the video's frames; `tmp/d59run.sh`).
  The layer is built on each frame from that frame's particles alone, with nothing carried: one disc for each
  cell of a lattice fixed in space that the zero set crosses (the cell's centre projected onto f = 0 and kept
  where it stays in its cell), the lattice's pitch h set once per run so that the target sample's surface takes
  M = 300 000 discs, sigma = h. This comes before the carried layer the proposal named for three reasons: the
  carried layer needs this extraction as its insertion step in any case; a lattice fixed in space gives the same
  discs for the same surface, so no random choice can flicker; and it costs one projection per disc (about four
  million field evaluations a frame, against 25 million for the Poisson-disk pool).
  Measured per frame, for the base display and the exterior: soft pixels of the crop and of the picture; the
  pixels that change their solid state for one frame only (blips: solid in frames k − 1 and k + 1 and not in k,
  or the reverse); the sets apart from the main one and the pixels they are seen on; the discs; seconds. On every
  40th frame the share of the zero set that lets more than a fifth through.
  Criteria: the fringe is gone on every frame (the crop's soft pixels below the base's on each); no flicker that
  the base does not have (the exterior's blips not above the base's, none visible in the crop video); thin parts
  drawn without holes as they grow (0 % of the zero set uncovered); under 15 s a frame.
  Expectation: clean edges over the whole morph; a disc count that grows with the surface from the sphere to
  the target; blips of the lattice's scale possible at edges. If it flickers, the discs are carried (moved with
  the kernel-weighted displacement of their particles, reprojected, thinned, the gaps filled from this
  extraction).
  **Result (2026-10-03 22:38 CDT; videos `output/video_2026-10-03/d59_exterior_layer/video`, panels: base display
  | exterior with the field's normals | exterior with the base display's normal treatment): the fringe is gone on
  every frame of the three runs and the exterior flickers less than the base display; early in the morph it
  draws spray as webs, and it does not find the smallest closed sets.**
  The runs (base code, own stop): bunny 300k 73 windows, 904 s, silhouette 0.9872, λ first window 0.249, g_share
  0.87; dragon 100k 107 windows, 917 s, 0.9809, 0.42, 0.91; dragon 300k 151 commits, 3 237 s, 0.9848, 0.396, 0.91.
  Rendering influence of this stage: none, the layer is drawn after the run.
  Cost: the field read through a neighbour tree took 16 s a frame (129 neighbours a point); with the particles
  binned in cells of the kernel's radius the same layer (299 995 against 300 019 discs on the bunny's target
  sample, the same IoU to four places) takes 2.4–4.0 s a frame at 300k.

  | run (frames) | lattice pitch, discs: sphere → most → end | crop's soft pixels, base → exterior (exterior / base, median and worst frame) | blips over the run, whole picture: base → exterior (frames where the exterior has more) | the same in the crop | IoU of the solid regions, median (least) | sets apart, pixels seen: median (most) |
  |---|---|---|---|---|---|---|
  | bunny 300k (245) | 0.398 a, 213 651 → 367 707 → 304 778 | 14 289 → 4 531 (0.31, 0.35) | 98 910 → 83 931 (94 of 243) | 50 965 → 37 169 (75) | 0.9844 (0.9836) | 0 (1 010) |
  | dragon 100k (358) | 0.340 a, 141 221 → 379 146 → 298 890 | 17 000 → 3 578 (0.21, 0.29) | 227 565 → 157 630 (42 of 356) | 73 490 → 39 674 (48) | 0.9814 (0.9582) | 1 031 (5 324) |
  | dragon 300k (491) | 0.498 a, 140 910 → 395 026 → 310 967 | 13 815 → 4 876 (0.36, 0.68) | 222 934 → 136 691 (85 of 487) | 58 190 → 28 766 (106) | 0.9825 (0.9681) | 547 (7 990) |

  The fringe: on no frame of any run does the exterior have more soft pixels than the base, in the crop or in the
  whole picture (the worst ratio, 0.68, is the dragon's first frames, when the crop is nearly empty).
  The flicker: the blips follow the particles (a sawtooth with the windows in both drawings, the breathing of D1);
  over a run the exterior has 15–39 % fewer of them than the base in the whole picture and 27–51 % fewer in the
  crop, more than the base on 12–39 % of the frames. Nothing was carried from frame to frame, so the carried
  layer, its insertion and deletion are not needed for display.
  What the layer draws that the base does not: early in the morph (windows 3–10) the spray between the bunny's
  ears, a faint fuzz in the base drawing, is a web joining the ears in the exterior, and the dragon's solid
  region is up to 4 % larger than the base's at windows 4–6 (IoU 0.958): the field joins sparse particles within
  its radius into solid material, the same property that bridges the detached horn tips. By window 12 the ears
  are apart and clean.
  What it does not draw: 0.2–0.35 % of the zero set on the bunny's frames, 0.4–0.8 % on the dragon's at 100k and
  1.0–2.3 % at 300k has no disc within 1.5 sigma (the target samples: 0.03–0.04 %). Every such point lies in a
  cell the search never reached: closed sets that hold no node of the search's coarse lattice (two pitches).
  82–84 % of them at the dragons' end frames have 57 or more particles within R (the open surface has 32, the
  deep interior 113): pockets under the surface, unseen. 34–99 points a frame at the ends (0.01–0.02 % of the
  zero set), and 755 at the dragon's window 15, lie in the open with 20 particles or fewer: small sets around
  sparse particles, among them the field's own blobs between two distant particles. Widening the search to the
  nodes that are outside by less than their cube's half diagonal did not find them (88 against 89 on the bunny)
  and is not kept. The Poisson-disk build of stage 1a drew all of these; the per-frame layer therefore hides the
  smallest floaters, while a set large enough to hold a coarse node is drawn apart (the blob under the dragon's
  tail in the last frame).
  The discs: at a fixed budget the lattice's sigma is its pitch (0.34–0.50 a), wider than the Poisson-disk set's
  r (0.24–0.34 a), so the edge is softer than stage 1a's (the bunny's end frame: 4 678 soft pixels in the crop
  against 2 965, the base 14 567).
  Against the criteria: fringe gone on every frame, yes; no flicker beyond the base's, yes by the run's totals
  and not on every frame; thin parts without holes, yes in the drawings, with the 0 % of the pre-registration
  not met (the closed sets above); under 15 s a frame, yes after the binning.
  Open, in the order they would be taken: the webs and the sheets in concavities are the field's (stage 1a's
  finding; the remedy is its definition); the sets the search misses; the lumpy surface under the field's own
  normal (hidden by the base's normal averaging in both drawings). Stages 2–4 (the render loss on this layer, a
  smaller interior, a variable-mass shell) are the user's decision.
- **D58, literature: resampling, and a simulated interior apart from a displayed surface (three agents, 2026-10-03
  18:45–19:00 CDT; the user: "resample/3DGS 에서의 resample 방법들 … Surface를 덮어야 하는 N 이 모자라서 … 내부와 외부를
  이제는 진짜 나눠야 할 거 같으니"; papers opened in full unless marked; extracted texts in the session's
  scratchpad).** No design is fixed by this entry.
  *How 3D Gaussian splatting adds, removes and moves primitives.* 3DGS (Kerbl, SIGGRAPH 2023): clone a small
  Gaussian and split a large one where the view-space position gradient exceeds 0.0002, prune low opacity,
  reset opacity. The later criteria change what triggers it, not where primitives go: per-Gaussian error
  (Rota Bulò, ECCV 2024), absolute or pixel-weighted gradients (AbsGS, ACM MM 2024; Pixel-GS, ECCV 2024; GOF,
  SIGGRAPH Asia 2024), the area on which a Gaussian is the largest contributor (Mini-Splatting, ECCV 2024), a
  saddle of the loss (SteepGS, CVPR 2025), a score under a budget (Taming 3DGS, SIGGRAPH Asia 2024). Only
  3DGS-MCMC (Kheradmand, NeurIPS 2024) works at a fixed count by moving: a dead Gaussian (opacity under 0.005)
  is put on a live one drawn in proportion to opacity, and the co-located copies get opacity 1 − (1 − o)^(1/N)
  and a shrunk covariance so the picture is unchanged. Mip-Splatting (CVPR 2024) floors a primitive's size at
  the sampling interval. None of these criteria measures how many primitives cover a piece of surface.
  *Primitives held on a surface.* 2DGS (Huang, SIGGRAPH 2024): oriented discs with normal and depth losses,
  3DGS's densification, no even sampling. SuGaR (CVPR 2024), Gaussian Frosting (ECCV 2024), GaussianAvatars
  (CVPR 2024): Gaussians bound to mesh triangles by barycentric coordinates, sampled once (Frosting: a fixed
  budget in a layer around the mesh); they follow the mesh and are as even as it is. DG-Mesh (ICLR 2025): one
  Gaussian per face, merged where several share a face and created at empty faces: the only explicit evenness
  rule found.
  *Physics with Gaussians.* PhysGaussian (CVPR 2024): one set, every Gaussian an MPM particle, the interior
  filled once before the simulation, no resampling; its anisotropy loss is the only measure against "plush"
  artefacts. Two sets, the simulated one driving the displayed one by an interpolation fixed at t = 0:
  Gaussian Splashing (arXiv 2401.15318; PBD particles, GMLS), VR-GS (SIGGRAPH 2024; a tetrahedral cage), GIC
  (NeurIPS 2024; Gaussians "for rendering only"), PhysDreamer and Spring-Gaus (ECCV 2024), PhysSplat (ICCV
  2025). None resamples the displayed set during deformation, and none moves interior primitives to the
  surface: a fixed embedding stretches with the material.
  *Resampling in particle simulation.* By distance to the surface: Adams (SIGGRAPH 2007; split when distance
  plus local feature size is under 2 radii, merge above 3), Winchenbach (SIGGRAPH 2017; mass linear in the
  surface distance, split, merge and share with exact mass), Ando 2013 (SIGGRAPH; a sizing function, new
  particles snapped onto the surface). In thinning sheets: Ando 2012 (TVCG; neighbour covariance σ3 ≤ 0.2 σ1
  at density under 0.7, a particle inserted between a separating pair). In MPM both known schemes leave the
  outermost layer alone: Yue (TOG 2015) inserts only deeper than 2.2 radii ("popping" nearer the surface), Gao
  (SIGGRAPH Asia 2017) splits below the visible layer and merges deep inside. Narrow-band FLIP (Ferstl 2016,
  Sato 2018) keeps particles only near the surface and the interior on the grid, without exact mass.
  *A displayed surface apart from the simulated points, resampled as it deforms.* Müller et al. (SCA 2004) and
  Keiser et al. (2005): mass-carrying phyxels and a separate set of massless surfels that follow the phyxels'
  displacement field; a surfel with too few surfel neighbours is split, with too many deleted (6 and 9 in a
  radius), then tangential repulsion and projection onto an implicit coat of the phyxels. Pauly et al.
  (SIGGRAPH 2003): each sample carries two tangent vectors deformed with the surface; when their stretch is
  too large the sample is replaced by two along the major axis; deletion deferred. Witkin and Heckbert
  (SIGGRAPH 1994), Meyer et al. (2005, 2007): particles on an implicit surface with repulsion, tangential
  motion and reprojection, fission and death by energy against the six-neighbour ideal with a hysteresis gap
  (0.35 and 1.75) against insert/delete cycles. Blue-noise selection (Bowers 2010, Corsini 2012, Yuksel 2015)
  resamples from scratch, with no coherence between frames. Learned upsampling (PU-Net, PU-GAN, Grad-PU):
  repulsion or uniformity losses, farthest-point selection.
  *Synthesis.* The standard rule for a stretching surface is to insert where the local count (or a carried
  stretch) falls under a fraction of the hexagonal ideal and to delete above an upper one, with a gap between
  the two; samples move only in the tangent plane and are reprojected. What a separate surface set needs from
  the volume particles is a smooth displacement field with its gradient and a scalar field whose level set is
  the surface. The combination the user describes (a simulated interior, a displayed exterior resampled as
  the area grows, with Gaussians) has its parts in the literature and no instance: the Gaussian-splatting
  papers fix the embedding at t = 0, the point-based papers of 2003–2005 resample but predate splatting.
  pre-registered 2026-10-03 18:38 CDT at the runs' launch; the user: "particle 문제인지 render 문제인지 시작해서, 모든
  frame에서 조사 들어가자. 그리고 나서 수정을 들어가야 할 거 같아. 40K 갤러리는 현재 이걸 보지 못하니까, bunny 300K와
  dragon 300K/100K를 기준으로"; the base e9a210f = repo_r64; `tmp/d57.sh`, archives kept; `tmp/fuzz_probe.py`;
  `output/gpu/d57`).** Three runs of the base with the default recipe: the bunny at 300k (GPU 0), the dragon at
  300k (GPU 1) and at 100k (GPU 3). No change to the algorithm or to the display; the probe reads the archive
  and draws with the base renderer's own primitives.
  The particles, on every simulated frame: the rendered particles that are not linked to the body within one
  layer spacing (single linkage; the body is the largest set), how many sets, how far from the body; the
  particles the display rule draws enlarged (8th-neighbour distance over the target's, the rule's own factor,
  above 1.5) and how many of those are detached; from the frames near the end also their distance to the target.
  The display, on every frame of the video (every 12th simulated frame), three drawings of the same particles
  with the renderer's own coverage buffer: as the base draws it; with every disc at the target spacing (no
  enlargement); with the detached particles left out. Per drawing the solid pixels (coverage 0.5 and more) and
  the soft pixels (0.02 to 0.5), whole picture and the thin part's crop (the bunny's ears, the dragon's horns);
  the same three drawings of the target's own sample as the level a finished surface has. The crops of every
  frame are kept side by side as a video.
  Readings. The display's part: the soft pixels the enlargement adds, base minus no-enlargement, against the
  same difference on the target's sample (what the rule adds to a perfect surface). The particles' part: the
  soft pixels the detached particles draw, base minus without-them. If the soft pixels above the target's
  level go with the enlargement and stay when the detached particles are left out, it is the display; if they
  go with the detached particles in both drawings, it is the particles, and the display only decides whether
  they read as fur or as beads. Expectation, uncertain (D34, D42 on today's other code): both, the particles
  first: most of the tuft's soft pixels go when the detached particles are left out, the enlargement makes
  what is left of them wider, and the target's own sample has a soft rim under the base rule that the
  no-enlargement drawing does not. 100k against 300k: the same share of particles detached, drawn wider at
  100k (the discs follow the spacing).
  Added to the probe after its first test (18:44, before any run's result): the surface's own sampling. The
  body's surface particles (neighbourhood asymmetry, not detached) counted against the target sample's own,
  whole and on the thin part (nearest target point below 2 MPM cells); their spacing on the surface against
  the target's (above 1.5: too few particles for that piece of surface); and per drawn frame the pixels that
  are solid only by the enlargement and only by the detached particles. The soft-pixel count alone did not see
  the tuft: on the test frames the target's own sample has more soft pixels in the ears' crop than the run.
  **Result, the bunny at 300k and the dragon at 100k (2026-10-03 19:00 CDT; `output/gpu/d57/probe_*`, locally
  `output/video_2026-10-03/d57_particle_or_render/`: curves, tables, the crops of every drawn frame as a video,
  the target sample's drawings).** Runs: bunny 728 s, 57 commits of 65, silhouette IoU 0.9879; dragon at 100k
  600 s, 96 of 106, 0.9799, world-thin 2.05 %; render λ of the first window 0.249 and 0.420, g_share 0.88 and
  0.91. Census on every simulated frame (2201 and 3841), drawings on every 12th (185 and 321).
  The target's own sample, drawn by the base renderer, has the fuzz. Bunny at 300k: the left ear's edge is
  fuzzy in the base drawing and hairy without the enlargement; 1.05 % of the solid pixels are solid only by
  the enlargement (2.6 % in the ears' crop). Dragon at 100k: the horns of the perfect sample are feathers in
  all three drawings; 1.76 % (4.0 % in the horns' crop). A finished morph cannot look better than this under
  this display with these particles.
  The run against that level, bunny (windows 3, 5, 8, 11, 17, 28, 39, 55): surface particles against the
  target sample's 15 583: 0.67, 0.78, 0.81, 0.83, 0.85, 0.86, 0.86, 0.85; on the thin part (6019): 0.62, 0.70,
  0.73, 0.80, 0.80, 0.83, 0.82, 0.81; surface particles with spacing above 1.5: 1.6, 0.7, 0.3, 0.4, 0.4, 0.2,
  0.2, 0.3 %; rendered detached particles 9959, 4281, 2791, 2233, 1698, 1356, 1376, 1528 (84–99 % within the
  berth from window 8); pixels solid only by the enlargement 2.4, 2.6, 1.6, 1.2, 1.0, 0.83, 0.77, 0.79 % (crop
  6.4, 9.3, 5.2, 3.8, 2.8, 2.1, 1.9, 2.0 %); only by the detached particles, in the crop: 102 353, 47 764, 16 145,
  5431, 1861, 486, 556, 834 pixels (60 % of the crop's solid pixels at window 3, under 1 % from window 17).
  Dragon at 100k (windows 5, 10, 14, 19, 29, 38, 58, 77, 96): surface count 0.57, 0.88, 0.97, 0.99, 1.02, 1.01,
  1.00, 0.98, 0.98; thin part 0.72, 0.73, 0.82, 0.81, 0.88, 0.88, 0.86, 0.85, 0.85; spacing above 1.5: 11.6, 0.9,
  0.8, 0.5, 0.7, 0.8, 0.7, 0.8, 0.8 %; detached 2632, 2468, 1720, 1572, 1099, 1070, 1038, 1101, 936; only by the
  enlargement 7.5, 2.6, 2.0, 1.7, 1.45, 1.3, 1.3, 1.2, 1.2 % (horns' crop 19.8, 9.5, 8.4, 6.3, 5.3, 4.7, 5.0, 3.8,
  4.3 %); only by the detached, crop: 27 870, 34 624, 18 136, 27 556, 12 880, 12 448, 12 117, 4618, 2171 pixels (a
  fifth of the crop's solid pixels until window 20, 7–8 % until window 67, 1–3 % at the end).
  Reading so far. Early and mid-morph it is the particles: the thin parts are first drawn by detached sets (the
  bunny's ears to window 10, the dragon's horns at 100k to window 70). At the end it is neither loose
  particles nor the enlargement in excess of a perfect sample: the end frames sit at or under the target
  sample's own level on the enlargement (0.8 against 1.05 %, 1.2 against 1.76 %), the detached particles draw
  under 1–3 % of the thin crop, and the surface carries 85–98 % of the particles a perfect sample puts there
  (80–88 % on the thin part). The fuzz that is left at the end is what a volume sample of this N looks like
  at a thin feature under this display: the user's reading ("the N that has to cover the surface is short")
  holds, and it holds for the target sample itself. The 300k dragon follows.
  **Result, the dragon at 300k (2026-10-03 19:56 CDT).** The run: 3168 s of simulation (3301 s of process), 162
  commits of 174 attempts (the event at window 111, stop at 173), silhouette IoU 0.9846, world-thin 0.44 %,
  chamfer 0.0589, last kinetic record 9.1e-4; render λ of the first window 0.396, g_share 0.92. Census on 6401
  simulated frames, drawings on 535.
  The target's own sample: 24 572 surface particles of 299 765 rendered (8.2 %), 11 650 of them on the thin
  part; its horns have a spiky fringe in all three drawings; 1.31 % of its solid pixels are solid only by the
  enlargement (3.5 % in the horns' crop).
  The run (windows 8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160): surface particles against the target
  sample's 0.71, 0.77, 0.81, 0.82, 0.82, 0.81, 0.80, 0.80, 0.79, 0.79, 0.78, 0.79; thin part 0.67, 0.69, 0.75, 0.75,
  0.77, 0.75, 0.73, 0.73, 0.73, 0.73, 0.72, 0.74; spacing above 1.5 on 1.7, 0.8, 1.1, 1.1, 0.9, 1.0, 1.0, 1.1, 1.1,
  1.0, 1.0, 1.0 % (thin 1.1–2.0 %); rendered detached particles 7624, 8801, 6543, 6145, 5604, 5529, 5463, 5007,
  4501, 4155, 4211, 3739 (93–96 % within the berth from window 24); solid only by the enlargement 4.0, 2.1, 1.6,
  1.45, 1.3, 1.2, 1.15, 1.1, 1.1, 1.15, 1.1, 1.1 % (crop 12.8, 11.0, 8.0, 6.8, 5.3, 4.8, 4.3, 4.0, 4.0, 4.3, 4.0, 3.7
  %); solid only by the detached particles, in the crop: 37 659, 20 148, 26 152, 21 916, 15 590, 10 751, 16 455,
  10 265, 9038, 9320, 11 734, 11 342 pixels: 6–8 % of the crop's solid pixels from window 64 to the end.
  Corrected reading for this mesh: at 300k the dragon's horn tips are still detached sets at window 160. Drawn
  without them the tips of both large horns and the top of the small left horn are missing. So here the
  particles' part does not die out with the run as it does on the bunny (under 1 %) and on the dragon at 100k
  (1–3 %); the enlargement's part ends at the target sample's own level (3.7–4.3 against 3.5 % in the crop),
  and the surface holds 79 % of a perfect sample's surface particles (74 % on the thin part), the lowest of
  the three and slowly falling from window 48. With the base display the run's horns at the end are smoother
  than the target sample's own (the relaxed layer against a raw volume sample).
  Reading over the three runs. Early and mid-morph: the particles (thin parts arrive as detached sets). At the
  end: the sample's own limit under this display on all three (the target's own sample is fuzzy), plus, on the
  dragon at 300k, thin tips that remain sets apart from the body. In every case the surface carries only 5–12 %
  of the particles, and 73–88 % of what a perfect sample puts on the thin part.
- **D56, the zero-row fix alone on the restored code: the 300k dragon and bunny (pre-registered 2026-10-03 17:24
  CDT at launch; the user: "그것만 되돌린 코드에 올리고 bunny랑 dragon 실험 해서 보여줄래?"; repo_r63 = the restored tree,
  repo_r64 = repo_r63 with `window/layer.py` and its test; `tmp/d56.sh`; `output/gpu/d56`; the restored code's
  own dragon is the rollback's confirmation run, `output/gpu/d55`).** The fix as in D39, written onto the layer
  code of 617a8ec: a layer particle whose weight row has no weight keeps its place (its row is itself) instead
  of being carried to the plane through the world's origin. The test
  `test_a_layer_particle_without_neighbours_is_not_relaxed` fails on repo_r63 (exit 1) and passes on repo_r64;
  suite 270 passed, exit 0. Runs, the default recipe (64 → 96 px, to their own stop), each alone on a GPU:
  `dragon_fix` (GPU 0) and `bunny_fix` (GPU 2) with the fix; without it `d55/dragon` (GPU 1, since 17:20) and
  `bunny_rollback` (GPU 3); the 45-minute video's run (d17, one code step earlier) as the dragon's second
  baseline, D16b's 300k bunny (0.9868–0.9876 in the earlier 300k runs) as the bunny's. Shown to the user: the
  4K videos and the same crops side by side.
  Expectation, from D39 on today's code: no visible change (the horns as the 45-minute video's with and
  without the fix); silhouette within the same code's run-to-run spread (0.002); the same windows and time
  within the stops' draw (the dragon's two baselines: 129 attempts, and d55's). One run an arm: a difference
  inside that spread is not the fix's.
  **Result (2026-10-03 18:10 CDT; sheets and videos in `output/video_2026-10-03/zero_row_fix/` locally,
  `output/gpu/d55`, `output/gpu/d56` on the server).** Simulation seconds, commits of attempts (the event's
  window), silhouette IoU, world-thin, thin, chamfer, last kinetic record; rendered particles beyond 3, 4.4, 6
  target spacings at the end:
  dragon, the 45-minute video's run (d17): 2704 s, 122 of 129 (105), 0.9838, 0.55 %, 18.8 %, 0.0589, 1.5e-4.
  dragon, the restored code (d55): 2572 s, 120 of 126 (98), 0.9843, 0.91 %, 20.4 %, 0.0590, 4.8e-5; 125, 0, 0.
  dragon, with the fix: 2306 s, 104 of 111 (58), 0.9845, 0.63 %, 17.7 %, 0.0589, 5.1e-4; 96, 0, 0; census 1474
  within the berth, 88 near band, none beyond a loss cell.
  bunny, D16b's run (d17): 726 s, 58 of 63, 0.9877, 0.02 %, 12.8 %, 0.0580, 8.6e-6.
  bunny, the restored code: 1031 s, 81 of 91 (57), 0.9866, 0.05 %, 13.6 %, 0.0579, 5.0e-6; 24, 0, 0; census 1099
  / 88 / 0.
  bunny, with the fix: 872 s, 71 of 78 (39), 0.9873, 0.02 %, 13.5 %, 0.0580, 5.2e-6; 22, 0, 0; census 1177 / 83
  / 0.
  Render: λ of the first window 0.396 (dragon) and 0.249 (bunny) in every run, median g_share 0.92–0.94 and
  0.86–0.88.
  The pictures (the restored renderer, the same crops): the dragon's horns are the 45-minute video's solid
  rounded tubes in both runs, with and without the fix; the whole bodies alike; the bunny's three runs alike,
  ears and body. Expectations met: no visible change from the fix, silhouettes within 0.002 of the same code's
  other runs (the bunny's two runs differ by 0.0007, the dragon's by 0.0002), windows and time inside the
  stops' draw (the event came at window 39–58 with the fix and 57–98 without: two runs each, not read as the
  fix's). The rollback's confirmation is d55: the restored code gives the 45-minute video's result (0.9843
  against 0.9838, the horns alike), in 43 minutes.
  What the user points at next (a crop of the bunny's ears, 17:50): tufts of loose material at the ear. In
  the fix run's frames the fuzz on the left ear's edge is there from window 18 and unchanged to the end at
  window 69 (`q_bunny_fix_ears_time.jpg`): a long run does not remove it.
  The fix is committed on the restored code.
- **ROLLBACK, 2026-10-03 17:19 CDT (the user: "일단 오늘 수정 다시 돌릴 수 있을까? dragon300K_current_schedule_45min_4k
  결과가 나왔었을 때로").** The code is back at 617a8ec, the last commit of 2026-10-02 (23:33): every file under
  `physmorph/`, `scripts/` and `tests/` and the README as they were then; only this log keeps today's entries
  (D20–D53 below), which now describe code that is not in the tree. Today's code is kept whole at the tag
  `settled-2026-10-03-d53` (commit 3d5a02b). What went out: the layer as the body's (D26) and its arrived scope
  (D48), the zero-row fix (D39: the bug is back in the tree, see its entry), the alternating Sinkhorn solve
  (D29), the runtime reuses (D47, D49, D50, D53) and their clocks, the display's surface-spacing disc rule
  (D35), the term dump's channel record, `--render_res`, and the probes of D20–D53. `render_res` is 64 again
  (the 64 → 96 schedule of the 45-minute video).
  The 45-minute run itself (d17's `dragon_on`, 2026-10-02 17:03–17:49) ran one code step earlier, f2720ad,
  before the diagnostic records were taken out of the production path (9ec502a, 18:05: 19.8 against 23.6 s a
  window, the run reading none of them). 617a8ec is that code with the records off.
  Checked after the rollback: the suite on the restored tree, and the 300k dragon run again with the default
  recipe to its own stop (results under this entry).
- **D52, the code as a whole: the arrived-scope layer with the three runtime steps (pre-registered 2026-10-03
  14:58 CDT at launch; repo_r58 = the working tree, file for file (checksums compared), suite 273 passed, exit 0;
  repo_r59 = the same at `render_res` 64; `tmp/d52.sh`, `tmp/d52run.sh`; `output/gpu/d52`).** Four things at
  once, one a GPU: the 40k gallery at 96 px (`d52_96_40k_*`, GPU 0) against D48's and D39's; the 40k gallery on
  the 64 → 96 schedule (`d52_64_40k_*`, GPU 3) against D19's as-is arm (the same schedule before D19;
  `d19_asis_40k_*`) and against the 96-px gallery; the 300k dragon on the 64 → 96 schedule to its own stop,
  alone on GPU 1 (`dragon_full`), against `berth_full` (the same without the runtime steps: 2003 s, 127
  attempts); the 300k bunny the same on GPU 2 (`bunny_full`).
  Criteria: both galleries' silhouette within 0.002 of the earlier runs' mean on at least 15 of 19, none lower
  by more than 0.003 but the known stops (beast's freeze; C's, V's and bob's rejection stops), no dense set
  beyond a loss cell on any mesh; the dragon's quality as `berth_full`'s (silhouette within 0.002, no rendered
  particle beyond 6 spacings at its end, horns solid by the same crop) at 15 % or more less time an attempt;
  the bunny within 0.002 of its earlier 300k runs (0.9868–0.9876) with no dense far set. What the 64 → 96
  gallery decides: whether the schedule of the 45-minute video goes back to being the default (`render_res` 64)
  now that the user asks for its result; expectation: silhouette as the 96-px gallery's, about twice the
  windows.
  **Result (2026-10-03 15:37 CDT; `tmp/d52_96_eval.py`, `tmp/d52_64_eval.py`, `output/gpu/d52/eval_96.txt`,
  `eval_64.txt`).** The gallery at 96 px against D31, D39 and D48: silhouette within 0.002 of their mean on 18 of
  19, median −0.0004; beast frozen (`domain`); C and V at their rejection stops (15 and 14 commits, −0.0013 and
  −0.0010); thin median +0.3 points; committed windows median 30; no dense set beyond a loss cell on a finished
  mesh (one dense particle in beast's frozen state); λ of the first window median 0.263, g_share 0.86.
  The gallery on 64 → 96 against D19's as-is arm and the two 96-px galleries of the new layer: within 0.002 on
  18 of 19, median −0.0001, inside the three runs' range on all but beast (frozen again, 15 commits); no early
  rejection stop on C, V or bob (22, 41, 40 commits); thin median +0.3 points; committed windows median 30 → 43,
  the minutes summed 69 → 104; no dense set beyond a loss cell; λ of the first window median 0.248, g_share
  0.88.
  The 300k dragon, 64 → 96 to its own stop: the event at window 117, stop at 137; 1799 s of simulation (1931 s
  of process), 127 commits of 137 attempts; silhouette IoU 0.9839, world-thin 0.83 %, thin 19.3 %, chamfer
  0.0591, holes 0.01 %, last kinetic record 2.2e-5; at the end 113, 0, 0 rendered particles beyond 3, 4.4, 6
  spacings; census 1713 / 110 / 0. Against `berth_full` (2003 s, 127 attempts, 0.9842): 13.1 s an attempt
  against 15.8 (−17 %), ten attempts more by the draw of its stops; render λ 0.391, g_share 0.91.
  The 300k bunny, the same: the event at window 62, converged at 83; 929 s (1013 s of process), 76 commits of
  84; silhouette IoU 0.9876 (0.9868–0.9876 before), world-thin 0 %, thin 12.3 %, chamfer 0.058, no hole, last
  kinetic record 2.1e-5; nothing beyond 4.4 spacings; census 1081 / 46 / 0; render λ 0.250, g_share 0.90.
  All criteria met. beast froze in both galleries (four of the last five gallery runs; D39: 35–45 % of runs of
  any code): the parked defect, now the one mesh the gallery loses in most runs.
  **Adopted and committed:** the arrived-scope layer (`window/layer.py`, `window/setup.py`), the three runtime
  steps and the record's reuse below (`losses/grid_ot.py`, `losses/support.py`, `window/rollout.py`,
  `window/solve.py`, `run/runner.py`), `render_res` 64 as the default again with `--render_res 96` for the
  short run (`pipeline/config.py`, `scripts/pipeline_run.py`; a two-window run of each on the 40k bunny reads
  64/96 and 96/96 in its config), the README. Suite 273 passed, exit 0, on the tree as committed (repo_r62,
  file for file).
- **D53, the clocks outside the optimiser, and the record's transport energy taken from the commit (2026-10-03
  15:20 and 15:25 CDT at their launches; repo_r60 = repo_r58 with `t_setup`, `t_record`, `t_total` in the
  record; repo_r61 = repo_r60 with one line in `run/runner.py`; `tmp/d53.sh`, 14 attempts of the 300k dragon at
  96 px, alone on GPU 2; `tmp/clock_mean.py`).** A window's turn of the loop, windows 4–13: construction 0.99 s
  (layer data, the trajectories and their captured graphs, the objective with the gate's transport solve),
  start 1.15, gradients 6.41, line search 3.55, commit 0.84, record 0.42 (promotion, assimilation, the frames,
  the record), 13.49 s in all with 0.12 s unaccounted: the earlier "2.2 s outside the clocks" was the
  construction and the record plus the slower first windows. The record evaluated the transport energy at the
  promoted state, which is the commit rollout's own end state unless a guard repaired it: it now takes the
  commit's potentials (`repeat`, set when nothing was clamped): the same measure, the same value; record 0.42 →
  0.17 s. Not yet tried, from the audits: the commit's forty per-step determinants with a host read each; the
  gate's transport solve at every construction (0.28 s); the graphs captured anew for every window.
  Where a 300k window stands: 14.1 s before today, about 12.7 s now by these clocks (D47, D49, D50, D53), of
  which the three adjoint sweeps are about 5.1 s and the Sinkhorn solves 3.0 s. Those two are untouched; the
  ways into them that change the arithmetic (the cleanup's gradient taken with the transport's: one sweep in
  three; a looser solve) are the user's to decide.
- **D51, the notch trap: what holds a set at the notch under the dragon's tail (diagnostic, pre-registered
  2026-10-03 14:40 CDT at launch; repo_r54, 96 px, 40 attempts with the term dump, the archive kept; `tmp/d51.sh`,
  GPU 2; `output/gpu/d51`).** Six of nine 40-window runs of today end with 30–60 rendered particles 6–8 target
  spacings from the target at (1.2, −1.9, −0.4), dense (8th neighbour at 0.3–0.45 coverage radii), detached or
  linked by a strand: the floater that is left. It is there with either layer, with and without the zero-row
  fix and the runtime changes, and gone at the end of a 160-window run. If this run has it: per window, the
  set's particles traced back (when they arrive, by which channel), each term's position gradient on them
  against the same on the body's surface, whether they are layer particles, detached, arrived, inside the spray
  gate or the near band. Expectation: they arrive by advection in the first ten windows, are beyond the near
  band (one loss cell) and denser than the spray gate, so only the transport acts, with a gradient small
  against the surface's; and the notch is where the transport's pull toward the tail and toward the body
  cancel. If the run does not have it (three of nine), it is repeated.
  **Result (2026-10-03 14:54 CDT; `tmp/trap_probe.py`, `output/gpu/d51/trap_probe.txt`).** The run (689 s, 39
  commits of 40, silhouette 0.9842, world-thin 0.45 %; λ of the first window 0.462, g_share 0.94) ends with 2
  particles there, and its dump shows the place emptying, which answers more than an occupied end would. The
  notch's centre is 6.6 target spacings from the nearest target point and lies inside the source sphere: at
  window 0 there are 325 particles within 12 spacings of it and beyond a loss cell from the target. Windows
  1–6: 370–570 (the sphere's material leaving through it), moved toward the target by the advection at 0.7–1.6
  spacings a window. Then 349, 256, 192, 142, 92, 64, 53, 48 (windows 7–14), and from window 15 a remnant of
  43 that goes 41, 38, 38, 38, 36, 36, 34, 33, 28, 27, 25, 24, 19, 17, 16, 16, 13, 8, 3 (windows 16–34): one or
  two particles a window. In those windows the remnant moves toward the target by 0.03–0.3 spacings a window
  from the advection and 0.0–0.2 from u; it is 6–7 spacings out, its 8th neighbour goes from 1.2 to 0.44
  coverage radii (it is drawn together as it shrinks), it is linked to the body by a strand until window 25
  and detached after. The gradients on it against their rms on the arrived layer: transport 6–7 times, spray
  3–19 times, render 2–4 times, the near band none (beyond a loss cell), the proximity none (a target-side
  term).
  Expectations: arrival by advection, refuted in its premise: nothing arrives, the material is the source's own
  and has to leave; beyond the near band, confirmed; "only the transport acts, with a small gradient" and "the
  pulls cancel", refuted: the transport's gradient on it is six times the surface's and the spray's larger
  still. The objective sees the remnant and pushes it; it moves 0.1–0.3 spacings a window against the 6–7 it
  has to go. A set in the air has no handle but u: its own dFc gives internal stress and no net force, and the
  grid couples it to the body only within two cells. u's step is the optimiser's common step (0.5e-3–1.5e-3
  world units an iteration, eight iterations: 0.1–0.35 spacings a window, a tenth of u's clamp of one layer
  spacing). So the remnant is not trapped; it is being walked out at u's pace, and whether a run has 2 or 60
  particles there at window 39 is how far that walk has got (six of nine runs: not far enough; at 160 windows:
  done). This is D32's parked item (u bounded by the accepted step) seen at its largest instance.
- **D50, runtime: the replay pair takes the warm start's evaluation as its first (pre-registered 2026-10-03
  14:40 CDT at launch; repo_r57 = repo_r56 with `window/solve.py`; `tmp/d50.sh`, alone on GPU 0;
  `output/gpu/d50`).** A window's start evaluates the zero control and the warm control, keeps one, and then
  evaluates the kept control twice more to measure how two rollouts of one control differ. The first of those
  two is a rollout the warm start already has: `warm_start` returns the kept evaluation and `replay_noise`
  rolls out one more against it. The measure is the same (two independent rollouts of the start control); the
  window's first evaluation with the tape still follows a solved evaluation at its point (D47). Suite 272
  passed, exit 0. Predictions: evaluations without the tape 14 → 13 a window, the window's start 1.4 → about 1.1
  s, the 40-attempt run 566 → about 555 s; quality inside the earlier runs' spread; the record `replay_rel` of
  the same size as before (median over windows within a factor of two of D49's). Fail: a quality number
  outside, or the replay record off by more than that.
  **Result (2026-10-03 14:56 CDT).** The 40-attempt run: 549 s of simulation (596 s of process) against 566;
  39 commits of 40; the window's start 1.50 → 1.12 s (median over the committed windows), the window by its own
  clocks 12.12 → 11.49 s; silhouette IoU 0.9836, world-thin 0.50 %, thin 18.9 %, chamfer 0.0595, holes 0.01 %,
  last kinetic record 1.1e-3; render: λ of the first window 0.463, median g_share 0.93. Beyond 4.4 and 6
  spacings at the end 71 and 31 (the notch remnant). The replay record where it is kept (the windows without a
  commit): 1.3e-7 against 1.0e-7. Predictions met. Kept.
  The three runtime steps together: the 40-attempt 300k dragon 664–669 s → 549 s (−18 %), the window by its
  clocks 14.1 → 11.5 s. A run's seconds an attempt are about 2.2 s more than its windows' clocks (549 s / 40 =
  13.7 s against 11.5): the window's construction (layer data, the captured graphs, the gate's transport
  solve), the record on the promoted state (one more Sinkhorn solve at the commit rollout's own end state),
  the frames copied to the host. That part has no clock yet; it is the next thing to measure.
- **D49, runtime: the proximity term asks for one neighbour, the trajectory's min det is taken step by step
  (pre-registered 2026-10-03 14:23 CDT at launch; repo_r56 = repo_r53 with `losses/support.py` and
  `window/rollout.py`; `tmp/d49.sh`, alone on GPU 0; `output/gpu/d49`; the parts timed by
  `tmp/bench_eval_parts.py` on D43's kept state, on a GPU shared with a running cell).** From the two code
  audits (agents, read-only). The surface proximity term built a KD-tree of the body and asked for each outer
  target point's 32 nearest particles, then took the minimum of the 32 distances: the nearest alone gives the
  same d_min on all 19 030 points (0 differ), the query 50 → 5 ms of the term's 66. The trajectory's min det
  stacked the 41 deformation gradients twice and added the control to all 40 steps; the control is zero in the
  released half, where det(F + dFc) before a step is det F after the step before: step by step on the
  trajectory's own buffers, the same minimum (0.808378041 both), 30 → 17 ms. Suite 272 passed, exit 0.
  Predictions: the proximity term 47 → about 20 ms a call (1.0 → 0.45 s a window); the line search 3.5 → about
  3.2 s; the window 12.1 → about 11.2 s; the 40-attempt run 590 → about 550 s; quality inside the earlier runs'
  spread. Fail: a quality number outside, or less than 0.6 s a window gained.
  **Result (2026-10-03 14:38 CDT).** With `--profile`, windows 4–13: the proximity term 46.6 → 25.9 ms a call
  (1.03 → 0.57 s a window), the line search 3.48 → 3.24 s; the window 12.10 → 11.82 s, because the three adjoint
  sweeps, which this step does not touch, read 217 ms in this run against 208 in D47's (219 in D44's): the
  parts changed gained 0.70 s, the total 0.28 s. The 40-attempt run: 566 s of simulation (613 s of process)
  against 588 and 591: 0.58 s a window; 38 commits of 40; silhouette IoU 0.9844, world-thin 0.46 %, thin 19.3 %,
  chamfer 0.0593, holes 0.02 %, last kinetic record 8.3e-4; render: λ of the first window 0.463, median g_share
  0.94. Beyond 4.4 and 6 spacings at the end: 44 and 20, of them 35 at the notch under the tail (the trap, D47;
  six of nine 40-window runs now). By the letter the gain is at the criterion's edge (0.58 s a window by the
  run, 0.28 by the profile's total, 0.70 by the parts); the values are the same by construction (the same
  nearest particle, the same minimum), and world-thin (0.46 %) is the highest of the five runs since the
  zero-row fix (0.16–0.32), inside the 0.2-point allowance. Kept.
  Also from the audits, not in this step: the window's start evaluates the start control twice (0.3 s a
  window); in each adjoint sweep, 443 MB of geometric-F gradients zeroed and a dead kernel swept, the control's
  gradient cloned and stacked over 40 steps of which 20 carry it (single-digit per cent of a sweep); the three
  sweeps themselves (5.0 s a window, P2G, G2P and the stress adjoint) have no exact reduction short of
  carrying three adjoints through one sweep. Merging the cleanup's gradient with the transport's would save a
  sweep in eight but is not the same arithmetic: the render gradient's projection is active at the first
  iteration of 269 of 871 windows (`tmp/pc_active.py`: the two dragons' windows 1–3, a third to a half of the
  windows of most 40k meshes), so it would change the result wherever it is active. The Sinkhorn ladder with a
  fixed number of blocks a level was tried in D25–D29 and costs more.
- **D48, a detached set is in the air only beyond the berth: an arrived particle is a layer particle like any
  other (pre-registered 2026-10-03 13:58 CDT at launch; repo_r54 = repo_r47 with `window/layer.py`,
  `window/setup.py` and a test, repo_r55 the same at `render_res` 64; `tmp/d46.sh`, tags `berth64`, `berth64b`
  (48 attempts, GPUs 1 and 3), `berth96` (40 attempts, GPU 2); `output/gpu/d46`).** D46 (below): the old layer
  makes the horns solid and brings the far dense sets back in one of its two 64-px runs and at 96 px; D26's
  normal with the relaxation given back keeps the far sets away and leaves the horns feathered. So what makes a
  thin feature solid is u drawing a set together along its own normals, the same motion that clumps a set in
  the air. The two differ in where the set is: 91–99 % of the loose material is within the berth of the
  target (D45), the floaters are beyond a loss cell. The definition: D26's membership (not relaxed, u along the
  direction away from the non-members) holds for the particles of a detached set that are beyond the
  objective's berth (`nn_berth_k` target spacings, the near band's inner edge: no new constant); a particle
  within it is at the surface it is to form and is treated as every layer particle. Precedent: Adams et al.
  2007 project the particles nearer to the surface than their support radius; none of the nine papers
  withholds the smoothing from near-surface material (D45). Tests: a group four spacings above a slab has
  self rows, and ordinary rows when marked arrived; suite 271 passed, exit 0.
  Predictions at window 39/40: the horns solid as the old layer's (D43) in both 64-px runs, and at 96 px as
  `old96`'s; rendered particles beyond 4.4 spacings at most 16 and no dense set beyond a loss cell in all
  three (D26's level; the old layer had 40–53 and 4–14 dense in two of three runs); silhouette within 0.002 of
  the same-schedule runs. Fail: horns feathered as D40's, or a dense far set. If it holds: the 40k gallery, a
  second 300k run at 96 px, and the full 64 → 96 run to its own stop against the 45-minute video.
  **Result of the three cells (2026-10-03 14:16 CDT; `d46/q_berth_64.jpg`, `q_berth_96.jpg`).** At window 39
  (`berth96`: 37 commits of 40, read at its end), rendered particles beyond 3, 4.4, 6 spacings; census at the
  end; silhouette, world-thin; the thin class's fill and under-half share:
  `berth64`: 302, 22, 5; 2112 / 155 / 14, none dense; 0.9836, 1.23 % (47 commits of 48); 0.936, 14.9 %.
  `berth64b`: 308, 43, 13; 1914 / 204 / 17, none dense; 0.9824, 1.33 % (46 of 48); 0.938, 14.8 %.
  `berth96`: 303, 72, 44; 2432 / 119 / 10, none dense; 0.9841, 0.73 %; (window 20: 0.906, 20.3 %).
  The horns, committed disc rule: in both 64-px runs the large horns are solid and rounded as the old layer's,
  with a feathered small horn in one and a rough patch on a large horn in the other (the old layer's second
  sample shows the same); at 96 px fuller than D26's, with a rough patch on the first horn. D26's feathering of
  every horn is gone in all three.
  Predictions: the horns as the old layer's (met, within the two samples' spread); no dense set beyond a loss
  cell (met in all three; the old layer had 4 and 14 dense particles there in two of three runs); "at most 16
  rendered particles beyond 4.4 spacings" missed (22, 43, 72): these are not detached sets (the census has 10–17
  detached there) but material linked to the body, as in D47's run (54, 31 beyond 6); what it is, is being
  located (`tmp/far_where.py`). Silhouette within 0.002 of the same-schedule runs (0.9822–0.9841 at 64 px,
  0.9840–0.9846 at 96 px). Render: λ of the first window 0.391 (64 px) and 0.462 (96 px), median g_share
  0.93–0.94.
  **Next, launched 14:16 CDT:** the full 64 → 96 run to its own stop (`berth_full`, repo_r55, GPU 1); a second
  96-px run (`berth96b`, GPU 2); the 40k gallery of repo_r54 (`tmp/d48.sh`, `tmp/d48.queue`, three workers on
  GPU 3; `output/gpu/d48`) against D39's (the same code without the change) and the three earlier arms.
  Criteria: the gallery's silhouette within 0.002 of the earlier runs' mean on at least 15 of 19, none lower by
  more than 0.003 except C at its early stop and beast and V at theirs (D39: 35–45 % and 1 in 3 of either
  code); the detached census beyond a loss cell with no dense set on any mesh; the full run's horns at its end
  against the 45-minute video's, both disc rules; `berth96b` inside the three 96-px runs' spread.
  **`berth96b` (14:38 CDT; `d46/q_berth_96b.jpg`).** 683 s, 40 commits of 40, silhouette 0.9835, world-thin
  0.49 %, thin 17.8 %; beyond 3, 4.4, 6 spacings at window 39: 307, 19, 6; census 1831 / 106 / 11, none dense;
  the thin class 0.956, 14.5 % under half density. Its horns are solid and rounded with a clean outline, the
  best of the 96-px runs. Inside the spread.
  **The 40k gallery (14:49 CDT; `tmp/d48_eval.py`, `output/gpu/d48/eval.txt`).** Against the three earlier arms
  (D27's old arm, D31, D39): silhouette within 0.002 of their mean on 18 of 19 (median −0.0004; against D39
  alone −0.0005, within 0.002 on 17); beast frozen (`domain`, its 35–45 %); none else lower by more than 0.003
  (teapot −0.0012 against the mean and −0.0030 against D39's own 0.9782, which was that mesh's high draw; bob
  −0.0015). bob stops at window 13 on three consecutive rejections with 10 commits (29–37 before) and a last
  kinetic record of 2.8e-2: the rejection rule's stop, as C's and V's; three repeats an arm are running
  (`tmp/d48bob.queue`, launched 14:49) with V's reading (D39): if a run without the change also stops before
  window 20, or none of three with it does, it is the rule's draw. Thin: median +0.8 points. Committed
  windows, median: 25 → 31. Detached census at the end, D39 → D48: beyond a loss cell no dense set on any mesh
  (0 → 0); beyond the berth 81 → 121 without beast; within the berth 1990 → 3067 without beast: more loose
  material is left at the surface by the census's count, where at 300k the horns are more solid: the count is
  of sets not linked within a spacing, which u draws together without linking them to the body. Render: λ of
  the first window median 0.263, g_share median 0.86. The gallery's criteria are met.
  bob's repeats (14:57 CDT): with the change 38, 28 and 32 commits, without it 31, 27 and 24; none stops early
  (silhouette 0.9824–0.9833 in all six): the gallery's stop at window 13 was the rejection rule's draw.
  **The full run (`berth_full`, 14:56 CDT; `d46/q_full_end_old_discs.jpg`, `q_full_end_new_discs.jpg`,
  `q_full_whole.jpg`).** 64 px to window 97 (the event, after three rejections at 96), then 96 px to its own stop
  at window 126 (three rejections; the best commit 121): 2003 s of simulation (2105 s of process), 117 commits
  of 127 attempts; silhouette IoU 0.9842, world-thin 0.73 %, thin 19.5 %, chamfer 0.0591, holes 0.02 %, last
  kinetic record 1.5e-5. Render: λ of the first window 0.391, median g_share 0.92. At the end, rendered
  particles beyond 3, 4.4 and 6 spacings: 98, 1, 0; census 1739 within the berth, 120 in the near band, 1
  beyond a loss cell, none dense. The old 45-minute run: 2704 s, 122 commits of 129, 0.9838, 0.55 %, 0.0589,
  1.5e-4. D40 (D26's layer on the same schedule): 2790 s, 161 of 169, 0.9846, 0.33 %, 0.0589, 1.0e-3.
  The horns at the end: drawn with the old disc rule they are the old video's (solid rounded tubes, the small
  horns solid; the second horn's top a little ragged); with the committed rule they are solid with a clean
  outline, where D40's end has the wispy small horn and the fringe. The whole body is the old video's, with one
  small bead at the tail fin's edge. So the 45-minute video's result is back, on the current code, in 33
  minutes, without the far sets (1 particle beyond a loss cell). World-thin is 0.73 % against 0.33 % with D26's
  layer on the same schedule: the thin share again reads the feathered cover as the better one.
  **Verdict.** The arrived-scope layer meets what was pre-registered (horns as the old layer's in five 300k
  runs, no dense far set in any 300k run or on any gallery mesh, the gallery's silhouette criteria, the full
  run against the 45-minute video) except the far count's line, which was the notch remnant's (D51: material
  being walked out at u's pace, with either layer). Adopted; committed after D52's validation of the code as a
  whole.
- **D47, runtime: the gradient's point is not solved a second time (pre-registered 2026-10-03 13:44 CDT, before
  its runs; repo_r53 = repo_r47 with `losses/grid_ot.py`, `window/solve.py` and a test;
  `output/gpu/d47`).** D44's first item. Every gradient is taken at a point that has just been evaluated without
  the tape (the replay pair at the window's start, then each accepted candidate); the two rollouts of one
  control differ by the transfers' atomics only (5e-6 spacings). `GridSinkhornLoss` keeps the potentials of its
  latest solved call, and a caller that sets `repeat` gets them back for one call, with no sweep taken from
  them (not a warm start: a call without the flag starts from zero duals as before). The optimiser sets it
  before each taped evaluation. Test: the repeated call solves nothing and returns the solved call's value and
  gradient bit for bit at the same measure and within 1e-3 under a 1e-6 perturbation; the flag lasts one call;
  suite 272 passed, exit 0. Runs when a GPU is free: the 300k dragon with `--profile`, 14 attempts, against D44;
  then 40 attempts alone on a GPU against D39's two runs. Predictions: Sinkhorn evaluations a window 22 → 14,
  the window 14.1 → about 12.4 s, the 40-window run 11.1 → about 9.9 minutes of simulation; silhouette and
  world-thin inside D39's two runs give or take the run-to-run 0.002 and 0.2 point; the merit of windows 0–2
  equal to D44's to four digits (the runs are the same until the atomics' noise has grown). Fail: a quality
  number outside, or less than 1 s a window gained.
  **Result (2026-10-03 14:10 CDT).** With `--profile`, windows 4–13: the window 14.08 → 12.10 s; Sinkhorn
  evaluations 22 → 14 a window (blocks 1379 → 871, 4.64 → 2.95 s), the gradients 8.33 → 6.35 s, everything else
  unchanged. The 40-attempt run: 588 s of simulation (636 s of process) against 664 and 669 s; 39 commits of 40;
  silhouette IoU 0.9838 (0.9844, 0.9846), world-thin 0.24 % (0.21, 0.16), thin 17.8 %, chamfer 0.0593, holes
  0.01 %, last kinetic record 3.3e-3 (0.9e-3, 1.1e-3); render: λ of the first window 0.463, median g_share 0.93.
  The merit of windows 0–3: 0.68633, 0.56888, 0.41853, 0.27390 (D44: 0.68633, 0.56887, 0.41853, 0.27388).
  Predictions met (2.0 s a window gained; silhouette and world-thin inside the allowance). Not in the
  criteria and outside D39's two runs: rendered particles beyond 3, 4.4 and 6 spacings at the end 235, 54, 31
  against 156–192, 7–16, 0–2 (census: 7 detached beyond a loss cell, none dense, so the 31 are linked to the
  body). The reuse has no path to it (the potentials are those of the same point), and before the zero-row fix
  one run had 63 and 35 there; a second sample with the far particles located is running (`tmp/d47b.sh`,
  `tmp/far_where.py`) before this is called a draw.
  **Second sample (14:21 CDT).** 591 s of simulation (638 s of process), 40 commits of 40, silhouette 0.9842,
  world-thin 0.32 %, thin 18.5 %, last kinetic record 3.3e-3. Beyond 4.4 and 6 spacings at the end: 69 and 65,
  of them one detached set of 62 particles, dense (8th neighbour at 0.32 coverage radii), 7.8 spacings from the
  target at (1.2, −1.91, −0.42): the notch under the tail, the recurring trap of D32. Whether a run ends with
  material in that trap, by the far part of its merit at window 39: D30's two runs (before the zero-row fix)
  4e-5 and 8e-5, D39's two 1e-5 and 0, D47's two 5e-5 and 9e-5, `relax96` 0, `berth96` 7e-5. So it is there
  in five of eight 40-window runs, with and without the fix, with and without this change, with either layer;
  D39's two runs were the two without it, and my "7–16 against 63" for the zero-row fix (D39's verdict) read a
  draw as the fix's effect: the fix's world-thin (0.16–0.32 % in four runs against 0.32–0.42 %) stands, its far
  count does not. The trap is empty at the end of the long run (D40: 1 particle beyond 4.4 spacings at window
  159). It is a set that D26 leaves alone and nothing else sees: denser than the spray gate, beyond the near
  band, 2e-4 of the mass for the transport.
  The reuse is adopted: the same potentials at the same point, 2.0 s a window.
- **D46, the layer's treatment of detached sets, four cells on the 300k dragon (pre-registered 2026-10-03 13:38
  CDT at launch; `tmp/d46.sh`, one cell a GPU; `output/gpu/d46`).** D43: with the layer as before D26 the 64-px
  stage's horns are solid again and no far dense set came back. D26 has two parts: a detached set is not relaxed,
  and u moves its members along the direction away from the non-members. D20's clump was u contracting a set
  along its own radial normals; the relaxation only kept the set regular. The literature (below) relaxes with
  whatever lies inside the kernel's support and nowhere withholds the smoothing from material that is not linked
  to the largest body. Cells: `old96` (repo_r50: no set detached, 96 px, 40 attempts): do the far dense sets of
  D20 return now that the zero rows are gone, and are the horns less feathered than D39's; `relax96` and
  `relax64` (repo_r51 and r52: D26's normal kept for a detached set, its relaxation given back; 96 px 40
  attempts, 64 px 48 attempts): does the normal alone keep the far sets from clumping while the horns come
  back; `old64b` (repo_r49 again, 48 attempts): D43's second sample. Read at window 39/40: the horn crop with
  both disc rules, far counts (beyond 4.4 and 6 spacings), dense sets beyond a loss cell, the thin class's
  fill, silhouette and world-thin. Expectations: `old64b` solid as D43; `relax64` solid as D43 with no dense far
  set; `relax96` less feathered than D39's runs, no dense far set; `old96`: uncertain whether the clump of D20
  returns (it was one set of 38 at 6.6 spacings). Adoption is not decided by these four: the candidate that
  holds goes to the 40k gallery and a second 300k run.
  **Result (2026-10-03 13:56 CDT; `d46/q_64_new_discs.jpg`, `q_96_new_discs.jpg`).** At window 39, rendered
  particles beyond 3, 4.4 and 6 target spacings; detached census at the end (within the berth / near band /
  beyond a loss cell, of them dense); silhouette, world-thin; the thin class's fill and under-half share:
  `old96`: 348, 40, 6; 2584 / 124 / 25, 4 dense; 0.9843, 0.54 %; 0.947, 15.5 %.
  `relax96`: 180, 0, 0; 2814 / 89 / 0; 0.9840, 0.61 %; 0.971, 15.3 %.
  D26 at 96 px (D39's two runs): 156–192, 7–16, 0–2; 2515–2836 / 116–146 / 4–6, none dense; 0.9844–0.9846,
  0.16–0.21 %; 0.966, 14.0 %.
  `old64b`: 380, 53, 6; 2072 / 194 / 31, 14 dense; 0.9841, 1.6 % (45 commits of 49); 0.932, 16.0 %.
  D43 (the same code, first sample): 253, 6, 2; 2142 / 184 / 2, none dense; 0.9822, 0.88 %; 0.947, 14.0 %.
  `relax64`: 189, 8, 0; 3280 / 100 / 0; 0.9833, 0.90 %; 0.965, 15.2 %.
  D26 at 64 px (D40 at window 39): 338, 77, 25.
  The horns at window 39, drawn with the committed disc rule: the old layer solid in D43, mostly solid with one
  feathered small horn in `old64b`, fairly solid at 96 px (`old96`); with the relaxation given back and D26's
  normal kept (`relax64`, `relax96`) feathered small horns and blotched large ones, no better than D26; D26
  feathered. Render: λ of the first window 0.468 (96 px) and 0.397 (64 px), median g_share 0.92–0.95.
  Expectations: `old64b` is solid less cleanly than D43 and has the far dense sets back (14 dense particles
  beyond a loss cell): D43's "no far set" was one draw, D26's reason stands. `relax64` and `relax96` keep the
  far sets away as expected (the best far counts of all: none beyond 4.4 spacings at 96 px) and do not bring
  the horns back: refuted. So it is not the relaxation that makes a thin feature solid but u acting along the
  set's own normals, which draws the set together: at the target that is the feature filling in, in the air it
  is the clump of D20. World-thin is 0.5–0.6 % in both cells without D26's exclusion against 0.2 % with it.
- **D45, what tells a set at the surface from a set in the air (read-only on kept states; pre-registered
  2026-10-03 13:26 CDT, before the probe's first run; `scripts/probes/settled/set_reach_probe.py`;
  `output/gpu/d45`).** The user, 13:15: the fringe is to be fixed. D42 left the question which sets are the
  body's. D20's pathology was a set measured against itself (its relaxation neighbourhood is its own
  members); D26 withholds the relaxation from every set not linked within one layer spacing, which also takes
  the sets that sit one spacing from the target on a sparse thin feature. Three readings of "measured against
  what", on the rendered detached particles of five states (the old layer at 96 px, d20's windows 20 and 39,
  where the far dense sets are; d32's end; D40 at window 40 and at its end), by distance to the target: the
  set's gap to the body in layer spacings (the relaxation's weight is a Gaussian of two layer spacings); the
  share of a member's ordinary relaxation row that lies on its own set; dense or not. A reading is usable as
  the definition if it puts the sets within the berth on the body's side and the far dense sets (the floaters
  of D20) on the other, with few in between. Expectation: the gap does (within the berth nearly all sets are
  within two layer spacings of the body, the far dense ones beyond); the own weight is the mechanism itself and
  should separate at least as well; the literature check (running) says which of the two established methods
  use.
  **Result (2026-10-03 13:30 CDT; `output/gpu/d45/set_reach_probe.txt`; D43's window 40 added at 13:37).** The
  layer spacing is 1.9 target spacings. Rendered detached particles within the berth, to one loss cell, beyond:
  d20 window 20 (old layer, 96 px) 7124, 590, 154; d20 window 39: 5546, 250, 39; d32's end (D26, 96 px) 4785,
  222, 10; D40 window 40 (D26, 64 px) 6087, 266, 47; D40's end 2951, 85, 0; D43 window 40 (old layer, 64 px)
  5337, 234, 2. The old layer does not have fewer loose sets at the surface than D26 (5546 against 4785 at 96
  px, 5337 against 6087 at 64 px): what differs between solid and feathered horns is not their number.
  The gap to the body: within the berth 91–97 % of them are in sets within 1.5 layer spacings of the body and
  93–99 % within 2; the rest are islands at the target, up to more than 4 layer spacings from the body (369
  particles at d20's window 39, 183 of them dense, 98 at d32's end): pieces of thin features assembled apart
  from the body. D20's floater is there at d20's window 39: 38 of the 39 particles beyond a loss cell are one
  dense set of 24 or more, 3–4 layer spacings from the body, with its members' ordinary rows wholly on
  themselves (own weight 1.00). The own weight elsewhere: median 0.38–0.41 within the berth with 28–37 % of the
  particles above one half; beyond a loss cell 0.43–1.00.
  Expectations: the gap separates the floater from the near sets (confirmed) but would leave the islands at the
  target on the wrong side; the own weight does not separate (a third of the near particles are above one
  half): refuted as a criterion. The distance to the target is the only one of the three that puts the
  islands with the surface.
  **The literature (agent, nine papers opened in full, 13:36 CDT).** Smoothing or projection takes whatever lies
  inside the kernel's support: Zhu and Bridson 2005 (R twice the particle spacing; "exactly reconstruct the
  signed distance field of an isolated particle"), Adams et al. 2007 (projection for particles nearer to the
  surface than their support radius), Alexa et al. 2003 (a point at most h/2 from its projection; beyond a
  neglect distance nothing contributes), Akinci et al. 2013 (cohesion to the support h), Yu and Turk 2010/2013
  (position smoothing with a kernel of about four spacings; a particle with 25 neighbours or fewer in it only
  gets a spherical kernel). Explicit connectivity appears once, Yu and Turk 2013 §4.3: two particles are
  connected within the average spacing and a component is smoothed against its own members only, "the
  relocation step pulls particles together even when they are further apart than r_a"; every component is
  still smoothed. Covariance eigenvalues detect thin sheets (Ando et al. 2012) or bound the projection's domain
  (Amenta and Kil 2004), never label material as detached. None of the nine withholds the smoothing from
  near-surface material because it is not linked to the largest body: that part of D26 has no precedent; its
  other part (what a detached set is measured against) is Yu and Turk's concern.
- **D44, where a 300k window's time goes now (pre-registered 2026-10-03 13:19 CDT at launch; `tmp/d44.sh`, the
  committed code with `--profile`, 14 attempts, alone on GPU 0; `output/gpu/d44`).** The user, 13:15: the long
  run's result is the one wanted, so the run may stay long and each operation is to be made faster. By the
  records' own clocks a 300k window is 14.3–15.1 s in every run of today: the window's start 1.6–1.8 s, eight
  gradients 8.4–8.6 s (1.05 s each), nine line-search trials 3.8–4.3 s (0.43 s each), the commit 0.55 s. The
  45-minute look took 120–160 windows; 160 windows in 15 minutes is 5.6 s a window, 2.6 times faster than now.
  This run splits the gradient and the trial into their parts (rollout, tape, adjoint, transport solve, render,
  the other terms). No expectation is set on the split; what follows is one change at a time on the largest
  part, each checked to return the same values.
  **Result (2026-10-03 13:29 CDT; windows 4–13, `tmp/prof_mean.py`).** A window is 14.1 s: start 1.5, gradients
  8.3, line search 3.5, commit 0.8. By part, seconds a window (calls, milliseconds a call): the Sinkhorn solves
  4.64 (22 evaluations, 210; 1379 blocks in 44 solves, 31 blocks a solve); the three adjoint sweeps 5.30 (8 of
  each; the transport and stability terms 219, the cleanup 217, the render 226); the rollout 1.88 (14 without
  the tape at 79, 8 with it at 95); the surface proximity 1.05 (22 at 48); the render 0.29 (22 at 13); the
  cleanup's value 0.02. The parts sum to 13.2 s.
  What each could give, in the order to be tried, each to return the same values: (1) every iteration evaluates
  its point twice, once as the accepted line-search trial and once more with the tape for the gradient: the
  second Sinkhorn solve is at the same point as the first and can take its duals (8 solves of 22: about 1.7 s);
  (2) the window's start runs four evaluations (zero control, warm start, two for the replay noise), of which
  the replay pair can reuse one (0.3 s); (3) the proximity term's neighbour search, 48 ms a call; (4) the
  adjoint: the three sweeps are three vector-Jacobian products through one tape, needed apart because the
  render gradient is projected against the transport gradient alone before the cleanup's is added; taking the
  cleanup's with the transport's would save a sweep in eight (1.7 s) but changes what the render gradient is
  projected against: not the same values, a change to be validated on the gallery if wanted. Without (4) the
  window comes to about 11.5 s, with it under 10 s: 1.2 to 1.5 times faster, not the 2.6 that 160 windows in 15
  minutes ask for. The rest of the factor has to come from the number of windows (the coarse stage runs to
  window 105–125 before its event) or from the solve's ladder (31 blocks a solve).
- **D43, which change took the solid horns of the 64-px stage: the layer as it was before D26 on the code as it
  is (pre-registered 2026-10-03 13:19 CDT at launch; repo_r49 = repo_r47 with `layer_relax_data`'s
  `group_query` 1, so no set is detached, and `render_res` 64; `tmp/d43.sh`; 48 attempts, alone on GPU 1;
  `output/gpu/d43`).** Of the four cells of D40 only the old code at 64 px has solid horns at window 40, and D42
  shows the fringe to be the sets D26 leaves unrelaxed. The old code differs from this one in the layer, the
  solve (D29) and the zero-row fix. One run with the layer alone put back: if its horns at window 40 are solid
  as the old run's (the same crop, drawn with both disc rules), D26's reach is what took them and the solve is
  cleared; the far material is then expected back as before D26 (dense sets beyond a loss cell). If they are
  feathered as D40's, the layer is not it and the solve is next. Expectation: solid, with the far sets back.
  **Result (2026-10-03 13:37 CDT).** 810 s of simulation (884 s of process), 48 commits of 48; silhouette IoU
  0.9822, world-thin 0.88 %, thin 21.4 %, chamfer 0.0593, holes 0.07 %, last kinetic record 4.0e-5; render: λ of
  the first window 0.397, median g_share 0.92 (all at 64 px: the coarse stage's numbers). At window 40 the horns
  are solid with both disc rules (`d43/q_w40_old_discs.jpg`, `q_w40_new_discs.jpg`): rounded as the old run's
  with the old rule, and with the committed rule solid where D40's and the 96-px run's are feathered; the second
  horn's top is a little rough. Rendered particles beyond 3, 4.4 and 6 target spacings at window 40: 253, 6, 2
  (D40, D26 at 64 px: 338, 77, 25); at the end 236, 4, 1. Census at the end: 2328 rendered detached, 2142 within
  the berth, 184 near band, 2 beyond a loss cell, none dense. The thin class's fill at window 40: 0.947, median
  0.87, 14.0 % under half density: as every other run's, so D41's measure does not see what the eye sees here
  either (the material is there in both; it is regular in one and loose in the other).
  The layer is what took the solid horns; the solve is cleared. The second expectation is refuted: no dense set
  beyond a loss cell came back, and the far counts are lower than with D26 on the same schedule. D20's clump
  was measured with the zero rows still in the code and at 96 px; whether it returns there is D46's first cell.
  The archive is deleted; four frames kept (`d43/dragon_keyframes.npz`).
- **D42, is the feathered fringe on thin features the detached sets (display-only diagnostic on kept frames, no
  simulation; pre-registered 2026-10-03 13:11 CDT, before the probe's first run;
  `scripts/probes/settled/fringe_probe.py`; `output/gpu/d42`).** D40 left the fringe on the horns as loose
  material within the berth, by a count (2515–3251 rendered detached particles at 40 windows, 1812 at 160) and
  not by which particles draw it. Three frames: d32's end (96 px, 39 windows), D40 at window 40 and at its end
  (64 → 96 px, 160 windows). For each: the particles not connected to the body within one layer spacing, by
  distance to the target and by the thickness of the nearest target point; and the frame drawn twice with the
  committed renderer, as it is and with every detached particle put deep inside the body, on the horn crop and
  the whole body. Expectation: the detached sets sit at the thin class far above its share (more than half of
  them on the class below 2 cells, which holds about a fifth of the surface), and without them the horns' wisps
  are gone while the outline stays where it is: then the fringe is these 2–3 thousand particles and the open
  question is what brings a set within the berth onto the surface. If the horns are as feathered without them,
  the fringe is the body's own outer layer and the detached count was the wrong reading.
  **Result (2026-10-03 13:14 CDT; `output/gpu/d42/fringe_probe.txt`, `q_fringe_horns_40.jpg`,
  `q_fringe_horns_end.jpg`, `q_fringe_whole.jpg`).** Particles not connected to the body within one layer
  spacing (singles included, which the census of D20–D40 leaves out): d32 at window 39, 5573 in 1891 sets, 5017
  rendered, of them 4785 within the berth, 222 to one loss cell, 10 beyond; D40 at window 40, 7042 in 2183 sets,
  6400 rendered, 6087, 266, 47; D40 at window 159, 3527 in 1352 sets, 3036 rendered, 2951, 85, 0. Their distance
  to the target: median 1.0 spacing, 90 % within 1.7–1.8, in all three frames. By the thickness of the nearest
  target point (below 2 cells, 2 to 4, 4 and more): 44, 28, 27 % (d32), 42, 31, 26 %, 47, 32, 21 %, where the
  classes hold 48, 30, 22 % of the target's surface points: they are spread as the surface is, not gathered on
  the thin class (my "a fifth of the surface" was wrong: the thin class is half of the surface); per 1000
  particles of the class 65, 24, 7 (d32).
  Drawn without them: at 40 windows the wisps around the head, the whiskers and the foot are gone and the
  outline is clean; and the tips of the small horns are gone with them, in D40's frame most of one large horn:
  those parts of the thin features are themselves detached sets. At 160 windows the horns stay (they are
  connected by then) and only the wisps go.
  Expectation: the wisps are the detached sets (confirmed); "the outline stays where it is" is refuted on the
  thin features at 40 windows: material that has reached a thin feature sits there as many small sets, one
  spacing from the target, not linked to the body or to each other within a layer spacing, because the thin
  class is under-filled (D41). Since D26 a set of 2–512 is not relaxed, so these stay as they arrived; the
  code before D26 relaxed them as a surface of their own, which is what clumped the far sets (D20) and
  smoothed the near ones.
  Reading: one definition serves two kinds of set. A set in the air has no surface of its own (D26 holds for
  it). A set within the berth of the target is at its place on the surface and is counted as detached only
  because the material around it is sparse. Which criterion tells the two apart (the distance to the target
  that the objective already uses, the berth; or what the set's neighbourhood is made of) is a design question
  with two failed neighbours (D24: relaxed against the surroundings, a pile-up on a horn; before D26: relaxed
  against itself, far clumps): not decided here.
- **D41, how much material the thin part of the target holds (read-only on d32's archive, then on D40's;
  pre-registered 2026-10-03 11:25 CDT, before the probe's first run; `scripts/probes/settled/thin_fill_probe.py`;
  `output/gpu/d32/thin_fill_probe.txt`).** D38: the thin share passed runs whose horns the eye rejects. It counts
  a thin target point as covered when one particle is within 1.5 target spacings, so a horn drawn by a few
  beads counts as much as a solid one. The body has as many particles as the target sample has points, each the
  same mass: a part of the target that holds M sample points is full when M particles sit in it. Per frame, by
  the target's local feature thickness (below 2 MPM cells, 2 to 4, 4 and more): the fill (particles whose
  nearest target point is of the class and within 2 target spacings, over the class's points), and at each of
  the class's points the local fill (particles within 2.5 target spacings over the other target points within
  2.5; 1 for an independent sample of the target): its median, the share of points below 0.5 and at 0.
  Expectation, uncertain: at d32's end (96 px from the start, 40 windows) the class below 2 cells is under-filled
  where the thick one is full (fill 0.7–0.9 against about 1.0), with 10–30 % of its points below half of the
  target's local count, while fewer than 1 % have no particle within 1.5 spacings: the cover is there, the
  material is not. If the thin class is as full as the thick one, the feathered look is the arrangement of
  material that has arrived, not missing material, and this measure does not see it either. Then the same on
  D40's archive, by window: if its horns are solid, the thin class's fill is to be higher there at the same
  window and at the end.
  **Result on d32 (2026-10-03 11:26 CDT).** The MPM cell is 8.8 target spacings. Target points below 2 cells:
  34 611 (11.5 %), 2 to 4 cells: 59 746, 4 cells and more: 205 643; their other target points within 2.5
  spacings: 14, 16, 17. Thin, middle, thick class at windows 5, 10, 20, 30 and the end (39): fill 0.60, 0.53,
  0.72; 0.85, 0.94, 0.98; 0.92, 0.97, 1.01; 0.95, 0.98, 1.01; 0.96, 0.97, 1.01. Local fill, median: 0.33, 0.43,
  0.75; 0.72, 0.89, 1.04; 0.82, 0.94, 1.05; 0.88, 0.94, 1.06; 0.88, 0.95, 1.06. Points with a local fill below
  0.5: 62, 54, 20 %; 31, 16, 5.6 %; 19, 11, 4.1 %; 16, 8.9, 3.6 %; 14.0, 8.2, 3.3 %. With no particle within 2.5
  spacings: 19, 15, 1.1 %; 2.7, 1.4, 0.08 %; 0.9, 0.16, 0.03 %; 0.5, 0.06, 0 %; 0.44, 0.05, 0 %. With none within
  1.5 spacings at the end: 12.2, 5.9, 2.5 %.
  Expectations: the thin class is fuller than expected by count (0.96, not 0.7–0.9: about 1400 particles short of
  34 611), and unevenly so: one thin point in seven has less than half of the target's local count around it
  (one in thirty in the thick class), and the class's median is 0.88 where the thick one's is 1.06. My "fewer
  than 1 % with no particle within 1.5 spacings" was the world threshold's number (2.9 spacings at 300k), not
  this one: wrong by my own confusion, 12 %. The thin class fills last (0.60 → 0.85 → 0.92 → 0.96) and after
  window 20 gains about 0.01 in five windows. So the measure sees the feathered cover (14 % of thin points at
  under half density) where world-thin read 0.3 %. Whether it separates solid horns from feathered ones is
  D40's reading.
  **On D39's second dragon and on D40 (12:19 CDT).** The fix's second 300k run at the end (window 40): fill
  0.966, 0.977, 1.004; local fill median 0.89, 0.95, 1.06; under half density 14.0, 7.9, 3.6 %: as d32. D40 (64
  px to window 125, then 96 px, 160 windows), the thin class at windows 5, 10, 20, 40, 60, 80, 100, 125, 140,
  159: fill 0.36, 0.62, 0.92, 0.954, 0.965, 0.972, 0.971, 0.974, 0.979, 0.985; local fill median 0.00, 0.36,
  0.80, 0.87, 0.89, 0.90, 0.91, 0.91, 0.92, 0.93; under half density 71, 59, 23.5, 14.9, 13.0, 12.4, 12.0, 11.4,
  10.8, 10.2 %; the thick class at the end 1.002, 1.06, 2.2 %. At window 40 the two schedules are level (14.9
  against 14.0 %; the coarse stage fills the thin class later in the first ten windows and has caught up by
  20); the under-filled share then falls by about 0.04 points a window, with no step at the event. The measure
  follows the picture in its order (40 windows 14–15 %, 160 windows 10 %) but slowly: it would take a run of
  several hundred windows to bring the thin class to the thick one's 2–3 %. No archive of the old code is left
  to read its horns with this measure.
- **D40, the 45-minute run's schedule with the code as it is now (launched 2026-10-03 11:20 CDT; this entry
  written at 11:23, before any result of the run; repo_r48 = repo_r47 with `render_res` 64; `tmp/d40.sh`;
  `output/gpu/d40`).** D38 left two candidates for the old run's solid horns: its schedule (the render at 64 px
  until the run would stop, then 96 px to its own stop) or the layer as it was. One run: the 300k dragon with
  the code as it is now (the layer as the body's, the new solve) and `render_res` 64, so the coarse stage and
  its event are back as in D17's run; no window budget; alone on GPU 1; then the census, the far count and the
  4K render. One thing rides along: repo_r48 also has D39's zero-row fix (one to nine particles a window); D39's
  own dragon run (96 px, 40 attempts) shows what the fix alone does to the horns.
  Read at window 40 and at the end, on the same horn crop as D38. If the schedule is the cause: the horns are
  solid blobs at window 40 and solid tubes at the end, as in the old run; world-thin stays at 1.5–2 % through
  the coarse stage; the run takes 100 windows or more and 35–45 minutes. If the horns are feathered as in D38:
  the layer change (or the solve) is the cause, and the old layer is then rerun on the new solve. Expectation,
  uncertain: the schedule, because of d20 against the old run at window 40. What this decides: whether D19's
  default (96 px from the start) buys its 11 minutes with the horns' solidity, which the thin share did not see.
  **Result (2026-10-03 12:19 CDT).** The run: 2790 s of simulation (2942 s of process), the event at window 125
  (the old run's at 105), stop at window 168 on three consecutive rejections, 161 commits of 169 attempts, the
  best commit window 163; silhouette IoU 0.9846, world-thin 0.33 %, thin 18.6 %, chamfer 0.0589, holes 0.02 %,
  last kinetic record 1.0e-3. Render: λ of the first window 0.392 (64 px; 0.277 after the event), median
  g_share 0.92. Rendered particles farther than 3, 4.4 and 6 target spacings from the target: 338, 77, 25 at
  window 39 and 40, 1, 0 at the end; census at the end 1866 rendered detached, 1812 within the berth, 54 in the
  near band, none beyond a loss cell. The 23 GB archive is deleted; five frames kept
  (`d40/dragon_keyframes.npz`: raw 0, 1600, 3200, 5000, 6360).
  The horns (the same crop; `d40/q_horns_w40.jpg`, `q_horns_end.jpg`, `q_discs_w40.jpg`, `q_discs_end.jpg`,
  `q_2x2_w40_old_discs.jpg`). At window 40 this run's horns are feathered, as feathered as the 96-px run's at
  the same window: the schedule alone does not make them solid; the prediction is refuted. At its end (160
  windows, 46 minutes) they are fuller than any shorter run's, with a wispy fringe left on the small left horn
  and on the right horn's edge; still not the old run's rounded tubes.
  Three things are in the old video's look, separated as far as single runs allow:
  the display: the old video is drawn with the old disc rule. D40's own frames drawn with it (the renderer of
  repo_r43 on the kept frames) are rounder and softer than with D35's rule, at window 40 and at the end: the
  old rule's inflated discs fill the fringe in (and blur the rest, D34);
  the length: from window 40 to 160 the far material goes 338 → 40 (beyond 3 spacings), the detached material
  within the berth to 1812 (2515–3251 at 40 windows in the five 96-px runs), the thin class's under-filled
  points 14.9 → 10.2 % (D41);
  the code: with the same schedule and the same (old) disc rule, at window 40, the old code's horns are solid
  and this code's are feathered; and at 96 px the old code's are feathered too (d20). All four cells of old or
  new code by 64 or 96 px, drawn alike: only old code with 64 px is solid. One run a cell, so the cell is one
  observation. What is counted: the detached material left within the berth at 40 windows is 2122 with the old
  layer (d20) and 2515–3251 with the layer as the body's (five runs); a detached set is no longer made regular
  or drawn together, which removed the far clumps (D26) and leaves more loose material at the surface (D32).
  Reading: the 45-minute video looks best because it is the old disc rule on a long run of the code before the
  layer change. The schedule is not what the short run lacks. What the short run lacks is in D32's parked
  items (what moves near-surface detached material in: off-layer particles have no position channel, u is
  bounded by the accepted step) and in the run's length.
- **D39, a layer particle without neighbours was relaxed toward the world's origin (bug; found on the user's
  question of 2026-10-03 "a zero-neighbour smoothing bug?"; measured 11:13 CDT on d32's term dump, fix
  pre-registered 11:17 CDT before its runs; `scripts/probes/settled/zero_row_probe.py`; repo_r47 = repo_r44 +
  `window/layer.py` and a test; `output/gpu/d32/zero_row_probe.txt`, `output/gpu/d39`).** `layer_relax_data`
  weights a layer particle's 24 nearest layer particles by a Gaussian of the distance times the normal agreement
  clamped at zero, and divides the row by its sum clamped at 1e-12. A row whose weights are all zero (every
  neighbour on the other side, or all beyond about ten layer spacings) stays zero; `k_layer_resid` then takes
  the row's centroid as (0, 0, 0), the residual as n · x, and `k_layer_project` moves the particle by −(1/T) of
  it every step: 87 % of the way to the plane through the world's origin in one window. In the code since the
  relaxation exists (2026-09-19); D24's probe had shown such rows and I had left them.
  Measured on d32 (the committed recipe, 39 windows): 100 such rows on 98 particles, none before window 3 and
  one to nine a window after; 97 of the 100 on particles connected to the body; their recorded relaxation
  displacement 31 target spacings a row (median 20–30 a window, up to 97) against 0.05–0.10 for the other layer
  particles, equal to the origin plane's prediction (cosine +1.000). At the end of the run the 98 particles are
  0.9 spacings from the target in the median, 60 % within one, one beyond 3: they are carried inside the body
  and back, and are not the end's far material (298 particles beyond 3 spacings).
  The fix, in the definition: a layer particle with no same-side neighbour within the weight's reach has no
  plane to be relaxed onto; its row is itself, as a detached group member's is (D26). Every layer row then sums
  to one (also when there are fewer layer particles than neighbours, where every row used to be zero). A test
  (`test_a_layer_particle_without_neighbours_is_not_relaxed`): fails on repo_r44 (exit 1), passes with the fix;
  the suite 270 passed, exit 0; on d32's states the fixed layer data has no zero row.
  Runs: the 300k dragon, 40 attempts, alone on GPU 0, with the term dump; the 40k gallery (19 meshes) on GPU 3,
  against D31 (the same code without the fix) and the two older arms. Predictions: no zero row in the dragon's
  dump; the dragon inside the three D30 runs' range (silhouette 0.9833–0.9846, world-thin 0.33–0.42 %) give or
  take the run-to-run 0.002 and 0.2 point, 11–12 minutes; the gallery within 0.002 of the earlier arms' mean on
  at least 15 of 19, none lower by more than 0.003 except C at its early stop. One to nine particles of 300 000
  a window: no visible change is expected; the fix is for correctness. Fail: any quality criterion missed.
  **Result, the 300k dragon (2026-10-03 11:31 CDT).** No zero row in any of the 40 windows (11 392 layer
  particles in window 0). 669 s of simulation (741 s of process with the term dump), 40 commits of 40; silhouette
  IoU 0.9844, world-thin 0.21 %, thin 16.8 %, chamfer 0.0592, holes 0.03 %, last kinetic record 9.3e-4; render: λ
  of the first window 0.463, median g_share 0.94. The five runs without the fix: 0.9833–0.9846, 0.32–0.42 %,
  18.4–19.2 %, 0.0593–0.0595. Rendered particles farther than 3, 4.4 and 6 target spacings from the target at
  the end: 192, 16, 2 (without the fix at window 39: 281, 63, 35 in d38; 194, 36, 3 only at d38's own stop after
  55 windows). Census at the end: 2667 rendered particles in detached sets, 2515 within the berth, 146 in the
  near band, 6 beyond a loss cell, none dense. Silhouette and time inside the predicted range; world-thin, thin
  and the far counts below every earlier run's, which the prediction "no visible change" did not expect: one
  run, so a second one (`tmp/d39d.sh`, no term dump, with the 4K frames) is running on GPU 0 since 11:32.
  **The gallery so far (11:33 CDT, 9 of 19): beast fails.** Eight of the nine are within 0.002 of the three
  earlier runs' mean (C at −0.0020, the same early stop as in two of them). beast: silhouette 0.8354 against
  0.9706, 0.9695, 0.9716; 10 commits, then five windows without a commit, every candidate (the zero control
  included) refused as `domain`: a particle inside the two-cell margin of the MPM domain. Its record equals
  d31's to five digits through window 2, the first difference is in window 3 (where the zero rows begin), |v|
  max 3.4 and 3.0 in windows 8 and 9. This is the parked beast freeze (2026-09-30: the head of a flung filament
  coasts into the domain margin at window 10–12; 6 of 12 runs of the codes carrying the end drift, 0 of 6
  without), which the three earlier arms of this recipe happened not to show. By the letter D39's gallery
  criterion is missed. Open: whether the fix raises the freeze's rate (the bug carried a layer particle with no
  neighbour in reach back toward the origin's plane, which is what the head of a flung filament is), or the run
  met the known rate.
  **Pre-registered 11:34 CDT, at launch: beast's freeze rate with and without the fix.** beast, 40k, seed 97,
  six runs of repo_r47 (the fix) and six of repo_r44 (without), three workers on GPU 2 (`tmp/d39beast.queue`);
  and one run of repo_r44 with the term dump for 16 windows (`tmp/d39e.sh`, launched 11:35): which particles
  have zero rows in windows 3–12, how far they are from the body, and where the bug carries them. Readings: the
  same rate in both arms (a difference of two runs or fewer of six): the fix is neutral and beast's freeze is
  the parked defect at its rate. Four or more freezes with the fix against one or none without: the bug was
  hiding the flung filament by carrying its head back, the fix is still the correct definition (a particle is
  not to be relaxed toward the world's origin), and what it uncovers is the parked ejection, not a new defect;
  adoption is then the user's decision, with the rate stated. Expectation, uncertain: the second, because the
  three earlier arms of this recipe did not freeze; against it, the bug was already in the code on 2026-09-30,
  when 6 of 12 froze.
  **Result, the second dragon and the whole gallery (2026-10-03 11:56 CDT).** The second 300k dragon with the fix
  (no term dump): 11.1 minutes of simulation, 11.9 of process, 40 commits of 40, silhouette IoU 0.9846,
  world-thin 0.16 %, thin 17.5 %, chamfer 0.0593, holes 0.01 %; far counts at the end 156, 7, 0 (beyond 3, 4.4, 6
  spacings); census 2956 rendered detached, 2836 within the berth, 116 near band, 4 beyond a loss cell, none
  dense. Both runs with the fix are below all five without it on world-thin (0.21, 0.16 against 0.32–0.42 %),
  on thin (16.8, 17.5 against 18.4–19.2 %) and on the far counts (beyond 4.4 spacings 16, 7 against 63; beyond
  6: 2, 0 against 35): the carried particles were not themselves the end's far material (measured above), but
  a particle put 31 spacings inside the body in one window displaces what is there, and the far material falls
  when that stops. The prediction "no visible change" was too modest. In the horn crop of the 4K frames the fix
  changes nothing (`output/gpu/d39/q_horns_fix.jpg`: feathered as without it at 40 windows), and the thin class's
  fill is the same (D41's measure: 0.966, median 0.89, 14.0 % of thin points under half density; d32: 0.960,
  0.88, 14.0 %).
  The 40k gallery, 19 meshes, against the three earlier runs (D19's arm, D27's old arm, D31): silhouette within
  0.002 of their mean on 16 (criterion: 15), median +0.0001; teapot +0.0025 (25 windows against 13–17); C −0.0020
  at its 16-window stop as in two of the three; V stops after 13 windows (31–37 before) at 0.9765, inside its
  range, with 25 rendered particles beyond the berth (2 before); beast −0.135 (the freeze above). Thin: median
  −0.4 points. Committed windows, median: 32 → 25. Render: λ of the first window median 0.263, g_share median
  0.85. Census without beast: beyond the berth 78 → 81, beyond a loss cell 0 → 3 (C), none dense. The criterion
  "none lower by more than 0.003 except C" is missed on beast.
  beast's zero rows (repo_r44 with the term dump, 16 windows, `beast_base_dump.zero_row.txt`, `.where.txt`):
  none before window 9, then 2, 2, 5, 0, 1, 1, 2 a window; all 13 are within 1.3 target spacings of the target
  and 20 MPM cells or more inside the domain, each carried 2–26 spacings toward the origin's plane. The
  filament's head is ten other particles (25211, 38521, …), 11–16 spacings from the target, layer particles
  with a row of their own (relaxation 0.0): at the starts of windows 9, 10, 11, 12 they are 2.7, 1.9, 1.6, 1.7
  cells inside the two-cell margin box, then they turn back (4.7 cells at window 15). So the reading "the bug
  carried the head back" is refuted: the zero rows are not the head, and nothing differs between the two codes
  before window 9. In this run the head misses the margin by 1.6 cells (7 target spacings); whether it crosses
  is decided in windows 9–11, which is where the two codes begin to differ. Also corrected: beast's record
  differs from d31's from window 3 in the fifth digit by the evaluation's own noise (the atomics), not by the
  fix.
  **beast's freeze rate so far, and twelve more runs an arm (2026-10-03 12:10 CDT, at their launch).** Finished:
  with the fix 5 frozen of 6 (the gallery's run, fix_1, 2, 4, 5; fix_3 runs 115 windows to 0.9698); without it 1
  frozen of 5 (base_5; base_1, 2, 3 and the dump run are sound) and none in the three earlier arms. Every freeze
  is `domain` with the first refused window 10 or 11. The two codes give the same layer data, row for row, on
  the dump run's states of windows 0–8 (`tmp/layer_out.py`, `tmp/layer_cmp.py`: no row with other neighbours
  or weights; 2, 2, 5 rows from window 9), and the runs of both arms branch alike on the line search's halvings
  from window 3 (base_5, the gallery's run and fix_5 share the branch "five accepted iterations, then the
  search exhausted" in window 5, and all three freeze). So the fix cannot act before the first zero row, and
  the freeze is decided by the end of window 9: the only difference the fix makes there is two particles on
  the body's surface that are not carried inside. No path from that to the head, 100 spacings away, has been
  measured; 5 of 6 against 1 of 5 is one-sided p = 0.04 and may be the draw. Twelve more runs an arm
  (`tmp/d39beast2.queue`, six workers on GPUs 0 and 3): if the rates stay apart (with the fix above 60 %,
  without it under 30 %), the fix changes beast's freeze rate through the optimiser's coupling (one step
  length for all particles) and that is reported as such; if they meet, the gallery's beast was the parked
  defect at its own rate.
  **V's early stop, three runs an arm (pre-registered 12:46 CDT at launch; `tmp/d39V.queue`, GPU 1).** The
  gallery's V with the fix stopped at window 16 on three consecutive outer rejections (best commit 11, last
  kinetic record 7.5e-3, silhouette 0.9765 inside its range, 25 rendered particles beyond the berth against 2);
  the three earlier runs took 31–37 windows. C stops this way in three of four runs of any code. If one of the
  three runs without the fix also stops before window 20, or none of the three with it does, V's stop is the
  rejection rule's draw (the parked `reject_stop` item); if two or three with the fix stop early and none
  without, the fix is the cause and is held back.
  **Verdict (2026-10-03 13:09 CDT; `tmp/beast_tally.py`, `tmp/beast_pre.py`, `tmp/zero_row_where.py`).** beast,
  40k, seed 97, all runs: with the fix 8 frozen of 19 (10 of 22 with the three dumped runs), without it 7 of 20;
  one-sided exact p = 0.45. The first six had read 5 of 6 against 1 of 6; the rates met as the runs came in.
  Every freeze is `domain` with the first refused window 10 or 11. Three runs with the fix and the term dump,
  read with the layer code as it was: the particles nearest the margin are the same ten in every run (25211,
  38521, 30390, 18115, …), none of them would have had a zero row; the old code's zero rows on those states are
  one to three particles within 1.1 spacings of the target, 16–22 cells inside, from window 8 or 9. In two of
  the three the head is 1.5 and 2.0 cells from the margin at the start of window 9, before any zero row exists.
  Seven of the 17 freezes follow a collapse of the line search's step in windows 5–7 (the step of window 7 under
  5e-4 in 12 of the 42 runs: 7 with the fix, 6 of them frozen; 5 without, 1 frozen), which is where the two
  codes are the same code: the first six runs' imbalance was drawn there.
  So: the bug neither held the filament's head back nor did the fix release it; the gallery's beast met the
  parked freeze, which this recipe has in 35–45 % of its runs, and the three earlier galleries' sound beast
  runs were three draws from that. One difference with the fix on beast: its sound runs commit 87–126 windows
  (median 114) against 49–100 (median 71) without it, at the same silhouette (median 0.9703 and 0.9697); the
  gallery as a whole goes the other way (median 32 → 25 committed windows), one run a mesh.
  V: without the fix 31, 29 and 13 commits (the third stops at window 16 on three outer rejections, last
  kinetic record 1.1e-2), with it 14, 32 and 35: the early stop occurs in both codes (1 of 6 before and without
  the fix, 2 of 4 with it): the rejection rule's draw, the parked `reject_stop` item, as C's.
  By the letter the gallery criterion was missed on beast; by the pre-registered follow-up the miss is the
  parked defect at its own rate, in both codes. The fix is adopted: a layer particle with no same-side
  neighbour within the weight's reach keeps its place instead of being carried toward the world's origin. What
  it buys at 300k (two runs against five): world-thin 0.16–0.21 % against 0.32–0.42 %, rendered particles beyond
  one loss cell 7–16 against 63 (d38 at the same window), the same 11 minutes and silhouette. What it does not change: the feathered
  horns (D40, D41), beast's freeze, the early stops. Render: λ of the first window and g_share unchanged (0.463
  and 0.94 on the 300k dragon; gallery medians 0.263 and 0.85).
  To the record of 2026-09-30 ("three clean galleries") add: beast's freeze rate has to be read from repeats;
  one gallery run of beast says nothing about a change.
- **D38, why the 45-minute video looks best: the same code run to its own stop (pre-registered 2026-10-03 11:00
  CDT at launch; this entry written 11:02, before any result of the run; repo_r44 = the committed tree;
  `tmp/d38.sh`; `output/gpu/d38`).** The user, comparing the videos: the 45-minute one (D17's run of the code
  before D19: 64 px then 96 px, the layer as it was, the old solve, to its own stop) looks best. Its record: 2704
  s, 122 commits of 129 attempts, silhouette IoU 0.9838, world-thin 0.55 %, chamfer 0.0589, last kinetic record
  1.5e-4; the 40-attempt runs of the committed code: 0.9833–0.9846, 0.33–0.42 %, 0.0593–0.0595, 0.75e-3–1.6e-3.
  By the numbers the short runs are as good or better; in the 4K frames the user is right: at the end the old
  run's horns are solid rounded tubes with a clean outline, the new run's are feathered, with wisps at the snout
  and the right foot and a rough tail fin. And at the same window (40) the old run's horns are already solid
  blobs (with a strand still across the mouth), where the new code's are feathered. The thin share counts a
  target point as covered when any particle is within reach; it does not see whether the cover is a solid
  surface or a few beads, so it rewarded what the eye rejects. Candidates for the difference: the window budget
  (40 against 122); 96 px from the start (thin features are covered early, by sparse particles: world-thin 0.5 %
  by window 30–40 against 1.5–2 %); the layer as the body's (detached sets are no longer made regular).
  The run: the committed code, the 300k dragon, no window budget, alone on GPU 1; then the census, the far count
  and the 4K render. Expectation: it stops after 60–130 windows in 15–30 minutes; if the horns are solid at its
  end as in the old run's, the 40-window budget is the cause and the 15-minute run trades that for time; if
  they stay feathered, the cause is in 96 px from the start or the layer change, and those two are then
  separated at 40 windows (64 → 96 px with the new layer; 96 px with the old layer is d20).
  **Result (2026-10-03 11:23 CDT).** The run stopped by itself at window 60 on three consecutive rejections (the
  best commit is window 55; 2081 frames delivered): 958 s of simulation (1037 s of process), 54 commits of 60
  attempts; silhouette IoU 0.9839, world-thin 0.32 %, thin 18.4 %, chamfer 0.0593, holes 0.03 %, last kinetic
  record 7.3e-4. Render: λ of the first window 0.463, median g_share 0.94. Rendered particles farther than 3
  target spacings from the target: 281 at window 39, 194 at the end; farther than 4.4: 63, 36; farther than 6:
  35, 3. Census at the end: 2666 rendered particles in detached sets, 2546 within the berth, 115 in the near
  band, 5 beyond a loss cell, no dense far set.
  The horns in the 4K frames (the same crop; `q45/q_horns.jpg`): the old run's end, solid rounded tubes; this
  run's end, more consolidated than at window 40 but the right horn still feathered; the share of strong-gradient
  pixels in the crop 6.1 % (old end), 6.6 % (this end), 6.4 % (the committed code at 40 windows). The whole body
  is comparable, with a slightly rougher tail fin.
  Expectation: the stop at the low edge of the expected range (60 windows, 16 minutes; the old code ran 129).
  The horns stay partly feathered at the run's own end, so the 40-window budget alone is not the cause: the
  committed code does not go on to 122 windows when it is allowed to, and the 15 further windows it takes remove
  far material (35 → 3 beyond 6 spacings) more than they consolidate the horns. What remains is 96 px from the
  start or the layer change; with the old layer, 96 px from the start (d20) is already feathered at window 40
  where 64 px (the old run) is solid, which points at the schedule. D40 separates it with the code as it is now.
- **D36 and D37, the bulk morph's sparse surface and where its beads come from (B1 and B2 of the plan; read-only
  on d32's archive, its archived F and its term dump; pre-registered 2026-10-03 10:58 CDT, before the run;
  `scripts/probes/settled/midmorph_probe.py`; `output/gpu/d32/midmorph_probe.txt`).** d32 is the committed recipe:
  the outer-layer relaxation on (D16b: it stays), the layer as the body's, the new solve. If the relaxation is
  ever taken out, the stretch statistic is to be read again on that trajectory before a design rests on it.
  D36 (B1): on the surface particles connected to the body, at raw 80–480 and the end, by the display's sparsity
  S / S_ref (D35's measure): the two lengths l1 ≥ l2 of the in-plane arrangement of the 8 nearest in-plane
  neighbours (from the covariance of their tangent-plane offsets), each against the same on the target's
  surface; and from the archived F the material's stretches s1 ≥ s2 in the tangent plane and det F. One
  direction stretched: l1 up with l2 as the target's (then a disc drawn longer along that direction covers it,
  display only). The area diluted: both up (then the surface holds too few particles and no disc shape gives
  them back). Expectation, uncertain: at windows 3–6 the sparse class (S ≥ 1.5) is mostly diluted in both
  directions (l2 ≥ 1.2 on more than half), because the sphere's surface has to grow into the dragon's in every
  direction; and the material's in-plane area stretch s1 s2 is above 1.5 there and follows S (correlation
  above 0.3).
  D37 (B2): the rendered particles in detached groups at least two layer spacings from the body at window 5's
  start, traced back to window 0: detached, on the layer, depth in the source, S, det F, the stretch of their
  eight source bonds, the window of first detachment, and what moves them apart from their source neighbours
  per window, by channel. Expectations: most first detach in windows 2–4; the separating displacement is the
  MPM advection's (u's gate is closed for the layer until window 5–6, so u cannot be it); they come from the
  source's outer four lattice steps no more often than the body's surface does (78 % in D12's addendum). If u or
  the rest carries a third or more of the separation, the control is breaking the neighbourhoods and the
  reading changes.
  **Result (2026-10-03 10:59 CDT).** The target's surface: S_ref 1.57 spacings, l1 1.34, l2 0.95, l1 / l2 1.39.
  D36. Surface particles connected to the body, the sparse class S ≥ 1.5 (and all of them), at windows 2, 3, 4,
  5, 6, 8, 12 and the end: particles 1491, 1965, 1762, 1220, 676, 255, 156, 178 (of 11 600–19 200); l1 and l2
  against the target's 1.56 and 1.56, 1.59 and 1.56, 1.60 and 1.59, 1.58 and 1.57, 1.58 and 1.57, 1.58 and 1.50,
  1.55 and 1.46, 1.51 and 1.41 (all of them: 1.30 and 1.30 at window 2, 1.16 and 1.13 at 5, 1.04 and 1.00 at 8,
  1.05 and 0.99 at the end); l1 / l2 1.39–1.51 (the target's own 1.39); stretched in one direction only 7–15 %,
  diluted in both 74–90 %. The surface is diluted, not stretched along a line: a disc drawn longer in one
  direction would cover a tenth of it. The dilution is transient on the body: the whole surface is at 1.30 of
  the target's spacing at window 2 and back at 1.0 by window 8.
  The material measure failed: in every class and frame the archived F gives in-plane stretches 1.00–1.02 and
  0.97–1.00 and det F 1.00–1.01, with no correlation to S (−0.08 to +0.07; −0.33 once on 156 particles). The
  stored F stays at the identity (it is smoothed every step and its strain is assimilated every window), so it
  does not record how far the surface has been pulled apart; the expectation "s1 s2 above 1.5, following S" is
  refuted, and F cannot be used for this. The source bonds can (D37).
  D37. 2893 rendered particles are in detached groups at least two layer spacings from the body at window 5's
  start (of 8736 rendered in detached groups). Their depth in the source: median 2.5 lattice steps, 24 % within
  2, 91 % within 4, 2 % deeper than 8; the body's surface at that moment: 2.5, 21 %, 91 %, 1 %: the same
  material. First detached in window 0–1: 3 %, window 2: 17 %, 3: 27 %, 4: 31 %, 5: 22 %. Per window 0–5: the
  mean stretch of their eight source bonds 1.00, 1.19, 1.51, 1.91, 2.38, 2.74 (the longest bond 1.00 … 4.65); S
  0.95, 1.10, 1.26, 1.39, 1.53, 1.59; on the layer 37 … 67 %; moved apart from the source neighbours along the
  bonds by the MPM advection +0.32, +0.54, +0.71, +0.83, +0.58, +0.21 spacings a bond a window, by u 0.00 (its
  gate open on 0–10 % of them), by the rest −0.01 to −0.03.
  Predictions: the dilution in both directions confirmed (expected on more than half: 74–90 %); the onset in
  windows 2–4 confirmed (75 %, another 22 % in window 5); the separation is the advection's, u and the rest
  carry none; the beads are the surface's own material. Refuted: the material stretch from F.
  Reading: in windows 1–5 the flow that makes the dragon out of the sphere pulls the surface's own material
  apart (source bonds at 2.7 times their length, up to 4.7), evenly in the plane. The body's surface thins to
  1.3 of the target's spacing and recovers by window 8; the strands that are pulled thinnest come off as groups
  of a few particles and travel on their own (D33's beads; 96–97 % arrive, D21). Neither the control u nor the
  relaxation makes them. What the plan called for next is the literature, then a design: the surface's
  particle count under a growing area, not a display shape.
- **D35, the display's measure of sparsity taken on the surface (display only, no simulation; pre-registered
  2026-10-03 10:51 CDT, before the run; `scripts/probes/settled/disc_rule_probe.py`; `output/gpu/d35`).** D34: the
  rule sigma = spacing × clamp(S / S_ref, 1, 4) takes S as the 8th-neighbour distance in space and S_ref as its
  median over all target particles, so it inflates the surface of a perfect sample. The user's plan of
  2026-10-03 (A, then B, then C): A is this entry. Candidates, each with S_ref = the median of the same S over
  the target's surface particles (neighbourhood asymmetry of half a coverage radius or more): R1, the
  8th-neighbour distance in space (only the reference moves); R2, the 8th smallest distance among the 32
  nearest neighbours measured in the particle's tangent plane, |(I − n nᵀ)(x_j − x_i)|, with the display's own
  normal; R2s, R2 over same-sheet neighbours only (n_i · n_j > 0). No new constant: the 8th neighbour, the 32
  neighbours of the normals and the clamp are the rule's own. Drawn with R0 (the rule as it is), each candidate
  and no inflation: the target sample; the morph's end (d32, raw 1560, head box); the horns alone (thin
  features); raw 192 (the stretched surface and the beads) and raw 480.
  What counts as success (the user's criteria): on the target the inflated share falls to almost nothing while
  the coverage holds; the morph's end has a narrower edge; the thin and stretched places stay closed. In
  numbers, for the candidate to adopt: target particles inflated by 1.1 times or more under 3 % (R0: 12.4 %);
  covered pixels lost against R0 at most 0.3 % of the frame on the target and on the morph's end (no inflation:
  1.3 and 1.4 %) and at most 2 % at raw 192 (no inflation: 7.7 %); the silhouette edge narrower than R0's on the
  target and at the end (9.5 and 9.3 px). Expectations, not targets: the end's edge about 7–8 px; R1 and R2 close
  on ordinary surface, apart on the one-particle sheets of raw 192 (R2 inflates them more) and on thin
  features (R2 may read the opposite face as a neighbour: R2s is there to measure whether that matters; if R2
  and R2s agree the filter is not needed). If no candidate holds the coverage, the measure is recorded as it is
  and the rule stays.
  **Result (2026-10-03 10:57 CDT; `output/gpu/d35/disc_rule_probe*.txt`, sheets per object): R2 adopted in the
  display renderer; my coverage criterion was the wrong measure and is missed by the letter.** The target: 8.3 %
  of its particles are on the surface; references in target spacings R0 1.97, R1 2.33, R2 1.57.
  Inflated by 1.1 times or more (R0 / R1 / R2 / none): target 12.4 / 1.4 / 2.7 / 0 %; the end 20.5 / 5.1 / 7.2 /
  0; raw 480 21.0 / 6.8 / 7.2 / 0; raw 192 53.7 / 18.3 / 20.0 / 0. Silhouette edge width (px): target 9.5 / 8.7 /
  8.0 / 7.8; the end 9.3 / 8.1 / 7.5 / 6.7; raw 480 12.0 / 11.1 / 10.1 / 9.1; raw 192 24.7 / 25.3 / 25.2 / 16.4.
  Covered pixels lost against R0, whole frame (R1 / R2 / none): target 0.88 / 1.10 / 1.31 %; the end 0.77 / 0.80 /
  1.43; raw 480 0.94 / 1.07 / 2.08; raw 192 2.09 / 2.30 / 7.70. The criterion (at most 0.3 % on the target and
  at the end, 2 % at raw 192) is missed by every candidate. What those pixels are, measured afterwards: of the
  lost pixels, those more than 10 px inside R0's outline (holes) are 0.010 / 0.011 / 0.024 % of the covered
  pixels on the target, 0.014 / 0.025 / 0.108 at the end (head 0.05 / 0.10 / 0.37; horns 0.06 / 0.06 / 1.04), 0.033
  / 0.033 / 0.30 at raw 480 and 0.25 / 0.29 / 3.46 at raw 192 (the bead box 1.1 / 1.8 / 12.9); the rest is the
  outline moving in. Against the mesh's own silhouette (three million surface samples of the fitted mesh, as in
  D12): 71–74 % of the lost pixels lie outside it, and the silhouette's IoU with it rises, target 0.9509 →
  0.9547 (R1), 0.9553 (R2), 0.9560 (none); the end 0.9459 → 0.9493, 0.9490, 0.9508. R0's "coverage" on those
  pixels was the inflated surface discs drawn beyond the shape. The criterion counted that as coverage; the
  user's criteria (the target's inflation almost gone, the holes kept closed, the end's edge narrower) hold.
  R2s equals R2 to the last digit on every object: the same-sheet filter is not needed. R2 against R1: the
  narrower edge everywhere (8.0 / 7.5 / 10.1 against 8.7 / 8.1 / 11.1 px) for 0.01 point more holes at the end.
  Expectations: the end's edge 7.5 px (7–8 expected); R1 and R2 apart on the sheets of raw 192 as expected, by
  little (18.3 and 20.0 % inflated).
  In the code: `render/support.py` `surface_spacing` and `surface_particles`; `scripts/render_splat_photoreal.py`
  takes sigma = spacing × clamp(surface spacing / its median on the target's surface, 1, 4); suite 269 passed,
  exit 0. d32 drawn again (`output/gpu/d35/frames_dragon`, 51 s): share of head pixels with a strong gradient
  3.9 → 4.8 % at the end, 4.0 → 4.9 at raw 960, 1.5 → 2.1 at raw 192 (d20 before the layer change, old rule:
  4.6–4.7); the mid-morph beads are smaller and still there (they are particles, D33). Video:
  `output/video_2026-10-03/dragon300k_11min_surface_spacing_discs_4k.mp4` (local). The probes that print a "disc
  inflation" (frame_forensics, box_probe, inflation_probe) still compute the rule before D35.
  Not touched: the opacity's live support (the same 8th-neighbour test against the all-particle median), the
  two-view `render_splat_gpu.py`, the `--surface-common` path.
- **D34, what the display's disc inflation costs and what it closes (read-only on d32's archive, 2026-10-03 10:15
  CDT, written after the reading, no prediction; `scripts/probes/settled/inflation_probe.py`;
  `output/gpu/d32/inflation_probe.txt`, `inflation/`).** The user's question: a disc grown where the particles are
  sparse must blur. The rule is sigma = target spacing × clamp(8th-neighbour distance / coverage radius, 1, 4);
  D12 varied the base disc, not this factor. Three frames drawn with the cap at 4 (the rule), 2, 1.5 and 1 (no
  inflation), everything else unchanged.
  Rendered particles with a disc inflated by 1.1 / 1.5 / 2 times or more, and the share of the summed disc area
  the inflation adds: raw 192 (window 4.8) 53.7 / 6.2 / 0.49 %, 29 %; raw 480 21.0 / 2.3 / 0.14 %, 14 %; the end
  (raw 1560) 20.5 / 1.0 / 0.05 %, 12 %.
  Cap 4 → 2 → 1.5 → 1, silhouette edge width (px): raw 192 24.7, 24.5, 21.1, 16.4; raw 480 12.0, 11.6, 10.4, 9.1;
  the end 9.3, 9.0, 7.9, 6.7. Shading detail inside against the rule as it is: 1.49, 1.17, 1.11 at cap 1 (1.00–
  1.06 at the caps between). Strong-gradient share of the box's object pixels: raw 192 (the bead box) 0.3 → 6.4 %;
  raw 480 (head) 2.1 → 4.1 %; the end (head) 2.8 → 3.9 %. Covered pixels lost against the rule as it is, whole
  frame and box: raw 192 1.0, 3.5, 7.7 % and 6.1, 17.4, 34.9 %; raw 480 0.07, 0.5, 2.1 % and 0.2, 1.4, 4.7 %; the
  end 0.07, 0.35, 1.4 % and 0.2, 1.0, 3.0 %.
  In the pictures: without the inflation the end frame's outline is sharper and the horns are torn into separate
  flecks; at raw 192 the beads shrink to dots and the stretched surface beside them opens into a scale pattern
  with dark gaps.
  Reading: yes. At the end the inflation widens the silhouette edge by 2.6 px of 9.3 (28 %) and lowers the
  inside detail by a tenth, to keep 1.4 % of the covered pixels (3 % of the head) closed; in the bulk morph it
  is half again as much blur and what it closes is 8 % of the picture. It is a display rule: it trades holes
  for blur on particles that are sparser than the target sample (a fifth of the rendered particles at the end,
  half at window 5), and neither choice of cap removes the cause, which is that spacing. Part of the detail
  gained at cap 1 is the arrangement showing through (D12: narrower discs show the lattice), not relief.
  Addendum (10:39 CDT, `inflation_probe_target.txt`): the same on the target sample itself (the one-frame
  archive `output/gpu/render/target_dragon300k`). Inflated by 1.1 / 1.5 times or more: 12.4 / 0.1 % of its
  particles (the morph's end: 20.5 / 1.0 %); the inflation adds 8 % of the disc area; silhouette edge 9.5 px
  with the rule and 7.8 px without; without it 1.3 % of the covered pixels are lost (2.0 % of the head). So
  three fifths of the end frame's inflated particles are inflated on a perfect sample too: the rule compares a
  particle's 8th-neighbour distance with the median over all particles, nine tenths of which are interior, and
  a surface particle has half its neighbourhood empty. At the end the rule reads "on the surface" as "sparse";
  what the morph adds is the 1.5-times class (1.0 against 0.1 %). And even the perfect sample opens by 1.3 %
  without the inflation: the thin features hold too few particles at 300k (D12).
- **D33, three crops the user asked about: what is drawn there (read-only, 2026-10-03 10:06 CDT, written after
  the reading; `scripts/probes/settled/box_probe.py`; `output/gpu/d32/box_probe.txt`, `box_probe_beads.txt`,
  `boxes/`).** Located by template matching against the rendered frames: two crops are frame 16 of the new video
  (D30; raw 192, window 4.8, 0.8 s in), 4K boxes (1014, 363)–(1732, 1446), the body's left edge with a row of
  beads, and (1873, 99)–(2869, 663), the top of the head with a blurred fringe (scores 0.994, 0.998); the third
  is frame 14 of the bunny video made by the code before D20 (D17's run, raw 168; score 0.974), not analysed
  further (its archive is gone). The same beads and the same fringe are at the same pixels in d32 (the same
  code) and in d20 (the code before the layer and solver changes). On d32's archive: each bead is 7–8 rendered
  particles (15 in a larger one), all of them members of detached groups 3–8 layer spacings from the body, drawn
  with discs inflated 2.2–2.6 times; all of them are within the berth of the target at the end, 0–25 % still
  detached there. In the whole boxes: 3223 and 2138 rendered particles in detached groups (disc inflation 1.7,
  90th percentile 2.1–2.2; 95–97 % within the berth at the end) and 3515 and 1856 attached particles with a disc
  inflated 1.5 times or more (96–98 %). So: material in transit during the bulk morph (D21: 96–97 % of what is
  detached at window 4 arrives), a handful of particles a bead, enlarged by the display's disc rule; not the
  end state's leftover of D32, and not changed by D26 or D29 (the peak of rendered detached particles at a
  window end is 20 283–20 452 with them, 20 307 before).
- **D32, what holds the detached material that is left near the surface (diagnostic, pre-registered 2026-10-03
  03:53 CDT, launched 03:53; repo_r46 = repo_r43 with u and its gate added to the channel record; `tmp/d32.sh`;
  `scripts/probes/settled/near_band_probe.py`; `output/gpu/d32`).** After D30 the far clumps are gone and
  90–154 rendered detached particles stay 2–5 spacings from the target in sets of one to eight; one of them
  draws a small disc at the notch under the tail. The 300k dragon again with both changes and the term dump,
  GPU 2. For the end state's rendered detached particles beyond the berth, at four window starts of the last
  twenty: the share that is off the layer (the asymmetry under its threshold: material on both sides), on it as
  a single particle (relaxed against its neighbours) and on it as a member of a group (not relaxed); per class
  the asymmetry against the threshold, the share inside u's gate, the size of u, the displacement toward the
  target per window by advection, u and the rest, and the two local terms' pull.
  Predictions from D21's numbers: u moves them 0.03–0.06 spacings a window toward the target and advection
  nothing (within ± 0.05); at least a third are off the layer or in a group; the near band's pull is on for more
  than half of them. If so the leftover is set by u's step (the accepted step, one for every coordinate) and by
  the layer test in a gap, and no definition of the neighbourhood removes it. If instead single particles on the
  layer stay with a relaxation residual of a spacing or more, the relaxation itself is not doing what D21 read.
  **Result (2026-10-03 04:10 CDT).** The run: 676 s, 39 commits, silhouette 0.9846, world-thin 0.41 %, last
  kinetic record 1.5e-3; λ 0.463, g_share 0.93. Rendered particles by distance from the target, whatever they
  are connected to (d20, the code before, in brackets): beyond 3 spacings 3170 at raw 480 (3349), 594 at 960
  (791), 284 at the end (341–346); beyond one loss cell 24 at the end; beyond 6 spacings 22 at raw 960, 2 at
  1280, none at the end (the clump of 38 at 6.6–7.5). The far material is gone; the population between 3
  spacings and one loss cell is a sixth smaller.
  The end's 229 rendered detached particles beyond the berth (median 2.3 spacings out, at most 5.7; three
  windows before the end, per class at the window's start):
  off the layer, 39: asymmetry 0.81 of the threshold (material on both sides), so no u and no relaxation;
  advection +0.01 spacings a window toward the target; the near band on for 90 % with a pull of 5.0 (units of
  the all-particle rms gradient); 2.7 spacings out, where they were 18 windows earlier.
  On the layer as single particles, 17: inside u's gate, |u| 0.08 layer spacings; u +0.08 a window, the rest
  (the relaxation) −0.06, advection +0.02; the near band on for 76 %, pull 4.2; 2.1–2.2 spacings out.
  On the layer in groups, 173 (88 of them 18 windows earlier): inside the gate, |u| 0.075; u +0.05, advection
  +0.04, no relaxation; the near band on for 57 %, pull 3.1; 2.3 spacings out.
  Predictions: u's 0.03–0.06 a window confirmed for the groups (0.05) and a little above for single particles
  (0.08); advection within ± 0.05 confirmed; a third off the layer or in a group confirmed (93 %); the near band
  on for more than half confirmed. The alternative (single particles on the layer with a large relaxation
  residual) is not what is there: the relaxation acts on them, against u.
  Reading. What is left sits at the berth's edge, pulled three to five times as hard as an average particle, and
  has no channel that can follow the pull: a particle with material on both sides is not a layer particle; a
  single one's u is undone by the relaxation (D14, D15: the relaxation removes u's per-particle offsets); a
  group's u brings it 0.05 spacings a window, bounded by the accepted step (0.65e-3–3.2e-3 for every
  coordinate). These are the parked items (u against the relaxation, the step), not the layer's neighbourhood.
- **D31, both changes on the gallery (pre-registered 2026-10-03 03:48 CDT, launched 03:48; repo_r43; `tmp/d31.sh`,
  `tmp/d31.queue`, `tmp/d31b.queue`, `tmp/d29_eval.py`; `output/gpu/d31`).** The 40k gallery (19 meshes, seed 97,
  each to its own stop, GPUs 1 and 3, the census of detached material after each run) and the 300k bunny to its
  own stop (GPU 0), against the two runs of the code before (D19's 96-px arm and D27's old arm). Each change has
  its gallery (D27, D29); this is the pair. Predictions: silhouette IoU within 0.002 of the mean of the two runs
  before on at least 15 of 19, median change within ± 0.001; lower by more than 0.003 only where a run stops
  early while moving (C does in three of four runs of any code tonight); thin median change within ± 1 point
  (the two runs before differ by a median 0.9); no frozen run, no guard, no collapsed detached set; the bunny
  within ± 0.002 of 0.9856–0.9868. Fail: five or more meshes lower by more than 0.003; a frozen run; a guard; a
  collapsed set.
  **Result (2026-10-03 04:08 CDT, 20 runs, no guard, no frozen run; `python3 tmp/d29_eval.py`, `tmp/d31_census.py`).**
  Silhouette IoU within 0.002 of the mean of the two runs before on 18 of 19, median change +0.0001, lower by
  more than 0.003 on none (C −0.0029 at its 16-window stop, V −0.0015), inside the two runs' range widened by
  0.002 on all 19; thin share median −0.8 point (bimba, cheburashka, nefertiti, armadilo lower by 1–3; A and
  teapot higher by 1.5–2); chamfer unchanged; committed windows median 32 → 32; windows without a commit summed
  57 and 72 → 76; last kinetic record above 5e-3 on C only. The census (19 meshes; D27's old arm → both
  changes): rendered detached particles beyond the berth lower or equal on 16, summed 110 → 78; beyond one loss
  cell 3 → 0; within the berth 3106 → 2469; not rendered 602 → 574; the densest detached set at 0.60 of the
  coverage radius. The 300k bunny to its own stop: 0.9873 (0.9856 before, 0.9868 with the layer alone),
  world-thin 0.02 %, thin 12.8 %, chamfer 0.0580, 65 commits, 15.2 minutes alone on a GPU. Every prediction
  holds. With D27, D29 and D30 the two changes pass, each alone and together: the gallery, the 300k dragon five
  times (twice the layer alone, three times both) and the 300k bunny twice.
- **D30, both changes together: the layer as the body's and the new solve (pre-registered 2026-10-03 03:18 CDT,
  launched 03:18; repo_r43 = repo_r35b + `window/layer.py` (D26) + `losses/grid_ot.py` and its two restated
  tests (D29) + the diagnostic channel record; the suite 269 passed, exit 0; `tmp/d30.sh`; `output/gpu/d30`).** The
  300k dragon, 40 attempts, alone on GPU 2, then the census of detached material and the 4K render outside the
  timed run. Predictions, from D26, D27 and D29 each alone: at most 12 minutes of simulation and 13 of process;
  silhouette IoU at least 0.982; world-thin at most 0.8 %; no rendered detached component of five or more
  beyond one loss cell and at most 5 rendered particles beyond one loss cell and dense at the end; at most three
  windows without a commit. Fail: any of these missed. Then repeats, and the gallery with both changes.
  **Result (2026-10-03 03:33 CDT, one run): every criterion met; floaters fewer, not none.** 666 s of simulation
  (11.1 minutes; the process 735 s, 12.3 minutes), 39 commits and one null window (37), no guard; silhouette IoU
  0.9839, world-thin 0.33 %, thin 19.2 %, chamfer 0.0593, holes 0.02 %, last kinetic record 7.5e-4; λ 0.463,
  g_share 0.94 (no λ = 0 twin). Rendered detached particles beyond one loss cell: 400 at raw 480, 78 at 960, 26
  at 1280, 10 at the end, none of them dense from raw 1280; in the near band 154 at the end (d20: 141). The
  components whose median distance is beyond the berth: 41, 119 particles, the largest of 8 (2.0 spacings out)
  and none above 4.8 spacings; the largest detached sets overall sit on the target (302 particles on a horn at
  0.9 spacings, density ratio 1.12; one of 42 at 1.7 spacings is dense, 0.41).
  The 4K frames beside d20's: the mouth empty at raw 960 and at the end; at the notch under the tail base,
  pixel (1640, 1500), the place where d21, d22 and d23 held clumps of 30–62, four particles 3.5 spacings out
  draw one small disc; the head softer (strong-gradient share 3.7 % against 4.7 % at the end, 3.6 against 4.2 at
  raw 960), as in D26. Video: `output/video_2026-10-03/dragon300k_no_floaters_11min_4k.mp4` (local, not in git).
  What is left is the near band's population (90–154 rendered particles in sets of one to eight within 2–5
  spacings), which the change does not reduce; two repeats launched 03:34 and 03:35 on GPUs 0 and 2.
  **The repeats (03:47 CDT), each alone on its GPU.** Three runs: 11.1, 10.8, 11.0 minutes of simulation (12.2,
  11.6, 11.8 of process); silhouette IoU 0.9839, 0.9833, 0.9835; world-thin 0.33, 0.41, 0.42 %; chamfer 0.0593,
  0.0595, 0.0594; 39, 40, 39 commits (one, none, one null window); last kinetic record 0.75e-3, 1.6e-3, 1.2e-3;
  no guard; λ 0.463, g_share 0.93–0.94. The code before (D19, d20): 16.4 minutes, 0.9841 and 0.9835, 0.53 and
  0.77 %. The 15-minute criterion holds in three of three with three minutes to spare; the silhouette is inside
  the earlier runs' range less 0.0002; world-thin is lower in all three. D29's three null windows did not recur.
- **D29, the cross Sinkhorn problem with alternating sweeps, the self problem at the blur (runtime; the
  formulation untouched; pre-registered 2026-10-03 03:03 CDT, launched 03:03; repo_r42 = repo_r35b +
  `losses/grid_ot.py`; `scripts/probes/settled/ot_ladder_probe.py`; `tmp/d29.sh`; `output/gpu/d29`,
  `output/gpu/d23/ot_alternating_probe.txt`).** Where the solve's sweeps go (D25's addendum): the cross problem
  takes 81–189 of a value call's blocks, most of them at the last four blur levels. Both problems are iterated
  with the parallel averaged update f, g ← (f + T(g)) / 2, (g + T(f)) / 2, which is what keeps the two potentials
  of the self problem equal; on the cross problem it converges the slow modes at half a step a sweep where
  alternating sweeps (f ← T(g), then g ← T(f)) take two. And the self problem is sent down the whole ladder
  although its plan is local. The change: the cross problem alternates; the self problem starts at the blur.
  No seed, no mode: a value is a function of its state alone, as before (the seeded solver of D22–D28 is dropped).
  Measured on a 300k dragon's states (seven windows; each variant against a solve converged to 1e-5; tolerance
  1e-3), as it is → changed: blocks per value call 113, 109, 140, 186, 213, 189, 155 → 56, 68, 59, 63, 80, 73, 70;
  seconds 0.35–0.74 → 0.21–0.29; the value's error −2.6e-6, −2.5e-5, −1.5e-3, −5.4e-3, −1.4e-2, −2.8e-2, −3.4e-2 →
  −8e-7, −1.2e-5, −3.7e-4, −1.3e-3, −4.1e-3, −8.1e-3, −7.7e-3; the gradient's 0.3, 0.7, 5.0, 9.5, 14.7, 20.1,
  22.3 % → 0.2, 0.5, 2.4, 4.6, 7.8, 10.5, 10.3 %; the error in the difference of two candidates at most 8e-5 →
  at most 3e-5 of the value. The self problem at the blur alone: 2 blocks against 25–27, value and gradient
  unchanged to every printed digit. Faster and nearer the converged solution at the same tolerance.
  The suite on repo_r42: 264 passed, 5 failed (exit 1), all in `tests/test_grid_ot.py`, two contracts of the
  old iteration: the self solve equal bit for bit to the cross solve of two equal measures at half the
  transforms; and a gradient of exactly zero at the target at tolerance 1e-3 (the parallel update keeps the cross
  potentials of equal measures equal at every sweep, so the debiased value cancels exactly whatever the
  tolerance; alternating, the gradient at the target is 1.6e-4 in that test's units and vanishes with the
  tolerance: at 1e-9 the equilibrium test passes). A property that changes, to be restated in the tests and
  measured at 300k against the gradient of a run's end state before adoption.
  The run: the 300k dragon, 40 attempts, alone on GPU 2. Predictions: 10.5–12 minutes of simulation (a window
  15–17 s against 24.6); silhouette IoU at least 0.982; world-thin at most 0.8 %; at most two windows without a
  commit. Then two repeats, the 40k gallery and the 300k bunny against D19's 96-px arm (silhouette within ± 0.002
  on at least 15 of 19, none lower by more than 0.003, thin median within ± 0.5 point, no early stop while
  moving). Fail: above 13 minutes; a quality criterion missed; C or V stopping as in D25.
  **Result of the dragon (2026-10-03 03:16 CDT, one run).** 682 s of simulation (11.4 minutes; the process 750 s,
  12.5 minutes; the suite ran on the same GPU for a minute of it, my mistake), 37 commits of 40 attempts: three
  windows without an accepted step (20, 25, 28; the cold runs had none or one), no guard; silhouette IoU 0.9837,
  world-thin 0.58 %, thin 18.8 %, chamfer 0.0593, holes 0.02 %, last kinetic record 1.4e-3; λ 0.468, g_share
  0.93 (no λ = 0 twin). A window: start 1.7 s, gradients 8.5 s, line search 3.9 s, commit 0.6 s (cold: 3, 11, 7,
  1.7). At its end state the cross solve takes 70 blocks (1–4 at the eleven upper levels, then 7, 9, 9, 9, 9,
  11) and the self solve 2; over perturbations of 1e-10 to 1e-4 the block schedule does not change and the
  value moves by 1e-10–1.4e-8 on 1.1e-3. The tests restated for the two contracts (the self solve symmetric,
  cheaper than the cross solve of equal measures and equal to the mean of its potentials; the gradient at the
  target under ten tolerances of the gradient a shift away: measured two): 269 passed, exit 0.
  Predictions: time, silhouette, world-thin confirmed; "at most two windows without a commit" refuted (three).
  Whether three is this solver or chance needs the repeats (D30's three runs on the same solver: one, none, one).
  **Result of the gallery (2026-10-03 03:47 CDT, 19 runs, no guard, no frozen run; `python3 tmp/d29_eval.py`),**
  against the two runs of the solver before (D19's 96-px arm, D27's old arm), which differ from each other by a
  median 0.0005 in silhouette (at most 0.0022) and 0.9 point in thin share. Silhouette IoU within 0.002 of
  their mean on 18 of 19, median change +0.0001, inside their range widened by 0.002 on 18; lower by more than
  0.003 on one, C (0.9737 against 0.9787 and 0.9765): C stops after 16 windows with the last kinetic record at
  3.1e-2, exactly as the old solver's second run does (16 windows, 3.1e-2, 0.9765) and as both arms of D27 do.
  Thin share median +0.7 point (teapot +2.7, A +2.3, V +1.7; spot −2.6, ogre −1.8). Committed windows median
  32 → 32; windows without a commit summed 57 and 72 → 79; chamfer unchanged. By the letter two predictions miss:
  "none lower by more than 0.003" (C, at its own early stop) and "thin median within ± 0.5" (+0.7, inside the
  0.9 by which the two old runs differ); and the fail line "C stopping as in D25" is met by a stop that the old
  code shows as well. No mesh is outside what two runs of the old solver span, apart from C's 0.0028 below the
  lower of them. The 300k bunny is run with both changes (D31).
  The equilibrium property at 300k (`ot_ladder_probe.py` on D30's archive, production code against the solver
  before): at the target sample itself the solver before gives value +9e-10 and a gradient of rms 2e-13 (zero by
  construction); the new one gives value −7.8e-6 (0.6 % of the end state's 1.29e-3) and a gradient of rms 1.8e-8,
  8.7 % of the gradient at the run's end state (2.1e-7). At the end state and through the run the new solver's
  gradient is off by 9–10 % from the converged one and the old by 17–21 %; its value by 0.6–0.7 % against
  1.8–3.3 %. The new error is the same size everywhere; the old one was twice it everywhere except exactly at
  the target, where it vanished.
- **D28, the seeded solve only where its trial is seen to converge (pre-registered 2026-10-03 02:49 CDT, before
  launch; repo_r41 = repo_r38 + `losses/grid_ot.py`; `tests/test_transport_support.py` 26 passed, exit 0;
  `tmp/d28.sh`, `tmp/d28.queue`; `output/gpu/d28`).** D25's failure is the trial of `seed()` passing with the old
  potentials unchanged. One change: the trial counts only if its residual was above the tolerance at a check and
  below at a later one; a first check already below is not a solve, and the window is solved cold. The log
  names each window's mode. (1) The four 40k meshes that failed or moved under repo_r38, each to its own stop,
  GPU 2: C, V, bob, bimba. Predictions: C and V do not stop while moving (at least 25 committed windows, last
  kinetic record under 2e-3, silhouette within 0.002 of the cold 0.9787 and 0.9773), their late windows cold in
  the log; bob and bimba thin within 2 points of the cold 10.9 and 8.2. (2) When a GPU is free, the 300k dragon,
  40 attempts, alone: the windows seeded from about window 12 as before, at most 13 minutes of process,
  silhouette at least 0.982, world-thin at most 0.8 %. Fail: C or V stop on rejections with the physics worse by
  more than a tenth; the dragon above 15 minutes. Then the whole gallery.
  **Result of (1) (2026-10-03 02:54 CDT): fails; (2) not run; the seeded solver is dropped.** The log's modes
  (c cold, S seeded): C cSccccccccSSccccccccc, V cSScccccSSSSSSSS, bob cS, nine c, eleven S, then c to the end,
  bimba ten c, fifteen S, nine c. C: 18 commits of 21 attempts, stopped on three rejections (physics 30 % worse
  than the commit before, reversal +0.48) in cold windows, last kinetic record 1.6e-2, silhouette 0.9762 (cold
  0.9787). V: 16 commits, stopped as converged with the kinetic record at 3.4e-3 (cold: 31 commits, 1.8e-4),
  silhouette 0.9776 (0.9773). bob: 42 commits (33), 0.9826 (0.9837), thin 11.4 (10.9). bimba: 32 commits (37),
  0.9780 (0.9785), thin 11.8 against 8.2 (D25: 12.5). The fully stale windows are gone, the runs still end
  sooner and bimba's thin share is up by 3.6 points twice: a solve started from the window's reference reports
  less change than there is (D22: by 10–20 % of a candidate difference at 300k), whatever the guard. A run of the
  cold code on C is not there to say how often C stops early by itself. Predictions refuted for C, V and bimba;
  bob within its band.
- **D27, the layer as the body's beyond one dragon run (pre-registered 2026-10-03 02:46 CDT, launched 02:46;
  `tmp/d27.sh`, `tmp/d27c.sh`, `tmp/d27.queue`; `output/gpu/d27`).** D26 passed on one 300k dragon. The same change
  (repo_r40) against the layer as it was (repo_r35b), the cold solver in both: the 40k gallery (19 meshes, seed
  97, each to its own stop, the two arms interleaved in one queue on GPUs 1 and 3), each run followed by the
  census of detached material on its archive (`detached_probe.py`, parts A and B); and on GPU 0 the 300k dragon
  again (40 attempts, census, 4K render), then the 300k bunny to its own stop (against D19's 0.9856, 40 windows).
  Predictions: silhouette IoU within ± 0.002 on at least 15 of 19, median change within ± 0.001, none lower by
  more than 0.003; thin share median change within ± 0.5 point; committed windows median within ± 20 %; windows
  without a commit (null and rejected, summed over the gallery) not up by more than a fifth; rendered detached
  particles beyond the berth at the end lower or equal on at least 15 of 19 and no mesh with a new dense detached
  set (8th-neighbour distance under 0.1 of the coverage radius). The dragon: no rendered detached component of
  five or more beyond one loss cell, at most 5 rendered particles beyond one loss cell and dense, silhouette
  0.9815–0.9865, world-thin at most 0.8 %, 40 attempts with at most two windows without a commit. The bunny within
  ± 0.002 of 0.9856. Fail: silhouette lower by more than 0.003 on five or more meshes; a frozen run; a guard; a
  collapsed set; a dragon clump. A mesh that fails alone is examined frame by frame before any decision.
  **Result (2026-10-03 03:21 CDT; 38 gallery runs and the dragon, no guard, no frozen run; `python3
  tmp/d27_eval.py`; the bunny still running).** The 40k gallery, new against old: silhouette IoU within ± 0.002
  on 17 of 19, median change +0.0001, lower by more than 0.003 on none (C −0.0025, teapot +0.0028); thin share
  median −0.3 points, lower on 11 (spot 8.8 → 4.3, maxplanck 8.5 → 5.9, bimba 14.3 → 11.3, dragon 14.4 → 12.0;
  bob 10.0 → 13.1, fandisk 6.0 → 7.6); chamfer median −0.0001; committed windows median 34 → 29; windows
  without a commit summed 72 → 73. The census at each run's end: rendered detached particles beyond the berth
  lower or equal on 16 of 19, summed 110 → 64 (dragon 24 → 9, fandisk 18 → 13, ogre 11 → 8; higher on beast 2 →
  3, C 3 → 6, cheburashka 2 → 5); beyond one loss cell 3 → 2; none dense and far in either arm; within the berth
  3106 → 2262; not rendered 602 → 553; the densest detached set at 0.52 (old) and 0.83 (new) of the coverage
  radius: no collapse. At 40k a loss cell is an MPM cell and there are few floaters to begin with.
  The 300k dragon again: 951 s, 39 commits and one null window, silhouette 0.9843, world-thin 0.37 %, chamfer
  0.0594, last kinetic record 1.2e-3; λ 0.463, g_share 0.93; rendered detached particles beyond one loss cell
  92 at raw 960, 16 at 1280, 7 at the end, none of them dense from raw 960 on; the largest detached sets all
  within 1.1 spacings of the target. With D26 two runs of two without a clump.
  Every prediction made for the gallery and the dragon holds. Two things the old arm shows about single runs at
  40k: C stops after 16 windows while moving in both arms here (last kinetic record 3.1e-2 and 2.4e-2; D19's
  cold C ran 67), and nefertiti's old run stops after 16 with 1.0e-2; bimba's thin share is 14.3 in this cold
  run and 8.2 in D19's. D25 and D28 read C's stop and bimba's thin share as effects of the seeded solver; one
  run a mesh cannot carry that (V's two early stops under it remain: 16 and 13 windows against 31 and 34 cold).
  The 300k bunny to its own stop (03:26 CDT): silhouette IoU 0.9868 (D19: 0.9856; within ± 0.002), world-thin
  0.05 %, thin 13.5 %, chamfer 0.0580, 98 commits and three windows without one (D19: 40; D25's seeded run 80),
  19.5 minutes on a GPU it had to itself, last kinetic record 4.0e-6, converged; λ 0.341, g_share 0.89; at the
  end 57 rendered detached particles beyond the berth, none beyond one loss cell, no dense set. Prediction
  confirmed.
- **D26, the layer is the body's: a detached group is not relaxed and u moves it along the direction away from
  the material around it (pre-registered 2026-10-03 02:11 CDT, launched 02:11 after `tests/test_layer_relax.py`
  passed, exit 0; repo_r40 = repo_r36 + `window/layer.py`; `tmp/d26.sh`; `output/gpu/d26`).** D24 kept the part of
  D21's cause that the relaxation acts on and failed there. The other part reads the objective: u. One change
  to the layer's definition for the members of a detached set of 2–512 particles (connected at one layer
  spacing, the body = the largest set): their row of the relaxation is themselves (residual zero: a detached
  group has no surface of its own to be made regular), and their normal is the asymmetry against the 32 nearest
  particles that are not members of the set. Single detached particles and the body's rows are as before; the
  objective, the solver (cold) and the schedule are untouched; 40 attempts, term dump with the channel record.
  On d21's states: the members' normals have cosine −0.70 to −0.74 with the direction to the target (−0.39 to
  −0.65 before), 82–90 % are on the layer, residual zero.
  What it can and cannot do, from D21's numbers: u brought single particles (whose normal is already this one)
  0.04–0.05 spacings a window, so a group 6 spacings out will not arrive in 40 windows; but nothing contracts it
  any more (u's normals are parallel across the group, the relaxation is off), so it stays sparse: the isolation
  gate keeps the spray cleanup on it, and the display, which draws by live support, does not draw it.
  Predictions. No collapsed set (no detached component with the 8th-neighbour distance under 0.1 of the coverage
  radius); at the end at most 5 rendered detached particles that are beyond one loss cell and dense (d20, d21,
  d22: 38–47; d23: 13) and no rendered component of five or more beyond one loss cell; detached particles that
  are not rendered up from 405–511 to at most 900; the contraction of the largest far group by u and by the rest
  under 0.03 spacings a window (0.04–0.15 before); the run to its 40 attempts with at most two windows without a
  commit; silhouette IoU within 0.002 of 0.9835–0.9844, world-thin 0.5–0.8 %, last kinetic record under 5e-3.
  Fail: a rendered clump of five or more beyond one loss cell; an early stop; a collapsed set; silhouette below
  0.981 or world-thin above 1.0 %. If the groups stay sparse but rendered (8th-neighbour ratio 1.1–1.3), that is
  recorded as not solved, with the count.
  **Result (2026-10-03 02:30 CDT, one run): passes.** 978 s, 40 commits of 40 attempts, no null window, no
  rejection, no guard; silhouette IoU 0.9842, world-thin 0.28 %, thin 18.5 %, chamfer 0.0593, holes 0.02 %, last
  kinetic record 2.7e-3; λ 0.463, g_share 0.94 (no λ = 0 twin). Rendered detached particles at the window ends
  (d20 in brackets): raw 160 20 283 (20 307), 480 5104 (4384), 960 3583 (2844), 1600 2838 (2301); of them beyond
  one loss cell 8753, 560, 38, 2 (8718, 615, 86, 38); beyond one loss cell and dense 0 from raw 1280 (37–47); in
  the near band 90 at the end (141); not rendered 385 (439). At the end 92 rendered detached particles lie
  beyond the berth, in 60 components, the largest of five particles at 2.1 spacings (d20: 179, with the clump of
  38 at 6.6; d21: 234, clumps of 62 and 31); the smallest 8th-neighbour ratio among the large detached sets is
  0.84 (no collapse). The far groups do not only stay sparse, they go (7 rendered particles beyond one loss cell
  at raw 1280 against 60). A reading, not measured on the particles that arrived: sparse, their members are
  farther apart than one layer spacing, so they are single particles to the layer and are relaxed against the
  body, which is what already cleaned single particles. In the 4K frames the mouth is empty at raw 960 and at the end where
  d20 shows the pale blob; the head is a little softer (share of head pixels with a strong gradient 3.8 % against
  4.6 % at the end, 3.8 against 4.2 at raw 960, 2.9 against 3.3 at raw 480; D19's two cold runs differed by 0.2)
  and there is a little more wisp at the snout tip and the right foot: more particles sit on thin target
  features (2746 detached within the berth against 2122; world-thin 0.28 % against 0.53–0.77 %) and their groups
  are no longer made regular.
  Predictions: no collapse, no dense far particles, no far component, the 40 commits, the silhouette and the
  kinetic record confirmed; not-rendered particles did not rise (385; predicted up to 900); world-thin better
  than its band (0.28 against 0.5–0.8); the contraction by u on what remains is 0.00–0.06 spacings a window on
  a five-particle set, above the 0.03 predicted, on a set that is no clump. One run: D27 for the repeat, the
  gallery and the bunny.
- **D25, the seeded Sinkhorn solve beyond the dragon (pre-registered 2026-10-03 01:52 CDT, launched 01:52;
  `tmp/d25.sh`, `tmp/d25.queue`; `output/gpu/d25`).** repo_r38 (D23's second form) on the 40k gallery (19 meshes,
  seed 97) and the 300k bunny, each to its own stop, three workers on GPU 3; against D19's 96-px arm, which is
  the same code with the cold solver (repo_r35b; `output/gpu/d19/d19_96_*`). At 40k the loss grid is the MPM
  grid, so a solve is cheaper and the seed saves less. Predictions: silhouette IoU within ± 0.002 on at least 15
  of 19, median change within ± 0.001, none lower by more than 0.003; thin share median within ± 0.5 point;
  committed windows within ± 20 % in the median; minutes summed lower (67 → at most 60); the 300k bunny within
  ± 0.002 of 0.9856 in at most 8.7 minutes. Adoption of the solver needs this and D23's dragon (with two repeats
  for the spread). Fail: a frozen run, a guard, or five or more meshes lower by more than 0.003.
  **Result (2026-10-03 02:44 CDT, 20 runs, no guard, no frozen run; `python3 tmp/d25_eval.py`): not adoptable as
  it is.** The 40k gallery, seeded against cold: silhouette IoU within ± 0.002 on 17 of 19, median change
  −0.0002, lower by more than 0.003 on one (C −0.0031; beast −0.0022); thin share median unchanged (bimba +4.3
  and bob +4.8 points, nefertiti and ogre −1.0 and −1.4); chamfer unchanged; committed windows median 33 → 31.
  Minutes summed 67 → 95 with three workers on one GPU (D19 ran otherwise, so the times do not compare; at 40k
  the loss grid is the MPM grid, a solve is cheap and the seed has nothing to save). The 300k bunny: 0.9872
  against 0.9856, world-thin 0.02 %, 80 windows against 40 (to its own stop; 14.7 min on the shared GPU). The 300k
  dragon, three runs alone on a GPU: 12.1, 11.0, 11.8 min of simulation (13.2, 11.8, 12.6 of process), silhouette
  0.9844, 0.9838, 0.9845, world-thin 0.73, 0.42, 0.40 %, 40 commits each, last kinetic record 0.7e-3–3.5e-3; cold:
  16.4 min, 0.9835–0.9842, 0.53–0.77 %.
  What is wrong: C stops after 16 windows and V after 13 (cold: 67 and 31) on three consecutive rejected
  candidates, still moving (last kinetic record 3.6e-2 and 9.2e-3 against 2.5e-4 and 1.8e-4), with the rejected
  windows' physics 49 % (C) and 4–5 % (V) worse than the commit before them at a positive reversal cosine: not a
  reversal, a commit whose merit was too low. On a coarse loss grid one window's motion changes the rasterised
  mass by less than the tolerance, so the seeded solve passes the marginal test in its first block with the
  reference's potentials: its value is then the linearisation of the transport about the window's start, lower
  than the true value by the curvature term; the window is optimised on that, its commit is scored on it, and the
  next windows, seeded where it ended, are scored against it and rejected. The same mechanism, weaker, is the
  error measured on the 300k dragon (the difference between two candidates off by up to 2e-4 of the value, 10–20 %
  of the difference). Predictions: silhouette band, median, thin, windows and the bunny's silhouette confirmed;
  "none lower by more than 0.003" refuted by C (by 0.0001, for the reason above); both time predictions refuted.
  Also measured for the cost (`ot_ladder_probe.py` on d21): of a cold value call's 110–216 blocks the cross
  problem takes 81–189 and the self problem 25–27; at the end state the cross ladder spends 1–3 blocks at each of
  the twelve upper levels and 16, 34, 42, 27 at the last four. A fixed one or two blocks a level instead of
  convergence costs more in total (199–411) and usually fails the budget. The self problem started at the blur
  from zero duals takes 2 blocks with the same value and gradient to every printed digit (its plan is local);
  the cross problem started there does not converge.
- **D24, a detached group measured against the material around it (pre-registered 2026-10-03 01:52 CDT, launched
  01:51 after `tests/test_layer_relax.py` passed, exit 0; repo_r39 = repo_r36 + `window/layer.py`; `tmp/d24.sh`;
  `scripts/probes/settled/layer_groups_probe.py`; `output/gpu/d24`).** D21's reading (3): the layer's normal and
  the relaxation's plane are taken from a particle's 24–32 nearest neighbours, and for the members of a detached
  group those are the group. One change to that definition, nothing else (the cold Sinkhorn solver, 96 px from
  the start, 40 attempts, term dump with the channel record): particles are connected when nearer than one layer
  spacing; the body is the largest connected set; for a member of any other set of 2–512 particles the asymmetry's
  32 neighbours and the relaxation's 24 layer neighbours are the nearest particles that are not members of its
  own set, as they already are for a detached single particle. No term, weight, switch or schedule is added; the
  objective is untouched.
  On d21's states the definition gives (members beyond 2 spacings of the target, windows 20–38): on the layer
  82–89 % (was 54–66 %); the normal's cosine with the direction to the target −0.70 to −0.74 (was −0.39 to
  −0.65); the relaxation residual along the normal +0.4 to +0.7 spacings in the median and +2.5 to +5.4 at the
  90th percentile (was 0.0 and +0.2 to +0.4); no row without weight; 0.16–0.24 s a window against 0.07. On the
  4600–7500 group members within 2 spacings of the target (the sparse cover of thin features) the residual
  becomes +0.2 in the median and +1.1 to +1.9 at the 90th percentile (was +0.02 and +0.1 to +0.3): the change
  also pulls beads that sit beyond the connected material of a thin feature back toward it.
  Predictions. At the end no detached component of five or more rendered particles beyond one loss cell (d20: one
  of 38; d21: 62 and 31); detached rendered particles beyond the berth under 100 (179, 234); the peak of
  detached rendered particles at a window end under 12 000 (20 307). The cost, from the bead figures above:
  world-thin up by at most 0.5 point (0.5–0.8 % → at most 1.3 %); silhouette IoU within 0.002 of the cold runs'
  0.9835–0.9842. Fail: a clump of five or more beyond one loss cell remains; or world-thin above 1.5 %, silhouette
  below 0.980, a guard, more than three null windows, end kinetic record above 1e-2 (tips retracting and
  regrowing). Then the 4K frames of the mouth and the horn tips beside d20's, before anything is adopted.
  **Result (2026-10-03 02:10 CDT, one run): the clumps are gone, the change fails.** 854 s; 29 commits of 33
  attempts: one null window (16) and three consecutive rejected candidates (31–33: selection merit up 9–13 % with
  the physics down 4–6 %, the silhouette loss 1.0e-3 → 1.5e-3 in one window, reversal −0.31), so the run stops at
  window 33 and delivers 1120 frames; silhouette IoU 0.9824, world-thin 0.70 %, thin 18.9 %, chamfer 0.0594, holes
  0.02 %, last kinetic record 6.5e-3; λ 0.467, g_share 0.92; no guard. Detached rendered particles at the window
  ends (d20 in brackets): raw 160 20 452 (20 307), 480 4532 (4384), 960 1761 (2844), 1120 1714 (2506); beyond one
  loss cell 8387, 447, 16, 4 (8718, 615, 86, 54); beyond one loss cell and dense 0 from raw 800 on (25–47); in
  the near band 83 at the end (168); the near band's far record 1.5e-5–2.6e-5 against 1.1e-4. In the 4K frames
  the mouth is empty at raw 480 and at the end (d20: the ring at 480, the pale blob at the end).
  What fails: (1) four windows without a commit (the criterion was at most three) and the early stop; (2) a new
  defect: 277 detached particles within 1.1 spacings of the target at the top of the head collapsed onto a curve
  (8th-neighbour distance 0.01 of the coverage radius; pixel (2056, 200)), drawn as a bright line on a horn, and
  the horn tips blunter than d20's feathered ones. Measured against a small convex neighbourhood (the connected
  tip of a thin feature), every member of a bead group is projected onto planes through the same few neighbours,
  window after window. Predictions: the clumps and the count beyond the berth confirmed (87 < 100); world-thin and
  silhouette inside their bands; the mid-morph peak refuted (unchanged: the change does not act on the transit
  webs); the stop and the collapse not predicted.
  Reading: the cause named in D21 is the cause (with the neighbourhood corrected no clump forms or survives
  beyond a loss cell), but the relaxation is the wrong channel for the correction: it does not read the
  objective, so it also drags the beads that the objective holds on thin features, and the two fight until the
  selection rejects. The part of the change that reads the objective is u's direction.
- **D22 and D23, the Sinkhorn solve started from the window's own reference (runtime; the formulation untouched;
  entry written 2026-10-03 01:47 CDT, after D22 was read and before D23's result; the solver and its probe
  `ot_seed_probe.py` are kept on the server in repo_r38 and repo_r41, not in the repository (dropped in D28);
  `tmp/d22.sh`, `tmp/d23.sh`; `output/gpu/d20/ot_seed_*.txt`, `output/gpu/d22`, `d23`).** A
  window attempt of the 300k dragon costs 24.6 s: start 3 s, eight gradients 11 s, line search 7 s, commit 1.7 s
  (d20). Every evaluation solves two entropic problems from zero duals down a 15-level ε ladder, each level to the
  tolerance.
  Measured on d20's states (the seed = a window's start state, the evaluated state = its committed end, a second
  candidate two steps before it), 85³ loss grid, tolerance 1e-3: a cold value call takes 109–200 four-sweep blocks
  (0.34–0.67 s); started at the blur from the window start's potentials it takes 11–41 blocks from window 16
  (0.04–0.14 s) and more than the cold solve before window 12 (213–497 blocks, the far windows not converging in
  the sweep budget). Against a solve converged to 1e-6 (3800–8000 blocks): the cold solver's value is low by 0.1 %
  (window 8), 1.3 % (window 20), 2.9 % (window 39) and its gradient off by 4 %, 14 %, 21 %; the seeded one's by
  0.05 %, 1.0 %, 2.7 % and 2.5 %, 12 %, 20.5 %: at the production tolerance both are the same class, the seeded
  slightly nearer. The difference between the two candidates (what a line search compares; true size 1e-3–2e-3
  of the value): cold errs by at most 5.5e-5 of the value, seeded by up to 2.1e-4, always toward a smaller gain
  for the state farther from the seed.
  D22, first form (repo_r37 as it was at 01:29 CDT: each solve tries the seed for as many sweeps as the reference's
  cold solve took, then restarts cold), the 300k dragon alone on GPU 3, launched 01:29 CDT without an entry (my
  estimate before the run was 11.6 min): 821.7 s of simulation (13.7 min; process 889 s), 39 commits and one null
  window, silhouette IoU 0.9818, world-thin 0.73 %, thin 19.5 %, chamfer 0.0595, holes 0.007 %, last kinetic
  record 2.3e-3, no guard; λ 0.468, g_share 0.93. The cold runs of the same recipe: 981 s, 0.9835, 0.77 % (d20);
  1015 s with the term dump, 0.9842, 0.57 % (d21); 16.4 min, 0.9841, 0.53 % (D19). The estimate missed by two
  minutes: before window 12 every solve pays the failed attempt (1.5–1.7 times cold); and in the windows between,
  values from the two paths (they differ by 1e-3–3e-3 of the value) are compared inside one line search. Not kept.
  D23, second form (repo_r38, `GridSinkhornLoss.seed`): one mode per window. At a window's start its state is
  solved cold (the reference); the previous window's reference potentials are then tried on this state at the
  blur, with the attempt given up as soon as the residual's decay predicts more sweeps than the cold solve took;
  if that converges, a state one window away is within reach of level 0 and every solve of the window starts from
  the window's reference, else every solve is cold. A value depends on its own state and the window's start,
  never on earlier trials. On d20's states: windows 1–11 cold with identical values, 12–39 seeded; value-call
  seconds over the run 0.34 of cold; the two seed calls 0.4–1.1 s a window. Launched 01:45 CDT, the 300k dragon
  alone on GPU 1. Prediction: 10.5–12 minutes of simulation; silhouette IoU ≥ 0.9815 (the cold runs' 0.9835–0.9842
  less the run-to-run 0.002), world-thin 0.5–0.8 %. For the 15-minute goal: process wall time ≤ 15 min with
  silhouette ≥ 0.982 and world-thin ≤ 0.8 %; before adoption three repeats, the 40k gallery and the 300k bunny
  against the cold solver (silhouette within ± 0.002 on at least 15 of 19, none lower by more than 0.003).
  **D23 result (2026-10-03 01:59 CDT, one run).** 724 s of simulation (12.1 minutes; the process 791 s, 13.2
  minutes), 40 commits of 40 attempts, no null window, no guard; silhouette IoU 0.9844, world-thin 0.73 %, thin
  18.8 %, chamfer 0.0592, holes 0.02 %, last kinetic record 7.0e-4; λ 0.468, g_share 0.93. A window costs
  10.7 s from window 20 on (gradients 6.7 s, line search 2.4 s, start 1.1 s, commit 0.35 s) against 24.6 s cold;
  the first twelve windows are solved cold and cost what they did. Detached rendered particles at the end 2439
  (163 in the near band, 13 beyond one loss cell; d20: 2301, 141, 38). Predictions: the time at the edge (12.1
  against 10.5–12); silhouette and world-thin confirmed. The 15-minute criterion is met on this run (13.2 minutes
  of process, 0.9844, 0.73 %); two repeats launched 01:59 on GPU 1 (`dragon_b`, `dragon_c`), D25 for the gallery.
- **D20 and D21, the floating Gaussians of the 300k dragon traced to particles and to what moves them
  (diagnostics; entry written 2026-10-03 01:47 CDT after the runs: d20 launched about 00:34, d21 01:08 CDT;
  `tmp/d20.sh`, `tmp/d21.sh`; `scripts/probes/settled/clump_probe.py`, `detached_probe.py`,
  `floater_channels.py`, `floater_fate.py`; `output/gpu/d20`, `d21`).** The 16-minute dragon again (repo_r35b, 96 px
  from the start, 40 attempts) with the archive and every term's per-particle gradient kept (d20), then on
  repo_r36, whose dump also holds each window's displacement per channel (MPM advection = the sum of dt v; the u
  control; the rest = relaxation on the layer or bond projection), the layer mask and normal, the control size and
  det F (d21; `telemetry.channel_record`, diagnostic only). d20: 981 s, silhouette 0.9835, world-thin 0.77 %;
  d21: 1015 s, 0.9842, 0.57 %, 39 commits and one null window; λ 0.468 and g_share 0.93–0.94 in both (no λ = 0
  twin here).
  Counting (connectivity: single linkage at 2 target spacings, the body = the largest component; rendered =
  live support above zero). d20, detached rendered particles at each window end: 20 307 at raw frame 160 (the
  peak), 10 719 at 240, 4384 at 480, 2844 at 960, 2301 at 1600. Of the last 2301: 2122 lie within the sampling
  berth of the target (1.97 spacings: a sparse cover of thin target features, not in mid-air), 141 in the near
  band, 38 beyond one loss cell. The 38 are one clump in the open mouth: 4K pixel (2319, 697), world (0.66, 2.46,
  3.41), 6.6 spacings from the target (1.49 loss cells; the opposite jaw at 10.2), 6.3 particle spacings from the
  body, 8th-neighbour distance 0.26 of the coverage radius. d21 ends with 234 beyond the berth in 53 components
  (90 beyond one loss cell): 62 particles at pixel (1653, 1524), 7.2 spacings out, density ratio 0.34; 31 at
  (2049, 379), 6.5 out, 0.25; the mouth holds four. The place changes from run to run, the kind does not.
  The mouth clump's history (d20): material from the source interior, in the gap from window 4. Windows 4–12:
  isolated (density ratio 2.0), the spray cleanup on for 95–100 % of it with an inward pull of 0.5–1.1 (units of
  the all-particle rms gradient), distance 5.8–6.7 spacings throughout. Windows 12–24: its rms radius falls
  4.2 → 1.8 spacings and the density ratio 1.74 → 0.30; the isolation gate closes (spray on 61 % at window 14,
  3 % at 28, then none); the near band is on for at most 16 % and for none from window 32 (beyond one loss cell);
  the transport pulls coherently at 0.15–0.26; 88–100 % of the members' relaxation weights are on fellow members
  and the layer normal's cosine with the direction to the target is −0.2 to +0.2. It ends where it was at window 6.
  Which channel does what (d21). Detached material beyond the berth at a window's start, by the size of its
  component then, to the end of the run (share within the berth at the end; displacement toward the target by
  advection, u, rest, in target spacings):
  window 4 (14 044 particles): 96–97 % arrive in every size class; advection +11.2 to +12.7, u +0.3–0.4, rest
  +0.02–0.13. Window 12 (1881): singles 86 %, +1.6, +1.15, +1.09; groups of 2–4: 87 %, +1.8, +1.1, +0.41; 5–23:
  90 %, +1.8, +1.1, +0.09; 24 or more: 96 %, +1.8, +0.7, −0.04. Window 24 (328): singles 59 %, +0.07, +0.77,
  +0.51; 2–4: 47 %, +0.30, +0.78, −0.05; 5–23: 51 %, +0.21, +0.90, −0.05; 24 or more: 26 %, +0.09, +0.38, −0.16.
  The share of a particle's 32 nearest neighbours that are members of its own component: 0, 6, 23, 46–67 %.
  On the two large clumps of d21 from window 8: the contraction comes from u (0.04–0.15 spacings a window; the
  62-particle clump's radius 3.7 → 1.8) and from the rest (0.05–0.15 on the 31-particle one), not from advection
  (± 0.03); advection moves their centres away from the target by 0.1–0.5 a window in the last third while the
  control on them is 2–11 times the all-particle median; det F stays 0.98–1.01; the decoupling flag is never set
  (the fragment test is occupancy dilated by one MPM cell, 26-connected: it needs three empty cells, 0.9 wu = 26
  target spacings). The accepted step is 0.65e-3 to 3.2e-3 from window 2 on (0.02 at window 0): u can move a
  particle by at most eight steps a window, about 0.35 spacings.
  Reading. (1) Detached material is a by-product of the bulk morph and 96–97 % of it is carried in by advection
  while the body still flows (to window 8). (2) Once the body has settled, advection delivers nothing to what is
  left: it rides the body's grid velocity, and the control spent on it does not bring it in. (3) What still moves
  it is the position channel, and that channel is defined against a particle's 24–32 nearest neighbours: a single
  particle's neighbours are the body's surface (the relaxation brings it +1.1 spacings, u +1.2), a group's
  neighbours are the group itself: the relaxation holds it where it is and u, along normals that radiate from
  the group's own centre, contracts it. (4) The contraction makes the group dense: the isolation gate shuts the
  spray cleanup, beyond one loss cell the near band is not defined, and the display draws it at full opacity (it
  drew nothing while the group was sparse). What the video shows is step (4); the defect is step (3).
  Not decided here: which definition changes. Any re-coupling that does not read the objective (the relaxation
  measured against the body instead of the group; the bond projection with a particle-scale detachment test)
  also acts on the 2122 detached particles that sit on thin target features.
- **D19, the 15-minute dragon and whether 96 px from the start holds beyond it (pre-registered 2026-10-02 22:48
  CDT, before launch; `tmp/d19c.sh`, `tmp/d19.sh`, `tmp/d19_eval.py`; `output/gpu/d19`).** D18: with the render at
  96 px from the start the 300k dragon is at 0.5 % world-thin by window 30–40, and the share rises afterwards
  while the merit falls. Two things at once, the formulation untouched, the records off (repo_r35 = the schedule
  as it is, repo_r35b = `render_res` 96, no coarse stage).
  (1) Confirmation, alone on GPU 0: the 300k dragon, repo_r35b, `--animations 40`. The window budget bounds the
  loop and the log interval only (checked in the code: nothing else reads it; with one resolution there is no
  c2f event), so 40 is forty window attempts, at most forty commits. Read: wall time of the run, silhouette IoU,
  world-thin, chamfer, the last window's kinetic record, and a 4K render with the target sample's. Prediction:
  13–15 minutes; world-thin 0.4–0.7 %; silhouette IoU 0.979–0.983, below the full run's 0.9837 (the silhouette
  loss at window 40 was 6.0e-4 against 3.4e-4 at the end).
  (2) Generalisation: the 40k gallery (19 meshes, seed 97) and the 300k bunny, repo_r35 against repo_r35b, each
  to its own stop; silhouette IoU, thin, chamfer, the last kinetic record and the end jitter, committed windows,
  minutes. Decision rule fixed before the run: the gallery the same or better with fewer windows → 96 px becomes
  the default and the coarse stage and its event go; the same quality at a little more time → still one
  resolution, 96; a regression at 40k (silhouette lower by more than 0.003 on five or more meshes, or the median
  thin share up by more than two points) → the event stays and the dependence on N has to be explained from data.
  Predictions: silhouette within ± 0.002 on at least 15 of 19 (median change within ± 0.001); thin lower on at
  least 12; fewer committed windows on at least 12 (one epoch instead of two); gallery time at most 1.2 times;
  the 300k bunny within ± 0.002 with no more windows.
  **Result (2026-10-02 23:32 CDT, 42 runs, no guard, no frozen run; `python3 tmp/d19_eval.py`).**
  (1) The 300k dragon, 96 px from the start, records off, 40 window attempts, alone on GPU 0: 16.4 minutes of
  simulation (17.1 for the process), 39 commits and one null window; silhouette IoU 0.9841, world-thin 0.53 %,
  thin share 17.7 %, chamfer 0.0593, holes 0.02 %, last kinetic record 8.9e-4, end jitter 8.3e-6; λ 0.468,
  g_share 0.94. Against the schedule as it is run to its own stop (45–54 min): 0.9836–0.9854, world-thin
  0.66–1.04 %, chamfer 0.0589–0.0591. In the 4K picture it is as crisp as the 45-minute run (share of head pixels
  with a strong gradient 4.8 against 4.6 %), with the horn tips a little more ragged and the mouth nearer the
  sample. Predictions: world-thin confirmed; the time missed (13–15 predicted): a window costs 24.6 s here against
  the whole-run mean of 19.8 s, because the first forty windows are the expensive ones (D10: the Sinkhorn sweeps
  peak in windows 11–35); the silhouette is better than predicted (0.979–0.983).
  (2) The 40k gallery, 96 px from the start against the schedule as it is: silhouette IoU higher on 13 of 19,
  median +0.0007, within ± 0.002 on 18 (beast +0.0027), lower by more than 0.003 on none (the largest falls:
  armadilo −0.0013, nefertiti −0.0007); thin share lower on 12, median −0.3 points; chamfer median unchanged;
  committed windows fewer on 18 of 19 (armadilo 50 → 67), median 48 → 33; minutes summed 90 → 67. The last
  kinetic record is higher under 96 px on most meshes (beast 3.2e-3 against 1.1e-4, cow 1.4e-3 against 1.1e-4,
  spot 1.9e-3 against 2.5e-4): the single epoch stops with a little more residual motion; end jitter the same
  order (3e-6 to 1.3e-5). The 300k bunny: 0.9856 against 0.9874, world-thin 0.00 both, 40 against 69 windows, 8.7
  against 14.0 minutes. Every prediction of (2) confirmed (the bunny at the edge of its band, −0.0018).
  By the rule fixed before the run (the gallery the same or better, fewer windows), 96 px from the start replaces
  the coarse stage and its event. One run an arm; the run-to-run spread is about ± 0.002 in silhouette and ± 1 point
  in thin.
- **D17 and D18, the runtime: the records out of the production path, and the fine render from the start
  (pre-registered 2026-10-02 18:09 CDT, before launch; repo_r35 = HEAD with the per-window diagnostic records and
  the steering telemetry behind the existing `work_telemetry` flag, now off by default and exposed as
  `--telemetry`; the suite passes, 269 (one flaky contract test passed 3 of 3 alone and in the second full run);
  repo_r35b = repo_r35 with `render_res` 96, so there is no coarse stage and no c2f event).** The formulation is
  untouched. Three 300k dragon runs to their own stop, seed 97, archives deleted (`output/gpu/d18`): (a) repo_r35
  with `--telemetry`, the schedule as it is (64 px, then 96 px after the coarse stop) — the reference with the
  per-window records (world-thin, E, the accepted step, λ per window); (b) repo_r35b with `--telemetry`, 96 px
  from the start; (c) repo_r35 without `--telemetry`, the schedule as it is — the production path. D17 reads (c)
  against (a): seconds per window attempt and the end state. D18 reads (b) against (a): world-thin, E, the
  accepted step and λ by window and by wall time. Predictions. D17: (c) is 1.5–2.5 s a window faster (8–12 %)
  and its end state lies within the run-to-run spread of (a) (silhouette ± 0.002, windows within 20 %). D18: the
  gain after the c2f event in the earlier run (world-thin 1.3 → 0.6 % after window 104) came with the epoch
  reset, which restored the accepted step four- to fivefold; so (b), with one epoch, reaches about 1.2 % by
  window 40 as (a) does, then settles as the step anneals and stops between windows 60 and 110 at a world-thin of
  0.8–1.2 %, above (a)'s 0.6, in at most 35 minutes. If instead (b) reaches 0.7 % or better by 25 minutes, the
  coarse stage is unnecessary at 300k and can go. Either way, "96 px from the start is faster" would not by itself
  mean the resolution is the cause (λ's calibration and the path change too); if the reading is unclear, a
  control with the schedule as it is but the epoch reset alone follows.
  **Result (runs done 18:56–19:05 CDT, read 2026-10-02 22:45; `python3 tmp/d18_eval.py`).**
  D17, the records: a window attempt costs 19.8 s without them against 23.6 s with them (−3.8 s, 16 %: the commit
  phase 2.5 → 0.8 s, the untimed remainder 2.1 → 1.5 s, the line search 6.2 → 5.4 s, the start 2.6 → 2.2 s). The
  end state is inside the spread: silhouette IoU 0.9854 against 0.9836, 138 against 133 committed windows, the
  c2f event at attempt 88 against 89; 46.5 against 53.6 minutes. Prediction confirmed, the saving larger than
  predicted (1.5–2.5 s).
  D18, 96 px from the start (b) against the schedule as it is (a), both with the records. World-thin by committed
  window, a / b: window 10 (4 min) 3.27 / 2.89 %; 20 (9 min) 1.99 / 0.79; 30 (14 min) 1.68 / 0.51; 40 (18 min)
  1.54 / 0.48; 60 (26 min) 1.57 / 0.69; 80 (34 min) 1.83 / 0.88; at the end 1.03 (133 windows, 53.6 min) / 0.74
  (110 windows, 44.9 min; the end metric 0.69). (b) passes 1.0 % at 9.2 min and 0.7 % at 10.2 min; (a) passes 1.2 %
  only at 43.6 min, after its c2f event (attempt 89, 37.6 min), and never passes 1.0 %. Silhouette IoU at the end
  0.9836 (a) and 0.9837 (b). The accepted step under (b) is two to three times (a)'s through window 30 (2.6e-3
  against 1.2e-3 at window 20); λ of the first window 0.468 (b) against 0.396, 0.281 after (a)'s event. (b) has
  no second epoch and stops on three rejected candidates at attempt 118.
  My prediction is refuted: without the epoch reset (b) does not settle at 0.8–1.2 %, it is at 0.5 % by window 30
  and better than (a)'s end at a quarter of (a)'s time. The registered alternative holds (0.7 % by 25 minutes: at
  10 minutes), so the 64-px stage is not needed for the 300k dragon; it holds the thin parts at about 1.5 % for
  thirty minutes. Not shown: that the resolution alone is the cause (λ and the path differ too).
  In both runs the world-thin share is lowest about window 40 and then rises while the merit and the transport
  energy go on falling ((b): 0.48 % at window 40, 0.88–0.90 % at windows 80–90, with E 1.26e-3 → 6.7e-4): the long
  tail buys merit, not thin coverage.
- **D16b, the same comparison read frame by frame at 300k (diagnostic, pre-registered 2026-10-02 17:01 CDT, before
  launch).** D16's bunny and dragon were run again at 40k with 4K renders and read frame by frame at equal raw
  frames (`output/gpu/d16r`, `tmp/pair_sheets.py`). With the relaxation off the bunny's upright ear has a ragged
  edge in raw frames 264–936 and a lump on its left side from raw frame 408 to the end (11 600), which neither the
  target sample's render nor the ON arm shows; the layer's rms roughness of D16 does not see a local lump. At 40k
  every picture is a blob (spacing 0.106 wu; the user: the 40k pictures carry little), so the question is asked
  again at 300k: bunny and dragon, the frozen recipe, ON to its own stop and OFF with a window budget (80 for the
  bunny, 150 for the dragon: OFF does not stop), each with the 4K render, the target sample's render and the layer
  roughness; the bunny's archives are kept and OFF's term gradients dumped, so that a lump can be traced to its
  particles (`lump_probe.py`: the motion a particle does not share with its neighbours, along the normal, is the
  position channels'). `output/gpu/d17`. Predictions: silhouette IoU OFF at or above ON on both; the ≤ 4-spacing
  offset under OFF about the target sample's; in the pictures OFF shows ragged ear edges in the early windows and
  at least one local bump of two spacings or more on the bunny's ears at the end, as at 40k.
  **Result (2026-10-02 17:55 CDT; `output/gpu/d17`, sheets in `sheets_bunny`, `sheets_dragon`, forensics in
  `forensics_bunny_*`, `lump_bunny_off.txt`).** Bunny: ON 0.9877 (58 windows, 12.1 min), OFF 0.9875 (78 of 80,
  15.7 min); thin share 12.8 against 14.2 %, world-thin 0.02 against 0.00 %; outer target points with no particle
  within two spacings 0.90 against 0.48 %; front particles drawn with an enlarged disc 0.9 against 2.7 %; the
  residual the relaxation removes 0.111 against 0.351 spacings; λ 0.249 / 0.209, g_share 0.88 / 0.87. Dragon: ON
  0.9838 (122 windows, 45 min), OFF 0.9882 (129 of 150, 47 min); thin 18.8 against 17.4 %, world-thin 0.55
  against 0.01 %, holes 0.05 against 0.01 %; λ 0.396 / 0.497, g_share 0.94 / 0.96.
  Frame by frame, at equal raw frames. Bunny: both arms carry the web between the ears (raw 204) and the tuft at
  the notch (raw 396); from raw 600 to the end OFF keeps a translucent fuzz on the upright ear's left edge and
  tip and along the slanted ear's upper edge, where ON's edges are clean and like the target sample's; ON has
  ripples on the slanted ear's underside that OFF has less of; no lump at 300k (the 40k "lump" was particles
  within one spacing of the target, 37 px at 40k, and not meaningful). Dragon: OFF's head is nearer the target
  sample's (the mouth open, the horns banded and feathered as in the sample); ON's is doughier with the mouth
  closed into a hump; OFF's horns carry fuzz. No oscillation in any tail.
  The fuzz, traced (bunny OFF, the upright ear's left edge, 4K box 2000–2090 × 300–520): 746 particles project
  there, 53 of them more than one spacing from the target sample (median 1.17, at most 1.6; none beyond 3), from
  the source's surface (depth 1.9 spacings). They are pushed out in windows 6–12 (raw 240–520), when the u gate
  opens (u_gate 0.92 → 1.0): their motion along the normal that their 24 neighbours do not share is +0.02 to
  +0.05 spacings a window (90th percentile +0.15 to +0.33) against +0.02 for the whole layer; the grid moves a
  neighbourhood together, so this is the position channel, u. From window 12 they sit at 1.1–1.2 spacings:
  inside the near band's berth (the band is on for 9–28 % of them), with no other term pulling (transport,
  surface and render within ±0.1 of the all-particle rms, the render slightly outward). With the relaxation on
  the same offsets are removed inside the window. The fuzz in the picture is the display rule drawing the
  resulting sparse front particles enlarged and translucent (2.7 % of front particles against 0.9 %).
  Predictions: silhouette OFF at or above ON on both: confirmed (equal on the bunny, +0.0044 on the dragon);
  OFF's ≤ 4-spacing offset about the target sample's: bunny 0.214 and dragon 0.218 against the samples' 0.387
  and 0.203 (the 300k fit of the mesh is coarser; read with care); ragged ear edges early: confirmed; a local
  bump of two spacings or more: refuted, the defect is a fringe of one spacing, not a bump.
  Reading, corrected after the user looked (18:10 CDT; the first reading called OFF "nearer the target" on the
  dragon from the silhouette IoU and the open mouth). Looked at properly, region by region at native resolution
  (`sheets_dragon_regions`, `tmp/triple_sheet.py`: head, tail and paw, feet), OFF is the blurrier picture on the
  dragon as on the bunny: feathered outlines on the horns, jaw, tail and claws, mottled shading where the
  first layer is uneven, and fewer sharp pixels than the target sample's own render (share of object pixels with
  a strong luminance gradient, head: ON 4.6 %, target sample 3.8 %, OFF 3.5 %). ON is the crispest of the three
  (a regular first layer: plane residual 0.21 spacings against the sample's 0.35 and OFF's 0.44), at the cost of
  rounder forms (the mouth closed into a hump, the horns as knobs). So the two rulers disagree: the silhouette
  IoU and the thin share favour OFF (+0.0044, 0.55 → 0.01 %), the picture favours ON, because the display rule
  turns an uneven or fringed first layer into enlarged translucent discs. What the relaxation really removes is
  u's per-particle normal offset and the layer's unevenness, the things it was introduced for; the relief it
  erases (11 spacings and less, 0.6 wu at 300k) is below what the sample and the display show anyway (D12). For
  the current paper (300k, this renderer) the relaxation stays; a redefinition, if ever, belongs on u's side
  (what u may inject), R5's direction (2026-09-30), with the slow tail (D5) as its open cost.
- **D15, the closed loop on ridged targets, with the layer relaxation on and off (diagnostic, pre-registered
  2026-10-02 12:36 CDT, before launch [first written as 12:37, a minute ahead of the server clock at the launch:
  corrected]; `make_ridge_slab.py`, `ridge_closed_loop.py`, `tmp/d15.sh`).** D14, open
  loop: the rollout can carry relief down to the particles' sampling limit, and the outer-layer relaxation removes
  relief of 6 spacings and less. The question now: does the objective, window after window, rebuild what the
  relaxation removes? Targets: a box of the gallery body's volume (4.454 × 2.4 × 4.454 wu) whose top face carries
  ridges y = A sin(2πx/λ), A = λ/8, with λ = 22, 11 and 5.6 particle spacings; at 300k (λ = 1.19, 0.60, 0.30 wu =
  3.9, 2.0, 1.0 cells) and at 40k (2.33, 1.17, 0.59 wu). Two arms, the frozen recipe from the sphere, seed 97: ON
  = repo_r34; OFF = repo_r34x, a copy with the relaxation's rate set to zero in `window/setup.py` (the u channel
  stays; a diagnostic arm only, never a candidate; its suite was not run, the relaxation tests would fail by
  construction). Per committed window, at the end of the controlled half and at the window's end: the share of
  the target sample's relief the body's top layer carries in phase (a least-squares sinusoid at λ with a
  quadratic surface), its amplitude whatever the phase, and the rms height error; the target sample's own relief
  against the mesh's; silhouette IoU, λ of the first window and g_share. Archives deleted after the probe.
  Reading agreed before the run: ON holds the 5.6-spacing ridge → the relaxation is not a hard limit in closed
  loop; ON fails and OFF succeeds → the relaxation is the cause; both fail → the objective, the control or the
  target sample is. Predictions (in-phase share at the window's end, mean of the last ten windows): 22 spacings:
  at least 0.8 in both arms; 11 spacings: 0.3–0.6 ON and at least 0.15 more OFF, with ON's share at the end of the
  controlled half above its share at the window's end; 5.6 spacings: under 0.3 in both arms, because the objective
  hardly sees it (the relief is 0.7 spacings, inside the proximity threshold of 1.53 and the berth of 1.97; at
  300k a loss cell holds half a wavelength; the 96-pixel render has four to five pixels a wavelength). The same
  in spacings at 40k.
  **Result (2026-10-02 13:57 CDT; `output/gpu/d15/*.ridge.txt`).** Two failures on the way, recorded: the 12:36
  launch hung for twenty minutes in the sampler (the first ridge mesh had sliver triangles 0.006 × 2.4 wu, which
  the voxeliser subdivides without end); the meshes were rebuilt with uniform triangles and the runs restarted at
  12:58. The 300k OFF runs at 22 and 11 spacings did not stop by themselves (240 and 180 windows and going, an
  archive of 43 GB each at the 300-window budget): stopped at 13:41 and run again with an 80-window budget.
  In-phase share of the target sample's relief at the window's end, mean of the last ten windows (ON / OFF):
  40k: 22 spacings 0.98 / 0.98; 11 spacings 0.54 / 0.97; 5.6 spacings 0.00 / 0.69.
  300k: 22 spacings 0.91 / 1.00; 11 spacings 0.18 / 0.66 (window 76, still rising: 0.52, 0.58, 0.63 at windows 20,
  40, 60); 5.6 spacings 0.04 / 0.70 (31 windows: 0.42, 0.58, 0.65, 0.71 at windows 10, 15, 20, 25).
  The target sample carries 0.90–0.99 of the mesh's relief at 22 spacings, 0.99 (40k) and 0.69 (300k) at 11, and
  0.53–0.61 at 5.6. ON, end of the controlled half against the window's end: 0.56 / 0.54 (40k, 11 spacings), 0.20 /
  0.18 (300k, 11 spacings; 0.26 / 0.21 at window 10): the relief is not built and then lost in the release, it
  sits at a low balance. OFF the two are equal.
  The whole shape, OFF against ON: at 40k silhouette IoU 0.9895–0.9902 against 0.9740–0.9766 and world-thin
  2.1–3.2 % against 7.7–9.3 %; at 300k silhouette IoU 0.9900, 0.9873, 0.9875 against 0.9894, 0.9874, 0.9845
  (world-thin 0 in every 300k run). Windows: ON 20–30 at 40k and 46–57 at 300k; OFF 119, 251, 222 at 40k and 31
  (5.6 spacings) or no stop within 180–240 windows at 300k. No guard in any run. λ of the first window 0.075–0.103,
  median g_share 0.79–0.92 in both arms.
  Predictions: 22 spacings confirmed. 11 spacings: ON inside the band at 40k (0.54), below it at 300k (0.18); OFF
  more than 0.15 above ON: confirmed; ON's share at the end of the controlled half above the window's end:
  confirmed, by 0.02. 5.6 spacings: ON under 0.3 confirmed; OFF under 0.3 refuted (0.69 and 0.70): the objective
  does see a 5.6-spacing ridge of 0.4 spacings' height and the control does make it.
  Reading, by the table agreed before the run: ON fails and OFF succeeds at 5.6 spacings at both N and at 11
  spacings, so the limit on fine relief in the closed loop is the outer-layer relaxation, not the objective, the
  control or the grid. With it off the same recipe also reaches a better silhouette and thin coverage on this
  box, and takes 5–10 times the windows or does not stop. One target family, one seed, one run an arm; what the
  relaxation was introduced for (the lumps of the u channel on the gallery's shapes) was not measured here.
  Second run of the six 300k arms, with pictures (launched 13:59, done 14:33; OFF with the 80-window budget;
  `output/gpu/d15r`: 4K frames and videos of every arm and of the three target samples, camera elevation 35):
  in-phase share ON / OFF 0.89 / 1.01 (22 spacings), 0.18 / 0.66 (11), 0.02 / 0.58 (5.6), against the first run's
  0.91 / 1.00, 0.18 / 0.66, 0.04 / 0.70. In the 4K pictures the 22-spacing ridges are plain in the target sample
  and in both arms; the 11-spacing ridges show as a scalloped edge and faint streaks in the target sample and in
  OFF, and as fewer, rounder scallops in ON; the 5.6-spacing ridges are hardly visible in any of the three,
  target sample included (the display rule's normals are blurred over 3 spacings, D12). ON rounds the box's
  edges and corners more than OFF.
  The relaxation's definition against the literature (two passes, 2026-09-30 for R5 and 2026-10-02): it is a
  point-set fairing step (the Adamson–Alexa plane residual moved along the normal, Taubin's averaging without a
  pass band), not a particle regularisation. Every particle method read leaves the free surface's normal
  direction alone: SPH shifting is tangential or switched off at the surface (Lind et al. 2012; Khayyer et al.
  2017; Sun et al. 2017), MPM resampling stays away from it (Yue et al. 2015; Gao et al. 2017), and position-based
  fluids call their normal effect an artefact (Macklin & Müller 2013). The noise (2–4 spacings) and the features
  wanted (5–11 spacings) are adjacent bands, so no linear filter separates them sharply; a narrower stencil shifts
  the cut-off, Taubin's λ|μ amplifies the pass band at this scale, and a bilateral weight needs a live
  normalisation that broke the adjoint before. On record from 2026-09-29/30 on the earlier objective (D3b, D3c,
  R5): without the forward relaxation the 40k gallery's silhouette IoU rose on 18 of 19 meshes (+0.006–0.007) at
  three times the run length, with a tail that improves about 1 % a window (D5: the two controls share one step
  length).
- **M1, the frozen recipe under other materials (characterisation, pre-registered 2026-10-01 23:52 CDT, before
  launch; repo_r33 = repo_r32 with run flags `--young --poisson --assim --drag --f_ext --floor --floor_friction`, all
  defaulting to the frozen recipe; the suite passes, 269).** Nothing in the objective changes. One factor at a time
  around the frozen material (E 1.4e5, ν 0.2, assimilation 0.5, drag 0.9) at 40k, seed 97, on bunny, dragon,
  armadilo and bob; the base twice for the spread: E × 0.1, 0.3, 3, 7 (CFL 0.06, 0.10, 0.33, 0.48; the time step is
  fixed, so 7 is the stiffest the step carries); ν 0, 0.4, 0.45; assimilation 0, 0.25, 1; drag 0, 3
  (`output/gpu/m1`, `tmp/m1_eval.py`). "Reached" = silhouette IoU not more than 0.01 below the lower base run, no
  guard. Recorded: world-thin, committed windows, minutes, the kinetic record's peak and end, the released motion at
  the end, λ of the first window and the median g_share (the render's influence under each material). Predictions:
  (1) E: reached at 0.3, 1 and 3; at 0.1 reached with at least 1.5 times the windows (a unit of control makes a
  tenth of the stress, and a wave crosses 2.3 cells a window instead of 7.3 [figures corrected 23:55 from the
  run's own log, c = 13.4 wu/s at the dynamics mass; the registered text said 0.85 and 2.7, from the 300k report's
  unit-mass sound speed; the prediction is unchanged]); at 7 a guard fires or world-thin
  worsens on at least two meshes. (2) ν: reached at every level; at 0.45 world-thin worse than both base runs on at
  least two meshes. (3) Assimilation: at 0 not reached on at least three meshes (the body keeps the sphere as its
  rest shape and springs back in the released half); 0.25 and 1 reached. (4) Drag: reached at both; at 0 more
  windows than the base on at least three meshes.
  **Result (2026-10-02 00:55 CDT, 56 runs, no guard in any).** 47 of the 48 non-base runs reached the target.
  Silhouette IoU, in the order bunny / dragon / armadilo / bob; base runs 0.9751, 0.9753 / 0.9752, 0.9758 / 0.9728,
  0.9717 / 0.9831, 0.9826 (world-thin 9.2, 9.4 / 14.4, 13.8 / 10.3, 9.2 / 11.8, 10.5 %; windows 41, 41 / 60, 74 /
  68, 83 / 44, 31):
  E × 0.1: 0.9779 / 0.9746 / 0.9684 / 0.9834 (thin 12.0 / 14.0 / 14.6 / 18.3; windows 47 / 62 / 59 / 68; det F
  minimum 0.83 / 0.58 / 0.76 / 0.72 against 0.92–0.96 at the base);
  E × 0.3: 0.9761 / 0.9752 / **0.7775** / 0.9832 (thin 11.3 / 12.9 / 49 / 8.7); the armadilo froze after 13
  windows on the parked `domain` ejection (commit_invalid, start state dead, stray 0.38 %), the beast's failure;
  E × 3: 0.9757 / 0.9755 / 0.9719 / 0.9832 (thin 9.8 / 14.1 / 10.1 / 7.4);
  E × 7 (CFL 0.48–0.49): 0.9753 / 0.9758 / 0.9728 / 0.9829 (thin 8.9 / 11.8 / 7.9 / 6.1; windows 53 / 116 / 47 / 63);
  ν 0: 0.9762 / 0.9750 / 0.9709 / 0.9823 (thin 7.8 / 11.6 / 10.8 / 7.0); ν 0.4: 0.9749 / 0.9745 / 0.9718 / 0.9825
  (thin 8.1 / 15.3 / 13.2 / 10.5); ν 0.45: 0.9756 / 0.9747 / 0.9710 / 0.9826 (thin 9.5 / 14.4 / 11.6 / 9.6; windows
  62 / 103 / 108 / 65);
  assimilation 0: 0.9757 / 0.9761 / 0.9728 / 0.9838 (thin 8.8 / 10.9 / 9.3 / 9.6; windows 57 / 48 / 61 / 37); 0.25:
  0.9761 / 0.9755 / 0.9718 / 0.9831; 1: 0.9749 / 0.9743 / 0.9717 / 0.9833;
  drag 0: 0.9749 / 0.9757 / 0.9729 / 0.9834 (windows 45 / 71 / 74 / 41); drag 3: 0.9756 / 0.9760 / 0.9727 / 0.9828.
  Predictions: (1) E × 0.3 reached on three of four (the armadilo's ejection); E × 0.1 reached on all four, but
  with 1.5 times the windows only on bob: refuted; E × 7 neither fires a guard nor worsens the thin parts, it has
  the lowest world-thin of every mesh: refuted. World-thin falls as E rises on bunny, armadilo and bob (12.0 → 8.9,
  14.6 → 7.9, 18.3 → 6.1) and on the dragon from the base up (14.1 → 11.8). (2) Reached at every ν: confirmed;
  ν 0.45 is worse than both base runs on three meshes by 0.1–1.3 points, inside the run-to-run spread (±1.1): no
  finding; it costs windows (62 / 103 / 108 / 65). (3) Refuted: with no assimilation the morph reaches the target
  on all four, with world-thin at or below the base. The body's rest shape staying the sphere does not stop the
  control. (4) Reached at both drags: confirmed; more windows at drag 0 only on the bunny: refuted.
  The render's influence under each material: λ of the first window follows the stiffness (bunny 0.095, 0.18, 0.24,
  0.39, 0.61 for E × 0.1, 0.3, 1, 3, 7; dragon 0.27 → 0.95; armadilo 0.42 → 1.26; bob 0.28 → 1.01), because it is
  calibrated on the physics gradient; the median g_share stays 0.78–0.92 at every level. No render-off twin was run.
  The kinetic record's peak follows E (0.18–0.35 at × 0.1, 3.1–5.3 at × 7).
- **F1, the frozen recipe under external forces (feasibility, pre-registered 2026-10-01 23:57 CDT, before launch;
  repo_r34 = repo_r33 with the centre of mass and its velocity in the window record).** Nothing in the objective
  changes: the target, the cameras and the domain stay fixed in the world. The simulator already carries a uniform
  acceleration on the grid, a separating floor with friction and separating domain walls. 40k, seed 97, bunny and
  dragon (`output/gpu/f1`, `tmp/m1_eval.py f1`): (a) the floor alone (at the source's lowest point, the target moved
  to stand on it); (b) the floor with gravity at G = ρ g H / E = 0.02, 0.1 and 0.5 (g = 0.65, 3.2, 16 wu/s²; G is
  the strain the body's own weight makes); (c) G = 0.1 with floor friction 0.5; (d) no floor, a sideways
  acceleration of 3.2 wu/s² (the body's centre of mass must move: internal stress cannot change it, and the drag
  0.9 /s gives a terminal speed of 3.6 wu/s, which reaches the wall at 5.5 wu in about two seconds, 12 windows).
  Predictions: (a) reached on both (within 0.01 of M1's lower base run); (b) reached at 0.02; at 0.1 reached on the
  bunny with world-thin worse than both base runs (thin parts sag in the released half); at 0.5 not reached (each
  commit makes half the supporting elastic strain permanent, so the body creeps like a fluid and the released
  motion never falls); (c) as (b) at 0.1; (d) not reached: the selection merit worsens as the body drifts off the
  world-fixed target, the brake rejects, and the run stops within 10 windows with a silhouette IoU below 0.9; the
  released-motion and end-drift terms carry the centre-of-mass velocity, which no control can change. If (d) fails
  as predicted, the moving case needs the objective written in the body's frame (the target and the cameras carried
  by the centre of mass, the settling terms on the velocity relative to it): a change of definition for the user to
  decide, not made here.
  **Result (2026-10-02 00:17 CDT, launched 00:01 after the suite passed on repo_r34; the first full run failed one
  test, `test_an_unsolved_transport_cannot_initialize_an_accepted_commit`, with three other jobs on the GPU; it
  passed 3 of 3 alone on repo_r34 and repo_r33 and the second full run passed, 269: recorded as the known
  nondeterminism).** Silhouette IoU / world-thin / committed windows, bunny then dragon (M1's base: bunny 0.9751,
  0.9753 / 9.2, 9.4 % / 41; dragon 0.9752, 0.9758 / 14.4, 13.8 % / 60, 74):
  floor alone 0.9732 / 11.6 % / 17 and 0.9742 / 13.7 % / 16; G 0.02: 0.9768 / 10.0 % / 19 and 0.9754 / 13.3 % / 33;
  G 0.1: 0.8724 / 41 % / 5 and 0.8611 / 48 % / 7; G 0.5: 0.7196 / 2 and 0.6079 / 2; G 0.1 with friction 0.5:
  0.8720 / 5 and 0.8599 / 7; sideways acceleration: 0.8728 / 4 and 0.7955 / 5. No guard in any run. λ of the first
  window 0.18–0.25 (bunny) and 0.33–0.37 (dragon), as without forces; g_share falls with the force (0.73–0.83 at
  the floor and G 0.02, 0.55–0.66 at G 0.1, 0.32–0.36 at G 0.5).
  (a) Confirmed as registered (within 0.01), with a cause not predicted: the floor-standing target's centre of
  mass is 0.21 wu below the source's, and without an external force the centre of mass stays where it is (+0.01):
  the run stops at a merit of 0.0136 (bunny), seven times the G 0.02 run's 0.0018–0.0023, where gravity carries the
  centre of mass down to −0.21 and the floor holds it there. (b) 0.02 confirmed. 0.1 refuted on the bunny, 0.5
  confirmed, both by a mechanism other than the predicted creep: the centre of mass falls past the target's
  (−0.70 after four windows at G 0.1, velocity −1.06 wu/s; −0.78 after one at G 0.5, −4.15 wu/s; near free fall
  under the drag), the merit turns upward at window 3 (0.105 → 0.109 → 0.162), the brake rejects, and three
  rejections stop the run. A sphere standing on a point is not held up until it has flattened. (c) Confirmed:
  friction changes nothing (0.8720 against 0.8724). (d) Confirmed: the centre of mass moves +0.59 (bunny) and +0.88
  (dragon) in four to five windows at 1.6–1.9 wu/s, the merit turns upward and the run stops.
  Reading: the frozen recipe morphs under a weak body force with a support (G 0.02) and stops within a few windows
  whenever the centre of mass moves faster than the morph proceeds, because the target, the cameras and the
  settling terms are fixed in the world.
- **FV, the final validation of the formulation (pre-registered 2026-10-01 19:36 CDT, launched 19:37 after the suite passed
  on repo_r30 (269 passed); repo_r30 = repo_r29 with the three regularisers deleted from the code).** The objective is now eight terms: the Sinkhorn
  transport, the surface proximity and the residual drift of the released end (geometry, one scale ot_scale); the
  released motion (settling); the near band between the berth and one loss cell and the spray cleanup (local); the
  silhouette and the shading (render, weight λ calibrated once). Constants left: w_nn, w_dt with its isolation
  ramp, λ's calibration target, and the render's own (sil_k, w_hole, w_spray, w_pbr, the ambient). Runs: the 40k
  gallery twice (19 meshes, seed 97) and the 300k dragon and bunny to their own stop (`output/gpu/fv`,
  `tmp/fv_eval.py`). By the stop rule this is the last experiment on the formulation: it is frozen unless a new
  catastrophic failure appears, defined before the launch as any of: a mesh other than beast and C (their known
  defects) whose silhouette is below its lowest current value by more than 0.004 in both runs, or whose thin is
  above its highest current value by more than 3 points in both runs; a freeze or a stop while moving on a mesh
  the current runs do not show; a guard firing; a det F minimum below 0.5; the 300k dragon ending below silhouette
  0.981 or above 2.4 % world-thin (R13b: 0.9840–0.9846, 0.66–0.87 %), or the 300k bunny below 0.983. Anything
  smaller is recorded and not acted on. Predictions: no catastrophic failure; the gallery inside the current
  runs' range. Not part of this run: the PBR renders and the video (archives are not kept here; they are made from
  the frozen code on the meshes the user picks) and the gradient analysis beyond the records every run carries
  (the active-set and scale records of D9, D9b).
- **R14b, the three regularisers off together (pre-registered 2026-10-01 18:44 CDT, launched 18:44; no code change:
  `repo_r29x` = repo_r29 with the defaults w_ctrl, w_creg and w_jvol at zero).** Arm X against the current
  formulation, whose range now has four gallery runs: R13 a, b, B a and a second B run started with it. Runs: the
  40k gallery twice (19 meshes, seed 97) and the 300k dragon and bunny at 35 windows (`output/gpu/r14`, tags
  `r14X…`, `tmp/r14_eval.py`). Criteria as R14 for the three terms together (geometry, stability, det F minimum and
  quantiles, anisotropy, control size and roughness, surface roughness, 300k). Read against the measured noise of
  the criteria: arms C and G differ from the current code by 1e-10 and 1e-8 of the merit and still showed one
  jitter median 3 % over its bound, bob's thin at +5.2 and one beast freeze; a miss by X of that kind and size is
  noise, a miss beyond it (a mesh other than bob or beast outside its limit in both runs, det F minimum below half
  the current one, a guard firing, a 300k silhouette or world-thin outside its limit) keeps the terms. Prediction:
  X is inside the noise on every criterion; the three terms and their constants (w_ctrl, w_creg with creg_k, w_jvol)
  are then deleted. Render influence: unchanged channel; first-window λ and g_share reported.
- **Decisions, 2026-10-01 15:59 CDT (the user).** (1) The dense distance is deleted from the selection (R13, R13b);
  closed. (2) The near band keeps its definition and its scale; no normalisation by N and no gate on the transport's
  progress is added. D9b's finding is recorded as a known interaction, not a defect to fix: the near band's pull
  relative to the transport grows with the particle count and opposes the transport on the band's particles early
  in a run (more at 300k), which costs early windows; no degradation of the converged geometry was found at 40k,
  100k or 300k (the 300k dragon to its own stop: silhouette 0.984–0.985, world-thin 0.7–0.9 %). The method is not
  claimed to be resolution-invariant in its local correction. (3) Next is R14. (4) Stop rule: the terms that
  survive R14 get one final validation (the 40k gallery, the 300k bunny and dragon, the PBR renders and video, the
  gradient analysis); unless a new catastrophic failure appears there, the formulation is frozen. A ratio that
  looks odd is recorded, not turned into a new experiment.
- **R14, leave-one-out of the remaining legacy terms (pre-registered 2026-10-01 15:59 CDT, launched 16:00 after the suite passed on
  repo_r29 (269 passed); no change of behaviour in the base code: repo_r29 = repo_r28 with records of det F and anisotropy quantiles of the
  stored F and of the two control regularisers' raw values; each arm is a copy of repo_r29 with one default set to
  zero).** Arms: C, the control magnitude off (w_ctrl 1e-3 → 0); G, the control smoothness off (w_creg 100 → 0); J,
  the volume prior off (w_jvol 50 → 0); S, the spray cleanup off (w_dt 0.2 → 0). B is repo_r29 itself, run once for
  the new records. Runs per arm: the 40k gallery twice (19 meshes, seed 97) and the 300k dragon and bunny at 35
  windows (`output/gpu/r14`, `tmp/r14_eval.py`); against the current formulation (R13's two gallery runs and B;
  D9b's and B's 300k runs). A term is removable if, on the two-run means: silhouette difference median within
  ±0.002 and no mesh but beast below by more than 0.004; `thin_uncovered` median within ±1 point and no mesh worse
  by more than 3 beyond the arms' own run-to-run difference; end `kin`, tail jitter, rejected windows, committed
  windows and delivered frames (medians, totals) not beyond the three current runs' range by more than that range's
  width; no run stopping before 15 windows moving and no guard counts that the current runs do not show; what the
  term exists for does not degrade: for C and G the median |dFc| maximum, the control roughness and the surface
  roughness, for J the minimum det F and the 1 % and 99 % quantiles of det F and the anisotropy p90 and p99, for S
  `stray_max`, `stray_final`, `out_dt_frac` (each not beyond the current range by more than its width; det F
  minimum not below 0.5 of the current minimum); 300k: silhouette within ±0.002 and world-thin within ±1 point of
  the current runs, the transport energy at window 34 not above the current larger value by more than 30 %. A term
  that misses any of these stays. If two or more are removable, one combined arm (all of them off) is run before
  anything is deleted. Predictions: C removable (its value is 1e-3 wu times a squared increment bounded by the clip,
  far below every other term); G not removable (the controls roughen, the surface roughness and the tail jitter
  rise); J not removable (det F quantiles spread, at 300k most); S removable in geometry and thin (D9b: it agrees
  with the transport and is the smaller term on its particles) but `stray_max` rises, so it stays. Render
  influence: the render channel and λ's rule are unchanged; λ is calibrated on the physics gradient, which loses a
  term in each arm: the first-window λ and g_share are reported per arm.
- **D9b, which term sets the direction of the particles the local terms act on (diagnostic, pre-registered 2026-10-01
  15:22 CDT, launched 15:23 after the suite passed on repo_r28 (269 passed); repo_r28 = repo_r27 with one more record).** D9 read per particle: the unit masses are 1
  at every N; the near band's active share is constant in N at matched progress (0.3–3 %) and its pull per active
  particle is exactly 0.200 wu, so it halves from 40k to 300k through wu alone; the transport's rms gradient per
  particle falls about 5×, so the near pull is 16× the average transport gradient at 40k and 37× at 300k (bunny, E
  2.4e-3; dragon 23× and 59×). The scale of a loss gradient means nothing by itself (Adam normalises per element);
  what moves a particle is the balance and the alignment of the terms on that particle. Recorded at every committed
  state, on each local term's active set (near band: eligible and beyond the berth; spray cleanup: non-zero
  gradient): the rms position gradient of the local term, the scaled transport, the surface term, the λ-weighted
  render term, the other local term and the sum of the three non-local ones; the cosines of the local pull with
  each; the share of the set's particles where the local pull opposes that sum. Runs: bunny and dragon at 40k (to
  the stop), 100k and 300k (35 windows), seed 97 (`output/gpu/d9b`, `tmp/d9b_eval.py`), read at matched transport
  energy. Reading (the user's rule): case 1, at every N the local term dominates the sum of the others on its set
  and the pattern of the cosines is the same → the scale is left alone, whatever wu does, and the next step is the
  regularisers (R14); case 2, the competition changes with N (a term comparable and opposed at 40k that is
  overwhelmed at 300k) → a real scaling defect: the local terms are separated from wu by a new definition, validated
  at 300k (removing wu's N-dependence alone would make them 2.1× stronger there). Predictions: near band: the local
  pull is at least 3× the sum of the others at every N, growing with N, all cosines non-negative (the band's
  particles sit outside the surface; the transport, the surface term and the silhouette's spray penalty all pull
  them inward); spray cleanup: ratio 2–8 with the same signs; case 1.
- **R13b, the 300k confirmation of the selection without the dense distance (sanity check, pre-registered 2026-10-01
  15:04 CDT, launched 15:05; no code change).** The user decided to delete the dense distance (R13: different window
  decisions, the same outcome at 40k; the one `kin` bound 3 % over in one run read as spread). One 300k dragon run
  per ruler to its own stop (no window budget), seed 97: the legacy merit (repo_r26, `output/gpu/r12f/
  r12fW1300000_dragon_stop`) and the state merit (repo_r27, `output/gpu/r13/r13MS300000_dragon_stop`). Single
  runs, so only a gross difference counts: silhouette apart by more than 0.003, world-thin by more than 1.5 points,
  the stop window by more than a third, or a freeze in one arm only. Prediction: no gross difference; the state
  merit's run stops within a third of the legacy run's window count. Reported: windows, minutes, rejections, end
  `kin`, tail jitter, the shadow's disagreements along the long tail (the 35-window runs showed none).
- **R13, the selection merit without the dense body-to-target distance (pre-registered 2026-10-01 13:51 CDT, launched
  13:52 after the suite passed on repo_r27 (269 passed); repo_r27 = repo_r26 with the merit's common form changed).** Names from here on: the objective's term on
  isolated particles is the spray cleanup; the distance field summed over every particle, which only the selection
  merit carried, is the dense distance (41 % of the merit at 40k, R12f). Arm M_state: the merit is the objective
  read at the committed state alone: the isolation gate and the near band are taken at the state itself (not from
  the window's start, so the merit is a function of the state), and the dense distance is removed. The objective
  and its gradient are unchanged. The shadow selection now reads the legacy merit (with the dense distance) on the
  same trajectory. Runs: the 40k gallery twice (19 meshes, seed 97; `output/gpu/r13`, `tmp/r13_eval.py`) against
  the legacy merit's runs (R12f's two; R12e's two as a second legacy pair). No 300k run: at 35 windows the two
  rulers did not differ there (0 flips, the dense distance 8–9 % of the merit); if M_state survives, one 300k run
  per ruler to its own stop follows. Decision, on the two-run means (the user's rule): (1) geometry and stability
  within spread → the dense distance is deleted from the selection; (2) geometry within spread but more
  rejections, earlier stops, more reversals or tail jitter → it stabilised the selection: kept, its mechanism
  measured and an N-independent scale defined (its share is 41 % at 40k and 8–9 % at 300k); (3) results worse →
  kept and redefined. "Within spread": silhouette difference median within ±0.002 and no mesh but beast below by
  more than 0.004; `thin_uncovered` median within ±1 point and no mesh worse by more than 3 beyond the arms' own
  run-to-run difference; and each of the end `kin` median, the tail jitter median, the share of late windows with
  reversal cosine below −0.5, the rejected-window total, the committed-window median and the delivered frames
  median not beyond the legacy runs' range (four runs) by more than that range's width; no run stopping before 15
  windows moving that the legacy runs do not show. Predictions: case (1); the shadow's 17 delivered-window flips
  were near ties (0.5 % of the merit, one window apart), so the delivered states do not differ beyond spread; C
  still stops near 18 windows. Render influence: the render channel and λ are unchanged; g_share reported; the
  merit's render share rises by construction (the dense distance no longer dilutes it).
- **D9, the size of the local terms against the transport as N grows (diagnostic, pre-registered 2026-10-01 13:51
  CDT, launched 13:52; repo_r27, records only).** The spray cleanup and the near band are sums with a constant pull
  per particle; the transport's gradient per particle falls with N. Recorded at every committed state: the
  position-space gradient norms of the scaled transport (ot_scale |∇S_ε|), the surface proximity, the spray
  cleanup and the near band, and the number of particles the two local terms act on. Runs: dragon and bunny at 40k
  (to the stop), 100k and 300k (35 windows), seed 97 (`output/gpu/d9`, `tmp/d9_eval.py`). Read at matched progress
  (the window whose transport energy is nearest to 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3): the ratios |∇spray| /
  |∇transport| and |∇near| / |∇transport| against N. No pass criterion: the result is the exponent with which each
  ratio grows from 40k to 300k, which decides what normalisation gives the two terms the same role at every N.
  Prediction: both ratios grow with N, the near band's by about (300/40)^(1/2) ≈ 2.7 at equal progress if its active
  count grows as N and the transport's norm falls as N^(−1/2); the spray's stays small (few isolated particles).
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

0. **Open after D20–D32 (2026-10-03).** (a) The near-surface leftover of the 300k dragon: about 280 rendered
   particles between 3 target spacings and one loss cell at the end (D32), one of them a visible disc at the
   notch under the tail. Its cause is measured and lies in the parked items: a particle with material on both
   sides is not a layer particle and has no position channel; the relaxation undoes a single particle's u; a
   group's u is bounded by the accepted step. A change there is the user's decision. (b) With `render_res` equal
   to `render_res_hi` (96, D19) the coarse-to-fine stage and its event are inert: remove the code and its test.
   (c) With the layer as the body's the dragon's head is softer in the 4K picture (strong-gradient share 3.7 %
   against 4.7 %): the detached sets on thin features are no longer made regular.
   **After D42–D53 (2026-10-03 15:40).** (d) is closed: the feathering was D26's reach (D43, D46); a detached
   set is in the air only beyond the berth (D48), and the 64 → 96 schedule is the default again (D52): the 300k
   dragon in 30 minutes with the 45-minute video's horns and nothing beyond a loss cell, the bunny in 16. Open:
   (h) the time. A window is 12.7 s (14.1 before), 130–140 windows: 15 minutes would need 6.5 s. What is left
   is the three adjoint sweeps (5.1 s) and the Sinkhorn solves (3.0 s); the exact items still listed in D53 are
   worth about 0.5 s. (i) The notch remnant (D51): material in the air has no handle but u, whose step is the
   optimiser's common step, a tenth of its clamp; the long run walks it out, the 40-window run leaves it in two
   runs of three. The parked u/step item. (j) beast's freeze, now in four of five gallery runs. (k) The thin
   share and world-thin read the feathered cover as the better one (0.3 % against 0.7–0.8 % for the solid
   horns): neither is a measure of thin features; `thin_fill_probe` does not separate them either (D43). A
   measure of how regular the material of a thin feature is, is missing.
   **After D33–D41 (2026-10-03 13:10).** (d) The feathered thin features (the horns): the 45-minute video's solid
   horns are the old disc rule on a long run of the code before the layer change, not the 64 → 96 px schedule
   (D38, D40). The thin class (below two MPM cells) has 14–15 % of its points under half of the target's local
   density at 40 windows and 10 % at 160 (D41; the thick class 2–3 %); the thin share does not see it. The loose
   material is within the berth, where no term acts, and what would move it in is (a)'s parked items; the
   decision is the user's. D42: the wisps, and at 40 windows parts of the thin features themselves, are sets
   not linked to the body within a layer spacing (5000–6400 rendered particles, one spacing from the target),
   which D26 leaves unrelaxed like the sets in the air; whether a set within the berth is the body's is the
   design question, after the literature. (e) The mid-morph dilution (D36, D37): the surface's material is pulled apart evenly
   in the plane in windows 1–5 and strands come off as beads; next is the literature (4–5 primary sources on
   the particle count of a growing surface), then a design. (f) beast's `domain` freeze in 35–45 % of 40k runs
   and the three-rejection stop on C and V (D39): parked defects, to be read from repeats. (g) `thin_fill_probe`
   as the measure for thin features in place of the thin share: not yet in the run's record.
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

**FV, 2026-10-01 20:51 CDT — the final validation of the eight-term objective: no catastrophic failure
(pre-registered 19:36).** 40k gallery, two runs, against the four runs of the eleven-term code (R13 a, b; B a, b):
no mesh trips a flag (silhouette below its lowest current value by more than 0.004 in both runs, thin above its
highest by more than 3 in both runs, a new freeze, a guard, det F below 0.5). Medians: thin 9.65, 9.73 (9.17–10.09);
silhouette 0.9769, 0.9766 (0.9766–0.9773); chamfer 0.1156, 0.1158 (0.1157–0.1159); end `kin` 7.0e-5, 1.04e-4
(7.2e-5–1.46e-4); committed windows 49, 39 (38–44); delivered frames 1881, 1521 (1441–1681); tail jitter 3.78e-6,
4.49e-6 (3.89e-6–4.57e-6); `stray_max` 0.005, 0.007; surface roughness 1.213, 1.201 (1.177–1.206); det F minimum
0.897, 0.898 (0.890–0.895), its 1 % and 99 % quantiles 0.988 and 1.013 (the same); anisotropy p90 1.070, 1.061
(1.063); control roughness 1.6e-9, 2.1e-9 (2.2e-9–2.4e-9); no guard. beast sound in both runs (108 and 99 windows);
C stops at 19 windows in one run (`kin` 1.4e-2, the parked alternation) and runs 46 in the other. Recorded and not
acted on: homer's thin 15.9 and 14.4 against a highest current value of 13.5; A's silhouette 0.9769 in one run
against a lowest current value of 0.9781. 300k to the run's own stop: dragon silhouette 0.9848, world-thin 0.70 %,
chamfer 0.0591, 118 windows, 42 minutes, det F minimum 0.901, anisotropy p90 1.089 (the eleven-term code: 0.9846,
0.66 %, 0.0591, 98 windows, 38 minutes); bunny 0.9876, 0.02 %, chamfer 0.0580, 51 windows, 11 minutes, det F minimum
0.939. Render influence: first-window λ 0.248 at the gallery median, 0.396 and 0.249 at 300k, as before; g_share at
the end 0.89–0.90 (0.91 and 0.82 at 300k); no render-off twin in this run. By the stop rule the formulation is
frozen here, subject to the user's confirmation: Sinkhorn transport, surface proximity, residual drift; released
motion; near band (berth to one loss cell), spray cleanup; silhouette, shading. Open and outside the formulation:
the config defaults (proximity and the N-following grid are still flags), the PBR renders and video from the
frozen code, the runtime items (near band on: 23 s per 300k window against 15.5 s), the parked defects (C's early
stop, the beast ejection, the rollout's nondeterminism).

**R14b, 2026-10-01 19:34 CDT — the three regularisers off together: inside the noise on every criterion
(pre-registered 18:44).** Arm X against the current formulation's four gallery runs (R13 a, b; B a, b). Medians:
thin 9.69, 9.38 (current 9.17–10.09); silhouette 0.9769, 0.9763 (0.9766–0.9773; per mesh the difference median
−0.0001, worst −0.0010); end `kin` 8.6e-5, 1.09e-4 (7.2e-5–1.46e-4); committed windows 38, 38 (38–44); rejected 5, 5;
delivered frames 1481, 1441 (1441–1681); `stray_max` 0.007, 0.005; surface roughness 1.211, 1.194 (1.177–1.206);
det F minimum 0.897, 0.899 (0.890–0.895), its 1 % and 99 % quantiles 0.987 and 1.014–1.015 (0.988 and 1.013);
anisotropy p90 1.063, 1.060 (1.063); control roughness 1.9e-9, 2.2e-9 (2.2e-9–2.4e-9); |dFc| maximum 0.0037, 0.0039
(0.0035–0.0040); no guard, no run stopping while moving. The one flag is bob's thin (+3.7, the mesh the dead-term
arms flagged too). The tail-jitter median is 4.76e-6 and 5.13e-6, inside the bound (5.25e-6) but above all four
current runs (3.89e-6–4.57e-6): noted, not a miss. 300k at 35 windows: dragon silhouette 0.9808, world-thin 1.5,
E at window 34 1.91e-3, det F minimum 0.896 (this arm's current runs 0.9824, 0.9827; the current code's 300k
dragon runs since R12d range 0.9807–0.9835); bunny 0.9857, world-thin 0.0, det F minimum 0.946 (0.9866, 0.9867):
inside the limits, both below the two reference runs. First-window λ identical (0.248; 0.396 and 0.249 at 300k).
The prediction held. The control magnitude, the control smoothness and the volume prior are deleted with their
constants (w_ctrl, w_creg, creg_k, w_jvol).

**R14, 2026-10-01 18:42 CDT — the two control regularisers are dead terms, the volume prior changes nothing that was
measured, the spray cleanup misses its stray criterion by a hair and stays (pre-registered 15:59).** Current
formulation: R13 a, b and B (three gallery runs). Sizes first: at the middle and the end of a run the control
magnitude term is 1e-11 to 1e-9 of the merit and the control smoothness term 6e-9 to 8e-8 (bunny, dragon, heart,
homer); arms C and G are therefore reruns of the current code, and what they miss measures the noise of the
criteria. C (w_ctrl 0): thin difference median −0.11, silhouette −0.0001; misses by the letter: bob thin +5.2 (bob
ranges 8.3–15.7 over every arm), the tail-jitter median 4.54e-6 in one run against a bound of 4.41e-6, and one beast
freeze (10 windows, the parked ejection). G (w_creg 0): thin 0.00 (worst +1.6), silhouette +0.0001 (worst −0.0009),
jitter 4.03e-6 and 4.26e-6, control roughness 1.7e-9 and 1.4e-9 against 2.4e-9 with the term on, surface roughness
1.197 and 1.167 against 1.201–1.206: every criterion passes; the prediction that the controls and the surface would
roughen was wrong. J (w_jvol 0): thin −0.05 (worst +2.1), silhouette −0.0001 (worst −0.0009), det F minimum 0.897
and 0.896 against 0.890–0.894, the 1 % and 99 % quantiles of det F 0.987–0.988 and 1.013–1.014 against 0.988 and
1.013, anisotropy p90 1.068 and 1.064 against 1.063; the one miss is the jitter median 4.60e-6 in one run, the same
size as the dead-term arm's; the prediction that det F would spread was wrong. S (spray cleanup off): thin +0.11
(worst +2.4), silhouette 0.0000 (worst −0.0011), but the median `stray_max` is 0.010 % in both runs against
0.005–0.0075 % (bound 0.009), the median `out_nn_frac` 0.10 % against 0.05–0.07 %, and the runs are longer (windows
median 48 and 52 against 38–44, delivered frames 1841 and 2081 against a bound of 1921); its largest `stray_max`
without beast is lower (0.25 against 0.37–0.39) and `stray_final` and `out_dt_frac` are inside the range. 300k at
35 windows (dragon, bunny; current: silhouette 0.9824–0.9827 and 0.9866–0.9867, world-thin 1.2–1.4 and 0.0): C
0.9834 / 0.9864, G 0.9825 / 0.9873, J 0.9827 / 0.9865 with world-thin 1.1–1.5 and 0.0–0.1, det F minimum 0.892–0.900
and 0.944–0.948, anisotropy p90 1.078–1.087 (current 1.083): inside the limits; under J the end velocity at window
34 is higher (3.2e-3 on the dragon, 4.6e-4 on bunny; not a criterion; the current code's 300k runs range from 1e-4 to
3.9e-3 on the dragon). S 0.9813 / 0.9853: inside ±0.002 but lower on both meshes by 0.0013. No guard fired in any
run. Render influence: the first-window λ is identical in every arm (median 0.248; 0.396 and 0.249 at 300k) and
g_share at the end 0.87–0.89. Verdict: G and J are removable; C is a term of 1e-10 of the merit whose arm shows only
the criteria's noise, removable on that reading; S stays (its own criterion, narrowly, in both runs). Three are
removable, so the combined arm runs before anything is deleted (R14b). What keeps the controls smooth and det F
within 0.98–1.02 without these terms was not measured (the increment clip and the released phase are candidates).

**D9b, 2026-10-01 15:49 CDT — on its own particles the near band does not dominate, the render term is its equal, and
early in a run it opposes the transport, more strongly at 300k; the spray cleanup agrees with the transport and is
the smaller term (pre-registered 15:22).** Both predictions for the near band were wrong. On the band's active set
(0.3–3.4 % of the particles on bunny, 0.6–10.7 % on the dragon; the share is the same at every N at matched
progress): (1) the local pull over the sum of transport, surface and render is 0.4–3.5, not ≥ 3: the λ-weighted
render gradient is as large as the near pull on these particles (local / render 0.4–2.5), while the surface term
is small there (local / surface 4–140) and the transport smaller than the pull (local / transport 2.3–34). (2)
The pull is not aligned with the others. Against the transport its cosine is −0.34 to −0.76 while the transport
energy is above about 1e-2 (bunny at E 4e-2: −0.49, −0.58, −0.61 at 40k, 100k, 300k; dragon at E 0.1: −0.58,
−0.61, −0.76), and the pull opposes the sum of the others on 67–94 % of the band's particles there (dragon 84 %,
87 %, 94 %); from E ≈ 3e-3 down the cosine is between −0.15 and +0.27 and the opposed share 23–52 %. Against the
render and the surface term the cosine is about zero throughout (−0.05 to +0.25): the near band is a direction of
its own. With N: local / transport on the set grows (bunny 2.3 → 3.5 → 6.6 at E 4e-2, 6.1 → 7.3 → 12.2 at 2.5e-3;
dragon 2.7 → 4.5 → 7.1 at 0.1, 11.2 → 12.9 → 21.8 at 3e-3) while local / render does not (late: 1.9, 1.5, 1.4 and
2.5, 1.8, 2.5). So early in a run the near band overrides a transport that points the other way on its particles,
2.3–2.7× at 40k and 6.6–7.1× at 300k: the user's case 2 for the near band against the transport in the early
phase (this is the remainder of what R12 removed at long range; R12c's arm with this band was at E 3.1e-2 at
window 8 of the 300k dragon against 7.5e-3 without a near band); late in a run the relation is the same at every
N (orthogonal, the render its equal). Spray cleanup, on its set (0.5–10 %, shrinking with N late): its gradient
is 0.2–0.7 of the others' sum (the render is larger), aligned with the transport (cosine +0.2 to +0.5 on bunny,
−0.09 to +0.23 on the dragon) and with the near band (+0.04 to +0.38), opposed to the sum on 12–49 % of its
particles; nothing in it changes sign with N. Reading: wu is not the lever (removing its N-dependence would make
the near pull 2.1× stronger at 300k, where it already overrides the transport most); the defect is the near band
acting against the transport while the transport still moves mass through the band, and that it does so more at
larger N because its pull per particle falls only with wu while the transport's falls as about 1/N^0.8. What to
do is not decided here.

**R13b, 2026-10-01 15:50 CDT — the 300k dragon to its own stop under each ruler: no gross difference
(pre-registered 15:04).** Legacy merit: silhouette 0.9840, world-thin 0.87 %, 95 committed windows, 3 rejected, end
`kin` 2.3e-5, E 7.4e-4, tail jitter 1.3e-6, 40 minutes. State merit (no dense distance): 0.9846, 0.66 %, 98
windows, 6 rejected, `kin` 2.1e-5, E 6.6e-4, jitter 1.4e-6, 38 minutes. Differences 0.0006, 0.21 points and 3
windows, far inside the limits (0.003, 1.5 points, a third); neither froze. Along the legacy run the state ruler
would have judged 3 of 98 windows differently; along the state run the legacy ruler none of 104. The dense
distance stays deleted from the selection. A 300k dragon run to its own stop is now about 100 windows and 40
minutes and ends at silhouette 0.984 and 0.7–0.9 % world-thin.

**R13, 2026-10-01 14:44 CDT — the selection without the dense distance: geometry and stability inside the legacy
runs' range, one bound missed by 3 % in one run (pre-registered 13:51).** Two runs against the legacy merit's four
(R12f a, b; R12e a, b). Silhouette: difference of the means against R12f, median +0.0001, worst −0.0007; medians
0.9767, 0.9768 against 0.9764–0.9771. `thin_uncovered`: difference median −0.34, higher on 7 of 19, worst +1.1
(bunny); medians 9.17, 10.09 against 9.32–11.03; bob 9.2, 9.6 against 14.8, 14.0 and 8.3, 13.5 under the legacy
merit (inside what bob shows). Stability, each against the four legacy runs' range widened by its own width:
rejected windows 103, 96 (legacy 98–101, bound 95–104: inside); committed windows median 44, 44 (41–48: inside);
delivered frames median 1681, 1681 (1561–1841: inside); tail jitter median 3.99e-6, 4.15e-6 (3.29e-6–4.53e-6:
inside); late windows with reversal cosine below −0.5: median share 0 in every run; end `kin` median 7.2e-5 and
1.08e-4 (legacy 8.0e-5–9.3e-5, bound 6.8e-5–1.05e-4): the second run is 3 % over the bound, the first below the
legacy minimum. No run stops before 15 windows moving (the legacy runs: beast froze once); C stops at 18 and 18
windows with `kin` 1.8e-2, 1.9e-2 as under the legacy merit; beast sound in both runs (79 and 96 windows). The
shadow reading the legacy merit on R13's trajectories disagrees on accept/reject in 98 of 1977 windows (5.0 %),
as the reverse shadow did in R12f (4.8 %): the two rulers still judge windows differently, and the runs end in the
same place. Render influence: λ at the first window identical (median 0.248), g_share at the end 0.88 against
0.88; the render's share of the merit is larger by construction. By the user's rule this is case (1), geometry and
stability within spread, with the one `kin` bound read as spread (one run 3 % over, the other under the legacy
minimum; the two-run mean 9.0e-5 inside the legacy range); the prediction held, including C. The dense distance
has no measurable function in the selection at 40k. Deleting it is the user's decision; the 300k confirmation (one
run per ruler to its own stop) follows it.

**D9, 2026-10-01 14:17 CDT — both local terms exceed the transport's gradient late in a run at every N, and the near
band's ratio grows from 40k to 300k by 1.7 to 2.6 (pre-registered 13:51).** Position-space gradient norms at
committed states of matched transport energy E. Near band over transport, 40k → 100k → 300k: bunny 0.85 → 1.31 →
2.25 at E 1.3e-2, 1.24 → 1.86 → 3.10 at 2.4e-3, 1.69 → 2.18 → 4.00 at 1e-3 (×2.4–2.6, about N^0.45); dragon 0.70 →
1.15 → 1.64 at 0.3, 2.72 → 3.13 → 4.53 near 8e-3, 2.93 → 3.55 → 5.75 at 3e-3, 3.95 → 5.03 → 7.02 near 1e-3
(×1.7–2.4, N^0.26–0.42). The prediction (×2.7) is the upper end. Why: the band's active count grows as N (bunny
at E 2.4e-3: 239, 744, 2069) and its pull per particle is constant up to the unit conversion (1.38e-5, 9.5e-6,
6.6e-6: wu falls 2.1×), so its norm grows as wu √N (×1.4), while the transport's norm falls (×0.56 on bunny,
about N^−0.3). Spray cleanup over transport: bunny 0.66 → 0.91 → 1.43 at 1.3e-2, 1.04 → 1.24 → 1.56 at 2.4e-3, 1.35
→ 1.26 → 1.71 at 1e-3; dragon 1.42 → 1.64 → 2.25 near 8e-3, 1.63 → 1.52 → 1.62 at 3e-3, 2.66 → 2.17 → 1.01 near
1e-3: no consistent growth late. The prediction that the spray term stays small was wrong: the isolation gate is
non-zero on about 14 % of the particles at every N (5.6k of 40k, 13k of 100k, 41k of 300k on the dragon; a ramp
from 1.2 to 1.8 median neighbour distances), and its gradient is 1–2.7× the transport's from E ≈ 1e-2 down. So is
the surface proximity's (ot_scale |∇| 1.7e-4 against the transport's 5.8e-5 on the 40k dragon at E 1e-3). Late in
a run the transport is the smallest of the four position gradients; the "cleanup" terms and the surface term
carry the end of the morph. For the scale step: a plain mean (÷N) with wu kept would turn the near band's N^+0.15
into about N^−0.85 and reverse the imbalance; with wu dropped it gives N^−0.5 against the transport's N^−0.3. What
has to be invariant (the ratio at matched progress, or the pull per particle against the transport's per
particle) is the definition to choose. A 300k run of 35 windows took 8 minutes on bunny.

**R12f addendum, 2026-10-01 13:45 CDT — offline replay: the flips are the dense distance's variation, not a dilution
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
