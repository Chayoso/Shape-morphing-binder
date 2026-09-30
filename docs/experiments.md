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
   thickest bin.
4. **Ear tips.** The ear tip holds 11–12 reference particles of the 13 required. Same approach as item 3.
5. **Constants.** `support_weight = 8` and `ot_iters = 1600` are validation values. Either derive them from the
   discretisation or show the result does not depend on them (a factor-of-two sweep each way on three meshes).
6. **Stopping.** After convergence the same rejected step can repeat identically each window until the patience runs
   out (the bunny with a longer budget: windows 50–68). Stop when a rejected step repeats unchanged, and confirm the
   end state is unaffected.
7. **Scaling.** Wall time and memory at 40k, 100k, 300k and 1M particles on one GPU.

## Results so far

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
