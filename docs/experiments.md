# Experiments

What to run next, what is running, and what has been measured. Every experiment runs on the GPU server (hyde06).
Before reading a result, add its entry here with the prediction and the pass criteria; then add the result, including
failures. Stamps use the server clock (CDT). The full record of the earlier pipelines is on the `v3-grid-gs` branch
(tag `v3-grid-gs-final`).

## Running now

- **S2, adoption gates.** Seed 97. The 19-mesh gallery at 40k (A, armadilo, beast, bimba, bob, bunny, cheburashka, C,
  cow, dragon, fandisk, heart, homer, maxplanck, nefertiti, ogre, spot, teapot, V) and the 300k dragon, settled
  transport against the earlier pipeline's adopted forms. Pass per mesh: silhouette IoU ≥ the earlier pipeline's
  − 0.005, no stall, second-half window reversals ≤ 25 %, centre-of-mass drift ≤ 0.1 spacing; the dragon's IoU ≥ the
  earlier pipeline's − 0.003. Outputs `$OUT/s2/`; table `scripts/probes/settled/gate_table.py $OUT/s2 40000 <meshes>`.
- **Line coverage** of the settled runs, the renderers and the tests (`$OUT/cov/`), to find code the pipeline never
  executes before the refactor.
- **Gradient dumps** of a 300k bunny run (`$OUT/viz/`), for a video of the per-particle render and physics gradients
  and the losses over the morph (large loss red, small green).

## To run next, in order

1. **GPU-only and refactor equivalence.** After the code runs GPU-only and every file is under 500 lines: the tests
   pass; the first window's objective matches the pre-refactor code to 1e-6 relative at the same seed; the 40k gallery
   and the 300k bunny and dragon match the S2 settled runs within the seed spread (silhouette IoU ±0.002). Anything
   outside is a bug in the port, not a new result.
2. **The gallery at 300k.** The 19 meshes at the delivery resolution, with 4K renders. Same measurements as S2.
3. **Render influence across meshes.** Render-off twins (`--render_weight_scale 0`) and a second seed on five meshes
   (bunny, dragon, C, V, nefertiti) at 300k: the render's effect against the seed spread, per mesh.
4. **Head and body relief.** Settled transport leaves 10.4–11.7 % of the bunny's target surface farther than 1.5
   spacings (the earlier pipeline 7.8 %). Measure where by region (`region_error.py`), then test a mechanism that
   sharpens the end state without pins or position edits. Pass: the gap at or below 8 % with no loss elsewhere.
5. **Ear tips.** The ear tip holds 11–12 reference particles of the 13 required. Same approach as item 4.
6. **Constants.** `support_weight = 8` and `ot_iters = 1600` are validation values. Either derive them from the
   discretisation or show the result does not depend on them (a factor-of-two sweep each way on three meshes).
7. **Stopping.** After convergence the same rejected step can repeat identically each window until the patience runs
   out (the bunny with a longer budget: windows 50–68). Stop when a rejected step repeats unchanged, and confirm the
   end state is unaffected.
8. **Scaling.** Wall time and memory at 40k, 100k, 300k and 1M particles on one GPU.

## Results so far

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
