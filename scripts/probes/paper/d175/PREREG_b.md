# D175b pre-registration: the keyframe compared as a projected-count (X-ray) picture

Written 2026-10-29 12:38 CDT, after the segment reading (output/gpu/d175/forensic/seg/seg_read.md) and before any D175b run.

## Why (the measurement)

The one-step reading at the start of segment windows 1-6 (S97 rerun, 40k), with the current comparison (exterior
silhouette + shading against P_k):

1. **The hole term carries the pull, and thin pole spikes satisfy it.**
   - The silhouette deficit (hole_core) carries most of the render gradient on dF_c: norm ratio 0.77-0.87, cos 0.84-0.96
     with the whole term.
   - About 97 % of that term lies in the pole pixels (keyframe covered, source sphere not).
   - From window 1 to 6, the pole pixels' coverage rises from 0 to 0.21 / 0.41 / 0.61 / 0.75 / 0.83, while the body's
     projected particle count there, over the keyframe's, rises only from 0 to 0.015 / 0.08 / 0.18 / 0.26 / 0.30.
   - The hole term falls by 87 %, while the projected count mismatch falls by 27 %. The aspect is 1.08 at window 6's
     start, against the keyframe's 1.48.
   - The soft alpha of the exterior discs saturates on a thin column, so the term is met with 30 % of the mass.
2. **Between windows the bulk push alternates.**
   - At windows 2 and 4 the step of every exterior term moves the bulk against the stretch: cos −0.45 / −0.43 below 3
     pitches, aspect −0.017 / −0.019. Windows 1, 3 and 6 move it with the stretch (+0.49 / +0.22 / +0.36).
   - The particles' own silhouette (cloud_sil) flips the same way.
   - The projected-count picture (column) moves the bulk with the stretch in all 6 windows: aspect gain +0.048, +0.014,
     +0.038, +0.001, +0.027, +0.046; cos below 3 pitches +0.62, +0.09, +0.55, +0.02, +0.19, +0.68.
   - Its dF_c gradient is a different direction from the silhouette's (cos 0.01-0.53 with the whole term).

## The definition change (the smallest; no weight, no schedule, no blocking)

Inside a keyframe segment, the keyframe comparison reads the particles' projected count picture, against the keyframe
samples' (the mean over its 8 samples):

  L_k = mean over views and pixels (c(x) − c_k)² / mean (c_k²)

- c = the CIC count of the particles per pixel. It is soft_silhouette_multi's coverage before its 1 − exp(−k ·): the same
  views, resolution and camera box.
- The normalisation by the keyframe's own mean square is part of the definition: a relative L2 that makes the term
  dimensionless, like the silhouette.
- It replaces the exterior silhouette + shading inside a segment.
- The final target's render term is unchanged (the exterior, bit-SAME).
- λ is held from the pre-pass on the final target, as before.
- Everything else is the D175 storyboard: the clock, PCGrad, the epochs, and Hu after t_k.

Code (wip/d175 3db83c19 line numbers), behind `--keyframe_compare column` (default `silhouette` = D175):
- physmorph/pipeline/render_loss.py:306 (beside d_exterior): `count_pictures(x, views, res, extent)` (the CIC count, the
  splat of soft_silhouette_multi) and `d_count(x, counts_k, views, res, extent)` (L_k above).
- physmorph/pipeline/target.py:28-41 (Exterior): a field `counts` (None on the final target's exterior).
  physmorph/pipeline/target.py:151 (keyframe_pictures): with `counts=True`, also the mean count pictures of the samples.
- physmorph/pipeline/run/storyboard.py:71-72: the keyframe's Exterior carries its counts when the comparison is `column`.
- physmorph/pipeline/window/objective.py:208: `if e.counts is not None: return d_count(xT, ...), zero`. A keyframe
  exterior only; the final target's has counts None, so its path is unchanged.
- physmorph/pipeline/config.py:153: `keyframe_compare` ("silhouette" | "column"); scripts/pipeline_run.py:83 the flag.

## Experiment (40k, sphere → bunny, keyframe E at t = 1 s, Hu)

| arm | seeds | flags |
|---|---|---|
| X: dF_c + layer u + render loss (full), storyboard, column comparison | 97, 99 | 208's S flags + `--keyframe_compare column` |

References, reused from job 208 (the same code path for them):
- D97 / D99 have no keyframe;
- the twin T97 has λ = 0, so it never reads the keyframe comparison.

Gate first: G1-G4 bit-SAME (defaults, Hu, Hu coin, settle with relaxation) and the unit tests.

## Metrics and bars: unchanged from PREREG.md (the same reader, tmp/d175/d175_read.py)

- **P1:**
  - G ≥ 0.6 on the run views and ≥ 0.5 on the held-out views (KF-IoU = the exterior silhouette IoU at alpha > 0.5, as
    before);
  - 3D IoU(X) ≥ 3D IoU(T97) + 0.10;
  - KF-IoU(X) ≥ KF-IoU(D, same seed) + 0.05.
- **P2:** final silhouette IoU ≥ min(D) − 0.005; Chamfer ≤ 1.1 × max(D); thin_uncovered ≤ max(D) + 0.02.
- **P3:** end v_mean ≤ 1.5 × max(D); Jmin_traj > 0 at every window; no serious piece.
- **PASS** = P1, P2 and P3 on X97 and X99.
- Also recorded:
  - the body's second-moment aspect at frame 240;
  - the share of the 6 segment windows whose committed step raised the aspect.

## Predictions

- Aspect at frame 240 ≥ 1.20 (D175 S: 1.06 / 1.07; dF_c toward a 3D target: 1.46).
- 3D IoU gain over the twin ≥ +0.10.
- KF-IoU may sit below S's 0.92: the count picture does not ask for the outline as sharply. A KF-IoU shortfall with a
  3D pass is a definitional trade to be read, not tuned.

## If it fails

The next measurement is named in the summary; no weight is added. If the aspect rises but the KF-IoU falls, the next
measurement is the one-step reading of X's segment windows for the outline, by layer.
