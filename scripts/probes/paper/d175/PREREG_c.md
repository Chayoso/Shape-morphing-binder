# D175c pre-registration: the segment's merit owned by the keyframe alone (soundness returns with the final target)

Written 2026-10-29 13:14 CDT, after the line-search split (output/gpu/d175/auto/200_ls/ls.md) and before any D175c run.

## Why (the measurement)

Three runs of the column arm (L_X97_a, L_X97_b, L_X99_a) recorded every segment line-search trial, split by term.

1. **One push, then braking.**
   - Window 1's first step is the push. λΔcount = −6.0e-3, and it sets the body moving: end speed 0 → 0.36 pitches per
     horizon, Δ released motion +1.9e-3, Δ end drift +0.9e-3.
   - Every later step in windows 1-6 is a brake. The released motion (about 2/3 of the soundness change) and the end
     drift fall, and λ·count rises: accepted steps median Δstab −3.6e-5 against λΔcount +3.7e-5.
   - The merit is nearly flat along this trade, and the optimiser slides along it toward rest. The end speed at the
     window starts goes 0.36 / 0.41 / 0.33 / 0.19 / 0.10 / 0.05 pitches per horizon.
2. **Every one of the 156 failed trials is "the count term rises":** the braking step costs more keyframe than it saves
   motion.
   - These trials do not leave the body moving: end and released speed ×1.00 of the start's.
   - So the soundness terms do their job (rest at each window end), and that job is the blocker.
   - A keyframe 6 windows ahead needs the body to keep moving across window ends.
   - What decides the trade is the scale of λ·count against the soundness terms, two quantities with no common unit.
3. **The window-1 replay null is gone.**
   - The segment check now reads D164's measure (start pair, taped pairs, one ulp of E).
   - It kept all 18 segment commits: noise 0.65–1.9e-7, tolerance 0.65–1.9e-6.

## The definition change (no weight, no schedule, no blocking)

Treuille / McNamara's ownership taken one step further: a storyboard time owns the whole objective of its windows.
Inside a keyframe segment the merit, the gradient and the Armijo test read the keyframe comparison alone (λ held ×
d_count). The soundness terms (the released motion and the end drift) are absent there, as the final target's geometry
already is, and return with the final target together with its transport.

Outside segments, nothing changes (gate: G1-G4 bit-SAME). PCGrad in a segment then sees g_p = 0 and passes the render
gradient through.

Code (wip/d175 d26730a8), behind `--keyframe_soundness final` (default `every` = D175 / D175b):
- physmorph/pipeline/target.py:95 (TargetPack): a field `soundness` (True).
- physmorph/pipeline/window/objective.py:185-190 (losses): with soundness False, lstab and the end drift enter as 0 × term
  (kept on the graph).
- physmorph/pipeline/run/storyboard.py:80-93 (apply): a keyframe entry sets soundness = (keyframe_soundness == "every");
  the final target sets True.
- physmorph/pipeline/config.py:153 and scripts/pipeline_run.py:83: the field and the flag.

## Experiment (40k, sphere → bunny, keyframe E at t = 1 s, Hu)

- Arm Z: D175b's X flags + `--keyframe_soundness final`, seeds 97 / 99.
- The segment replay measure (D164's) is in the code for every keyframe arm.
- References reused from job 208: D97 / D99 (no keyframe) and T97 (λ = 0, so the keyframe comparison has no effect; its
  segment merit becomes 0 under Z's definition, which is why the 208 twin is kept as the reference).

## Metrics and bars

Unchanged (PREREG.md, d175_read.py). Also recorded: the aspect at frame 240, and the segment windows' accepted steps and
holds.

- **P1:**
  - G ≥ 0.6 on the run views and ≥ 0.5 on the held-out views;
  - 3D IoU(Z) ≥ 3D IoU(T97) + 0.10;
  - KF-IoU(Z) ≥ KF-IoU(D, same seed) + 0.05.
- **P2:** final silhouette IoU ≥ min(D) − 0.005; Chamfer ≤ 1.1 × max(D); thin_uncovered ≤ max(D) + 0.02.
- **P3:** end v_mean ≤ 1.5 × max(D); Jmin_traj > 0 at every window; no serious piece.
- **PASS** = P1, P2 and P3 on Z97 and Z99.

## Predictions

- Aspect at frame 240 ≥ 1.35 on both seeds (X: 1.17 / 1.31; dF_c toward a 3D target: 1.46).
- 3D IoU gain over the twin ≥ +0.10 on both.

Risks:
- the body arrives at t_k still moving and overshoots after it;
- without the soundness terms the segment's push may spray or fold the surface (P3's pieces, Jmin);
- the final shape and the rest are read after the settle as before.

## If it fails

The next measurement is named; no weight is added. If P3 fails, the next measurement is the per-window speed and pieces
across t_k.
