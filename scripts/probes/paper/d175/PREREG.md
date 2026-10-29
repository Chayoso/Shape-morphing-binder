# D175 pre-registration: picture keyframes, first experiment

Written 2026-10-29 11:15 CDT, before any keyframe run. Design: docs/design/keyframe_render_control.md (2026-10-28), section 6,
with the method in use now (Hu) in place of D166. Code: wip/d175 = wip/d172 e370038a + the storyboard (server repo_d175).

## Method (what the runs execute)

- Storyboard ownership (design option C). A window whose end frame lies in (0, 240] serves the keyframe:
  - the render reads the keyframe pictures P_k;
  - the final target's transport, surface proximity and cleanup (spray W1, near band) are absent; the relaxation has no relief;
  - only the released motion and the end drift stay;
  - PCGrad (`pcgrad()`) is unchanged and protects only those two terms.
- Clock: a segment window with no accepted step (null, gradient converged, or outer reject) commits its zero-control rollout,
  the settle window's code. So frame 240 is t = 1 s on every arm. No stop rule fires inside a segment. Each storyboard entry
  opens a selection epoch, and the stops and the delivery read the final target's epoch.
- **Render weight per control block (the decision).**
  - λ is calibrated ONCE, by a pre-pass: one gradient at the source on the FINAL target, the value a run without keyframes
    takes at its first window. It is then held (`--lambda_ema 0`).
  - Inside a keyframe segment, the held λ goes on BOTH blocks (dF_c and u). The rule's per-window value would read
    |g_p| of the soundness terms alone (≈ 0), and a bulk pose change needs the render's authority on dF_c.
  - From t_k on, Hu as it stands: the held λ on u, the rule's per-window value on dF_c (`--render_split u`). The merit,
    line search and stops keep the held λ.
- Keyframe E: the run's own source sample (and 7 further draws of the isosphere, seeds s + 1000 + 100k, k = 1..7, in the
  source's frame and density-matched scale), stretched 1.3× along y and 1/√1.3 along x and z about the source sample's
  centroid. So the volume and the centre are kept, and the map is rotation-free.
  - File: output/gpu/d175/assets/kf_s{97,99}.npz, built by tmp/d175/make_kf.py.
  - The pictures are drawn with the final pack's pitch, kernel, lattice and camera box, as the mean over the 8 samples.
  - An independent ninth draw (seed s + 2000) is the 3D reference only. The run never reads it.

## Arms (sphere → bunny, 40k, Hu = `--lambda_ema 0 --settle_windows 30 --no-settle_relax --render_split u`)

| arm | seeds | flags beyond Hu |
|---|---|---|
| D: dF_c + layer u + render loss (full), no keyframe | 97, 99 | (none) |
| S: dF_c + layer u + render loss (full), with the storyboard | 97, 99 | `--keyframe kf_sS.npz --keyframe_time 1.0` |
| S twin: dF_c + layer u (twin), with the storyboard | 97 | the same + `--render_weight_scale 0` |
| N: naive picture swap (full) | 97 | the same as S + `--keyframe_geometry final` (pictures swapped, the final geometry, cleanup and relief kept; the clock and the held λ on both blocks as S) |

## Metrics (reader tmp/d175/d175_read.py, on the card after the runs)

- KF-IoU: the silhouette IoU (alpha > 0.5) at frame 240 against P_k.
  - Body pictures: the run's own exterior operator (a field found at the frame's state, its discs, the soft silhouette).
  - P_k: the mean of the 8 keyframe samples' alpha.
  - Averaged over the 18 run views, and over 12 held-out views (6 azimuths offset half a step, elevations ±0.25).
- 3D IoU: occupancy (≥ 1 particle) on the loss grid (the loss cell) at frame 240, against the independent keyframe draw.
- The KF-IoU curve over the kept frames 160–320 (run views).
- G = (IoU − IoU_twin)/(1 − IoU_twin). S97 and S99 are both read against the one twin, seed 97.
- Final shape, after the settle (the delivered last frame):
  - the silhouette IoU (alpha > 0.5, 18 run views) against an independent target sample (seed s + 2);
  - the run JSON's Chamfer and thin_uncovered.
- Rest:
  - the end v_mean (the delivered last record);
  - `Jmin_traj` > 0 at every committed window (optimised, hold and settle);
  - serious pieces: a non-largest set (1.5-pitch link) of ≥ 10 particles in kept frames spanning ≥ 120 frames (3 windows).
- Render influence:
  - λ (held, from the pre-pass);
  - g_share, mean over the segment windows and over the final-target windows;
  - g_raw_cos in the segment windows (arm N: the conflict);
  - the twin against the full arm and against the two D seeds' spread.

## Pass bars (the design's, on both S seeds)

- P1 (keyframe):
  - G ≥ 0.6 on the run views and ≥ 0.5 on the held-out views;
  - 3D IoU(S) ≥ 3D IoU(twin) + 0.10;
  - KF-IoU(S) ≥ KF-IoU(D, same seed) + 0.05.
- P2 (final shape):
  - final silhouette IoU ≥ min(D97, D99) − 0.005;
  - Chamfer ≤ 1.1 × max(D97, D99);
  - thin_uncovered ≤ max(D97, D99) + 0.02 (within 2 points).
- P3 (rest):
  - end v_mean ≤ 1.5 × max(D97, D99);
  - `Jmin_traj` > 0 at every window;
  - no serious piece.
- PASS = P1 and P2 and P3 on S97 and on S99.

## Predictions (the design's)

- The twin stays a sphere: KF-IoU ≈ 0.79.
- S reaches ≥ 0.92.
- N: `g_raw_cos` < 0 in ≥ 4 of the 6 segment windows, and G_N < 0.2.
- S ends like D, about 1 s later.

## If it fails

- No limitation write-up. The next measurement is named in the summary:
  - S rises but falls short: Treuille's blurred comparison (the design's main risk).
  - P2 or P3 fails: the per-window terms of the first final-target windows (transport, render share, step) against D's.
- If it passes: one showcase at 300k, bunny → dragon, with a letter-shaped or pose silhouette keyframe.
  - Deliverables: frames at t_k against the keyframe pictures, and a twin | full video.
