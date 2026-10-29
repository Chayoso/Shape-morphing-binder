
## Segment one-step reading (2026-10-29 12:37 CDT; output/gpu/d175/forensic/seg/seg_read.md, states state_w1..6.npz)

**When and why the bulk push collapses** (S97 rerun, 40k; Adam's first step at each segment window's start).

1. **The hole term carries the pull, and thin pole spikes meet it.**
   - The silhouette deficit (hole_core) carries 77–87 % of the render's dF_c gradient norm (cos 0.84–0.96 with the
     whole term). About 97 % of that term sits in the pole pixels.
   - From window 1 to 6, the pole pixels' coverage rises from 0 to 0.83, while the body's projected particle count there
     reaches only 0.30 of the keyframe's.
   - The hole term falls by 87 %, the projected-count mismatch by only 27 %, and the aspect only to 1.08.
   - The exterior's soft alpha saturates on a thin column, so the hole term is met with 30 % of the mass.
2. **The exterior steps flip.**
   - At windows 2 and 4 every exterior term (hole, area, edge, shading) and the particles' own silhouette move the bulk
     against the stretch: cos −0.45 / −0.43 below 3 pitches, aspect −0.017 / −0.019 per step.
   - Windows 1, 3 and 6 move it with the stretch (+0.49 / +0.22 / +0.36).
   - The projected-count (X-ray) picture moves it with the stretch in 6 of 6 windows (aspect gain +0.001 … +0.048 per
     step).

**Proposal, pre-registered in tmp/d175/PREREG_b.md (12:38 CDT, before any run):**
- `--keyframe_compare column`: inside a segment the keyframe is compared as the particles' projected count picture,
  mean (c(x) − c_k)² / mean c_k².
- It lives in render_loss.count_pictures / d_count, target.keyframe_counts, Exterior.counts (keyframe exteriors only), and
  objective.render_terms (the counts branch).
- The final-target render term is unchanged.
- Arm X = 97 / 99 at 40k under the 208 bars, against 208's D97 / D99 / T97 (job 200_d175_col, gate c first).
