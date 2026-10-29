---
name: frame-by-frame-particle-forensics
description: "User directive 2026-10-01 ~23:30 CDT: quality defects (blur, floating Gaussians, holes, oscillation) are now investigated frame by frame and particle by particle, not by aggregate metrics: find the frame where the artifact appears, read that frame's JSON values, identify WHICH particles sit where and why, and trace the causal chain to the visible defect."
metadata:
  type: feedback
---

From now on a visual defect is investigated as a forensic case, not as a metric:

1. Look at the rendered frames one by one and name the frames where blur, a floating Gaussian, a hole (e.g. near
   the ears) or oscillation appears.
2. For those frames read the run's JSON values of the matching window (energies, rejections, velocities, the
   active-set records …).
3. Identify the particles responsible by index: which particle is where, why it is there (what pulled or left it
   there: which term, which window, isolated or not, its neighbours, its splat size), what problem that causes, and
   how that produces the blur / oscillation / hole seen in the frame.

**Why:** the user said it after two days of gallery metrics: medians (silhouette IoU, thin-uncovered, jitter) said
the frozen formulation is fine while the 300k dragon still looks badly blurred. Aggregate numbers do not explain a
visible defect; a particle-level cause does ("이젠 메트릭이 아니라, frame by frame으로 … 몇 번 particle이 왜 저기에
위치해서 어떤 문제를 야기해서 그 결과 blur/oscillation/hole이 발생하는지").

**How to apply:** when asked about render quality, render the frames (4K splats), view them (Read the images, full
frame and crops), map an artifact's pixels back to particle indices with the renderer's camera, follow those
particles through the archive frames and the per-window records, and report the chain cause → particle state →
visible artifact with frame numbers and particle indices. Metrics may support the story but never replace it. Do not
propose a fix before the particle-level cause is shown ([[no-parameter-only-fixes]]). Keep the archive of the run
being examined (disk rule: [[cleanup-over-100gb]]).

See [[formulation-stop-rule]], [[tail-flicker-and-ear-dropout]], [[doc-only-current-gsplat]].

**Added 2026-10-02 (what the first two cases taught):** before reading a morph's frames, draw the run's OWN TARGET
SAMPLE with the same 4K renderer (`scripts/probes/settled/target_ref.py` + `compare_ref.py`) and put it beside the
last frame. On the 300k bunny and dragon the target sample is as blurred as the morph's end state (feathered ears and
horns, no scales): the end blur is the display renderer's and the sampling's, not the particle arrangement's. The
morph's own defects are transient (material in the concave gaps: the bunny's web between the ears, the dragon's
mouth / neck / tail folds). Tools: `frame_sheets.py`, `frame_forensics.py`, `web_probe.py`, `--term_dump` +
`term_probe.py` / `flux_probe.py` / `web_terms.py` (which term pulls a particle, whether the pull is followed).
Results are in docs/experiments.md under D11.

**D12 (2026-10-02):** the display blur is mostly the SAMPLE, not a filter: at 300k the surface has one particle per
13.8 px at 4K (lattice step 0.062 wu); the dragon's teeth and scales are smaller and are absent from the target
sample itself. Normal error vs the mesh: 18.5° (300k) → 13.3° (1.2M) → 10.9° (2.4M); narrowing the renderer's
widths gains ≤ 20 % on the target and worsens the morph frames. `scripts/probes/settled/render_axes.py`. The user
also confirmed the formulation freeze on 2026-10-02 by adopting the plan renderer → runtime → video (body-frame
objective deferred as future work).
