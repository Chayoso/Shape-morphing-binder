---
name: resume-2026-10-05-afternoon
description: "Handoff 2026-10-05 ~14:00 CDT — D94 kept (u free of rigid modes), D95 gradient videos (loss_viz style), D96 speed cause (exterior searches), D97 drift-source record running; server folders, disk, what to read next"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-05T18:53:31.227Z
---

State at 2026-10-05 13:55 CDT (server clock), branch settled-base, HEAD c7798d7, all pushed.

- **D94 kept** (0793cf9): u projected free of the 6 rigid modes after every step. Both arms drift 2–6× less (dragon L rot 0.072°→0.018°, COM 0.019→0.0096 pitch). Display ahead of the physics-only twin on all measures (common range after window 10: bunny 1−IoU −25/−27/−28 %, dragon −28/−38/−30 %; yardstick sil −47/−58 %). Still behind the twin on small remainders: dragon COM 0.0096 vs 0.0074 pitch, rotation 0.0178° vs 0.0168°; bunny rest at the end (+10 % kinetic at matched windows). Repeat spread at 300k ≈ 0.0013 IoU (not 0.0006).
- **D95** gradient visualisation (user asked "loss_viz.mp4처럼", 300k): `scripts/probes/settled/loss_video.py` now has a 4th panel (render push along the normal, red out / blue in) and colours each window on its OWN p99 (λ falls ~10×; window-1 scale left late frames blank). Local: `output/video_2026-10-05/gradient/` (README.md in Korean, gradviz_bunny300k_w.mp4, summary/maps/effect PNGs). Bunny numbers: render gradient 98–99.6 % in top 5 % particles, 85–95 % on the outer layer, pushes toward the target (+0.36…+0.82), nearly orthogonal to physics (cos 0.01–0.2). Dragon run with dump on GPU 2 started 13:44 (d95.sh: gradviz at stride 3 from the full archive, then frames12, render_influence, prune); `tmp/slim_dumps.py` slims non-10th dumps on the fly (disk).
- **D96** speed: 300k bunny 8.7 min (10-02) → 22–28 min now = windows ×1.6 (44→67–89) × window time ×1.75. Profile: the exterior's disc searches (Zhu–Bridson at 3.2M lattice nodes, ~2.3 per window) are the window's extra ~5 s; render pictures at 188 px only 0.6 s. Exact speed-up committed (4e49d17): field value-only at corners, search 1.77→1.16 s, bitwise-identical discs.
- **D97 running** (repo_r82): record grid/spacing/u/relax/body net translation+rotation per window. Bunny L/P on GPUs 0/1 from 13:53 (+runeval3); dragon L/P wait for the D95 dragon archive to go (GPUs 3/2). Read with the history fields `*_com`, `*_rot`; expectation: relaxation carries the drift → next fix = same definition as D94 for that part.
- Server: connect `ssh chayo@hyde06.dabh.io` directly; output 28 GB at 13:50 (dragon archive will add ~17 GB temporarily). Helpers in tmp/: allframes.py, matched_rest.py, time_rows.py, ext_time.py, ext_same.py, slim_dumps.py, d95*.sh, d96.sh, d97.sh; repos r80 (D94), r81 (+exterior speed-up), r82 (+D97 record).
- Lessons today: read display gaps over the common range after window 10 (render_twin_plot) and at matched windows, not per-run means (run lengths differ); profile before attributing slowness; an outer no_grad context breaks autograd.grad outside an enabled block.

Related: [[render-must-beat-physics-300k]], [[resume-2026-10-04-night]], [[runtime-findings-2026-10-02]], [[hyde06-ops-etiquette]], [[cleanup-over-100gb]]
