---
name: local-results-layout
description: "Local results layout: output/results_YYYY-MM-DD/ + output/README.md; cleanups archived to C:\dev\Shape-morphing-binder_archive_<date>.zip (10-07, 10-09, 10-14, 10-21); since 10-21 only demo + paper/Ch4 evidence + running-agent folders are kept"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-09T02:58:07.013Z
---

User request 2026-10-07: "중간 결과들 정리 … image 관련 된 것들 (ppt) 다 지워버리고, 날짜 별로 딱 필요한 결과들만". Chosen: archive then delete.

**Kept (output/, 52 MB):** `results_2026-10-06/` (D105 base 4K bunny/dragon, D115 detail-layer 4K bunny/dragon, dragon side + front|side, `render_proof/` = D121 figure + numbers) and `results_2026-10-07/` (D120 D105|D120 full morphs ×3, D123 r 3.0|2.5 full ×2, detail-layer before/after ×6). `output/README.md` lists each file.

**Archived:** `C:\dev\Shape-morphing-binder_archive_2026-10-07.zip` (408 MB): `docs_figures/` (every pipeline-overview variant incl. the confirmed final = `pipeline_overview_final.pptx` (= v2), parts/, surface_before_after, relief_relaxation, gradient_flow banners, outer_layer_with_discs, surface_disc_rect) and `output/<old folders>` (video_2026-10-03…07, forensics, r300_page, before/corrected, index.html).

**Standing rule (user 2026-10-08: "날짜별로 결과 나오면 local에 잘 정리 해 줘"):** every stage's viewable/reportable result is copied to `output/results_<YYYY-MM-DD>/` (production date, CDT) as it is produced, with one line per file in `output/README.md` under that date; only what matters (final videos, side views, one table per stage), studio look; large intermediates stay on hyde06. Agents running experiments get this instruction.

**How to apply:** new local results go into `output/results_<date>/` with a line in `output/README.md`; figures for the deck are not kept in the repo (the user has them in their deck; extract from the zip if needed). Server results are separate (B/output/gpu, < 45–50 GB rule).

**2026-10-14 tidy (user: "일단 이거 고정하고 한 번 정리하자"):**
- Superseded files of the day are archived in C:\dev\Shape-morphing-binder_archive_2026-10-14.zip (47 entries) and removed locally:
  - the RGV first design V1/V2/V3 and its preview tiles;
  - the D147 old-definition and follow-camera videos and frames;
  - the 300k render logs.
- output/README.md was updated, and results_2026-10-14 is at 129 MB.
- The RGV design is FIXED as A · B2 · C · D (+U).

**2026-10-21 cleanup (user: "이전 결과들 다 지워버리고, 이제 demo 및 진짜 특별한 결과들만 남기자"):**
- All of output/ (809 files) was archived in C:\dev\Shape-morphing-binder_archive_2026-10-21.zip (739 MB; the entry count was checked against the disk), then deleted except the files below. output/ is now 373 MB, 324 files.
- Kept: results_2026-10-21/{demo_studio, showcase300k, d157_metrics, b9b_gravity_world, d156_image_target}, results_2026-10-20/d145_external, and results_2026-10-16/render_gradient_viz_tag.
- Also kept, because docs/figs_ch4 make_*.sh re-extract frames from them:
  - results_2026-10-15/d151_gate/{bunny,dragon}300k_side_SL_vs_NL.mp4;
  - results_2026-10-17/d147_throw_fixed/stiff300k_s97_4k_cut.mp4.
- The rule now: keep only the demo, paper / Chapter 4 evidence, and the folders of running agents. Everything else gets archived and removed.
