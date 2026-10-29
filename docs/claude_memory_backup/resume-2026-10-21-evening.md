---
name: resume-2026-10-21-evening
description: "Handoff 2026-10-21 ~18:00 CDT before a local shutdown: demo done, local cleanup, new overview figure, D156 FAIL, D158 cause, render-own-contribution tracks D159/D160/D161 + D156b, B9b assim gate, SHOWCASE 2 cards, D155 dragon re-run; per-agent RESUME files on hyde06"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-21T23:11:33.773Z
---

**Done today (after the 17:00 demo):**
- **Demo folder:** output/results_2026-10-21/demo_studio is relabelled with the arm names ([[arm-naming]]).
  - Drop scene: drop_bunny_300k_{main,higher,gentler}. The bunny tips onto its head after the bounce in every version; the highest drop is only 2.4 body heights.
  - D155 bunny 300k frozen | ramp side-by-side.
- **Local cleanup:** everything was archived to C:\dev\Shape-morphing-binder_archive_2026-10-21.zip; only the demo, paper evidence and running-agent folders are kept ([[local-results-layout]]).
- **New pipeline overview** (the user's sketch layout, SCA style):
  - Layout: the morph strip on top, a funnel for one window, three stages (MLS-MPM / Exterior Surface / Differentiable Rendering), the adjoint box, and the control update on the left.
  - **Spray / cleanup is out of the overview only.** The user insisted: the text and code keep E_clean.
  - Files: docs/figs_ch4/pipeline_overview.{svg,tikz} plus a preview PNG and pipeline_overview_tikz.pdf. The old one is in src_pipeline_overview/previous_detailed/.
  - Chapter4(.tex, _ko) captions were updated.
  - The Chapter 4 files are still untracked/uncommitted (the user has not asked for a commit).
- **B9b** (world-fixed standing target with gravity) passed 40k and 300k. The end is not at rest because assimilation absorbs the sag. A 40k gate of "assimilation reads only the control's share" was told to run on wip/b9b, default off.
- **D156:** visual hull + true pictures, 40k. FAIL: the render cannot carve concavities. D156b (photo hull) is running.
- **D158:** the in-loop render cannot write D115 detail because its TARGET PICTURES are blurry. Neither the pixel size nor u is the cause. D159 (detailed target pictures) is running.
- **User goal ([[render-own-contribution]]):** the render's own contribution.
  - D160 (colour layout = the physics null space) was chosen by the user.
  - D161 (render-carved relief) is the user's idea.
  - The user widened the goal: anything by which the render changes the physics' direction counts, e.g. dragon scales and bunny wrinkles.
- **SHOWCASE** may use 2 cards (about 5 h left).
- **D155 dragon re-run** chain: tmp/d155/chain_demo300k_dragon.sh, PID 215529, output in output/gpu/d155/demo300k/.
- From ~17:50 another user (nurshat) occupies all 4 GPUs; our queues wait.

**On resume:**
1. Read each agent's RESUME file on hyde06:
   - tmp/d159 or tmp/d158/RESUME.md
   - tmp/d160/RESUME.md
   - tmp/d161/RESUME.md
   - tmp/d156/RESUME_d156b.md
   - tmp/b9b/RESUME.md
   - tmp/showcase/RESUME.md
   - tmp/d145/RESUME_r.md
2. Respawn readers or continue from them.
3. Pending user questions:
   - Whether the earlier "약간 후처리를 render loss가 하게" meant the spray cleanup. If so, run the experiment where the render replaces spray pull and near band.
   - The drop-scene height and the tip-over.
4. Slips today: `pgrep -u chayo -f` was used once by me and once by D158, read-only. The rule is restated to all agents.
