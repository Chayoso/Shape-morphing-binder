---
name: resume-2026-10-15-night
description: "21:40 CDT handoff (user ended the day). Tag freeze-2026-10-15 cut; post-tag evidence (446 items) running; D147 final scene s97/twin/s99 all NOT clean (escape → inf transport → uncontrolled spring-back); B13 300k waiting for a card; open user decisions (floater display rule, D147 delivery)"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-16T02:43:27.127Z
---

State at 2026-10-15 21:40 CDT. The user said "일단 실험 돌려놓고, 오늘은 여기까지 하자". Every chain below runs detached on hyde06 (parent 1); the local agents die with the session. Disk: 328 G of the 500 G limit. GPU 3 belongs to another user (nurshat, fluidrecon); never touch it.

**Freeze.** Tag `freeze-2026-10-15` = fc02a1a on refreeze-cand (pushed): D135 + D146b + D149 as defaults, B13 merged (default off), no floater fix. See [[freeze-plan-2026-10-18]].
- Post-tag determinism gate passed at 19:01. repo_ts and repo_tgf both match repo_t: SAME on 4 arrays, 37 metrics, 556 history fields.
- D145 pool: `tmp/d145/pool.sh` r1–r3 runs main.queue with the 446 post-tag items appended. posttag_watch set max_cards to 2. The first B1 runs on the tag started at 20:28 / 20:58.
- Outputs go to output/gpu/d145t_*. The reading is the D145 evidence work (no agent running now): B1/B2/B3/B7/B8/B9/B10, plus A6 ablations NRD/NUC/NSS.

**D151 `--layer_link`: NOT adopted.** Read and committed on settled-base: c19d4fb + 731ba8a.
- Dragon visible pieces: −22 % (arm) and −47 % (twin) at 1.5–6 s. The early crevice beads stay.
- The bunny's blobs nearly vanish.
- Momentum fails on all 4 runs: the link breaks the per-step rigid-mode removal.
- Local: output/results_2026-10-15/d151_gate/ (4 side-by-side videos, sheets, README_ko).

**D147 final throw scene.**
- Setup: `repo_d147fin`, chain `repo_d147fin/scripts/probes/d147/chain19.sh`; log output/gpu/d147/chain19.log; runs in output/gpu/d147/runs/.
  - wip/d147 caff236 = the tag merged with the scene. Suite 435 passed after a known flaky test, amended in 50a5fbf. Pre-registration: 89a59ee.
- s97: t_cut 16.0. NOT clean: 67 far frames, largest piece 14 particles at 6.4–15.35 s, 5.22 sp.
- s97 render-off twin: t_cut 15.0. NOT clean: 32 far frames, a 24-particle piece, 5.84 sp.
- s99: FAILS the judge with end error 0.093 (silhouette 0.937 / 0.877); t_cut 13.8. The agent's mechanism:
  1. After the rebound, body 0 stretches to 7.06 wu, with up to 1133 particles outside the loss raster.
  2. The transport goes inf for 14 windows (12.67–14.83 s), so both bodies get zero-control commits for 2.3 s.
  3. With `--assim 0` the rest shape is the sphere, so both bodies spring back toward it.
  The agent says s97 shows the same mechanism, weaker.
  - s99 is also the worst on clean before its cut: 58 far frames, 2688 particle-frames, up to 103 particles at 8.69 sp. All of it is body 1's ears and head (0.75–0.94 of its height), from 10.95 s (2.1 s after landing), and it grows once the controls stop at 12.67 s.
- Renders were in progress at 21:40 (s97 done, the twin's 4K contact done).
- On resume:
  1. Collect the table and videos to output/results_2026-10-15/d147_throw_final/.
  2. Compare with the interim s97 (`output/results_2026-10-14/d147_throw/300k_s97_4k/`: cut 15.30, 2 far frames of 1 particle).
  3. Give the user the choice: deliver the interim, or fix the "escape → inf transport → uncontrolled" defect. That is a definition question about loss-raster coverage and inf handling, so it is the user's decision after a measurement.

**B13 300k.** `tmp/b13/b13_chain300k.sh` on repo_ts: 36 runs, one card, claimed under the 10-minute-idle rule. Waiting for D147's card. The 40k reading is in docs (ce39674): force at scale 1, force_mf at 1/4.

**B12 read and committed** (wip/b12 a29e096 + aea2fe4, not merged). Table at output/results_2026-10-15/b12_render_only/. See [[competitor-landscape]].

**RGV.**
- The relabelled dragon 1080p is in output/results_2026-10-14/render_gradient_viz/v2_dragon/; the LinkedIn video is `dragon300k_AB2CD_2x2_side_1080p.mp4`.
- `tmp/rgv/rgv_queue3.sh` still waits for idle cards for the dragon 4K side/front and held3.
- The RGV redo on the tag is B7 in the post-tag queue (repo_tgf with rgv_patch2).

**User confirmation (21:55 CDT).** "gradient visualization 했던 방법이 제일 괜찮았던 거 같아. 그 쪽으로 fix하고... 이제 그냥 실험 돌리자고". Asked to choose, the user KEPT the tag freeze-2026-10-15 (the visualization code D135 + D146b + D149); D146b/D149 are not dropped.
- From now on: no more algorithm changes; just run and read the experiments on this tag.

**Open user decisions.**
1. Floater display: DEFERRED by the user at 21:50 ("떠 있는 거 뭐 어떻게 나중에 봐 보자고"). Revisit later, for the dragon in particular. Do not block anything on it. My recommendation: the main text and video stay unfiltered, with the numbers in the limitations. Any filter must be a disclosed component-size rule applied to every method, with the unfiltered video in the supplementary. Context: the user asked whether a post-process is enough for SIGGRAPH.
2. D147: which run to deliver, and whether to fix the escape defect.

**Ops notes.**
- Off campus, hyde06.dabh.io does not resolve; use `ssh hyde06j` through hyde01. The user re-added the key on 2026-10-15 ~17:50.
- posttag_local.sh uses the direct host name, so its steps were done by hand.

**D147 forensic (22:00 CDT; docs 2cd7b12 on settled-base).** The final runs differ from the interim in the controls dropping out during the recovery after landing, not in the landing squash, which is identical in all four runs. The grid moves the ears; u and the relaxation do not.
- Zero-control windows: interim 5, s97 13, twin 15, s99 20.
- Trigger (1): a commit-replay null in body 0's landing window at 11.33 s, in every final run and never in the interim.
- Trigger (2): the per-body loss raster ends at 5.28 wu. Body 0's rebound pushes particles past it, transport goes inf, and BOTH bodies lose control. The body keeps stretching while uncontrolled. The interim missed that line by 0.25 wu.
- With `--assim 0` (rest shape = the sphere) and `--drag 0` in the scene (pure fixed-corotated elastic, E 1.4e5, ν 0.2: NO damping at all), an uncontrolled body springs back toward the sphere.
- Attribution is not possible from one run each. The smallest set: the tag vs the 09b path at s97, 2 runs each, ~2.9 card-hours. NOT launched.
- Open user decisions (scene code only, not the frozen morph):
  - (a) the loss grid is too small for landings;
  - (b) one body's inf stops both bodies;
  - (c) a replay null commits the uncontrolled continuation;
  - (d) my proposal: damping on each body's deformation velocity only (rate 0.9/s, the single-body drag; it keeps each body's linear and angular momentum);
  - (e) run the attribution or not.
- The user posted on LinkedIn: "PhysMorph – Contact", the 8-s cut of s99 (`final300k_s99_1080p_cut_8s.mp4`, ending before the landing).
