# D159 resume (agent D159), written 2026-10-21 22:37 CDT, updated 22:45

## What D159 is
The render target pictures carry the mesh relief: `--render_target mesh_shell` (wip/d159, now 0480659 = rt1 + the option,
default off). The target surface is the mesh grown by the run own exterior offset, measured LOCALLY over the exterior
field radius (amendment 2). Pre-registration and amendments 1-2: docs/experiments.md, D159 entry (settled-base).
Regions (fixed before any run): output/gpu/d159/step0/regions_{bunny,dragon}_b5.4.npz (amendment 1).

## Running
- Chain PID 2247716 (restarted 22:56 with a card free 600 s: D159 now has the queued agents priority; earlier chains 650919, 2079254, 2121612 stopped before any card): `tmp/d159/d159_chain.sh "V R X G"`, log output/gpu/d159/status.txt.
  - Card: lock gpu_locks/gpuN.lock with info "agent D159". It claims a card free for 600 s (D159_IDLE), never GPU 2.
  - It releases the card between phases V+R / X / G.
  - V: the full suite on repo_d159 (gates/suite.log, suite.rc), the digest, det S1 vs rt1 S1 (SAME), det M1 mesh_shell.
  - R: 300k bunny/dragon M97 (--render_target mesh_shell, seed 97, --term_dump output/gpu/d159/terms/NAME), then P99 twins
    (--render_weight_scale 0, seed 99), then T97 (the tag path again, seed 97, --term_dump): amendment 3.
    Two runs at a time on the card; each run is followed by tmp/d145/read300.sh (yardstick t3end_, display e3end_,
    a_, m_). After R, `tmp/d159/d159_readcpu.sh MESH` starts detached (PIDs in output/gpu/d159/readcpu_MESH.pid). It runs:
    - apart, pulse, visible_apart, pieces145;
    - R1 relief (read/relief_MESH.jsonl);
    - R2 course (read/course_MESH.jsonl);
    - R3 D157 (read/d157_MESH.jsonl).
  - X: studio renders of the tag L97 and D159 M97 (render_deliverable.py +fit=0 +field=zbs, az 35 el 18), then the
    1080p side-by-side output/gpu/d159/video/MESH300k_tag_vs_d159_1080p.mp4 (4K frames removed, cleanup.log).
  - G: 40k gallery M97 x 19 meshes, output/gpu/d159/g40/.
- History: the first chain (21:04) stopped at V1. The CUDA test asked for bit equality of two GPU builds (index_add
  atomics). The test was rewritten (sentinel mesh + measured run-to-run spread) in wip/d159 0480659. CPU part passed.

## Already read (no run needed)
- step0/step0.jsonl (one-number c) and step0_local.jsonl (local c): amendment 2.
- read/relief_tag_{bunny,dragon}.jsonl: R1 on the tag L97-L105, P97, P99, own.
- read/course_tag_{bunny,dragon}.jsonl: R2 L97 vs P97.

## Next steps (by hand after the chain)
1. After G is done: `bash /data/relcfd/chayo/physmorph_v2/tmp/d159/d159_g40read.sh` (CPU, D154 reader, read/g40_m97.jsonl).
2. After both readcpu are done (status "readcpu MESH: R3 (D157) rc"), run on the server with env
   `source tmp/d146/cache_env.sh; export CUDA_VISIBLE_DEVICES="" PYTHONPATH=repo_d159`:
   `/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python tmp/d159/d159_table.py output/gpu/d159/table`
   This writes table.txt, table.json, course.png and relief.png.
3. Close-ups, not scripted yet:
   - What: render_deliverable.py +frames=K-K +look=x,y,z +dist=0.3 at K = 20, 40, 80 and the last frame.
   - Arms: the tag L97 and M97.
   - Regions: the bunny ears and wrinkled region, the dragon horns/head and belly scutes.
   - Card: one card under the lock protocol.
4. Reading:
   - Pass P1-P4 as pre-registered. Note: D157 decision set = 14 measures (4 sil, 4 3D, 6 image), not 11.
   - Report the render->control->outcome summary (g_share, lambda) and use the arm names.
   - Write the D159 result into docs/experiments.md from a separate worktree (Co-Authored-By line).
   - Local folder: output/results_2026-10-21/d158_render_detail/d159/ plus a D159 section in its README_ko.md.
   - Report to the coordinator in Korean.

## Amendment 3 (22:42): R5 the per-term hold on the detail band
- The term dumps are in output/gpu/d159/terms/{bunny,dragon}300k_{M97,T97}/terms_NNNN.npz: x, g_ot, g_surf, g_near,
  g_spray and g_rend, at every committed window. Each run makes about 2-4 GB.
- DISK: /data/relcfd/chayo is at 459 of 500 GB. Read the dumps (d159_terms.py, to be written) and delete them at once.
  Log the deletion in output/gpu/cleanup.log.
- If D159 fails P1 with P2-P4 held: pre-register the combined arm (mesh_shell + D161b proximity reach). Get the
  definition from the coordinator.

## Close-ups (added 22:50)
- tmp/d159/d159_closeups.sh is running as PID 2247793. It waits until the main chain (PID 2247716) has exited, then
  claims ONE card by the same rule (d159_claim.sh).
- It renders the tag L97 and D159 M97 at kept frames 20, 40, 80 and the last:
  - bunny ears (look -0.316,0.71,0.15; az 35 el 18; dist 0.35);
  - bunny haunch wrinkles (0.315,-0.287,-0.103; az 83 el 0; dist 0.3);
  - dragon head and horns (-0.059,0.688,0.255; az 35 el 18; dist 0.35);
  - dragon belly scutes (0.29,-0.299,0.29; az 125 el 5; dist 0.3).
- Sheets: output/gpu/d159/closeups/{bunny_ears,bunny_wrinkles,dragon_head,dragon_scales}.jpg (d159_closeup_sheet.py).

## State at 2026-10-22 01:00 CDT (first reading in docs, commit a1545ff0)
- Done:
  - the gates (V1 442 passed exit 0, V2, V3 SAME, V4);
  - the runs bunny/dragon300k_M97 and P99;
  - the early readings in read/: relief_tag_*, relief_early_*, course_tag_*, course_early_*, course_floor_*,
    terms_early_*, window.jsonl.
- Verdict so far: P1 FAILED on both meshes.
  - The particles carry about 4/5 of the region relief anyway; the exterior (r = 3) hides it.
  - After arrival the render is starved: lambda and the step fall 20-70x and the body moves 0.01 spacing per window.
  - The 40k gallery is NOT run. drop_g.sh (PID in tmp/d159/drop_g.pid) stops the chain after X while it waits for G.
- Running:
  - chain 2247716 (T97 pair, then readcpu detached, then X renders);
  - drop_g.sh;
  - the close-up waiter 3280248 (starts when the chain has exited; replaced 2247793).
- Next by hand:
  1. Run d159_table.py (output/gpu/d159/table): table.txt, course.png, relief.png, holds.png.
  2. Copy the figures, videos (output/gpu/d159/video/*.mp4) and close-up sheets (output/gpu/d159/closeups/*.jpg)
     to the local output/results_2026-10-21/d158_render_detail/d159/.
  3. Finish the README D159 section and the docs reading: P2-P4 from R3/R4, T97s R5 against M97s.
  4. Delete the frames12 of P99/T97 after reading and log it.
  5. Proposal to the user (via the coordinator): the render authority after arrival (the frozen lambda rule), together
     with D159 pictures and exterior radius 2.5. D161bs proximity reach is not the main detail-band gradient at 300k
     (R5: the near band is the largest detail-band term).

## D159c (pre-registered 01:04, commit f65e8574): STAGED, NOT LAUNCHED
- What: D159 pictures + --lambda_ema 0 (lambda held from the first window), 300k bunny/dragon seed 97, term dumps.
  The runs are named bunny300k_C97 / dragon300k_C97.
- Launch ONLY when D160c (output/gpu/d160/gate_H/) shows the colour still moving after arrival with lambda held, or
  when the coordinator says so. If D160c shows no movement, do NOT launch.
- Launch command (one card, lock protocol inside):
  `cd /data/relcfd/chayo/physmorph_v2/tmp/d159 && (setsid nohup ./d159c.sh > d159c.out 2>&1 < /dev/null & echo $! > d159c.pid)`
- When the runs end, d159c_read.sh MESH starts by itself. It writes R1 relief_c_*, R2 course_c_*, R5 terms_* (the dumps
  are deleted after reading), R6 window_c.jsonl, pieces and visible blobs.
- Then add C97 to d159_table.py (R5 names, R1/R2 labels) and read P1c-P3c.

## Close-ups (updated 01:04)
- The waiter is now PID 3280248; it adds a DISPLAY READING pair (bunny wrinkles and dragon scutes at 4 s and the end, tag
  and D159) drawn through the r = 2.5 field.
- The sheets are closeups/bunny_wrinkles25.jpg and dragon_scales25.jpg. Label them as a display reading only.
- The M97 term dumps were deleted at 01:03 (logged); their R5 is in read/terms_{bunny,dragon}.jsonl.

## 2026-10-22 02:52 CDT
- D159 second reading committed (c3b4c64a); the launch note for D159c is 3d2a3305 (settled-base).
- D160c after-arrival reading: the colour keeps falling after arrival in all 10 held-lambda runs (12-67 %) -> D159c ARMED:
  d159c_after.sh PID 4161998 (d159c.pid) waits for chain 2247716 (X renders on gpu3) and the close-up waiter 3280248,
  then execs d159c.sh (claims ONE card, runs bunny300k_C97 + dragon300k_C97, read300, then d159c_read.sh on the CPU).
- After D159c: read with d159c_read.sh outputs, delete output/gpu/d159/terms/*C97* and *_C97_frames12.npz after reading,
  log in output/gpu/cleanup.log. Also delete bunny/dragon300k_M97_frames12.npz after the close-ups.
- Still to do locally: re-run d159_table.py (short holds.png titles), copy videos + closeups + figures to
  output/results_2026-10-21/d158_render_detail/d159/, finish README_ko.md D159 section.

## CHECKPOINT 2026-10-22 03:57 CDT (local shutdown)
Running, all detached (PPID 1, own session, no tty):
- 2247716 d159_chain.sh V R X G: X step on gpu3 (lock gpu_locks/gpu3.lock "agent D159"); bunny side-by-side + video done
  (output/gpu/d159/video/bunny300k_tag_vs_d159_1080p.mp4), dragon renders done 03:55, dragon side-by-side + video next,
  then the 4K frames removed and gpu3 released. G already ran (nothing left after X).
- 3280248 d159_closeups.sh: waits for 2247716, claims ONE card (free 900 s, never GPU 2), sheets in
  output/gpu/d159/closeups/{bunny_ears,bunny_wrinkles,dragon_head,dragon_scales,bunny_wrinkles25,dragon_scales25}.jpg
  (the *25 pair = r = 2.5 field, labelled DISPLAY READING).
- 4161998 d159c_after.sh: waits for 2247716 and 3280248, then execs d159c.sh (ONE card; bunny300k_C97 + dragon300k_C97,
  --render_target mesh_shell --lambda_ema 0 --term_dump; read300 on the card; then d159c_read.sh bunny/dragon on the CPU,
  which deletes the C97 term dumps and logs it). Launch condition read and committed (3d2a3305).
Progress: tail output/gpu/d159/status.txt (lines "D159c ... rc/wall", "D159c readcpu MESH done").
After D159c (reader commands, python = /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python, CPU only):
- R6 / R1 / R2 / R5 land in output/gpu/d159/read/{window_c.jsonl, relief_c_*.jsonl, course_c_*.jsonl, terms_*.jsonl}.
- Add C97 to tmp/d159/d159_table.py (R5 names, R1/R2 labels), then
  $PY tmp/d159/d159_table.py output/gpu/d159/table ; read P1c-P3c of the D159c pre-registration (docs/experiments.md).
- At rest / pieces: output/gpu/d159/pieces.jsonl (run bunny300k_C97 / dragon300k_C97), pulse_/apart_/vis_ logs.
Cleanup after reading: output/gpu/d159/{bunny,dragon}300k_{M97,C97}_frames12.npz (after the close-ups), log in
output/gpu/cleanup.log. /data/relcfd/chayo at 467 GB at 02:50 (limit 500).
Local (when back): copy video/*.mp4, closeups/*.jpg, table/* to
output/results_2026-10-21/d158_render_detail/d159/ ; bunny video already copied; README_ko.md D159 section updated
through the D159c launch.

## 2026-10-22 10:50 CDT: D159 and D159c READ, nothing running, no lock held
- Everything ran overnight: X renders (videos in output/gpu/d159/video/), close-ups (04:17-04:24), D159c (04:39-04:58,
  bunny300k_C97 / dragon300k_C97, rc 0). The CPU readers lost vis / pieces / R5 / R6 to a variable clash (cache_env.sh
  sets C); re-read 10:19-10:28 with d159c_readfix.sh, d159c_terms.sh, plus D157 by d159c_d157.sh.
- D159c close-ups rendered on gpu1 (10:21-10:24, released), sheets now three rows (tag | D159 | D159c).
- Verdict: P1c / P2c / P3c fail; not a candidate. Reading in docs/experiments.md (D159 entry, "D159c read").
- Deleted and logged: C97 term dumps (2.0 GB), M97 / C97 kept frames (3.6 GB). Left: *_end1.npz (7 MB each), JSON, logs.
- Table: $PY tmp/d159/d159_table.py output/gpu/d159/table (R1-R6 with C97).
