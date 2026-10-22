# D159 resume (agent D159), written 2026-10-21 22:37 CDT, updated 22:45

## What D159 is
The render target pictures carry the mesh relief: `--render_target mesh_shell` (wip/d159, now 0480659 = rt1 + the option,
default off). The target surface is the mesh grown by the run own exterior offset, measured LOCALLY over the exterior
field radius (amendment 2). Pre-registration and amendments 1-2: docs/experiments.md, D159 entry (settled-base).
Regions (fixed before any run): output/gpu/d159/step0/regions_{bunny,dragon}_b5.4.npz (amendment 1).

## Running
- Chain PID 2121612 (restarted 22:42 after amendment 3; earlier chains 650919, 2079254 stopped): `tmp/d159/d159_chain.sh "V R X G"`, log output/gpu/d159/status.txt.
  - Card: lock gpu_locks/gpuN.lock with info "agent D159". It claims a card free for 900 s, never GPU 2.
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
- tmp/d159/d159_closeups.sh is running as PID 2144242. It waits until the main chain (PID 2121612) has exited, then
  claims ONE card by the same rule (d159_claim.sh).
- It renders the tag L97 and D159 M97 at kept frames 20, 40, 80 and the last:
  - bunny ears (look -0.316,0.71,0.15; az 35 el 18; dist 0.35);
  - bunny haunch wrinkles (0.315,-0.287,-0.103; az 83 el 0; dist 0.3);
  - dragon head and horns (-0.059,0.688,0.255; az 35 el 18; dist 0.35);
  - dragon belly scutes (0.29,-0.299,0.29; az 125 el 5; dist 0.3).
- Sheets: output/gpu/d159/closeups/{bunny_ears,bunny_wrinkles,dragon_head,dragon_scales}.jpg (d159_closeup_sheet.py).
