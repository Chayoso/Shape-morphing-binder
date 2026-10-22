# D159 resume (agent D159), written 2026-10-21 22:37 CDT

## What D159 is
The render target pictures carry the mesh relief: `--render_target mesh_shell` (wip/d159, now 0480659 = rt1 + the option,
default off). The target surface is the mesh grown by the run own exterior offset, measured LOCALLY over the exterior
field radius (amendment 2). Pre-registration and amendments 1-2: docs/experiments.md, D159 entry (settled-base).
Regions (fixed before any run): output/gpu/d159/step0/regions_{bunny,dragon}_b5.4.npz (amendment 1).

## Running
- Chain PID 2079254: `tmp/d159/d159_chain.sh "V R X G"`, log output/gpu/d159/status.txt.
  - Card: lock gpu_locks/gpuN.lock with info "agent D159". It claims a card free for 900 s, never GPU 2.
  - It releases the card between phases V+R / X / G.
  - V: the full suite on repo_d159 (gates/suite.log, suite.rc), the digest, det S1 vs rt1 S1 (SAME), det M1 mesh_shell.
  - R: 300k bunny/dragon M97 (--render_target mesh_shell, seed 97), then P99 twins (--render_weight_scale 0, seed 99).
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
