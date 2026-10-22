# SHOWCASE resume note (written 2026-10-21 18:16 CDT, before the local machine shut down)

Pre-registration: docs/experiments.md "SHOWCASE" (commit 0fb226d, 12:14 CDT). Scripts: repo scripts/probes/paper/showcase/
(server copies here in tmp/showcase/). Code under test: repo_t1s (tag freeze-2026-10-15-rt1), defaults, seed 97.

## Running (all detached with setsid nohup; stop only by these PIDs or by touching output/gpu/showcase/STOP_<id>)
- c7  pid 205170  tmp/showcase/sc_chain3.sh c7 queue.txt  (10-min idle rule; holds gpu3; HOLD 3: bimba, cheburashka, ogre)
- c8  pid 264061  tmp/showcase/sc_chain4.sh c8 queue.txt  (15-min idle rule; has claimed heart, waits for a card)
- swap_c7 pid 264063  tmp/showcase/swap_c7.sh: when c7 releases gpu3 after ogre it stops c7 and starts c9 on sc_chain4.sh
- Locks: gpu_locks/gpuN.lock/info "agent SHOWCASE" (at most two). The coordinator: at most two cards; a card only after
  it has been idle (sc_chain4: 15 min, so queued agents go first); one card while any lock is "agent DEMO".
- Status log: output/gpu/showcase/status.txt (every run, its rc/wall/peak and its kept-frames path; renders; CPU readers).

## Done (rc 0, read)
- Gallery, sphere -> X, render arm L97: nefertiti, spot, maxplanck, cow, fandisk, bimba, cheburashka (ogre running).
  Renders made: nefertiti, spot, maxplanck (re-render queued at dist 1.12: neck cut), cow (re-render queued at dist
  0.75: small), bimba, cheburashka. Fandisk render deferred (queued).
- Pair: bunny -> dragon L97 (27.7 min; kept frames given to DEMO; studio render queued at the end).

## Queue left (tmp/showcase/queue.txt, read before every item; claims in output/gpu/showcase/claims/)
heart (claimed by c8), beast, A, C, V; twins maxplanck300k_P97, fandisk300k_P97; pairs teapot2bob, spot2cow,
armadilo2homer, bunny2happy (Happy Buddha in tmp/showcase/assets_extra/happy.obj); twins bunny2dragon300k_P97,
teapot2bob300k_P97; happy300k_L97 (sphere -> happy); renders: fandisk, cow (dist 0.75), maxplanck (1.12), the six B1 L97
(b1_*: d145t_b1 symlinks), bunny2dragon. Cameras: tmp/showcase/cams.txt (NAME AZ EL [DIST]).
When the queue is done the chains wait (polling every 2 min, no card held); touch output/gpu/showcase/END to let them exit.

## Where things land (output/gpu/showcase/)
- NAME.json / NAME.log / NAME_frames12.npz / NAME_end1.npz; readings t3end_, e3end_, a_, m_, apart_, vis_*, pieces_,
  endgen.jsonl, mid/NAME.json
- videos: video/NAME_main_4k.mp4, video/NAME_main_1080p.mp4; stills: stills/NAME_main/NNNN.jpg; sheets/NAME_main_sheet.jpg
- render frame folders are removed after both videos exist (cleanup.log); twins' kept frames removed after their readings.

## After the queue: table, montage (on hyde06), then collect locally
B=/data/relcfd/chayo/physmorph_v2; D=$B/output/gpu/showcase; source $B/repo_t1s/scripts/ops/hyde06_env.sh
cd $D; CUDA_VISIBLE_DEVICES="" $PY $B/tmp/showcase/sc_table.py $D/showcase_table \
  nefertiti300k_L97 maxplanck300k_L97:maxplanck300k_P97 spot300k_L97 cow300k_L97 fandisk300k_L97:fandisk300k_P97 \
  bimba300k_L97 heart300k_L97 cheburashka300k_L97 ogre300k_L97 beast300k_L97 A300k_L97 C300k_L97 V300k_L97 \
  bunny2dragon300k_L97:bunny2dragon300k_P97 teapot2bob300k_L97:teapot2bob300k_P97 spot2cow300k_L97 \
  armadilo2homer300k_L97 bunny2happy300k_L97 happy300k_L97 b1_bunny300k_L97 b1_dragon300k_L97 b1_teapot300k_L97 \
  b1_bob300k_L97 b1_armadilo300k_L97 b1_homer300k_L97
CUDA_VISIBLE_DEVICES="" $PY $B/tmp/showcase/sc_montage.py $D/montage/showcase300k_montage.jpg 3 400 <the same names>
# locally (Git Bash):
L=/c/dev/Shape-morphing-binder/output/results_2026-10-21/showcase300k
scp "hyde06j:$D/video/*_1080p.mp4" $L/video_1080p/; scp "hyde06j:$D/video/*_4k.mp4" $L/video_4k/
scp "hyde06j:$D/sheets/*.jpg" $L/sheets/; scp "hyde06j:$D/montage/*.jpg" $L/montage/
scp "hyde06j:$D/showcase_table.*" "hyde06j:$D/endgen.jsonl" $L/data/
Then: README_ko.md in $L, the result in docs/experiments.md (SHOWCASE entry), commit only showcase paths
(other agents share the checkout), push; Korean report to the coordinator.

## Update 2026-10-21 22:45 CDT (the user via the coordinator, ~23:00: render experiments D159 / D160 / D161 first)
- One card only from now on. c9 (pid 952104, gpu3) stops after its current item (teapot2bob300k_L97: run done, render
  running) by STOP_c9 and releases gpu3; it takes no new card. Its remaining items stay in the shared queue.
- c8 (pid 264061, gpu1) finishes spot2cow300k_L97 and one more item, releases gpu1; swap_c8.sh (pid 2099371) then stops
  it and starts c10 = tmp/showcase/sc_chain5.sh with MAXC=1 (never a second card) and render_waiting(): no claim while a
  D159 / D160 / D161 chain process is alive without a lock of its agent (it is waiting for a card), plus the 15-min idle
  rule. Stop c10 only by its recorded PID (output/gpu/showcase/chain_c10.pid) or touch output/gpu/showcase/STOP_c10.
- Done (rc 0, read, rendered): the 13 gallery meshes (nefertiti, spot, maxplanck, cow, fandisk, bimba, cheburashka,
  ogre, heart, beast, A, C, V; fandisk render deferred), twins maxplanck / fandisk P97, bunny -> dragon (render queued),
  teapot -> bob (render finishing). Running: spot -> cow (c8).
- Left in the queue (one card, in this order): armadilo2homer L97, bunny2happy L97, bunny2dragon P97, teapot2bob P97,
  happy L97; then the renders: fandisk, cow (dist 0.75), maxplanck (1.12), the six B1 L97, bunny2dragon.

## Update 2026-10-21 22:45 CDT (the user via the coordinator, ~23:00: render experiments D159 / D160 / D161 first)
- One card only from now on. c9 (pid 952104, gpu3) stops after its current item (teapot2bob300k_L97: run done, render
  running) by STOP_c9 and releases gpu3; it takes no new card. Its remaining items stay in the shared queue.
- c8 (pid 264061, gpu1) finishes spot2cow300k_L97 and one more item, then releases gpu1. swap_c8.sh (pid 2099371) then
  stops it and starts c10 = tmp/showcase/sc_chain5.sh.
  - c10 runs with MAXC=1, so it never takes a second card.
  - render_waiting(): c10 claims nothing while a D159 / D160 / D161 chain process is alive without a lock of its agent
    (such a chain is waiting for a card).
  - The 15-min idle rule still applies.
  - Stop c10 only by its recorded PID (output/gpu/showcase/chain_c10.pid) or by touching output/gpu/showcase/STOP_c10.
- Done (rc 0, read, rendered):
  - the 13 gallery meshes: nefertiti, spot, maxplanck, cow, fandisk (render deferred), bimba, cheburashka, ogre, heart,
    beast, A, C, V;
  - the twins maxplanck P97 and fandisk P97;
  - bunny -> dragon (render queued);
  - teapot -> bob (render finishing).
- Running: spot -> cow (c8).
- Left in the queue, one card, in this order:
  - runs: armadilo2homer L97, bunny2happy L97, bunny2dragon P97, teapot2bob P97, happy L97;
  - then the renders: fandisk, cow (dist 0.75), maxplanck (1.12), the six B1 L97, bunny2dragon.

## Update 2026-10-21 23:53 CDT
- c8 finished spot2cow and armadilo2homer and stopped; c10 replaced by c11 = sc_chain6.sh (pid 2646984), MAXC=1;
  render_waiting() now matches the render agents' chains as they run (d159_chain.sh, d160_chain.sh, d161_chain_b.sh):
  no claim while one of them is alive without a lock of its agent. Stop c11 by pid or touch output/gpu/showcase/STOP_c11.
- Added at the end of the queue: spotx2cow300k_L97 (spot yawed +90 deg to face the cow's +x; samples prepared).
