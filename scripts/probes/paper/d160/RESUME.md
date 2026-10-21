# D160 RESUME (agent D160; written 2026-10-21 18:12 CDT before a local shutdown)

Pre-registration: docs/experiments.md "D160, the render routes material by its colour" (settled-base 30b9640, 17:58 CDT).
Code: branch wip/d160 (origin), 89bf653 = rt1 6c9b0e9 + --appearance SCENE (default off). Deployed: repo_d160, repo_d160s
(frames12 variant via tmp/d160/patch_s.pl). Scripts: tmp/d160/ (copies in scripts/probes/paper/d160/).

## Running (detached, own sessions; check with ps -p PID only)
- PID 573927  `bash tmp/d160/d160_chain.sh "V R C M"` -- waiting for a card (0/1/3, never 2) idle 12 one-minute samples;
  claims gpu_locks/gpuN.lock (info "agent D160"), releases by trap. Log: output/gpu/d160/status.txt, tmp/d160/chain.out.
  - V gates (~40 min): suite repo_d160, pinned + D160 tests, det runs S1/WH vs rt1 S1 (SAME), T0 vs TB traj SAME, LB acts.
    Any failure -> "STOP:" in status.txt, no runs.
  - R: 20 runs 40k (bunny half/band/ears/turn, dragon band) x L/P x 97/99 -> output/gpu/d160/runs40/ (~1.5-2 min each)
    + D155 card readers (t3end_, e3end_, ind.log). Ends with "R done".
  - C: tmp/d160/d160_read.py on all runs -> output/gpu/d160/read_colour.jsonl ("C colour readings" line).
  - M: tmp/d160/d160_studio.py seed-97 pairs -> output/gpu/d160/video/{mesh}_{scene}_97/{side_by_side.mp4,end.png}.
- PID 601586  `bash tmp/d160/d160_after.sh` (CPU, no card) -- waits for "R done": pieces145 (pieces.jsonl), d155_gate_read
  (gate_d160.jsonl), D157 reader on L runs (d157_L.jsonl); then waits for "C colour readings" and runs d160_gate.py ->
  output/gpu/d160/gate/{gate.json,table.md,fig_*.png}; logs "[after] gate read: ..." in status.txt.

## If a process died, re-run (idempotent: finished runs are skipped by rc files)
  cd /data/relcfd/chayo/physmorph_v2/tmp/d160 && nohup setsid bash d160_chain.sh "R C M" > chain2.out 2>&1 < /dev/null &
  (use "V R C M" if the gates did not pass yet; check gates/ *.rc and status.txt first)
  nohup setsid bash d160_after.sh > after2.out 2>&1 < /dev/null &
Manual gate read (CPU): source ../d146/cache_env.sh; CUDA_VISIBLE_DEVICES="" PYTHONPATH=/data/relcfd/chayo/physmorph_v2/repo_d160 \
  /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python d160_gate.py /data/relcfd/chayo/physmorph_v2/output/gpu/d160/gate

## Next steps after the gate
1. Read gate/table.md + gate.json by the pre-registered C1-C4; write the D160 read entry in docs/experiments.md
   (separate worktree, push), with the render->control->outcome summary (vs_twin red_outer_shift, outer |dx|, timeline).
2. Seed-99 end sheets: d160_studio.py OUT SCENE L99 P99 ind_MESH.npz +endonly=1 (on a card, lock protocol).
3. Copy figures/videos/table to local output/results_2026-10-21/d160_appearance/ (README_ko.md drafted there).
4. If the gate passes: PROPOSE 300k runs to the coordinator; do not start them without the go.
   If C1 fails on the long scene: next method = Sinkhorn divergence between rendered and target colour pictures over
   (pixel position, colour) (DROT), scale fixed by the pictures.
Reference numbers (tag twins, read_tag.jsonl): mis half 0.057 / band 0.099 / ears 0.127 / turn 0.433.
Note: one stray local `python3 -` call was made by mistake at ~17:30 local (no effect); to be reported.
