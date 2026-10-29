# D172 resume (agent D172, started 2026-10-28 21:10 CDT after the ssh outage)

Read ONE file: output/gpu/D172_SUMMARY.md (rewritten by every job's reader): the B table first (Hu / Hf against Rx and R).

## What D172 is
- B: the render's authority split by control block. Hu = dF_c + layer u + render loss (full), the held weight on u and the
  rule's per-window weight on dF_c; Hf = the reverse. Both with Rx's settle (--lambda_ema 0 --settle_windows 30
  --no-settle_relax); the merit (line search, stop rules) keeps the held weight, as Rx's. Flag `--render_split u|dFc`.
  Where: physmorph/pipeline/window/solve.py gradient() (the two places PCGrad's projected render gradient joins g_p:
  `[a + lam * b for a, b, lam in zip(gp, gr, self._block_lams())]`), _block_lams(), the rule's value in _calibrate_lambda
  (the balancer's target with its cap). Default off = one weight on both (bit-SAME).
- C: the render_x paint fix (physmorph/pipeline/target.py, the appearance block of build_target: the pictures' first sample is
  render_x, so its albedos are appearance.albedo_at(render_x), as run/runner.py:120 / :136 do for the floor). No flag (a bug fix:
  without --render_tgt or --appearance nothing changes; gate check 4).
- D: stylized dragon scales (tmp/d172/d172_mesh.py: D161's builder with a hexagonal lattice of domes on the smoothed dragon's
  flank). Representability gate before any run: a perfect 300k sample must read corr >= 0.25 (tmp/d172/d172_detail.py, the
  pattern region). A/B (packed domes) blurred flat; C/D (gaps) read corr 0.09 / 0.14; E/F/G (taller, wider) being probed;
  the chosen variant is output/gpu/d172/assets/variant.txt (job 189 waits for it).
- Code: repo_d172 (server) = repo_d170 + the two changes; local branch wip/d172 4a0b2537 (worktree C:/dev/smb_d172), pushed.

## Queue (tmp/gpu1_queue; card_runner_n.sh on cards 0 / 1 / 3, PIDs tmp/d172/runner_card*.pid; never gpu2)
- 184 V3 gate (defaults SAME vs d153/rt1/S1; Rx flags SAME vs repo_d166's X0; --render_split differs; band SAME vs repo_d166).
- 185 Hu / Hf on coin / bunny relief / band 40k x 97 / 99. 186 Hu / Hf on the 6-mesh gallery 40k seed 97 (191 = seed 99 for an
  arm holding 6/6, armed by post172.sh). 187 Hu / Hf on the 300k plaque. 188 combined scene 40k (T, Rx, Hu, Hf x 97 / 99;
  192 = the painted end frames, armed after the read). 189 scales 40k (T, Rx, Hu, Hf x 97 / 99; 190 = 300k with the best arm).
- Readers: tmp/d172/after.sh MODE JOB QUEUE_NAMES -> output/gpu/d172/auto/<job>/summary.md + verdict.json, then
  D172_SUMMARY.md, then tmp/d172/post172.sh (follow-ups, deliverables in output/gpu/d172/deliver/). PIDs auto/<job>/after.pid.
- Readers used unchanged: D166's scene reader (Hu / Hf scenes), D170's gallery / carve readers, D161's relief and D160's colour
  readers (combined), d172_detail.py (scales).

## D170 (finished 21:10-21:30)
- Ripple 300k twin | full video done (output/gpu/d170/deliver/rip/), band video, plaque close-ups; copied to local
  output/results_2026-10-28/d170/. carve_b (coin, bunny flank) read still running at 21:30 (the flank read ~2 h, 92 GB).
- dsm300k_H0mr15_97: CUDA OOM twice on a 48 GB card (noted in d170/auto/177_dragon300/summary.md); superseded by D.
- R0 INCOMPLETE = kept undefined where nothing was written (noted in d170/auto/178_r0h0/summary.md).

## Disk
- 21:10: 467 GB -> deleted 29 GB of read dumps (d145t_b7 cf, d146/d146g dumps, d165 stamps, d129/d130/d148 dumps, render terms)
  + 6 GB (d145_ext r4 end1/display, cache/samples older than 2 days) -> 433 GB (tmp/d172/disk_base.txt); all logged in
  output/gpu/cleanup.log. Next candidate if needed: the external baselines' intermediate checkpoints (~20 GB in
  baselines/*/exp_b14/*/checkpoints), not touched without the user's word.

## Resume
- A failed run: delete its .rc and rerun its job script (finished runs skip). A reader by hand:
  `cd repo_d166; PYTHONPATH=$PWD $PY ../tmp/d172/d172_read.py MODE ../output/gpu/d172/auto/JOB`, then `... aggregate ../output/gpu/D172_SUMMARY.md`.
- Stop: kill the runner PIDs (their traps release the locks), then the after.pid processes. Never pkill / pgrep -f.
