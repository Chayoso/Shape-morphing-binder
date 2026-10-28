# D170 resume (agent D170, 2026-10-28; last edit 17:45 CDT)

Read ONE file: output/gpu/D170_SUMMARY.md (rewritten by every job's reader): the dragon's scales first, then "Adopt Rx?",
the reject-stop diagnostic, R0 / H0, the combined carve + colour note, then every job summary.

## What D170 is
- Rx = dF_c + layer u + render loss (full), weight held (--lambda_ema 0), then 30 zero-control windows with the outer-layer
  relaxation off (--settle_windows 30 --no-settle_relax). Twin = dF_c + layer u (lambda = 0, --render_weight_scale 0).
- Code: repo_d166 (unchanged) for Rx; repo_d170 = repo_d166 + `--layer_relax/--no-layer_relax` (default on; branch wip/d170
  2664ca56, local worktree C:/dev/smb_d170, pushed) for the relaxation-off-throughout arms. V3 gate (job 168) PASS: defaults
  bit-SAME vs d153/rt1/S1, Rx's flags bit-SAME vs repo_d166, --no-layer_relax differs. The grad and eval trajectories read the
  same layer tuple (window/setup.py:100 / :114), so the switch reaches tr_eval.

## Read so far (17:45)
- Gallery 40k (job 174): Rx FAIL, holds 0/6 (bob, teapot miss by <= 0.002; dragon / armadillo / homer lose 0.012-0.033 sil_iou
  and +0.09-0.19 thin); rest ok (v_end ~0). 300k shape (175): FAIL 0/2 (dragon sil -0.010, thin +0.088). Cause read in the
  logs: the held weight's runs stop by the frozen reject rule before arrival (gallery dragon window 41 vs the rule's 69; plaque
  300k window 21 / 3.5 s vs the twin's 112). Diagnostic queued: job 179 (Rx + --reject_stop 0, gallery seed 97 + 300k plaque).
- Dragon render-only scales (TOP item): below the 300k resolution. Size probe (output/gpu/d170/size_probe/): the scales'
  height rms 0.09 pitch, half-correlation ~1.1 pitch at 300k (0.4 pitch at 40k); a perfect N-sample of the detailed dragon
  carries corr 0.05 (40k), 0.04 (300k), 0.10 (1M), 0.14 (1.5M), 0.19 (3M) with its pattern (the bunny's wrinkles: 0.27 at
  300k). 40k arms (job 173 + 168: Rx, Rxm, H0, H0m, R0m x 97/99): shares swing -0.6..2.4 with corr 0.02-0.08, inside the
  reading's own noise (the detailed mesh's two samples read 0.41 vs 0.85 at seed 99): no carving signal; post.sh's 300k gate
  (share >= 0.40 and >= twin + 2 x that noise at both seeds) did not pass. 300k blocker test queued anyway as a measurement
  (jobs 180 / 181: twin, Rxm, H0m, H0m with --exterior_radius 1.5), reader auto/177_dragon300.
- Mesh-relief pictures with the detailed 3D target (job 172): the twin already carries the wrinkles / scales via the layer
  relaxation's target relief (bunny share 1.33, dragon 2.99 of a random sample); Rx adds nothing on the bunny (1.30) and +0.40
  on the dragon at a shape cost (thin 0.131 vs 0.036).

## Running / queued (tmp/gpu1_queue; card_runner_n.sh on cards 0 / 1 / 3, PIDs tmp/d170/runner_card*.pid; never gpu2)
- 178 R0 / H0 (repo_d170) on coin / bunny / band 40k + R0 gallery 97 -> auto/178_r0h0. 179 reject-stop diagnostic -> auto/179a,
  179b. 180 / 181 dragon 300k -> auto/177_dragon300.
- CPU readers (after2.sh / after.sh, PIDs auto/<job>/after.pid): carve_a (plaque read, ripples read ~2 h: done ~19:00),
  carve_b (coin, then the bunny flank; tmp/d170/mem_guard.sh holds the flank read with SIGSTOP until the ripple read exits, both
  ~95 GB), band300, D157 gallery (auto/174b_gal_d157, tmp/d170/d157_gal.sh).
- Deliverables (post.sh, after each read): output/gpu/d170/deliver/<scene>/ close-ups at 84 deg rake for every content-passing
  scene; twin | full 4K + 1080p videos and studio stills for PASS scenes (d170_render.py: D163's renderer, the true camera, no
  flip). The 500k stamp redux is armed only if the 300k plaque passes (it fails shape: early stop at window 21).

## Not run, with the reason
- Combined carve + colour scene: target.py:215 paints the first picture sample from target_x while target.py:156 draws it
  from render_x (--render_tgt): needs sample 0's albedo = appearance.albedo_at(render_x) (code change; not made).
- R0 / H0 needed a switch (none existed: window/setup.py:59 only for --baseline xu*, run/runner.py:313-316 only in the settle):
  added in repo_d170 as above.

## Disk
- 16:30: /data/relcfd/chayo 477 GB -> deleted 35 GB of read bulk dumps (logged in output/gpu/cleanup.log) -> 444 GB
  (tmp/d170/disk_base.txt); d170 holds ~14 GB at 17:40; runs stop at 495.

## Resume
- Failed run: delete its .rc and rerun the job script (finished runs skip). A reader by hand:
  `cd repo_d166; PYTHONPATH=$PWD $PY ../tmp/d170/d170_read.py MODE ../output/gpu/d170/auto/JOB` (read_*.json are cached), then
  `$PY ../tmp/d170/d170_read.py aggregate ../output/gpu/D170_SUMMARY.md`.
- Stop: kill the runner PIDs (their traps release the locks), then the after.pid processes. Never pkill / pgrep -f.
- Scripts: tmp/d170/ (copies in wip/d170 scripts/probes/paper/d170/).
