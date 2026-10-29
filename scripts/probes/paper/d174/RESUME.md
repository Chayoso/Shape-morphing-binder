# D174 resume (agent D174: runtime ~10 min per 300k Hu run, and the memory track toward ~2M; 2026-10-29 09:40-12:15 CDT)

Read ONE file first: output/gpu/D174_SUMMARY.md (profile table, each speed-up with its saving and gate, the A/B, memory, the
2M plan). Nothing is running for D174 when this was written; nothing is adopted.

## Code (local worktree C:/dev/smb_d174, branch wip/d174, pushed)
- a623a0b6 exact batch 1: settle windows without the unread objective (window/rollout.release_rollout, run/runner.py),
  target draws cached (sampling/mesh._cached_draws), thin set cached (thin.thin_set_cached). Gate e1 PASS.
- 211d290f exact batch 2: Sinkhorn axis log-sum-exp from a shared-memory CUDA kernel (losses/lse_cuda.py; SHARED_LSE in
  losses/grid_ot.py, False = the Warp kernel), the u gate's map on the layer's rows (grid_transport_displacement rows=).
  Gate e2 PASS.
- 0f6689c0 A/B switches (default off): --render_every K, --ls_max K. Gates ab / e2 PASS with them off.
- 2cf6caa0 memory m1: driven-only control buffers (tape, its gradients, the eval dc_buf), no kept velocity / frame stacks.
  Gate m1 PASS; bit-SAME at 300k deterministic (gates/d300_cmp.txt).
- 3b5dfd77 memory m2: the exterior lattice's ring of nodes in chunks (render/exterior.Lattice.near_nodes). Gate m2 PASS,
  chunk check gates/m2_chunks.txt.
- Server copies: repo_d174 = HEAD (all of the above; synced after the last timing run), repo_d174ab (batch 1 + switches:
  the A/B runs), repo_d174m / repo_d174m2 (the memory gates' copies; superseded by repo_d174).

## Tools (tmp/d174; local copies in the session scratchpad d174/)
- d174_gate.sh TAG [REPO]: V3 gate (deterministic, bunny 40k, 4 windows; G1 defaults vs d153/rt1/S1, G2 Hu settle 30 vs
  gates/ref_H0, G3 Hu coin settle 2 vs d172/gates/X2, G4 settle 3 with relaxation vs gates/ref_R0); NOLOCK=1 under a queue
  runner. d174_common.sh: the queue jobs' functions (run1 / runp, flags HU, F_coin / F_bunny / F_band / F_scl).
- d174_prof.py RUN: phases from the stamped log, --profile sections, GPU busy per phase. d174_cut.py: metrics at every 10th
  kept window vs the wall a run stopped there would take. d174_mem.py OUT.json [args]: memory phases + every device buffer by
  owner; d174_memtab.py AUTO: the table and the linear fit. check_m2.py: near_nodes chunked vs one chunk.
- d174_read.py MODE OUT (scenes / gal / scales / aggregate) on D172's readers; d174_after.sh (finished).
- bench_lse2.py / bench_lse3.py: the kernel's bit-for-bit check and times; warm_cache.py; d174_run.sh / onecard.sh (own-lock
  launchers, used before the cards went to the shared queue runners).

## Outputs (output/gpu/d174)
- runs: P_* (solo profile runs), T_* (solo timing runs, JSON + util CSV), M_* (memory probes), the A/B runs (D172's names,
  arms Hu2 / Hu4). Bulk frames12 of P_ / T_ / M_ deleted after reading; the A/B runs' kept.
- auto: cut_dragon.md / cut_bunny.md, mem_scl{300,500}k{,_m1,_m2}.{json,log}, ab_scenes / ab_gal / ab_scales; AB_SUMMARY.md.
- gates: e1, e2, ab, m1, m2 (.md / .ok), d300_cmp.txt, m2_chunks.txt; refs ref_H0 / ref_R0 (repo_d172).

## Next (the user's / coordinator's call)
- Runtime: adoption of batches 1-2 (exact). Hu4 passes by the letter but loses 44 % of the 300k scales' content: not
  recommended. A 10-minute dragon needs an earlier stop (cost in the summary): a stop-rule A/B only if the user asks.
- Memory toward 2M (summary section 4): the tape's gradient ring, the eval trajectory on the tape's buffers, in-place
  line-search candidates, then tape checkpoints; each to be gated bit-SAME (V3) and probed at 300k / 500k. No run above 500k.

## Resume
- A queue job: write tmp/gpu1_queue/NNN_d174_name.sh.tmp, then mv to .sh (the runners on cards 0 / 1 / 3 take it).
- Readers by hand: `cd repo_d166; PYTHONPATH=$PWD:$B/deps/d157_lpips CUDA_VISIBLE_DEVICES= $PY ../tmp/d174/d174_read.py MODE OUT`.
- Never pkill / pgrep -f; never gpu2.
- Process note: once in this session a local python3 was used to substitute text in this summary's draft (no project code
  was run locally); not repeated.

## Stop A/B (2026-10-29 12:20-14:10 CDT, user decision: a <= 10 min 300k dragon through the existing stop rule)
- Arms = repo_d174 copies with only PipelineConfig.tol changed: repo_d174t10 / t07 / t06 / t05 (Ht10 / Ht07 / Ht06 / Ht05).
  Runs output/gpu/d174/runs/{dragon,bunny,scl,plq}300k_HtNN_97, {coin,bunny,band}_HtNN_{97,99}, gal_*_HtNN_97.
- Readings: tmp/d174/d174_read2.py (fast readers) via d174_after4.sh (finished) -> output/gpu/d174/auto/st_*,
  output/gpu/d174/STOP_SUMMARY.md; the trade table and the definition-change proposal in D174_SUMMARY.md section 5.
- tmp/d174/d174_stop.py replays the stop rule for other tolerances on recorded runs (no anneal feedback).
- Withdrawn before they ran (redundant once Ht07 stopped like Ht10): tmp/d174/*07.sh.withdrawn, the t10 gate.
- Nothing committed as a default; the stop-tolerance definition change awaits the user.

## --stop_tol (user GO; 14:10-15:55 CDT)
- Code wip/d174 228ee3a7 (selection.py second counter set; config stop_tol; CLI --stop_tol); gate s PASS; repo_d174 and
  repo_d174s = HEAD. Chosen value 0.12 (dragon 300k 8.0 min); A/B in D174_SUMMARY.md section 6 (STOPTOL_SUMMARY.md,
  auto/ss_*). The 0.10 arm's scales (share 1.13) and plaque are read by tmp/d174/d174_after6.sh (PID after6.pid) into
  auto/ss10_*; it was still on the plaque at 15:55 (hyde06 load ~600). Runs Hs10 / Hs12 in output/gpu/d174/runs.
