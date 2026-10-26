# D166 / D167 resume (agent D166) -- RERUN FROM SCRATCH, queued 2026-10-24 01:53 CDT

Read ONE file: output/gpu/RENDER_REST_WEIGHT_SUMMARY.md ("What to adopt" first; rewritten every 10 min by job 199).

## State
- Code: repo_d166 = repo_d162hs 58029ddd + `--settle_u zero|last`, `--settle_relax/--no-settle_relax` (both default = old
  settle) + the coordinator's capture fix (run/runner.py 315-316: `opt.win.tr.capture()` after `layer_frac = 0`). Every
  --no-settle_relax result before the fix is invalid.
- The first round (2026-10-23) moved to output/gpu/d16{6,7}/old_20261023/ (summaries, json, logs kept; its .npz deleted for
  the 500 GB rule). Codex's material (codex_2026-10-23/) is not used.
- Reader tmp/d166/d166_read.py = Codex's hardened reader + this rerun's edits (tmp/d166/edit_reader.py; copy before them:
  d166_read.py.before_rerun_20261024): a stop between kept frames is interpolated (12-frame stride), the master's criteria.
- Measurement (step 1, repo_d162hs): u is a per-window increment, zero at each window start (window/solve.py:110); the
  settle commits a fresh optimizer's zero u (run/runner.py:296-305) while the layer relaxation toward the smooth target's
  relief keeps acting (window/setup.py:61-62, mpm/kernels.py:548-568, target.py:93-105).

## Queue (tmp/gpu1_queue, card_runner_n.sh on cards 0 / 1 / 3; PIDs tmp/d166/runner_card*.pid; never gpu2)
- 60 V3 gate: defaults SAME vs d153/rt1/S1; settle defaults SAME vs repo_d162hs; --no-settle_relax must differ.
- D166 (40k coin + bunny relief + band, seeds 97/99, lambda held, settle 30): 61 S0 the settle as it is; 62 Rx
  --no-settle_relax; 63 E3 E x3 (+ its twin TE3).
- D167: 100 TP twin; 101 F/FP, 102 S/SP, 103 H/HP, 104 R/RP (40k scenes); 110-114 the same on the 6-mesh gallery (seed 97);
  116 cross = best definition + --no-settle_relax, only if D166 reads the relaxation as the eraser; 120-129 300k bunny +
  dragon seed 97 (T F S H R TP FP SP HP RP).
- 199: detached CPU loop writing the master until all 25 summaries in tmp/d166/expected.txt exist (PID tmp/d166/aggregate.pid).
- ETA: 40k by ~03:30, 300k by ~05:30, master final ~06:00 CDT.

## Where
- Runs output/gpu/d16{6,7}/runs; per job output/gpu/d16{6,7}/auto/<NN_name>/summary.md + verdict.json; status.txt beside.
- Disk guard: base tmp/d166/disk_base.txt (436 GB) + d166 + d167 < 495 GB, else runs are skipped (logged).

## Resume
- Failed run: delete its .rc, copy the job's .done back to .sh in tmp/gpu1_queue; finished runs skip.
- Missing summary: rerun the after line from the job's status.txt (`bash tmp/d166/after.sh MODE OUT RUNDIR "TITLE" ARM:TWIN`).
- Master by hand: `$PY tmp/d166/d166_read.py aggregate output/gpu/RENDER_REST_WEIGHT_SUMMARY.md`.
- Stop: kill the runner PIDs (their traps release the locks), then the after.pid / aggregate.pid processes.
