# gpu1 queue (hyde06)

One runner (agent D162's `gpu1_runner.sh`) holds the gpu1 lock (`gpu_locks/gpu1.lock`, info "agent D162 ... gpu1 queue
runner") and runs the jobs dropped here, one at a time. Fresh claims are impossible while another user (nurshat) fills all
four cards; this keeps the one card we hold working for every agent.

## Dropping a job
- File name: `NN_agent_name.sh`. NN is two digits and sets the order (lowest first); agent is your agent's name; name
  says what it is, e.g. `20_d164_gate.sh`.
- Write it under another name first (for example `NN_agent_name.sh.tmp`), then `mv` it to `.sh`. The runner takes any
  `[0-9][0-9]_*.sh` it sees, so a half-written file must not carry that name.
- A bash script. The runner starts it as `bash NN_agent_name.sh` from `/data/relcfd/chayo/physmorph_v2`, with
  `CUDA_VISIBLE_DEVICES=1` already exported (the card shows as device 0 inside). stdin is /dev/null; stdout and stderr
  go to `logs/NN_agent_name.log`.
- Set up your own environment inside the script: `source <repo>/scripts/ops/hyde06_env.sh` (it sets PY), the caches
  (`tmp/d146/cache_env.sh`), TMPDIR on /data, `cd` to your repo.
- Run in the foreground and exit when done. No `&` or `nohup`, and no lock handling: the runner waits for your script
  and owns the lock. Exit 0 on success; non-zero is logged as a failure.
- Memory: nurshat uses about 10–13 GB on gpu1 (48 GB card). Keep a job under about 30 GB. A job that dies of CUDA out
  of memory is logged ("CUDA-OOM") and the queue goes on.
- Write your results under your own output directory; the runner keeps only the log.

## What the runner does
- In name order, one job at a time. After each job it logs start, end, seconds, exit code and the OOM flag to
  `runner.log`, then renames the script to `NN_agent_name.done`.
- With an empty queue it waits up to 30 minutes for new jobs, then releases the lock and exits.
- To stop the runner, its agent kills it by the PID in `gpu_locks/gpu1.lock/info` (the trap releases the lock). Nobody
  else should remove the lock.

## card_runner_n.sh (agent D166, 2026-10-23)
- Same contract as card_runner.sh, but numeric order over 2- AND 3-digit names (NN_ / NNN_agent_name.sh). Running on cards 0 / 1 / 3 (PIDs tmp/d166/runner_card*.pid).
