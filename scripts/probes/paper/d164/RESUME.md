# D164 RESUME (2026-10-22 ~19:35 CDT, before the local shutdown)

State: code wip/d164 (003075a8 + scripts up to the yield change), pre-registration settled-base abdeded6 (docs/experiments.md, D164).
Server: repo_d164 (default "window"), repo_d164s (kept frames), repo_d164_start (config/CLI default "start", V2 comparison only).
Running: d164_chain.sh "P V R40 R300", PID 2040818 (setsid, PPID 1), waiting for a card under the protocol
(idle 10 min, no other-user job; yields to D162 / D162b only). No card held, nothing has run yet. Status: output/gpu/d164/status.txt.
Steps: P = pilot (bunny/dragon 40k L97, --replay_noise start + tmp/d164/replay_probe.py) -> output/gpu/d164/pilot/, then
d164_pilot_read.py checks the pre-registered stop rules mechanically (STOP / AMEND ends the chain). V = gates (suite, 20x
contracts test on repo_d164 + reports on repo_d164_start, refs + digest, deterministic S1/S2 SAME + vs rt1 S1) -> output/gpu/d164/gates/
(PASSED / FAILED). R40 = tmp/d164/d164_runs40.txt (76 runs) -> runs40/. R300 = d164_runs300.txt (4 runs) + read300.sh.
After the runs (CPU): bash tmp/d164/d164_after.sh 1  -> output/gpu/d164/d164_tables.md (+ .json), G verdict.

PENDING coordinator instruction (19:30): stop the claim-waiting chain by its PID and submit P and V as a job script to
tmp/gpu1_queue/ (CUDA_VISIBLE_DEVICES=1, after the RGV job) once the D162 agent's README is there; R40 / R300 later.
At 19:35 tmp/gpu1_queue/ did not exist yet, and the checkpoint message asked to keep the chain alive, so it was left running.
On resume: if tmp/gpu1_queue/README exists, `kill 2040818` (mine; it holds no card while waiting), write the job as
`bash tmp/d164/d164_chain.sh "P V"` with CARDS=1 and the queue's conventions, and report in Korean when V is read.
Reminder: no local python in any form; everything via ssh hyde06j.
