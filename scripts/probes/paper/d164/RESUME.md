# D164 RESUME (2026-10-22 ~19:35 CDT, before the local shutdown)

State: code wip/d164 (003075a8 + scripts up to the yield change), pre-registration settled-base abdeded6 (docs/experiments.md, D164).
Server: repo_d164 (default "window"), repo_d164s (kept frames), repo_d164_start (config/CLI default "start", V2 comparison only).
Running: chain 2040818 stopped 19:40 by its PID (it never held a card). P + V are queued on the gpu1 queue as
tmp/gpu1_queue/20_d164_pilot_gates.sh (QUEUE=1 bash tmp/d164/d164_chain.sh "P V"; after 01_ri_rgv_held_bunny and 02_d162_d162d).
Status: output/gpu/d164/status.txt; the job log tmp/gpu1_queue/logs/20_d164_pilot_gates.log; runner.log.
Steps: P = pilot (bunny/dragon 40k L97, --replay_noise start + tmp/d164/replay_probe.py) -> output/gpu/d164/pilot/, then
d164_pilot_read.py checks the pre-registered stop rules mechanically (STOP / AMEND ends the chain). V = gates (suite, 20x
contracts test on repo_d164 + reports on repo_d164_start, refs + digest, deterministic S1/S2 SAME + vs rt1 S1) -> output/gpu/d164/gates/
(PASSED / FAILED). R40 = tmp/d164/d164_runs40.txt (76 runs) -> runs40/. R300 = d164_runs300.txt (4 runs) + read300.sh.
After the runs (CPU): bash tmp/d164/d164_after.sh 1  -> output/gpu/d164/d164_tables.md (+ .json), G verdict.

Next: when V is read (gates/PASSED or FAILED), report in Korean; R40 / R300 go to the same queue (e.g.
`QUEUE=1 bash tmp/d164/d164_chain.sh "R40"` as 3x_d164_runs40.sh) or a freed card, as the coordinator says.
Reminder: no local python in any form; everything via ssh hyde06j.
