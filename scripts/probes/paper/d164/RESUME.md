# D164 RESUME (2026-10-22 23:08 CDT)

Code wip/d164 fc730244 (amendment 1: the commit reads the accepted candidate's potentials and lists; amendment 2: the
scene keeps the tag's commit). Docs settled-base cecd8d22 (amendment 1), f3b64ee7 (amendment 2).
Server repo_d164 / repo_d164s / repo_d164b = fc730244; repo_d164_start = 11aab852 with the "start" default (V2 reports).
Done: P fired AMEND (21:52, as pre-registered); P2 quiet (|c-a| ~30x smaller); V passed (V1 by amendment 2: only the
at-rest MPM test fails, the card's, d164b 3/10 vs tag 1/10 on identical MPM code; V2 0/0/0 of 20, start-tree 1/0/0;
V3 15 passed incl. digest; V4 SAME run to run and vs rt1 S1). Marker output/gpu/d164/gates/PASSED.
Queued: tmp/gpu1_queue/40_d164_r40_r300.sh (R40 76 runs then R300 4 runs + read300). Detached CPU watcher
tmp/d164/d164_auto.sh (PID 3514607) waits for "R300 done", runs d164_after.sh 1, writes output/gpu/d164/auto/summary.md.
Next: read auto/summary.md and d164_tables.md; report in Korean (null fractions, gallery G, 300k, render->control->outcome).
No local python in any form; everything via ssh hyde06j.
