#!/bin/bash
# 40_d164_r40_r300.sh -- gpu1 queue job (agent D164): R40 (the 76 40k gallery runs) then R300 (bunny / dragon 300k L97 /
# P97 with B1's readings) on repo_d164s (fc730244), QUEUE=1 (the runner holds gpu1). The CPU readings and
# output/gpu/d164/auto/summary.md come from tmp/d164/d164_auto.sh (detached, waits for this job's "R300 done").
B=/data/relcfd/chayo/physmorph_v2
cd $B
QUEUE=1 bash $B/tmp/d164/d164_chain.sh "R40 R300"
