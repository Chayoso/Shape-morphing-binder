#!/bin/bash
# B12 follow-up chain (one card at a time): after the first queue, the bunny's 12 runs again (the readout amendment of
# 12:20 CDT: every Large Steps step a candidate iterate, the recording timed apart), then D145's harness on seed 97.
B12=/data/relcfd/chayo/physmorph_v2/baselines/b12
until grep -q "queue done" $B12/logs/queue.log; do sleep 60; done
$B12/scripts/b12_queue.sh bunny
S=$B12/logs/status.txt
until [ $(grep -c "read done .*_bunny_\|read FAILED .*_bunny_" $S) -ge 24 ] && [ $(grep -c "read start" $S) -eq $(grep -c "read done\|read FAILED" $S) ]; do sleep 60; done
$B12/scripts/b12_gpu.sh 600 "b12 harness: D145 harness.py on seed 97 of 4 arms x 6 meshes (~30 min)" $B12/scripts/b12_harness.sh bunny dragon teapot bob armadilo homer
echo "$(date '+%F %T %Z') after-chain done"
