#!/bin/bash
# 2NN_d174_t_bunny172.sh -- queue job (agent D174): the timing run of repo_d172 on the 300k bunny with Hu's flags (seed 97), alone on
# this card, no --profile; the caches warmed first (untimed), as a repeated run of the same target finds them.
J=t_bunny172
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr repo_d172; [ 172 = 174 ] && $PY $W/warm_cache.py --tgt assets/bunny.obj --n 300000 --seed 97 > $O/runs/W_bunny_d172.log 2>&1 )
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 1000 > $O/runs/T_bunny300k_Hu_d172.util.csv 2>/dev/null &
SP=$!
run1 repo_d172 T_bunny300k_Hu_d172 --tgt assets/bunny.obj --n 300000 --seed 97 $HU
kill $SP 2>/dev/null
log "done"
