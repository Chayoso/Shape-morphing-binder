#!/bin/bash
# 2NN_d174_t_dragon174.sh -- queue job (agent D174): the timing run of repo_d174 on the 300k dragon with Hu's flags (seed 97), alone on
# this card, no --profile; the caches warmed first (untimed), as a repeated run of the same target finds them.
J=t_dragon174
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
gate_wait e2
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr repo_d174; [ 174 = 174 ] && $PY $W/warm_cache.py --tgt assets/dragon.obj --n 300000 --seed 97 > $O/runs/W_dragon_d174.log 2>&1 )
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 1000 > $O/runs/T_dragon300k_Hu_d174.util.csv 2>/dev/null &
SP=$!
run1 repo_d174 T_dragon300k_Hu_d174 --tgt assets/dragon.obj --n 300000 --seed 97 $HU
kill $SP 2>/dev/null
log "done"
