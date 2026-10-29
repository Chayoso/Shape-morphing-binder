#!/bin/bash
# 2NN_d174_t_dragon172.sh -- queue job (agent D174): the timing run of repo_d172 on the 300k dragon with Hu's flags (seed 97), alone on
# this card, no --profile; the caches warmed first (untimed), as a repeated run of the same target finds them.
J=t_dragon172
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr repo_d172; [ 172 = 174 ] && $PY $W/warm_cache.py --tgt assets/dragon.obj --n 300000 --seed 97 > $O/runs/W_dragon_d172.log 2>&1 )
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 1000 > $O/runs/T_dragon300k_Hu_d172.util.csv 2>/dev/null &
SP=$!
run1 repo_d172 T_dragon300k_Hu_d172 --tgt assets/dragon.obj --n 300000 --seed 97 $HU
kill $SP 2>/dev/null
( envr repo_d174; $PY $W/d174_cut.py $O/runs/P_dragon300k_Hu_97 86 32 > $O/auto/cut_dragon.md 2> $O/auto/cut_dragon.err; $PY $W/d174_cut.py $O/runs/P_bunny300k_Hu_97 61 27 > $O/auto/cut_bunny.md 2> $O/auto/cut_bunny.err )
log "done"
