#!/bin/bash
# 196_d174_sd10.sh -- queue job (agent D174): Hu --stop_tol 0.10 (repo_d174s: the stop's own tolerance; the anneal and the
# latch keep tol 0.003) on the 300k dragon, seed 97, alone on this card; caches warmed first. Picks the value (<= 10 min).
J=sd10
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr repo_d174s; $PY $W/warm_cache.py --tgt assets/dragon.obj --n 300000 --seed 97 > $O/runs/W_dragon_Hs10.log 2>&1 )
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 1000 > $O/runs/dragon300k_Hs10_97.util.csv 2>/dev/null &
SP=$!
run1 repo_d174s dragon300k_Hs10_97 --tgt assets/dragon.obj --n 300000 --seed 97 $HU --stop_tol 0.10
kill $SP 2>/dev/null
rm -f $O/runs/dragon300k_Hs10_97_frames12.npz
log "done"
