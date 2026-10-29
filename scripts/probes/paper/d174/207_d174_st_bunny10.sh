#!/bin/bash
# 203_d174_st_bunny10.sh -- queue job (agent D174, stop A/B): Hu with PipelineConfig.tol 0.10 (repo_d174t10; the existing stop rule's
# tolerance, 0.003 before) on the 300k bunny, seed 97, alone on this card (time, sil_iou, chamfer, thin, rest); caches warmed first.
J=st_bunny10
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr repo_d174t10; $PY $W/warm_cache.py --tgt assets/bunny.obj --n 300000 --seed 97 > $O/runs/W_bunny_Ht10.log 2>&1 )
nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -lms 1000 > $O/runs/bunny300k_Ht10_97.util.csv 2>/dev/null &
SP=$!
run1 repo_d174t10 bunny300k_Ht10_97 --tgt assets/bunny.obj --n 300000 --seed 97 $HU
kill $SP 2>/dev/null
rm -f $O/runs/bunny300k_Ht10_97_frames12.npz
log "done"
