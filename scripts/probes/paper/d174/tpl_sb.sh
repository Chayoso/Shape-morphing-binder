#!/bin/bash
# 196_d174_sbVV.sh -- queue job (agent D174): Hu --stop_tol 0.VV (repo_d174s) on the 300k bunny, seed 97, alone on this card.
J=sbVV
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr repo_d174s; $PY $W/warm_cache.py --tgt assets/bunny.obj --n 300000 --seed 97 > $O/runs/W_bunny_HsVV.log 2>&1 )
run1 repo_d174s bunny300k_HsVV_97 --tgt assets/bunny.obj --n 300000 --seed 97 $HU --stop_tol 0.VV
rm -f $O/runs/bunny300k_HsVV_97_frames12.npz
log "done"
