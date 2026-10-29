#!/bin/bash
# 196_d174_sbig12.sh -- queue job (agent D174): Hu --stop_tol 0.12 (repo_d174s) on the 300k stylized dragon scales and the 300k
# SIGGRAPH plaque (D172's arguments), seed 97, PAR 2.
J=sbig12
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"; PAR=2
runp repo_d174s scl300k_Hs12_97 $F_scl $HU --stop_tol 0.12
runp repo_d174s plq300k_Hs12_97 --tgt $G/d163/assets/d163plaque.obj --render_tgt $G/d163/assets/d163R300plaque.obj --n 300000 --seed 97 $HU --stop_tol 0.12
wait; log "done"
