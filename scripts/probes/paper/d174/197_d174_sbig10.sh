#!/bin/bash
# 197_d174_sbig10.sh -- queue job (agent D174): Hu --stop_tol 0.10 (repo_d174s) on the 300k stylized dragon scales and the 300k
# SIGGRAPH plaque (D172's arguments), seed 97, PAR 2.
J=sbig10
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"; PAR=2
runp repo_d174s scl300k_Hs10_97 $F_scl $HU --stop_tol 0.10
runp repo_d174s plq300k_Hs10_97 --tgt $G/d163/assets/d163plaque.obj --render_tgt $G/d163/assets/d163R300plaque.obj --n 300000 --seed 97 $HU --stop_tol 0.10
wait; log "done"
