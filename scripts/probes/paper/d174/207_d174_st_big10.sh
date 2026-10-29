#!/bin/bash
# 202_d174_st_big10.sh -- queue job (agent D174, stop A/B): Hu with tol 0.10 (repo_d174t10) on the 300k stylized dragon scales (D172's
# variant E) and the 300k SIGGRAPH plaque (D172 job 187's arguments), seed 97, PAR 2.
J=st_big10
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"; PAR=2
runp repo_d174t10 scl300k_Ht10_97 $F_scl $HU
runp repo_d174t10 plq300k_Ht10_97 --tgt $G/d163/assets/d163plaque.obj --render_tgt $G/d163/assets/d163R300plaque.obj --n 300000 --seed 97 $HU
wait; log "done"
