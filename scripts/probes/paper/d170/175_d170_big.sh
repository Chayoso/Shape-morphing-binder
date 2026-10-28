#!/bin/bash
# 175_d170_big.sh -- queue job (agent D170): Rx at 300k on bunny + dragon, seed 97 (shape). Refs reused: d167/runs
# {bunny,dragon}300k_R_97 (the rule) and _T_97 (the twin).
J=175_big; TITLE="D170 300k shape: Rx on bunny + dragon, seed 97"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
log "start"; PAR=2
for m in bunny dragon; do runp ${m}300k_Rx_97 --tgt assets/$m.obj --n 300000 --seed 97 $RX; done
wait; log "done"
