#!/bin/bash
# 221_d173_strp300.sh -- queue job (agent D173): the stripes scene at 300k, seed 97: the twin and Rx with the nearest colour.
J=221_strp300; TITLE="D173 stripes 300k"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
gate_t; log "start"; PAR=2
runp strp300k_T_97 --tgt assets/bunny.obj --appearance stripes --n 300000 --seed 97 $TW
runp strp300k_Rxn_97 --tgt assets/bunny.obj --appearance stripes --n 300000 --seed 97 $RX --colour_kernel nearest
wait; log "done"
