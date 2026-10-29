#!/bin/bash
# 201_d173_cmb300.sh -- queue job (agent D173): the combined relief + paint scene at 300k (the smoothed bunny; the pictures
# D161's "PM" relief at the 300k pitch painted with D160's band; repo_d173 has D172's render_x paint fix): Rx and the twin.
J=201_cmb300; TITLE="D173 combined relief + paint, 300k"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
for k in $(seq 1 360); do [ -e $O/gates/V3.ok ] && break; [ -e $O/gates/V3.fail ] && { log "V3 failed: stop"; exit 1; }; sleep 20; done
log "start"; PAR=2
C="--tgt $A1/d161smooth_bunny.obj --render_tgt $G/d170/assets/d170L300_bunny.obj --appearance band --n 300000 --seed 97"
runp cmb300k_T_97 $C $TW
runp cmb300k_Rx_97 $C $RX
wait; log "done"
