#!/bin/bash
# 202_d173_tex300.sh -- queue job (agent D173): the textured duck at 300k, seed 97: the twin and Rx.
J=202_tex300; TITLE="D173 texture 300k"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
for k in $(seq 1 360); do [ -e $O/gates/V3.ok ] && break; [ -e $O/gates/V3.fail ] && { log "V3 failed: stop"; exit 1; }; sleep 20; done
TX="--appearance texture --texture $A/d173bob_tex.obj,$A/d173bob_diffuse.png"
log "start"; PAR=2
runp tex300k_T_97 --tgt assets/bob.obj $TX --n 300000 --seed 97 $TW
runp tex300k_Rx_97 --tgt assets/bob.obj $TX --n 300000 --seed 97 $RX
wait; log "done"
