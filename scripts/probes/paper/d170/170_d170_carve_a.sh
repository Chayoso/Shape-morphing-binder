#!/bin/bash
# 170_d170_carve_a.sh -- queue job (agent D170): the 300k carve scenes whose assets exist, Rx, seed 97: the SIGGRAPH plaque
# (D163) and the D162f ripples. Twins reused: d163/runs300/plq300k_T_97, d162f/runs/rip300k_T_97.
J=170_carve_a; TITLE="D170: 300k carve scenes (plaque, ripples), Rx vs twin"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
log "start"; PAR=2
runp plq300k_Rx_97 --tgt $G/d163/assets/d163plaque.obj --render_tgt $G/d163/assets/d163R300plaque.obj --n 300000 --seed 97 $RX
runp rip300k_Rx_97 --tgt $A1/d161smooth_bunny.obj --render_tgt $G/d162f/assets/d162fR300ripple_bunny.obj --n 300000 --seed 97 $RX
wait; log "done"
