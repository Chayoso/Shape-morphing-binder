#!/bin/bash
# 176_d170_band300.sh -- queue job (agent D170): the D160 colour band at 300k, Rx, seed 97. Twin reused: d162f/runs/band300k_T_97.
J=176_band300; TITLE="D170: 300k colour band, Rx vs twin"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
log "start"; PAR=1
run1 band300k_Rx_97 --tgt assets/bunny.obj --appearance band --n 300000 --seed 97 $RX
log "done"
