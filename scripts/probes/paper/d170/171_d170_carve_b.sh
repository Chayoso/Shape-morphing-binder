#!/bin/bash
# 171_d170_carve_b.sh -- queue job (agent D170): the 300k "PM" reliefs (D161's coin and smoothed bunny flank, the relief
# meshes rebuilt at the 300k pitch: output/gpu/d170/assets), Rx, seed 97, and the coin's 300k twin (none existed).
# The bunny relief's twin is d162f/runs/rip300k_T_97 (the smoothed bunny, lambda = 0, seed 97: the render target is unread).
J=171_carve_b; TITLE="D170: 300k PM reliefs (coin, bunny flank), Rx vs twin"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
log "start"; PAR=2
wait_file $A/L300bunny.ok
runp rel300k_Rx_97 --tgt $A1/d161smooth_bunny.obj --render_tgt $A/d170L300_bunny.obj --n 300000 --seed 97 $RX
wait_file $A/L300coin.ok
runp coin300k_T_97 --tgt $A1/d161coin.obj --render_tgt $A/d170L300coin.obj --n 300000 --seed 97 $TW
runp coin300k_Rx_97 --tgt $A1/d161coin.obj --render_tgt $A/d170L300coin.obj --n 300000 --seed 97 $RX
wait; log "done"
