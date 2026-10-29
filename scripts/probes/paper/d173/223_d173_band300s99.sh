#!/bin/bash
# 223_d173_band300s99.sh -- queue job (agent D173): the band at 300k, seed 99: the twin, Rx and Rx + nearest colour. Seed 97's Rxn
# stopped on the delivery-merit plateau at window 42 (edge 2.93 pitches) while Rx ran to the physics convergence at window 82 (2.68);
# at the same time (raw 1680) Rxn's edge was 2.94 and Rx's 4.28. A second seed of all three says which stop each colour path takes.
J=223_band300s99; TITLE="D173 band 300k seed 99 (T, Rx, Rxn)"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
gate_t; log "start"; PAR=2
runp band300k_T_99 --tgt assets/bunny.obj --appearance band --n 300000 --seed 99 $TW
runp band300k_Rxn_99 --tgt assets/bunny.obj --appearance band --n 300000 --seed 99 $RX --colour_kernel nearest
wait
runp band300k_Rx_99 --tgt assets/bunny.obj --appearance band --n 300000 --seed 99 $RX
wait; log "done"
