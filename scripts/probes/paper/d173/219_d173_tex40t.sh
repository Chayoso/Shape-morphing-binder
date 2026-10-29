#!/bin/bash
# 219_d173_tex40t.sh -- queue job (agent D173): Bob at 40k, Rx and Rx + nearest colour rerun on the code that records the render's
# share of each block's step (the runs of jobs 200 / 205 / 209 predate it; the records only, the same flags): seeds 97 / 99.
J=219_tex40t; TITLE="D173 texture 40k reruns with the block records"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
TX="--appearance texture --texture $A/d173bob_tex.obj,$A/d173bob_diffuse.png"
gate_t; log "start"; PAR=4
for s in 97 99; do
  runp tex40k_Rxt_$s --tgt assets/bob.obj $TX --n 40000 --seed $s $RX
  runp tex40k_Rxnt_$s --tgt assets/bob.obj $TX --n 40000 --seed $s $RX --colour_kernel nearest
done
wait; log "done"
