#!/bin/bash
# 188_d172_combined.sh -- queue job (agent D172): C, the combined carve + colour scene at 40k (repo_d172's render_x paint
# fix): the smoothed bunny, the render's pictures = D161's "PM" relief bunny painted with D160's band; Rx, Hu, Hf and the
# twin, seeds 97 / 99.
J=188_combined; TITLE="D172: the combined relief + paint scene, 40k"
source /data/relcfd/chayo/physmorph_v2/tmp/d172/job_common.sh
gate_wait; log "start"; PAR=4
C="--tgt $A1/d161smooth_bunny.obj --render_tgt $A1/d161L40_bunny.obj --appearance band --n 40000"
for s in 97 99; do
  runp cmb_T_$s $C --seed $s $TW
  runp cmb_Rx_$s $C --seed $s $RX
  runp cmb_Hu_$s $C --seed $s $HU
  runp cmb_Hf_$s $C --seed $s $HF
done
wait; log "done"
