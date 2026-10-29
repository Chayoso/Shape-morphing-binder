#!/bin/bash
# 185_d172_scenes.sh -- queue job (agent D172): B, the render's authority split by control block, 40k with Rx's settle:
# Hu = held on u, the rule on dFc; Hf = held on dFc, the rule on u; coin "PM" / bunny relief / band, seeds 97 / 99.
J=185_scenes; TITLE="D172: Hu / Hf on the 40k scenes"
source /data/relcfd/chayo/physmorph_v2/tmp/d172/job_common.sh
gate_wait; log "start"; PAR=4
F_coin="--tgt $A1/d161coin.obj --render_tgt $A1/d161L40coin.obj"
F_bunny="--tgt $A1/d161smooth_bunny.obj --render_tgt $A1/d161L40_bunny.obj"
F_band="--tgt assets/bunny.obj --appearance band"
for s in 97 99; do for sc in coin bunny band; do eval f=\$F_$sc
  runp ${sc}_Hu_$s $f --n 40000 --seed $s $HU
  runp ${sc}_Hf_$s $f --n 40000 --seed $s $HF
done; done
wait; log "done"
