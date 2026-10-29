#!/bin/bash
# 220_d173_strp40.sh -- queue job (agent D173): the stripes scene (three vertical stripes in the front view, the middle red; the
# source's top third red: the twin's layout meets it at chance) on the bunny at 40k, seeds 97 / 99: the twin and Rx with the
# nearest-particle colour in the loss.
J=220_strp40; TITLE="D173 stripes 40k"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
gate_t; log "start"; PAR=4
for s in 97 99; do
  runp strp40k_T_$s --tgt assets/bunny.obj --appearance stripes --n 40000 --seed $s $TW
  runp strp40k_Rxn_$s --tgt assets/bunny.obj --appearance stripes --n 40000 --seed $s $RX --colour_kernel nearest
done
wait; log "done"
