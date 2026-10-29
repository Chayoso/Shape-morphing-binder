#!/bin/bash
# 186_d172_gal.sh -- queue job (agent D172): Hu / Hf on the 6-mesh 40k gallery, seed 97 (99 follows only for an arm that holds).
J=186_gal; TITLE="D172: Hu / Hf on the 40k gallery, seed 97"
source /data/relcfd/chayo/physmorph_v2/tmp/d172/job_common.sh
gate_wait; log "start"; PAR=4
for m in dragon armadilo homer bunny bob teapot; do
  runp gal_${m}_Hu_97 --tgt assets/$m.obj --n 40000 --seed 97 $HU
  runp gal_${m}_Hf_97 --tgt assets/$m.obj --n 40000 --seed 97 $HF
done
wait; log "done"
