#!/bin/bash
# 206_d174_st_40k07.sh -- queue job (agent D174, stop A/B): Hu with tol 0.07 (repo_d174t07) on coin / bunny relief / band x 97 / 99 and the
# 6-mesh 40k gallery seed 97 (D172's arguments), PAR 3.
J=st_40k07
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"; PAR=3
for s in 97 99; do for sc in coin bunny band; do eval f=\$F_$sc; runp repo_d174t07 ${sc}_Ht07_$s $f --n 40000 --seed $s $HU; done; done
for m in dragon armadilo homer bunny bob teapot; do runp repo_d174t07 gal_${m}_Ht07_97 --tgt assets/$m.obj --n 40000 --seed 97 $HU; done
wait; log "done"
