#!/bin/bash
# 196_d174_s40kVV.sh -- queue job (agent D174): Hu --stop_tol 0.VV (repo_d174s) on coin / bunny relief / band x 97 / 99 and the
# 6-mesh 40k gallery seed 97 (D172's arguments), PAR 3.
J=s40kVV
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"; PAR=3
for s in 97 99; do for sc in coin bunny band; do eval f=\$F_$sc; runp repo_d174s ${sc}_HsVV_$s $f --n 40000 --seed $s $HU --stop_tol 0.VV; done; done
for m in dragon armadilo homer bunny bob teapot; do runp repo_d174s gal_${m}_HsVV_97 --tgt assets/$m.obj --n 40000 --seed 97 $HU --stop_tol 0.VV; done
wait; log "done"
