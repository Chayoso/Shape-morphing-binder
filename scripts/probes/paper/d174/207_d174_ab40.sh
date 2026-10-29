#!/bin/bash
# 207_d174_ab40.sh -- queue job (agent D174): the gate of repo_d174ab (batch 1 + --render_every / --ls_max, default off:
# bit-SAME as repo_d172 on G1-G4), then the runtime A/B at 40k against Hu: Hu2 = Hu --render_every 2, Hu4 = Hu --render_every 4
# on coin / bunny relief / band x seeds 97 / 99 and the 6-mesh gallery seed 97 (D172's run arguments, PAR 3).
J=207_ab40
source /data/relcfd/chayo/physmorph_v2/tmp/d174/d174_common.sh
log "start (card $CUDA_VISIBLE_DEVICES)"
NOLOCK=1 bash $W/d174_gate.sh ab repo_d174ab
gate_wait ab
PAR=3
for a in Hu2:2 Hu4:4; do arm=${a%%:*}; k=${a##*:}
  for s in 97 99; do for sc in coin bunny band; do eval f=\$F_$sc; runp repo_d174ab ${sc}_${arm}_$s $f --n 40000 --seed $s $HU --render_every $k; done; done
  for m in dragon armadilo homer bunny bob teapot; do runp repo_d174ab gal_${m}_${arm}_97 --tgt assets/$m.obj --n 40000 --seed 97 $HU --render_every $k; done
done
wait; log "done"
