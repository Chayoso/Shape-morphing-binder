#!/bin/bash
# 178_d170_r0h0.sh -- queue job (agent D170): the coordinator's 40k diagnostic on repo_d170 (after the V3 gate): does the
# layer relaxation fight the render during the whole run? R0 = the rule (tag default) + --no-layer_relax, H0 = held +
# --no-layer_relax, each with settle 30 and --no-settle_relax (as Rx); coin / bunny relief / band, seeds 97 / 99 (D166's
# scenes; twins D161 / D160's); then R0 on the 6-mesh gallery, seed 97.
J=178_r0h0; TITLE="D170: the relaxation off throughout (R0 / H0), 40k scenes + R0 gallery"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
for k in $(seq 1 360); do [ -e $O/gates/V3.ok ] && break; [ -e $O/gates/V3.fail ] && { log "V3 failed: stop"; exit 1; }; sleep 20; done
[ -e $O/gates/V3.ok ] || { log "no V3 gate: stop"; exit 1; }
REPO=repo_d170; PAR=4; log "start"
F_coin="--tgt $A1/d161coin.obj --render_tgt $A1/d161L40coin.obj"
F_bunny="--tgt $A1/d161smooth_bunny.obj --render_tgt $A1/d161L40_bunny.obj"
F_band="--tgt assets/bunny.obj --appearance band"
OFF="--settle_windows 30 --no-settle_relax --no-layer_relax"
for s in 97 99; do for sc in coin bunny band; do eval f=\$F_$sc
  runp ${sc}_R0_$s $f --n 40000 --seed $s $OFF
  runp ${sc}_H0_$s $f --n 40000 --seed $s --lambda_ema 0 $OFF
done; done
for m in bunny dragon armadilo bob homer teapot; do runp gal_${m}_R0_97 --tgt assets/$m.obj --n 40000 --seed 97 $OFF; done
wait; log "done"
