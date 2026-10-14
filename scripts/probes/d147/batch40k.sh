#!/bin/bash
# D147 40k batch, run inside d147_job.sh's environment ($PY, $REPO): V0, V1, the scene (arm, twin).
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147/runs; mkdir -p $O
SP=scripts/probes/d147/specs
run() { tag=$1; shift; echo "$tag start $(date '+%F %T')" >> $O/batch.log; $PY scripts/scene_run.py "$@" --out $O/$tag > $O/$tag.log 2>&1; echo "$tag exit $? $(date '+%F %T')" >> $O/batch.log; }
for spec in ${D147_ONLY:-v0 v1 scene}; do
  case $spec in
    v0) for mesh in bunny dragon; do for s in 97 99; do for arm in fixed body; do
          run v0_${mesh}_${arm}_s$s --n 40000 --seed $s --tgt assets/$mesh.obj --scene $SP/v0_${mesh}_${arm}.json; done; done; done;;
    v1) for g in 005 010; do run v1_bunny_g$g --n 40000 --seed 97 --scene $SP/v1_g$g.json; done;;
    scene) run scene40k_L --n 40000 --seed 97; run scene40k_P --n 40000 --seed 97 --render_weight_scale 0
           run scene40k_LB --n 40000 --seed 97 --scene $SP/scene_long.json;;
  esac
done
echo "batch_end $(date '+%F %T')" >> $O/batch.log
