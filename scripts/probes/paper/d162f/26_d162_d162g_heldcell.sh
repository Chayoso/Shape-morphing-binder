#!/bin/bash
# 26_d162_d162g_heldcell.sh -- gpu1 queue job (agent D162): D162g, the held weight with the placement terms at the loss
# cell (H+C: --lambda_ema 0 --placement_at_cell) on D161's coin and bunny relief, 40k, seeds 97 / 99. Twin: D162's TC.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162g; mkdir -p $O/runs
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
A=$G/d161/assets
log "job 26 start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
for s in 97 99; do
  run coin40k_HC_$s repo_d162ds --tgt $A/d161coin.obj --render_tgt $A/d161L40coin.obj --n 40000 --seed $s --lambda_ema 0 --placement_at_cell
  run bunny40k_HC_$s repo_d162ds --tgt $A/d161smooth_bunny.obj --render_tgt $A/d161L40_bunny.obj --n 40000 --seed $s --lambda_ema 0 --placement_at_cell
done
log "R done (d162g)"
