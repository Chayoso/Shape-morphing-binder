#!/bin/bash
# 22_d162_d162e_uoff.sh -- gpu1 queue job (agent D162): D162e, is it the render or u? The lambda-held full arm (D161c's
# flags, --lambda_ema 0) with the layer's u off (--u_off) on D161's coin and bunny relief, 40k, seeds 97 / 99. Its
# references: D161c (the same arm, u on) and D161's twins (u on, lambda = 0). Output output/gpu/d162e/.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162e; mkdir -p $O/runs
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
A=$G/d161/assets
log "job 22 start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
for s in 97 99; do
  run coin40k_Hu_$s repo_d162ds --tgt $A/d161coin.obj --render_tgt $A/d161L40coin.obj --n 40000 --seed $s --lambda_ema 0 --u_off
  run bunny40k_Hu_$s repo_d162ds --tgt $A/d161smooth_bunny.obj --render_tgt $A/d161L40_bunny.obj --n 40000 --seed $s --lambda_ema 0 --u_off
done
log "R done (d162e)"
