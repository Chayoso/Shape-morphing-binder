#!/bin/bash
# 50_d162_settle.sh (27 rerun after the Jmin_traj fix) -- gpu1 queue job (agent D162): D162h, the settle test. The lambda-held arm (--lambda_ema 0)
# with 30 windows of zero-control released physics after its stop (--settle_windows 30, about 5 s, delivered) on D161's
# coin and bunny relief and D160's band, 40k, seeds 97 / 99 (repo_d162hs). V3 (SAME at the defaults) first.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162h; mkdir -p $O/runs $O/gates
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
A=$G/d161/assets
log "job 27 start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, code $(cat $B/repo_d162h/VERSION))"
( envr repo_d162hs; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 --tgt assets/bunny.obj \
    --out $O/gates/S1 > $O/gates/S1.log 2>&1 < /dev/null; $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $O/gates/S1 > $O/gates/cmp.txt 2>&1 )
log "V3: $(tail -1 $O/gates/cmp.txt)"
tail -1 $O/gates/cmp.txt | grep -q ": SAME$" || { log "V3 STOP"; exit 1; }
for s in 97 99; do
  run coin40k_Hs_$s repo_d162hs --tgt $A/d161coin.obj --render_tgt $A/d161L40coin.obj --n 40000 --seed $s --lambda_ema 0 --settle_windows 30
  run bunny40k_Hs_$s repo_d162hs --tgt $A/d161smooth_bunny.obj --render_tgt $A/d161L40_bunny.obj --n 40000 --seed $s --lambda_ema 0 --settle_windows 30
  run bunny40k_band_Hs_$s repo_d162hs --tgt assets/bunny.obj --appearance band --n 40000 --seed $s --lambda_ema 0 --settle_windows 30
done
log "R done (d162h2)"
