#!/bin/bash
# 51_d162_stiff.sh (29 rerun after the Jmin_traj fix) -- gpu1 queue job (agent D162): the held arm (--lambda_ema 0) on D161's coin and bunny relief, 40k,
# seed 97, with Young's modulus x3 and x7 (--young 4.2e5 / 9.8e5; default 1.4e5), each settled 30 zero-control windows
# after its stop (--settle_windows 30). References: D161c (E x1), D161's twins.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162s; mkdir -p $O/runs
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
A=$G/d161/assets
log "job 29 start"
for e in 3:4.2e5 7:9.8e5; do k=${e%%:*}; y=${e##*:}
  run coin40k_E${k}_97 repo_d162hs --tgt $A/d161coin.obj --render_tgt $A/d161L40coin.obj --n 40000 --seed 97 --lambda_ema 0 --young $y --settle_windows 30
  run bunny40k_E${k}_97 repo_d162hs --tgt $A/d161smooth_bunny.obj --render_tgt $A/d161L40_bunny.obj --n 40000 --seed 97 --lambda_ema 0 --young $y --settle_windows 30
done
log "R done (d162s2)"
