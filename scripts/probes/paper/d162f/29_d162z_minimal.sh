#!/bin/bash
# 29_d162z_minimal.sh -- gpu1 queue job (agent D162), job "29b": the render-centric minimal recipe. The render keeps its
# weight (--lambda_ema 0; with a settled end, --settle_windows 30, if D162h read A), the cleanup off (--cleanup_off:
# w_dt = w_nn = 0), the placement terms at the loss cell (--placement_at_cell). 40k, seed 97: coin lettering, colour band,
# six gallery meshes. Against the tag's V / W (D155 gate, D154 g40runs).
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162z; mkdir -p $O/runs
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
A=$G/d161/assets
S=""; grep -q "verdict A" $G/d162h/status.txt 2>/dev/null && S="--settle_windows 30"
M="--lambda_ema 0 --cleanup_off --placement_at_cell $S"
log "job 29b start (recipe: $M)"
run coin40k_M_97 repo_d162zs --tgt $A/d161coin.obj --render_tgt $A/d161L40coin.obj --n 40000 --seed 97 $M
run bunny40k_band_M_97 repo_d162zs --tgt assets/bunny.obj --appearance band --n 40000 --seed 97 $M
for m in bunny dragon armadilo bob homer teapot; do run gal_${m}40k_M_97 repo_d162zs --tgt assets/$m.obj --n 40000 --seed 97 $M; done
log "R done (d162z)"
