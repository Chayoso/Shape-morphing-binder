#!/bin/bash
# 82_d162_500k.sh -- gpu1 queue job (agent D162): the deliverables at 500k (300k if the card's free memory is < 36 GB;
# noted) with the winning definition, chosen here from the auto summaries: D162h verdict A -> held + settle 30; else
# D162g (H+C) passing rest at both seeds -> held + placement at the cell; else held. Plaque and ripples (full + twin),
# band (D160d + twin). Output output/gpu/d162k/.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d162k; mkdir -p $O/runs
source /data/relcfd/chayo/physmorph_v2/tmp/d162/d162f_job_common.sh
AU=$G/d162/auto; A1=$G/d161/assets; A3=$G/d163/assets; AF=$G/d162f/assets
DEF="--lambda_ema 0"
if grep -q "VERDICT: A" $AU/27_settle/summary.md 2>/dev/null; then DEF="--lambda_ema 0 --settle_windows 30"
elif [ -s $AU/26_heldcell/summary.md ] && ! grep "P4" $AU/26_heldcell/summary.md | head -2 | grep -q FAIL; then DEF="--lambda_ema 0 --placement_at_cell"; fi
used=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits); tot=$(nvidia-smi -i 1 --query-gpu=memory.total --format=csv,noheader,nounits)
N=500000; [ $((tot - used)) -lt 36000 ] && N=300000
k=$((N / 1000))k
log "job 82 start: N $N (free $((tot - used)) MiB), definition '$DEF'"
run plq${k}_X_97 repo_d162zs --tgt $A3/d163plaque.obj --render_tgt $A3/d163R300plaque.obj --n $N --seed 97 $DEF
run plq${k}_T_97 repo_d162zs --tgt $A3/d163plaque.obj --render_tgt $A3/d163R300plaque.obj --n $N --seed 97 --render_weight_scale 0
run rip${k}_X_97 repo_d162zs --tgt $A1/d161smooth_bunny.obj --render_tgt $AF/d162fR300ripple_bunny.obj --n $N --seed 97 $DEF
run rip${k}_T_97 repo_d162zs --tgt $A1/d161smooth_bunny.obj --render_tgt $AF/d162fR300ripple_bunny.obj --n $N --seed 97 --render_weight_scale 0
run band${k}_D_97 repo_d160s --tgt assets/bunny.obj --appearance band --n $N --seed 97 --appearance_weight held
run band${k}_T_97 repo_d160s --tgt assets/bunny.obj --appearance band --n $N --seed 97 --render_weight_scale 0
{ echo "# Job 82: the deliverables at $N (definition: $DEF)"; echo; grep -h "R .* exit\|CUDA out of memory" $O/status.txt; } > $AU/82/summary.md 2>/dev/null || { mkdir -p $AU/82; { echo "# Job 82: N $N, definition $DEF"; grep -h "R .* exit" $O/status.txt; } > $AU/82/summary.md; }
log "R done (d162k)"
