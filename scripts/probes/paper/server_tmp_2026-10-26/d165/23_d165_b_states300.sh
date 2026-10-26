#!/bin/bash
# D165 job b (gpu1 queue, 2026-10-22): D163's two 300k runs again with their end state kept (repo_d165s = repo_d163s +
# the --save_end_state dump, default off; tmp/d165/repo_d165s.patch): H (dF_c + layer u + render loss, the render weight
# held, --lambda_ema 0) and the twin T (dF_c + layer u, --render_weight_scale 0), D163's flags otherwise.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d165; W=$B/tmp/d165; D3=$B/output/gpu/d163
mkdir -p $O/runs300 $W/tmpdir
source $B/tmp/d146/cache_env.sh
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%F %T') [b] $*" >> $O/status.txt; }
log "start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
( export PHYSMORPH_RUN_REPO=$B/repo_d165s PHYSMORPH_OUT=$B/output/gpu; source $B/repo_d165s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
  export TMPDIR=$W/tmpdir; cd $B/repo_d165s
  for spec in "plq300k_H_97 --lambda_ema 0" "plq300k_T_97 --render_weight_scale 0"; do
    set -- $spec; name=$1; shift
    [ -f $O/runs300/${name}_end.npz ] && continue
    s=$(date +%s)
    $PY scripts/pipeline_run.py --tgt $D3/assets/d163plaque.obj --render_tgt $D3/assets/d163R300plaque.obj --n 300000 \
      --seed 97 "$@" --save_end_state --out $O/runs300/$name > $O/runs300/$name.log 2>&1 < /dev/null
    log "$name exit $? ($(( $(date +%s) - s )) s): $(grep -m1 'ARM ' $O/runs300/$name.log | cut -c1-200)"
  done )
log "end"
