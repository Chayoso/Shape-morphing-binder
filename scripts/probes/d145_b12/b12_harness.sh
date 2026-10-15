#!/bin/bash
# b12_harness.sh MESH... -- D145's harness (tmp/d145/harness.py, unchanged: progress p on one grid per mesh, apart
# particles and occupied volume at p = 0.25 / 0.5 / 0.75 and the end) on seed 97 of each B12 arm (LSa97, LSb97, FCa97,
# FCb97), in the environment harness.sh gives it (repo_d145, the pipeline's python); run on the card b12_gpu.sh claimed.
# The display step of harness.sh is not run here (its frames go to the D145 agent).
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d145_b12
export PHYSMORPH_RUN_REPO=$B/repo_d145
source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
cd $B/repo_d145
for mesh in "$@"; do
  specs=""
  for label in LSa97 LSb97 FCa97 FCb97; do
    f=$O/${mesh}300k_${label}_frames12.npz
    [ -f $f ] && specs="$specs $label=$f"
  done
  [ -n "$specs" ] && $PY $B/tmp/d145/harness.py $O/harness_$mesh.jsonl $mesh $B/output/gpu/d90/${mesh}_ind300k.npz $specs \
      > $O/harness_$mesh.log 2>&1
  echo "$(date '+%F %T %Z') harness $mesh rc $?" >> /data/relcfd/chayo/physmorph_v2/baselines/b12/logs/status.txt
done
