#!/bin/bash
set -euo pipefail
test "$#" -eq 2
task=$1
repo=$2
[[ "$task" =~ ^[a-z0-9_]+$ ]]
[[ "$repo" =~ ^repo_[a-z0-9_.]+$ ]]
B=/data/relcfd/chayo/physmorph_v2
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
D=$B/tmp/$task
test ! -e "$D"
mkdir "$D"
tar -xf "$B/tmp/$task.tar" -C "$D"
source "$B/repo_d168/scripts/ops/hyde06_env.sh"
source "$B/tmp/d146/cache_env.sh"
export PYTHONPATH="$B/deps/gpu_pipeline:$B/$repo"
export CUDA_VISIBLE_DEVICES='' PHYSMORPH_CPU_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset NPY_PROMOTION_STATE
exec "$PY" -B -u "$B/tmp/render_validity_overlay_v1/run_guarded.py" \
    --out "$B/output/gpu/${task}_fixtures" \
    --resource-helper "$B/tmp/reader_context_bounded_v3/resource_scan.py" \
    --fixture-plan "$D/fixture_plan.json"
