#!/bin/bash
set -euo pipefail
B=/data/relcfd/chayo/physmorph_v2
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
source "$B/repo_d168/scripts/ops/hyde06_env.sh"
source "$B/tmp/d146/cache_env.sh"
export PYTHONPATH="$B/deps/gpu_pipeline:$B/repo_d166"
export CUDA_VISIBLE_DEVICES='' PHYSMORPH_CPU_THREADS=8 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
unset NPY_PROMOTION_STATE
exec "$PY" -B -u "$B/tmp/dragon_dense_checkpoint_v3/freeze.py" \
    --proof-sha b5bbbbaff7f7a482d4bb95c4dd8b0dbcb024a077b1b1ffcda99f37c2400c6b34
