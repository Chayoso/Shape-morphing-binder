#!/bin/bash
# Usage: run_gpu_pipeline.sh GPU <pipeline_run arguments, including target_reference>
set -euo pipefail
HERE=$(cd "$(dirname "$0")/../.." && pwd)
export PHYSMORPH_RUN_REPO=$HERE
source "$HERE/scripts/ops/hyde06_env.sh"
source "$HERE/scripts/ops/gpu_env.sh"
if [ "$#" -lt 1 ]; then echo 'usage: run_gpu_pipeline.sh GPU [pipeline arguments]' >&2; exit 2; fi
GPU=$1; shift
[[ "$GPU" == 0 || "$GPU" == 2 ]] || { echo 'Use assigned GPU 0 or 2' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=$GPU
cd "$HERE"
exec "$PY" scripts/ops/cuda_python.py scripts/pipeline_run.py "$@" --compute_backend cuda
