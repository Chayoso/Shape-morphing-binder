#!/bin/bash
set -euo pipefail
base=/data/relcfd/chayo/physmorph_v2
source_dir=${1:?audited source copy under /data}
build_dir=${2:?new isolated build directory under /data}
source_dir=$(realpath "$source_dir")
build_dir=$(realpath -m "$build_dir")
case "$source_dir" in "$base"/*) ;; *) exit 2;; esac
case "$build_dir" in "$base"/deps/*) ;; *) exit 2;; esac
test ! -e "$build_dir"
export PHYSMORPH_RUN_REPO=$(cd "$(dirname "$0")/../.." && pwd)
source "$PHYSMORPH_RUN_REPO/scripts/ops/hyde06_env.sh"
source "$PHYSMORPH_RUN_REPO/scripts/ops/gpu_env.sh"
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=8.6
"$PY" "$REPO/scripts/ops/prepare_continuous_raster.py" --source "$source_dir" --destination "$build_dir"
cd "$build_dir"
"$PY" setup.py build_ext --inplace
