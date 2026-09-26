#!/bin/bash
# Source after hyde06_env.sh; isolated dependencies, no changes to the base env.
GPU_DEPS=/data/relcfd/chayo/physmorph_v2/deps/gpu_pipeline
test -d "$GPU_DEPS/cupy" || { echo "Missing CUDA pipeline dependencies: $GPU_DEPS" >&2; return 2; }
unset CONDA_PREFIX
export CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
export CUPY_CACHE_DIR=/data/relcfd/chayo/physmorph_v2/cache/cupy
export PYTHONPATH="$GPU_DEPS:$REPO${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CUPY_CACHE_DIR"
