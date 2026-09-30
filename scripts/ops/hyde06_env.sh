#!/bin/bash
# Common environment for every hyde06-side ops script. Everything lives under /data: the deployed
# repo, the outputs, the caches. Nothing in $HOME.
export REPO=${PHYSMORPH_RUN_REPO:-/data/relcfd/chayo/physmorph_v2/repo_settled}
export OUT=${PHYSMORPH_OUT:-/data/relcfd/chayo/physmorph_v2/output/settled}
export PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
export STATUS=$OUT/status.log
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export XDG_CACHE_HOME=/data/relcfd/chayo/physmorph_v2/cache
export WARP_CACHE_PATH=$XDG_CACHE_HOME/warp
export CUDA_CACHE_PATH=$XDG_CACHE_HOME/cuda
export TORCH_HOME=$XDG_CACHE_HOME/torch
export TORCH_EXTENSIONS_DIR=$XDG_CACHE_HOME/torch_extensions
export MPLCONFIGDIR=$XDG_CACHE_HOME/matplotlib
export TMPDIR=/data/relcfd/chayo/physmorph_v2/tmp
export PHYSMORPH_CACHE=$XDG_CACHE_HOME/samples      # the prepare stage's sample cache, kept across deploys
# the GPU-only run stage: CuPy (device KD-tree, ndimage) from an isolated folder, no changes to the env
GPU_DEPS=/data/relcfd/chayo/physmorph_v2/deps/gpu_pipeline
test -d "$GPU_DEPS/cupy" || echo "missing the CUDA pipeline dependencies: $GPU_DEPS" >&2
unset CONDA_PREFIX
export CUDA_HOME=/usr/local/cuda-12.8 CUDA_PATH=/usr/local/cuda-12.8
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:${LD_LIBRARY_PATH:-}"
export CUPY_CACHE_DIR=$XDG_CACHE_HOME/cupy
export PYTHONPATH="$GPU_DEPS:$REPO"
mkdir -p "$XDG_CACHE_HOME" "$WARP_CACHE_PATH" "$CUDA_CACHE_PATH" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
         "$MPLCONFIGDIR" "$TMPDIR" "$CUPY_CACHE_DIR" "$OUT"
cd $REPO
