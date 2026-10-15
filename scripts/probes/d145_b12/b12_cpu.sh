#!/bin/bash
# b12_cpu.sh SCRIPT ARGS... -- B11's CPU runner (baselines/arap/scripts/run_b11.sh) with B12's temp dir: B11's venv
# (read-only; libigl with CGAL, trimesh), repo_d145 on PYTHONPATH for the pipeline's sampler and metrics, B11's
# scripts dir on PYTHONPATH (b11_common, b11_endcheck), no card.
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
export PHYSMORPH_RUN_REPO=/data/relcfd/chayo/physmorph_v2/repo_d145
export PHYSMORPH_OUT=/data/relcfd/chayo/physmorph_v2/output/gpu/d145_b12
source /data/relcfd/chayo/physmorph_v2/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
A=/data/relcfd/chayo/physmorph_v2/baselines/arap
B12=/data/relcfd/chayo/physmorph_v2/baselines/b12
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export TMPDIR=$B12/cache/tmp PIP_CACHE_DIR=$B12/cache/pip
export PYTHONPATH="$A/scripts:$PYTHONPATH"
cd $A/scripts
exec $A/venv/bin/python -W ignore "$@"
