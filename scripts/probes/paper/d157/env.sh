# env.sh -- D157 (agent D157): every cache and temp dir on /data; the tag's code (repo_t1) on the path; CPU readers only.
source /data/relcfd/chayo/physmorph_v2/tmp/d146/cache_env.sh
export PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
export D157=/data/relcfd/chayo/physmorph_v2/tmp/d157
export D157_OUT=/data/relcfd/chayo/physmorph_v2/output/gpu/d157
export PYTHONPATH=/data/relcfd/chayo/physmorph_v2/deps/d157_lpips:/data/relcfd/chayo/physmorph_v2/repo_t1
export OMP_NUM_THREADS=${D157_THREADS:-6} MKL_NUM_THREADS=${D157_THREADS:-6} OPENBLAS_NUM_THREADS=${D157_THREADS:-6}
export CUDA_VISIBLE_DEVICES=""
