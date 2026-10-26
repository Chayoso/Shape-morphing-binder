#!/bin/bash
cd /data/relcfd/chayo/physmorph_v2/repo_d163
source ../tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 PHYSMORPH_CACHE=/data/relcfd/chayo/physmorph_v2/cache/samples
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
A=/data/relcfd/chayo/physmorph_v2/output/gpu/d163/assets
for N in 300 40; do
  ( s=$(date +%s); nice -n 10 $PY ../tmp/d163/d163_mesh.py relief $A/d163R${N}plaque.obj ../tmp/d163/relief${N}_spec.json > $A/d163R${N}plaque.log 2>&1; echo "rc $? secs $(( $(date +%s) - s ))" >> $A/d163R${N}plaque.log ) &
done
wait
