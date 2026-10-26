#!/bin/bash
# ri_endmap.sh -- render influence (2026-10-22, CPU, nice 10): the end-surface maps, bunny then dragon (one process).
B=/data/relcfd/chayo/physmorph_v2
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 D157_THREADS=8 PYTHONPATH=$B/repo_t1
O=$B/output/gpu/render_influence
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
for M in ${@:-bunny dragon}; do
  nice -n 10 $PY $B/tmp/render_influence/ri_endmap.py $M $O > $O/endmap_$M.log 2>&1
  echo "$(date +%H:%M:%S) endmap $M rc $?" >> $O/status.txt
done
