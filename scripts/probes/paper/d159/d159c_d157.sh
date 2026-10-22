#!/bin/bash
# d159c_d157.sh -- D157 decision set (R3) on D159c C97 ends (CPU), as d159_readcpu.sh did for M97.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d159; T=$B/tmp; R=$O/read; S=$O/status.txt
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
source $T/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
for mesh in bunny dragon; do
  ( source $T/d157/env.sh; cd $T/d157
    echo "$mesh 97 300k C $O/${mesh}300k_C97_end1.npz end" > $R/d157c_jobs_$mesh.txt
    nice -n 10 $PY d157_read.py $R/d157c_$mesh.jsonl $R/d157c_jobs_$mesh.txt > $R/d157c_$mesh.log 2>&1 < /dev/null )
  echo "$(date "+%m-%d %H:%M:%S") D159c R3 (D157) $mesh rc $?" >> $S
done
