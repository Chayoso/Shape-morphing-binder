#!/bin/bash
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; T1=$B/tmp/d161; O=$G/d162s
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=6 D157_THREADS=6 MPLCONFIGDIR=$B/cache/matplotlib PYTHONPATH=$B/repo_d162d:$B/deps/d157_lpips
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
cd $B/repo_d162d
for s in coin bunny; do nice -n 10 $PY $T1/d161_read.py $O/after/read_$s.json $O/after/scene_$s.json > $O/after/read_$s.log 2>&1 & done
wait
mkdir -p $G/d162/auto/51_stiff
$PY $B/tmp/d162/d162_auto.py stiff $G/d162/auto/51_stiff/summary.md > $O/after/auto51.log 2>&1
echo rc=$? >> $O/after/auto51.log
