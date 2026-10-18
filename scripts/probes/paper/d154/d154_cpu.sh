#!/bin/bash
# d154_cpu.sh TAG JOBS_PREFIX [yard] -- D154's CPU readers: d154_mesh_read.py on every JOBS_PREFIX_K.txt in parallel
# (one process each, D154_THREADS threads, nice 10, no card), into output/gpu/d154/read/TAG_K.jsonl; a status line each.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d154; O=$B/output/gpu/d154; mkdir -p $O/read
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" D154_THREADS=${D154_THREADS:-8} OMP_NUM_THREADS=${D154_THREADS:-8}
P=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
tag=$1; pre=$2; mode=$3
cd $W
for f in ${pre}_*.txt; do
  k=$(basename $f .txt); k=${k##*_}
  ( nice -n 10 $P d154_mesh_read.py $O/read/${tag}_$k.jsonl $f $mode > $O/read/${tag}_$k.log 2>&1 < /dev/null
    echo "$(date '+%m-%d %H:%M:%S') cpu $tag $k exit $? rows $(wc -l < $O/read/${tag}_$k.jsonl)" >> $O/status.txt ) &
done
wait
echo "$(date '+%m-%d %H:%M:%S') cpu $tag done" >> $O/status.txt
