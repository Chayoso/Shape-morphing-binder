#!/bin/bash
# d156_cpu.sh JOBS OUT_PREFIX [SLOTS] -- D156: d156_read.py over JOBS in SLOTS (default 3) CPU readers at nice 10, six
# threads each, no card. A context (mesh seed frame) is five consecutive lines (H M Mp and two states); the contexts are
# dealt round-robin to the slots, each slot writes OUT_PREFIX_sK.jsonl. PIDs and the start in OUT_PREFIX.pids.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d156
source $W/env.sh
export CUDA_VISIBLE_DEVICES="" D157_THREADS=6 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export PYTHONPATH=$B/deps/d157_lpips:$B/repo_d156:$W
JOBS=$1; OUT=$2; SLOTS=${3:-3}
cd $B/repo_d156
grep -v '^#' $JOBS | awk 'NF' > $OUT.all
awk -v s=$SLOTS -v o=$OUT '{k=$1" "$2" "$3; if (k!=last) {c++; last=k} print > (o "_s" (c % s) ".jobs")}' $OUT.all
: > $OUT.pids
for f in ${OUT}_s*.jobs; do
  k=${f%.jobs}
  nohup nice -n 10 $PY $W/d156_read.py $k.jsonl $f > $k.log 2>&1 < /dev/null &
  echo "$! $f $(date '+%F %T')" >> $OUT.pids
done
cat $OUT.pids
