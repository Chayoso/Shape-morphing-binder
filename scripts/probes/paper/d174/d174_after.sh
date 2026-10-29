#!/bin/bash
# d174_after.sh -- agent D174's detached CPU reader of the runtime A/B: waits for tmp/gpu1_queue/207_d174_ab40.done, reads the
# scenes and the gallery; waits for 208_d174_ab300.done, reads the 300k scales; writes output/gpu/d174/AB_SUMMARY.md after each.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d174; O=$B/output/gpu/d174; Q=$B/tmp/gpu1_queue
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=8 D157_THREADS=8 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d166:$B/deps/d157_lpips D170_WAIT_H=1
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%F %T') [after] $*" >> $O/status.txt; }
cd $B/repo_d166
read1() { local mode=$1 dir=$O/auto/ab_$1; mkdir -p $dir; local s=$(date +%s)
  nice -n 10 $PY $W/d174_read.py $mode $dir > $dir/read.out 2>&1; local rc=$?
  log "read $mode exit $rc ($(( $(date +%s) - s )) s): $(grep VERDICT $dir/summary.md 2>/dev/null | cut -c1-300)"
  nice -n 10 $PY $W/d174_read.py aggregate $O/AB_SUMMARY.md >> $dir/read.out 2>&1; }
for i in $(seq 1 2880); do [ -e $Q/207_d174_ab40.done ] && break; sleep 30; done
read1 scenes; read1 gal
for i in $(seq 1 2880); do [ -e $Q/208_d174_ab300.done ] && break; sleep 30; done
read1 scales
log "A/B readers done"
