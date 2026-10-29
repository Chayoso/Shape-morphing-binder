#!/bin/bash
# d174_after5.sh -- agent D174's detached CPU reader of the --stop_tol 0.12 A/B (arm Hs12 against Hu): waits for each group's queue
# jobs, reads it with tmp/d174/d174_read3.py (fast readers), rewrites output/gpu/d174/STOPTOL_SUMMARY.md.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d174; O=$B/output/gpu/d174; Q=$B/tmp/gpu1_queue
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=8 D157_THREADS=8 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d166:$B/deps/d157_lpips D170_WAIT_H=1 D174_ARM=Hs12
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%F %T') [after5] $*" >> $O/status.txt; }
cd $B/repo_d166
waitq() { for j in "$@"; do for i in $(seq 1 2880); do [ -e $Q/$j.done ] && break; sleep 30; done; done; }
read1() { local mode=$1 dir=$O/auto/ss_$1; mkdir -p $dir; local s=$(date +%s)
  nice -n 10 $PY $W/d174_read3.py $mode $dir > $dir/read.out 2>&1; local rc=$?
  log "read $mode exit $rc ($(( $(date +%s) - s )) s): $(grep VERDICT $dir/summary.md 2>/dev/null | cut -c1-300)"
  nice -n 10 $PY $W/d174_read3.py aggregate $O/STOPTOL_SUMMARY.md >> $dir/read.out 2>&1; }
waitq 196_d174_sbig12; read1 scales; read1 plq
waitq 194_d174_sd12 196_d174_sb12; read1 big
waitq 196_d174_s40k12; read1 scenes; read1 gal
log "stop_tol A/B readers done"
