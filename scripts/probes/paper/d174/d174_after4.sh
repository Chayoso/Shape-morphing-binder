#!/bin/bash
# d174_after2.sh -- agent D174's detached CPU reader of the stop A/B (tol 0.10 / 0.05 against Hu): waits for each group's queue
# jobs (tmp/gpu1_queue/2NN_d174_st_*.done), reads it with tmp/d174/d174_read2.py, rewrites output/gpu/d174/STOP_SUMMARY.md.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d174; O=$B/output/gpu/d174; Q=$B/tmp/gpu1_queue
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=8 D157_THREADS=8 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d166:$B/deps/d157_lpips D170_WAIT_H=1
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%F %T') [after2] $*" >> $O/status.txt; }
cd $B/repo_d166
waitq() { for j in "$@"; do for i in $(seq 1 2880); do [ -e $Q/$j.done ] && break; sleep 30; done; done; }
read1() { local mode=$1 dir=$O/auto/st_$1; mkdir -p $dir; local s=$(date +%s)
  nice -n 10 $PY $W/d174_read2.py $mode $dir > $dir/read.out 2>&1; local rc=$?
  log "read $mode exit $rc ($(( $(date +%s) - s )) s): $(grep VERDICT $dir/summary.md 2>/dev/null | cut -c1-300)"
  nice -n 10 $PY $W/d174_read2.py aggregate $O/STOP_SUMMARY.md >> $dir/read.out 2>&1; }
waitq 202_d174_st_big10 204_d174_st_big07 207_d174_st_big05; read1 scales; read1 plq
waitq 201_d174_st_dragon10 203_d174_st_bunny10 202_d174_st_dragon05 208_d174_st_bunny05 203_d174_st_dragon07; read1 big
waitq 204_d174_st_40k10 209_d174_st_40k05; read1 scenes; read1 gal
log "stop A/B readers done"
