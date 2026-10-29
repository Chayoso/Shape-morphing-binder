#!/bin/bash
# d174_after6.sh -- agent D174's detached CPU reader: the --stop_tol 0.10 arm's 300k scales and plaque (Hs10), for the trade
# against 0.12; writes output/gpu/d174/auto/ss10_scales, ss10_plq.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d174; O=$B/output/gpu/d174; Q=$B/tmp/gpu1_queue
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=8 D157_THREADS=8 MPLCONFIGDIR=$B/cache/matplotlib
export PYTHONPATH=$B/repo_d166:$B/deps/d157_lpips D170_WAIT_H=1 D174_ARM=Hs10
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%F %T') [after6] $*" >> $O/status.txt; }
cd $B/repo_d166
for i in $(seq 1 2880); do [ -e $Q/197_d174_sbig10.done ] && break; sleep 30; done
for mode in scales plq; do dir=$O/auto/ss10_$mode; mkdir -p $dir; s=$(date +%s)
  nice -n 10 $PY $W/d174_read3.py $mode $dir > $dir/read.out 2>&1
  log "read $mode exit $? ($(( $(date +%s) - s )) s): $(grep VERDICT $dir/summary.md 2>/dev/null | cut -c1-300)"; done
