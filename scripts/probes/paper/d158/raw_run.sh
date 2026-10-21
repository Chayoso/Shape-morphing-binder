#!/bin/bash
# raw_run.sh MESH KEPT -- D158 follow-up on D158 card (GPU 0, held by d158_run.sh lock)
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d158; G=$B/output/gpu
export PHYSMORPH_RUN_REPO=$B/repo_freeze1015 PHYSMORPH_OUT=$W/tmpdir/pm_out
source $B/repo_freeze1015/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir D158_REPO=$B/repo_freeze1015 CUDA_VISIBLE_DEVICES=0
m=$1
$PY $W/d158_raw.py $G/d158 $m $G/d111/${m}_dense15.npz,$G/d111/${m}_dense15_s9.npz,$G/d111/${m}_dense15_s10.npz,$G/d111/${m}_dense15_s11.npz $2 > $G/d158/raw_$m.log 2>&1 < /dev/null
echo "raw_$m rc $? $(date +%H:%M:%S)" >> $G/d158/status.txt
