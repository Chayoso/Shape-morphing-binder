#!/bin/bash
# raw_cpu.sh -- D158 follow-up for the dragon on CPU (no card after 17:00): waits for the dragon fit, then d158_raw.py
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d158; G=$B/output/gpu; S=$G/d158/status.txt
until grep -q "dragon300k_L97 rc" $S; do sleep 30; done
export PHYSMORPH_RUN_REPO=$B/repo_freeze1015 PHYSMORPH_OUT=$W/tmpdir/pm_out
source $B/repo_freeze1015/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir D158_REPO=$B/repo_freeze1015 CUDA_VISIBLE_DEVICES="" D158_DEV=cpu OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
last=$(ls $G/d158/dragon300k_L97/dump_*.npz | sed "s/.*dump_0*//; s/.npz//" | sort -n | tail -1)
echo "raw_cpu dragon 80,$last start $(date +%H:%M:%S)" >> $S
m=dragon
nice -n 10 $PY $W/d158_raw.py $G/d158 $m $G/d111/${m}_dense15.npz,$G/d111/${m}_dense15_s9.npz,$G/d111/${m}_dense15_s10.npz,$G/d111/${m}_dense15_s11.npz 80,$last > $G/d158/raw_$m.log 2>&1 < /dev/null
echo "raw_dragon (cpu) rc $? $(date +%H:%M:%S)" >> $S
