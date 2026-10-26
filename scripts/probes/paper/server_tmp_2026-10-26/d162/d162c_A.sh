#!/bin/bash
# d162c_A.sh -- D162c (A) on hyde06's CPU, detached: the 40k runs and the 300k plaque runs as two processes at nice 10.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d162c; W=$B/tmp/d162
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 PYTHONPATH=$B/repo_d162b
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
cd $B/repo_d162b
R40="F_coin_97 F_coin_99 S_coin_97 S_coin_99 D161c_coin_97 D161c_coin_99 twin_coin_97 twin_coin_99 F_bunny_97 F_bunny_99 S_bunny_97 S_bunny_99 D161c_bunny_97 D161c_bunny_99 twin_bunny_97 twin_bunny_99 F_band_97 F_band_99 S_band_97 S_band_99 D160d_band_97 D160d_band_99 P_band_97 P_band_99"
( nice -n 10 $PY $W/d162c_motion.py $O/A.jsonl $R40 > $O/A40.log 2>&1; echo "A40 rc $? $(date '+%T')" >> $O/state.txt ) &
( nice -n 10 $PY $W/d162c_motion.py $O/A300.jsonl D163H_plaque_97 T_plaque_97 > $O/A300.log 2>&1; echo "A300 rc $? $(date '+%T')" >> $O/state.txt ) &
wait
