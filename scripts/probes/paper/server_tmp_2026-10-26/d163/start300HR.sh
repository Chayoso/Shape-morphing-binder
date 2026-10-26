#!/bin/bash
# start300HR.sh -- once the 300k twin has exited, the 300k list H, R on the card D163 holds (gpu0): the lock re-made if
# the twin's chain has released it (the card never left D163's use: the twin's run, the 40k lanes)
B=/data/relcfd/chayo/physmorph_v2; S=$B/output/gpu/d163/status.txt; L=$B/gpu_locks
until grep -q "R plq300k_T_97 exit" $S; do sleep 30; done
if [ ! -d $L/gpu0.lock ] && mkdir $L/gpu0.lock 2>/dev/null; then
  printf "agent D163\nrun d163/d163_runs300_HR.txt (1 lanes)\npid %s\nstart %s\n" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu0.lock/info
  echo "$(date '+%m-%d %H:%M:%S') re-made the lock of gpu0 (start300HR)" >> $S
fi
sleep 5
[ -s $L/gpu0.lock/info ] || printf "agent D163\nrun d163/d163_runs300_HR.txt (1 lanes)\npid %s\n" "$$" > $L/gpu0.lock/info
cd $B/tmp/d163
USE_CARD=0 OUTD=$B/output/gpu/d163/runs300 ./d163_chain2.sh $B/tmp/d163/d163_runs300_HR.txt 1
