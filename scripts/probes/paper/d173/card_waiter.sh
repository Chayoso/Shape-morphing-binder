#!/bin/bash
# card_waiter.sh -- D173: cards 0 and 1 are held by agent D174; when one is free (no lock, < 2000 MiB), start our queue
# runner on it (card_runner_n.sh takes the lock). Stops after 12 h or when no D173 job waits in the queue.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; Q=$B/tmp/gpu1_queue
for k in $(seq 1 144); do
  ls $Q | grep -q "^2[0-9][0-9]_d173_.*\.sh$\|^19[57]_d173_.*\.sh$" || exit 0
  for c in 0 1; do
    [ -e $W/runner_card$c.pid ] && ps -p $(cat $W/runner_card$c.pid) > /dev/null 2>&1 && continue
    [ -d $B/gpu_locks/gpu$c.lock ] && continue
    u=$(nvidia-smi -i $c --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -lt 2000 ] || continue
    setsid nohup bash $Q/card_runner_n.sh $c > /dev/null 2>&1 < /dev/null & echo $! > $W/runner_card$c.pid
    echo "$(date "+%m-%d %H:%M:%S") [waiter] started our runner on card $c (pid $!)" >> $B/output/gpu/d173/status.txt
  done
  sleep 300
done
