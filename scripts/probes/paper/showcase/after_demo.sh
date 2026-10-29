#!/bin/bash
# after_demo.sh -- SHOWCASE: the coordinator (2026-10-21 13:12): one card while the DEMO agent holds locks. Once no lock is
# "agent DEMO" for 10 minutes (and c5 has started), start the second chain c6 on sc_chain3.sh.
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/showcase; D=$B/output/gpu/showcase; L=$B/gpu_locks
until grep -q "\[swap\] c5 started" $D/status.txt; do sleep 60; done
free=0
while [ $free -lt 10 ]; do
  if grep -q "^agent DEMO" $L/gpu*.lock/info 2>/dev/null; then free=0; else free=$((free + 1)); fi
  [ -e $D/END ] && exit 0
  sleep 60
done
setsid nohup bash $T/sc_chain3.sh c6 $T/queue.txt > $D/chain_c6.out 2>&1 < /dev/null &
echo "$(date "+%m-%d %H:%M:%S") [after_demo] no DEMO lock for 10 min: c6 started pid $! on sc_chain3.sh" >> $D/status.txt
