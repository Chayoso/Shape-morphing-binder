#!/bin/bash
# swap_c7.sh -- SHOWCASE: when c7 (sc_chain3.sh, 10-min idle rule) releases its card after three items, stop it while it
# waits (its claim is freed) and start c9 on sc_chain4.sh (15-min idle rule; the coordinator, 17:10).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/showcase; D=$B/output/gpu/showcase
until grep -q "\[c7\] released gpu" $D/status.txt; do sleep 30; done
touch $D/STOP_c7
until grep -q "\[c7\] stopped" $D/status.txt; do sleep 20; done
rm -f $D/STOP_c7
setsid nohup bash $T/sc_chain4.sh c9 $T/queue.txt > $D/chain_c9.out 2>&1 < /dev/null &
echo "$(date "+%m-%d %H:%M:%S") [swap] c9 started pid $! on sc_chain4.sh" >> $D/status.txt
