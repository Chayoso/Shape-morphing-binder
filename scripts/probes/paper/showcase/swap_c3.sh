#!/bin/bash
# swap_c3.sh -- SHOWCASE: once c3 (sc_chain2.sh) starts its third item, ask it to stop after that item (it releases its
# card at the item end anyway, HOLD 3) and then start c5 on sc_chain3.sh (the one-card rule while DEMO renders, the kept
# frames in the status file).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/showcase; D=$B/output/gpu/showcase
until grep -q "\[c3\] run maxplanck300k_L97" $D/status.txt; do sleep 60; done
touch $D/STOP_c3
until grep -q "\[c3\] stopped by STOP_c3" $D/status.txt; do sleep 30; done
rm -f $D/STOP_c3
setsid nohup bash $T/sc_chain3.sh c5 $T/queue.txt > $D/chain_c5.out 2>&1 < /dev/null &
echo "$(date "+%m-%d %H:%M:%S") [swap] c5 started pid $! on sc_chain3.sh" >> $D/status.txt
