#!/bin/bash
# swap_c3b.sh -- SHOWCASE: when c3 (sc_chain2.sh) releases its card after its third item, stop it while it waits for
# the next card (it frees that item's claim) and start c5 on sc_chain3.sh (one card while DEMO renders, kept frames in the
# status file, camera distance).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/showcase; D=$B/output/gpu/showcase
until grep -q "\[c3\] released gpu" $D/status.txt; do sleep 30; done
touch $D/STOP_c3
until grep -q "\[c3\] stopped" $D/status.txt; do sleep 20; done
rm -f $D/STOP_c3
setsid nohup bash $T/sc_chain3.sh c5 $T/queue.txt > $D/chain_c5.out 2>&1 < /dev/null &
echo "$(date "+%m-%d %H:%M:%S") [swap] c5 started pid $! on sc_chain3.sh" >> $D/status.txt
