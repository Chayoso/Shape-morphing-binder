#!/bin/bash
# swap_c8.sh -- SHOWCASE (the user via the coordinator, 23:00: one card; D159 / D160 / D161 first): when c8 (sc_chain4.sh)
# releases its card after its third item, stop it while it waits and start c10 on sc_chain5.sh with MAXC=1 (never a
# second card) and the render-agent check (no claim while a D159 / D160 / D161 chain waits for a card).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/showcase; D=$B/output/gpu/showcase
until grep -q "\[c8\] released gpu1" $D/status.txt; do sleep 30; done
touch $D/STOP_c8
until grep -q "\[c8\] stopped" $D/status.txt; do sleep 20; done
rm -f $D/STOP_c8
MAXC=1 setsid nohup bash $T/sc_chain5.sh c10 $T/queue.txt > $D/chain_c10.out 2>&1 < /dev/null &
echo "$(date "+%m-%d %H:%M:%S") [swap] c10 started pid $! on sc_chain5.sh (MAXC=1, render agents first)" >> $D/status.txt
