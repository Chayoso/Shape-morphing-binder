#!/bin/bash
# resume_1700.sh -- SHOWCASE: the coordinator (2026-10-21 14:46): pause until 17:00 CDT, then resume on one card.
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/showcase; D=$B/output/gpu/showcase
until [ "$(date +%H%M)" -ge 1700 ]; do sleep 60; done
until ! ps -p $(cat $D/chain_c5.pid) > /dev/null 2>&1; do sleep 30; done
rm -f $D/STOP_c5 $D/video/fandisk300k_L97_main_4k.mp4 $D/video/fandisk300k_L97_DEFERRED.txt
HOLD=3 setsid nohup bash $T/sc_chain3.sh c7 $T/queue.txt > $D/chain_c7.out 2>&1 < /dev/null &
echo "$(date "+%m-%d %H:%M:%S") [resume] 17:00: c7 started pid $! on sc_chain3.sh (one card while DEMO holds locks)" >> $D/status.txt
