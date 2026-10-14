#!/bin/bash
# relaunch.sh N -- wait until no D145 pool worker runs (ps, read-only), clear STOP, start N workers on pool.sh
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/d145; G=$B/output/gpu
while [ "$(ps -eo args | grep -c '[b]ash pool.sh')" != 0 ]; do sleep 30; done
rm -f $T/STOP
echo "$(date '+%m-%d %H:%M') relaunch: $1 workers on the current pool.sh / main.queue" >> $G/d145_pool.log
cd $T
for w in $(seq 1 $1); do setsid nohup bash pool.sh $T/main.queue r$w > $G/d145_pool_r$w.out 2>&1 < /dev/null & sleep 15; done
