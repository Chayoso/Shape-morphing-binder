#!/bin/bash
# d159c_after.sh -- starts d159c.sh once D159s own card users are done (chain 2247716 = X renders, waiter 3280248 =
# close-ups), so D159 never holds more than one card. Launched 2026-10-22 after reading D160cs after-arrival timeline.
S=/data/relcfd/chayo/physmorph_v2/output/gpu/d159/status.txt
echo "$(date "+%m-%d %H:%M:%S") D159c armed (pid $$): waits for 2247716 and 3280248" >> $S
for p in 2247716 3280248; do while ps -p $p > /dev/null 2>&1; do sleep 120; done; done
exec bash /data/relcfd/chayo/physmorph_v2/tmp/d159/d159c.sh
