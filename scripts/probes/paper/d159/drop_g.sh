#!/bin/bash
# drop_g.sh -- D159: P1 failed on both meshes, so the 40k gallery (P4, only read for adoption) is not run; the card goes to
# D160 / D161 (priority). Stops the main chain by its recorded PID only once the render step X has released its card and the
# chain waits for a card for G (no D159 lock held, no newer "claimed" line); the close-up step then follows (it waits for
# the chain to exit).
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d159; S=$O/status.txt; L=/data/relcfd/chayo/physmorph_v2/gpu_locks
P=$(cat /data/relcfd/chayo/physmorph_v2/tmp/d159/chain_VRXG3.pid)
while ps -p $P > /dev/null 2>&1; do
  tail1=$(grep -A400 "chain start pid $P" $S | grep -E "claimed|released|X dragon side-by-side" | tail -1)
  if grep -A400 "chain start pid $P" $S | grep -q "X dragon side-by-side" && echo "$tail1" | grep -q "released" && ! grep -lq "pid $P" $L/*/info 2>/dev/null; then
    kill $P; echo "$(date "+%m-%d %H:%M:%S") chain $P stopped after X while waiting for a card for G: P1 failed on both meshes, the 40k gallery (P4, read for adoption only) is not run; the card is left to D160 / D161" >> $S
    break
  fi
  sleep 20
done
