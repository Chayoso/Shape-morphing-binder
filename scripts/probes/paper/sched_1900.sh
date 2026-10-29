#!/bin/bash
# sched_1900.sh -- replaces sched_d148.sh (2026-10-14 ~16:00, the coordinator): at 19:00 only the no-dragon flag is lifted;
# max_cards stays 1 (D148 2 cards, D147 1, D145 the rest, at least 1; idle cards through the pool's 10-min rule). Logged.
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
T=/data/relcfd/chayo/physmorph_v2/tmp/d145; L=/data/relcfd/chayo/physmorph_v2/output/gpu/d145_pool.log
until [ "$(date +%H%M)" -ge 1900 ]; do sleep 60; done; rm -f $T/no_dragon; echo 1 > $T/max_cards; echo "$(date "+%m-%d %H:%M") 300k dragons allowed again; max_cards 1 + idle cards (10-min rule)" >> $L
