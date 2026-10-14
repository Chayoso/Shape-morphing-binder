#!/bin/bash
# sched_d148.sh -- the coordinator's card plan for 2026-10-14 (D148 finished today): from 13:00 D145 holds at most 1 card;
# 15:00-19:00 also no 300k dragon starts (tmp/d145/no_dragon); from 19:00 back to 3 cards (B5 waiting). Logged.
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
T=/data/relcfd/chayo/physmorph_v2/tmp/d145; L=/data/relcfd/chayo/physmorph_v2/output/gpu/d145_pool.log
until [ "$(date +%H%M)" -ge 1300 ]; do sleep 60; done; echo 1 > $T/max_cards; echo "$(date "+%m-%d %H:%M") max_cards 1 (D148 priority)" >> $L
until [ "$(date +%H%M)" -ge 1500 ]; do sleep 60; done; touch $T/no_dragon; echo "$(date "+%m-%d %H:%M") no 300k dragon starts until 19:00 (D148 gate)" >> $L
until [ "$(date +%H%M)" -ge 1900 ]; do sleep 60; done; rm -f $T/no_dragon; echo 3 > $T/max_cards; echo "$(date "+%m-%d %H:%M") max_cards 3, dragons allowed again" >> $L
