#!/bin/bash
# posttag_watch.sh -- D145 post-tag: once detproof_tag's verdict reads SAME for both copies, put the evidence queue
# (main_posttag.queue) into the pool's queue ahead of B14's seeds 2-3 (the external baselines' extra seeds wait behind
# it), and set the post-tag card cap; else log and stop.
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/d145; P=$T/posttag; Q=$T/main.queue
V=$B/output/gpu/d145t_detproof/verdict.txt; L=$B/output/gpu/d145_pool.log
while [ ! -s $V ]; do sleep 120; done
if [ "$(grep -o "SAME" $V | wc -l)" = 2 ]; then
  if grep -q "^# --- B14 seeds 2-3" $Q; then
    awk -v f=$P/main_posttag.queue '/^# --- B14 seeds 2-3/ && !d {while ((getline l < f) > 0) print l; d=1} {print}' $Q > $Q.new && mv $Q.new $Q
  else cat $P/main_posttag.queue >> $Q; fi
  echo ${POSTTAG_CARDS:-2} > $T/max_cards
  echo "$(date '+%m-%d %H:%M') post-tag: determinism gate passed ($(cat $V)); main_posttag.queue put in ($(grep -vc '^#' $P/main_posttag.queue) items, ahead of B14 seeds 2-3); max_cards ${POSTTAG_CARDS:-2}" >> $L
else
  echo "$(date '+%m-%d %H:%M') post-tag: determinism gate FAILED ($(cat $V)); nothing put in" >> $L
fi
