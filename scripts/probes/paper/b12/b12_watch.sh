#!/bin/bash
# b12_watch.sh -- D145 B12: once the bunny ISD smoke run has passed (converted, end-checked), insert the rest of seed 1,
# the radial arm and the bunny/dragon readings after it in the pool's queue, and append seeds 2-3 at the end; on a
# failure stop and log (nothing else is queued).
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d145/b12; R=$B/baselines/b12; Q=$B/tmp/d145/main.queue; PL=$B/output/gpu/d145_pool.log
while true; do
  if grep -q "bunny300k_ISDn1 endcheck + volume done" $R/status.txt 2>/dev/null; then
    awk -v f=$W/b12_rest.lines '{print} /^b12_bunny_ISDn1 / {while ((getline l < f) > 0) print l}' $Q > $Q.b12 && mv $Q.b12 $Q
    cat $W/b12_seeds23.lines >> $Q
    echo "$(date '+%m-%d %H:%M') queue: B12 smoke (bunny ISD seed 1) passed; seed 1 on the six meshes, the radial arm and the bunny/dragon readings inserted after it; seeds 2-3 appended at the end" >> $PL
    exit 0
  fi
  if grep -q "bunny300k_ISDn1 FAILED" $R/status.txt 2>/dev/null; then
    echo "$(date '+%m-%d %H:%M') B12 smoke (bunny ISD seed 1) FAILED: nothing else of B12 queued" >> $PL
    exit 1
  fi
  sleep 120
done
