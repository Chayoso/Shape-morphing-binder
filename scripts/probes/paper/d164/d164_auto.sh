#!/bin/bash
# d164_auto.sh -- detached CPU watcher (agent D164): waits for the queue job's "R300 done" (or the job's .done file), then
# runs d164_after.sh 1 (the exact-mesh readings and d164_read.py) and writes output/gpu/d164/auto/summary.md: the gates,
# the null fractions, the gallery verdict G and the 300k rows.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d164; Q=$B/tmp/gpu1_queue
mkdir -p $O/auto
until grep -q "R300 done" $O/status.txt || [ -e $Q/40_d164_r40_r300.done ]; do sleep 300; done
nice -n 10 bash $B/tmp/d164/d164_after.sh 1
{ echo "# D164 auto summary ($(date '+%F %H:%M %Z'))"; echo
  echo "## Gates (status.txt)"; grep "P2 read\|V1 suite\|V1b\|V2 \|V3 \|V4 " $O/status.txt | sed 's/^/- /'
  echo "- V verdict: V1 by amendment 2 (only the shared card's at-rest MPM test fails; MPM code byte-identical to the tag), V2 0/0/0 of 20, V3 exit 0, V4 SAME twice"
  echo; sed -n '/^## (R1)/,/^## (R3)/p' $O/d164_tables.md | sed '$d'
  echo "## Gallery verdict"; grep '^- ' $O/d164_tables.md
  echo; sed -n '/^## 300k/,$p' $O/d164_tables.md; } > $O/auto/summary.md
echo "$(date '+%m-%d %H:%M:%S') [auto] summary written: $O/auto/summary.md" >> $O/status.txt
