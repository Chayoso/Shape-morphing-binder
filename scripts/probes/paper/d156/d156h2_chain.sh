#!/bin/bash
# d156h2_chain.sh -- D156h2: wait until one of the D156g processes (output/gpu/d156g/sfsg.pids) has ended, then run
# d156h_floor.py on the bunny and the dragon at the five run seeds, one process at nice 10 (so at most three of the
# agent's CPU processes run at once).
B=/data/relcfd/chayo/physmorph_v2
source $B/tmp/d156/env.sh
P=$(awk "{print \$1}" $B/output/gpu/d156g/sfsg.pids)
while true; do
  for p in $P; do kill -0 $p 2>/dev/null || break 2; done
  sleep 60
done
O=$B/output/gpu/d156h
echo "start $(date "+%F %T")" >> $O/d156h2.state
cd $B/repo_d156
for m in bunny dragon; do
  nice -n 10 $PY $B/tmp/d156/d156h_floor.py $O $B/output/gpu/d145t_b1 $m 97 99 101 103 105 >> $O/d156h2_$m.log 2>&1 < /dev/null
  echo "$m rc $? $(date "+%F %T")" >> $O/d156h2.state
done
