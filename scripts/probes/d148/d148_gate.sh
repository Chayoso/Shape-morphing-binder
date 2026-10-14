#!/bin/bash
# d148_gate.sh VARIANT -- D148's pre-registered 300k gate for --gap_split VARIANT (on = A, interface = B): five waiters
# through gpu_locks (agent D148, at most D148_MAX cards, default 3): dragon render arm (GA, side + front) and twin (GT,
# side), bunny render arm (BA, side + front) and twin (BT, no render), a second dragon render arm at seed 99 (GA99,
# side); each d148_run.sh (seed 97 unless given). Then d148_bands.sh (the relief) once every run has its discs.
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/d148; D=$B/output/gpu/d148
v=$1; export D148_MAX=${D148_MAX:-3}
cd $T
setsid nohup ./d148_run.sh dragon 300000 GA 1 --gap_split $v > $T/q_GA.log 2>&1 < /dev/null &
sleep 40
setsid nohup ./d148_run.sh dragon 300000 GT 2 --gap_split $v --render_weight_scale 0 > $T/q_GT.log 2>&1 < /dev/null &
sleep 40
setsid nohup ./d148_run.sh bunny 300000 BA 1 --gap_split $v > $T/q_BA.log 2>&1 < /dev/null &
sleep 40
setsid nohup ./d148_run.sh bunny 300000 BT 0 --gap_split $v --render_weight_scale 0 > $T/q_BT.log 2>&1 < /dev/null &
sleep 40
setsid nohup ./d148_run.sh dragon 300000 GA99 2 --gap_split $v --seed 99 > $T/q_GA99.log 2>&1 < /dev/null &
echo "gate queued variant $v $(date '+%m-%d %H:%M')" >> $D/status.txt
