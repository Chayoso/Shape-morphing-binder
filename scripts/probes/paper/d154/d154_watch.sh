#!/bin/bash
# d154_watch.sh -- D154 (2): reads each run of the chain on the CPU as it lands (status.txt "R NAME exit 0"): the
# yardstick on its kept frames (d154_jobs2.py yard, 3 workers), the floor-free measures every 0.5 s (exact, 2 workers), one run at a time,
# momentum (d154_momentum.py); one marker per run (read/NAME.queued); polls every 2 min until the chain is done.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d154; O=$B/output/gpu/d154; R=$O/read; mkdir -p $R $W/jobs2
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" D154_THREADS=2 OMP_NUM_THREADS=2   # up to 3 runs at a time, 12 workers x 2 threads each (a frame barely speeds up with threads)
P=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
S=$O/status.txt
read_run() {   # NAME FRAMES12 MESH SEED LABEL
  local name=$1 f=$2 mesh=$3 seed=$4 label=$5
  rm -f $W/jobs2/${name}_yard_*.txt $W/jobs2/${name}_exact_*.txt
  case $label in L|P) ;; *) $P $W/d154_jobs2.py $W/jobs2/${name}_yard 8 yard $mesh $seed $label $f > /dev/null;; esac   # replicates: exact only
  $P $W/d154_jobs2.py $W/jobs2/${name}_exact 4 exact $mesh $seed $label $f > /dev/null
  cd $W
  for k in 0 1 2 3 4 5 6 7; do [ -e $W/jobs2/${name}_yard_$k.txt ] && nice -n 10 $P d154_mesh_read.py $R/sw_yard_${name}_$k.jsonl $W/jobs2/${name}_yard_$k.txt yard > $R/sw_yard_${name}_$k.log 2>&1 < /dev/null & done
  for k in 0 1 2 3; do D154_THREADS=4 OMP_NUM_THREADS=4 nice -n 10 $P d154_mesh_read.py $R/sw_exact_${name}_$k.jsonl $W/jobs2/${name}_exact_$k.txt > $R/sw_exact_${name}_$k.log 2>&1 < /dev/null & done
  OMP_NUM_THREADS=2 nice -n 10 $P d154_momentum.py $R/sw_mom_$name.jsonl $label=$f > $R/sw_mom_$name.log 2>&1 < /dev/null &
  wait
  echo "$(date '+%m-%d %H:%M:%S') cpu read $name done: yard $(cat $R/sw_yard_${name}_*.jsonl | wc -l), exact $(cat $R/sw_exact_${name}_*.jsonl | wc -l), mom $(grep -c summary $R/sw_mom_$name.jsonl)" >> $S
}
echo "$(date '+%m-%d %H:%M:%S') watcher start pid $$" >> $S
start=$(wc -l < $S)                                          # the chain's end is looked for after this line only
while true; do
  for name in $(grep -o "R [a-z]*300k_[A-Za-z0-9]*_[0-9]* exit 0" $S | awk '{print $2}'); do
    [ -e $R/$name.queued ] && continue
    touch $R/$name.queued
    mesh=${name%%300k_*}; seed=${name##*_}; label=${name#${mesh}300k_}; label=${label%_*}
    echo "$(date '+%m-%d %H:%M:%S') cpu read $name queued" >> $S
    while [ $(jobs -rp | wc -l) -ge 3 ]; do sleep 30; done           # up to three runs read at once
    ( read_run $name $O/runs/${name}_frames12.npz $mesh $seed $label ) &
  done
  if tail -n +$((start + 1)) $S | grep -Eq "chain done|STOP: "; then
    pending=$(grep -o "R [a-z]*300k_[A-Za-z0-9]*_[0-9]* exit 0" $S | awk '{print $2}' | while read n; do [ -e $R/$n.queued ] || echo $n; done | wc -l)
    [ "$pending" = 0 ] && break
  fi
  sleep 120
done
wait
echo "$(date '+%m-%d %H:%M:%S') watcher done" >> $S
