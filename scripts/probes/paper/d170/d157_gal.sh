#!/bin/bash
# d157_gal.sh -- D170 (CPU, detached; run by post.sh after the gallery read): D157's reader (tmp/d157/d157_read.py, unchanged)
# on the Rx gallery runs (seeds 97 / 99) and d167's rule runs (seed 97), three slots, then tmp/d170/d170_d157.py against
# D157's own V / W reads -> output/gpu/d170/auto/174b_gal_d157/summary.md.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d170; O=$B/output/gpu/d170; R=$O/auto/174b_gal_d157; mkdir -p $R
export D157_THREADS=8; source $B/tmp/d157/env.sh
cd $D157
J=$R/jobs40.txt; : > $J
for m in bunny dragon armadilo bob homer teapot; do
  for s in 97 99; do
    x=$O/runs/gal_${m}_Rx_${s}_frames12.npz
    [ -e $x ] || continue
    echo "$m $s 40k own $x tgt" >> $J
    echo "$m $s 40k Rx $x end" >> $J
    [ $s = 97 ] && [ -e $B/output/gpu/d167/runs/gal_${m}_R_97_frames12.npz ] && echo "$m 97 40k R $B/output/gpu/d167/runs/gal_${m}_R_97_frames12.npz end" >> $J
  done
done
n=$(wc -l < $J)
# three chunks on mesh-seed boundaries (a mesh-seed's lines stay together)
b=$(awk '{k=$1" "$2; if (k != p) {print NR - 1; p = k}}' $J | tr '\n' ' ')
set -- $b
cuts=(0)
for i in $(seq 1 2); do t=$(( n * i / 3 )); best=0; for c in "$@"; do [ $c -le $t ] && best=$c; done; cuts+=($best); done
cuts+=($n)
for i in 0 1 2; do
  f=${cuts[$i]}; l=${cuts[$((i + 1))]}
  [ $l -gt $f ] || continue
  ( nice -n 10 $PY d157_read.py $R/read40_$i.jsonl $J $f $l > $R/read40_$i.log 2>&1; echo "rc $?" >> $R/read40_$i.log ) &
done
wait
$PY $W/d170_d157.py $R/summary.md $R/read40_*.jsonl > $R/summary.log 2>&1
echo "$(date '+%m-%d %H:%M:%S') [d157 gallery] $(tail -1 $R/summary.log | cut -c1-300)" >> $O/status.txt
