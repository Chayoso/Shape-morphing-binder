#!/bin/bash
# d154_read_one.sh NAME -- D154 (2): the floor-free reads (d154_jobs2.py exact, 4 workers x 4 threads) and momentum of one
# run of output/gpu/d154/runs (used for the replicates, whose yardstick B1 already has); one status line at the end.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d154; O=$B/output/gpu/d154; R=$O/read
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" D154_THREADS=4 OMP_NUM_THREADS=4
P=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
name=$1; f=$O/runs/${name}_frames12.npz; mesh=${name%%300k_*}; seed=${name##*_}; label=${name#${mesh}300k_}; label=${label%_*}
rm -f $W/jobs2/${name}_exact_*.txt
$P $W/d154_jobs2.py $W/jobs2/${name}_exact 4 exact $mesh $seed $label $f > /dev/null
cd $W
for k in 0 1 2 3; do nice -n 10 $P d154_mesh_read.py $R/sw_exact_${name}_$k.jsonl $W/jobs2/${name}_exact_$k.txt > $R/sw_exact_${name}_$k.log 2>&1 < /dev/null & done
OMP_NUM_THREADS=2 nice -n 10 $P d154_momentum.py $R/sw_mom_$name.jsonl $label=$f > $R/sw_mom_$name.log 2>&1 < /dev/null &
wait
echo "$(date '+%m-%d %H:%M:%S') cpu read $name done (exact only): exact $(cat $R/sw_exact_${name}_*.jsonl | wc -l), mom $(grep -c summary $R/sw_mom_$name.jsonl)" >> $O/status.txt
