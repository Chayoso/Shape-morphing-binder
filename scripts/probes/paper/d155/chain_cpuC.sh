#!/bin/bash
# chain_cpuC.sh -- D155c's CPU readers (no card, nice, one at a time): d120_apart.py and visible_apart.py for each A3C run
# as its card readers finish; then d155_gate_read.py over every gate run (gate_all.jsonl).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp; G=$B/output/gpu; W=$T/d155; O=$G/d155/gate
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
S=$O/status.txt
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
names=""
for m in bunny dragon teapot armadilo homer; do names="$names ${m}40k_A3CL97 ${m}40k_A3CP97"; done
for n in $names; do
  until grep -q "^$n card readers done" $S 2>/dev/null || { [ -e $O/rc_$n ] && [ "$(cat $O/rc_$n)" != 0 ]; } || grep -q "stopped by STOP" $S; do sleep 30; done
  [ "$(cat $O/rc_$n 2>/dev/null)" = 0 ] || { echo "cpuC: $n failed or missing, skipped" >> $S; continue; }
  F=$O/${n}_frames12.npz
  ( source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
    nice -n 15 $PY $T/d120_apart.py $F 1.0 > $O/apart_$n.log 2>&1 < /dev/null )
  ( export PHYSMORPH_RUN_REPO=$B/repo_d145; source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
    export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4; cd $B/repo_d145
    nice -n 15 $PY scripts/probes/paper/visible_apart.py $F --label $n --jsonl $O/vis_w3_10_$n.jsonl --threads 4 > $O/vis_$n.log 2>&1 < /dev/null )
  echo "cpuC readers done $n $(date '+%m-%d %H:%M:%S')" >> $S
done
args=""
for f in $O/*40k_*.json; do n=$(basename $f .json); args="$args $n=$O"; done
cd $W
nice -n 15 $PY d155_gate_read.py $O/gate_all.jsonl $args > $O/gate_read_all.log 2>&1
echo "cpuC done $(date '+%m-%d %H:%M:%S')" >> $S
