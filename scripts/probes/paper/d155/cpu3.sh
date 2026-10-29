#!/bin/bash
# cpu3.sh -- D155 gate part 3 (diagnostic u_off runs): the CPU readers as chain_cpu.sh, one at a time (nice), then
# d155_gate_read.py over all 54 gate runs (gate_all.jsonl).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp; G=$B/output/gpu; W=$T/d155; O=$G/d155/gate
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
S=$O/status.txt
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
for n in bunny40k_A3U97 bunny40k_U97 dragon40k_A3U97 dragon40k_U97; do
  F=$O/${n}_frames12.npz
  ( source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4
    nice -n 15 $PY $T/d120_apart.py $F 1.0 > $O/apart_$n.log 2>&1 < /dev/null )
  ( export PHYSMORPH_RUN_REPO=$B/repo_d145; source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
    export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4; cd $B/repo_d145
    nice -n 15 $PY scripts/probes/paper/visible_apart.py $F --label $n --jsonl $O/vis_w3_10_$n.jsonl --threads 4 > $O/vis_$n.log 2>&1 < /dev/null )
  echo "cpu3 readers done $n $(date "+%m-%d %H:%M:%S")" >> $S
done
args=""
for f in $O/*40k_*.json; do n=$(basename $f .json); args="$args $n=$O"; done
cd $W
nice -n 15 $PY d155_gate_read.py $O/gate_all.jsonl $args > $O/gate_read_all.log 2>&1
echo "cpu3 done $(date "+%m-%d %H:%M:%S")" >> $S
