#!/bin/bash
# chain_cpu.sh -- D155's gate, the CPU readers (no card, nice): for each run as soon as its card readers are done:
# d120_apart.py over every kept frame (apart_NAME.log) and visible_apart.py over windows 3-10 (vis_w3_10_NAME.jsonl);
# at the end d155_gate_read.py over every run (gate.jsonl).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp; G=$B/output/gpu; W=$T/d155; O=$G/d155/gate
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
S=$O/status.txt
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
MESHES="bunny dragon teapot armadilo homer"
names=""
for m in $MESHES; do names="$names ${m}40k_A3L97 ${m}40k_A3P97 ${m}40k_V97 ${m}40k_W97 ${m}40k_A1L97 ${m}40k_A1P97"; done
for s in 99 101; do for m in $MESHES; do names="$names ${m}40k_V$s ${m}40k_W$s"; done; done
for n in $names; do
  until grep -q "^$n card readers done" $S 2>/dev/null || { [ -e $O/rc_$n ] && [ "$(cat $O/rc_$n)" != 0 ]; }; do sleep 30; done
  [ "$(cat $O/rc_$n)" = 0 ] || { echo "cpu: $n failed, skipped" >> $S; continue; }
  F=$O/${n}_frames12.npz
  (
    source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null
    export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=""
    nice -n 10 $PY $T/d120_apart.py $F 1.0 > $O/apart_$n.log 2>&1 < /dev/null
  )
  (
    export PHYSMORPH_RUN_REPO=$B/repo_d145
    source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
    export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=""
    cd $B/repo_d145
    nice -n 10 $PY scripts/probes/paper/visible_apart.py $F --label $n --jsonl $O/vis_w3_10_$n.jsonl --threads 8 > $O/vis_$n.log 2>&1 < /dev/null
  )
  echo "cpu readers done $n $(date '+%m-%d %H:%M:%S')" >> $S
done
args=""
for n in $names; do [ -e $O/$n.json ] && args="$args $n=$O"; done
cd $W
nice -n 10 $PY d155_gate_read.py $O/gate.jsonl $args > $O/gate_read.log 2>&1
echo "cpu chain done $(date '+%m-%d %H:%M:%S')" >> $S
