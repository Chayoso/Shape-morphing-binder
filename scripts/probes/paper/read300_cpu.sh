#!/bin/bash
# read300_cpu.sh DIR NAME MESH KEEP -- D145: the CPU readings of a run (no card): apart particles over the first 35 % of
# the kept frames (d120_apart.py; a 40k run: 0.35 as in d136.sh), the tail pulse (d116_pulse.py, 300k bunny and
# dragon); then the kept frames removed unless KEEP = 1 (logged in cleanup.log); "done NAME" in the status file.
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp; G=$B/output/gpu
dir=$1; name=$2; mesh=$3; keep=$4
D=$G/$dir; S=$D/status.txt; F=$D/${name}_frames12.npz
source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES="" $PY $T/d120_apart.py $F 0.35 > $D/apart_$name.log 2>&1 < /dev/null
case $name in bunny300k_*|dragon300k_*) CUDA_VISIBLE_DEVICES="" $PY $T/d116_pulse.py $F $D/$name.json 0.3 > $D/pulse_$name.log 2>&1 < /dev/null;; esac
if [ "$keep" = 0 ]; then
  sz=$(du -m $F | cut -f1); rm -f $F
  echo "$(date '+%Y-%m-%d %H:%M') $dir: removed $F ($sz MB; D145 run $name, read: its logs in $dir)" >> $G/cleanup.log
fi
echo "done $name $(date '+%m-%d %H:%M')" >> $S
