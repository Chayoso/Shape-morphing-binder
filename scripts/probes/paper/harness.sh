#!/bin/bash
# harness.sh OUTDIR MESH IND_NPZ LABEL=FRAMES_NPZ [...] -- D145's harness (harness.py: progress p on one grid, apart
# particles and occupied volume at p = 0.25 / 0.5 / 0.75 and the end) and then the display (surface_layer_probe, the
# base display against IND at those frames: apart discs and sets, holes) for each method. Runs on the card the pool
# gave it (CUDA_VISIBLE_DEVICES set by the caller).
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp
out=$1; mesh=$2; ind=$3; shift 3; mkdir -p $out
case $mesh in bunny) box=1900,100,2800,700;; dragon) box=1900,60,2500,560;; *) box=0,0,3840,2160;; esac
export PHYSMORPH_RUN_REPO=$B/repo_d145
source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
cd $B/repo_d145
$PY $T/d145/harness.py $out/harness_$mesh.jsonl $mesh $ind "$@" > $out/harness_$mesh.log 2>&1
unset PHYSMORPH_RUN_REPO     # the readers' own environment, as runeval3.sh and every earlier reading
source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null
cd $B/repo_r77
for spec in "$@"; do
  label=${spec%%=*}; F=${spec#*=}
  raws=$(grep "^RAWS $label " $out/harness_$mesh.log | tail -1 | cut -d' ' -f3-)
  [ -n "$raws" ] && $PY scripts/probes/settled/surface_layer_probe.py $F $out/e3h_${mesh}_$label $box measure ref=$ind target own $raws > $out/e3h_${mesh}_$label.log 2>&1 < /dev/null
done
echo "harness $mesh done $(date '+%m-%d %H:%M')" >> $out/status.txt
