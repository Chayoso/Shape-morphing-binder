#!/bin/bash
# b12_read.sh METHOD MESH ARM SEED -- B12's CPU readout of one run (no card): b12_frames.py (topology, watertight,
# self-intersection, volume per frame; the 300k samples) in B11's venv (b12_cpu.sh), then B11's end check (b11_endcheck.py,
# unchanged) and D145's volume routine (tmp/d145/vol145.py, unchanged). At most four readouts at once (a slot lock).
B12=/data/relcfd/chayo/physmorph_v2/baselines/b12
A=/data/relcfd/chayo/physmorph_v2/baselines/arap
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d145_b12
method=$1; mesh=$2; arm=$3; seed=$4
tag=${method}_${mesh}_${arm}_s${seed}
label=$(echo $method | tr a-z A-Z)$arm$seed
run=$B12/results/$method/$mesh"_"$arm"_s"$seed
mkdir -p $O $B12/logs/slots
slot=""
while [ -z "$slot" ]; do
  for s in 1 2 3 4; do mkdir $B12/logs/slots/$s 2>/dev/null && { slot=$s; break; }; done
  [ -z "$slot" ] && sleep 20
done
trap 'rmdir $B12/logs/slots/$slot' EXIT
ST=$B12/logs/status.txt
echo "$(date '+%F %T %Z') read start $tag" >> $ST
B12_POOL=6 $B12/scripts/b12_cpu.sh $B12/scripts/b12_frames.py $run $mesh $label $O > $B12/logs/frames_$tag.log 2>&1 || { echo "$(date '+%F %T %Z') read FAILED frames $tag" >> $ST; exit 1; }
npz=$O/${mesh}300k_${label}_frames12.npz
$B12/scripts/b12_cpu.sh b11_endcheck.py $O/endcheck.jsonl ${mesh}300k_$label=$npz,$mesh > $B12/logs/endcheck_$tag.log 2>&1
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
OMP_NUM_THREADS=8 $A/venv/bin/python /data/relcfd/chayo/physmorph_v2/tmp/d145/vol145.py $O/vol.jsonl $mesh \
    /data/relcfd/chayo/physmorph_v2/output/gpu/d90/${mesh}_ind300k.npz $label=$npz > $B12/logs/vol_$tag.log 2>&1
echo "$(date '+%F %T %Z') read done $tag" >> $ST
