#!/bin/bash
# r5_runcpu.sh NAME MESH -- D145 reading 5 (1), the CPU readings of one cell-size run (output/gpu/d145t_cell/NAME_*), run
# as a task of r5_queue.sh (nice 10): B1's CPU readings (apart over the first 35 %, the dragon's tail pulse, visible_apart
# over windows 3-10 and over the run every fourth kept frame), the far pieces (pieces145.py), the F-score, the display
# mesh at the end and every 0.1 s (teapot / homer to 4 s, the dragon to 6 s and every 0.05 s over 1.5-3.0 s) with B11 /
# B12's readers, the extra / missing material against the own target sample, the teapot's handle plane, and the end
# through D157's reader (r5_d157x.py, context from B1's L97).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp; G=$B/output/gpu; R=$B/tmp/d145r
[ -z "$R5_NICED" ] && exec env R5_NICED=1 nice -n 10 bash "$0" "$@"      # every reading below at nice 10
name=$1; mesh=$2; D=$G/d145t_cell; X=$G/d145_ext/r5; F=$D/${name}_frames12.npz; E=$D/${name}_end1.npz
source $T/d145/env_d145.sh
(
  source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null
  export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6
  $PY $T/d120_apart.py $F 0.35 > $D/apart_$name.log 2>&1 < /dev/null
  [ "$mesh" = dragon ] && $PY $T/d116_pulse.py $F $D/$name.json 0.3 > $D/pulse_$name.log 2>&1 < /dev/null
  $PY $R/pieces145.py $X/pieces_cell.jsonl $name=$F --every 1 --threads 6 > $X/pieces_$name.log 2>&1 < /dev/null
  $PY $R/r4_fscore.py $X/fscore_cell.jsonl $mesh:cell:$name=$E > $X/fscore_$name.log 2>&1 < /dev/null
  $PY $R/r4_extra.py $X/extra_cell.jsonl $X/png $mesh:$name=$E > $X/extra_$name.log 2>&1 < /dev/null
  [ "$mesh" = teapot ] && $PY $R/r4_slice.py $X/slice_$name teapot x 0.0 -1.7 2.3 -4.0 3.8 $name=$E > $X/slice_$name.log 2>&1 < /dev/null
)
(
  export PHYSMORPH_RUN_REPO=$B/repo_t
  source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
  cd $B/repo_t
  export CUDA_VISIBLE_DEVICES=
  $PY scripts/probes/paper/visible_apart.py $F --label $name --jsonl $D/vis_w3_10_$name.jsonl --threads 6 > $D/vis_$name.log 2>&1 < /dev/null
  $PY scripts/probes/paper/visible_apart.py $F --label $name --w0 0 --w1 100000 --every 4 --jsonl $D/vis_full_$name.jsonl --threads 6 >> $D/vis_$name.log 2>&1 < /dev/null
)
bash $R/r5_timeline.sh $name $mesh $F > /dev/null 2>&1
bash $R/r5_d157x.sh $X/d157x_cell_$name.jsonl "$mesh pts $name $E" > /dev/null 2>&1
echo "done cpu $name $(date '+%m-%d %H:%M')" >> $D/status.txt
