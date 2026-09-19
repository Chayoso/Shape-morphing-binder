#!/bin/bash
# Surface-smoothness acid test (docs/experiments.md 2026-09-19): every candidate on the TARGET
# cloud (--still -2) and on one morph frame, stills + meshes into $OUT/surf_test/<T>/, then
# scripts/probes/surface_gt.py against the true mesh. Usage: surface_test.sh <gpu> <run> <frame> [candidates...]
#   e.g. surface_test.sh 0 h150v8_bunny 500
source /data/relcfd/chayo/physmorph_v2/repo/scripts/ops/hyde06_env.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 EGL_PLATFORM=surfaceless
GPU=$1; RUN=$2; FR=$3; shift 3
ARM=render_full_dt_iso_nn
D=$OUT/surf_test/$RUN; mkdir -p $D
LOG=$D/log.txt; : > $LOG
declare -A FLAGS=( [mc]="" [pca]="--kernel pca" [poisson]="--surface poisson" [poisson_raw]="--surface poisson --pull 0"
                   [imls]="--surface imls" [surfel]="--surface surfel" [bilateral]="--post bilateral"
                   [poisson_dl]="--surface poisson --layer density"
                   [poisson_c23]="--surface poisson --poisson_cell 0.667" [poisson_c12]="--surface poisson --poisson_cell 0.5" )
CANDS=${@:-mc pca poisson poisson_raw imls surfel bilateral}
PLYS=""
for C in $CANDS; do
  for F in -2 $FR; do
    TAG=${C}_f${F#-}; [ "$F" = "-2" ] && TAG=${C}_target
    echo "== $RUN $C frame $F $(date +%H:%M:%S)" | tee -a $LOG
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/render_photoreal.py --npz $OUT/${RUN}_${ARM}.npz \
      --out $D/$TAG.png --save_mesh $D/$TAG.ply --still $F --res 720 --label "$RUN $C" ${FLAGS[$C]} \
      2>&1 | grep -E "still|surface|Traceback|Error|saved .*png" | tee -a $LOG
    [ -f $D/$TAG.ply ] && PLYS="$PLYS $D/$TAG.ply"
  done
done
echo "== probe $(date +%H:%M:%S)" | tee -a $LOG
$PY scripts/probes/surface_gt.py $OUT/$RUN.json $PLYS 2>&1 | grep -v "^\[sampling\]" | tee -a $LOG
echo "SURFTEST $RUN DONE $(date)" >> $STATUS
