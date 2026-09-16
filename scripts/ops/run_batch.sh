#!/bin/bash
# Production batch on one GPU: the recipe per target, live packets for the viewer, a DONE marker
# per target, post-processing after each. Usage: run_batch.sh <gpu> <n> <prefix> <targets...>
#   e.g. run_batch.sh 0 150000 h150 bunny armadilo dragon spot bob      (names h150_<T>)
#        run_batch.sh 2 40000 n40 cow homer                              (names n40_<T>)
# Extra flags via EXTRA="..." (e.g. EXTRA="--archive_stride 8" for 150k).
source /data/relcfd/chayo/physmorph_v2/repo/scripts/ops/hyde06_env.sh
GPU=$1; N=$2; PFX=$3; shift 3
UP=$(echo $PFX | tr a-z A-Z)
for T in "$@"; do
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/pipeline_run.py --arms render_full_dt_iso_nn --tgt assets/$T.obj \
    --n $N $RECIPE $EXTRA --live_dir $OUT/live --out $OUT/${PFX}_$T > $OUT/${PFX}_$T.log 2>&1
  echo "$UP $T DONE $(date)" >> $STATUS
  bash $REPO/scripts/ops/post_run.sh $PFX $T $GPU > $OUT/post_${PFX}_$T.log 2>&1
done
echo "$UP BATCH gpu$GPU DONE $(date)" >> $STATUS
