#!/bin/bash
# One mechanism trial: the 40k recipe plus the mechanism's flags on one target.
# Usage: run_trial.sh <gpu> <name> <target> [flags...]   -> $OUT/<name>_<target>, marker "RT <name>_<target> DONE"
#   e.g. run_trial.sh 2 eje2 dragon --grad_h1
source /data/relcfd/chayo/physmorph_v2/repo/scripts/ops/hyde06_env.sh
GPU=$1; NAME=$2; T=$3; shift 3
N=${N:-40000}
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/pipeline_run.py --arms render_full_dt_iso_nn --tgt assets/$T.obj \
  --n $N $RECIPE "$@" --out $OUT/${NAME}_$T > $OUT/${NAME}_$T.log 2>&1
echo "RT ${NAME}_$T DONE $(date)" >> $STATUS
