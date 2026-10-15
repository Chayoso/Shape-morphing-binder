#!/bin/bash
# b14_convert.sh NAME TAG RUN DATA_DIR -- D145 B14: the CPU half of one run (no card): convert_b14.py (pipeline frame,
# mesh properties, 300k sample -> archive), the end check, the volume over time by the common routine.
NAME=$1; TAG=$2; RUN=$3; DATA=$4
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d145/b14; R=$B/baselines/b14; G=$B/output/gpu
say() { echo "$(date '+%Y-%m-%d %H:%M:%S %Z') $RUN $*" >> $R/status.txt; }
export PHYSMORPH_RUN_REPO=$B/repo_d145 PHYSMORPH_OUT=$R/out_env
source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null 2>&1
GPUV=${CUDA_VISIBLE_DEVICES:-}; export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 TMPDIR=$R/cache/tmp
mkdir -p $R/out_env $G/d145_b14
say "convert start"; t0=$(date +%s)
$PY $W/convert_b14.py $NAME $TAG $R/recon/$RUN $DATA > $R/logs/${RUN}_convert.log 2>&1 || { say "FAILED convert"; exit 1; }
say "convert done $(( $(date +%s) - t0 )) s"
CUDA_VISIBLE_DEVICES=$GPUV OMP_NUM_THREADS=8 $PY $W/endcheck_b14.py $R/results/endcheck.jsonl $NAME $TAG >> $R/logs/${RUN}_convert.log 2>&1 || { say "FAILED endcheck"; exit 1; }
OMP_NUM_THREADS=8 $PY $B/tmp/d145/vol145.py $R/results/vol.jsonl $NAME $G/d90/${NAME}_ind300k.npz \
  "$TAG=$G/d145_b14/${RUN}_frames12.npz" >> $R/logs/${RUN}_convert.log 2>&1 || { say "FAILED vol145"; exit 1; }
say "endcheck + volume done"
