#!/bin/bash
# Clean wall-clock of the 150k batch recipe (no cProfile), 6 windows, per-window timing
# breakdown (PHYSMORPH_TIMING=1). Usage: timed150.sh <tag> <gpu> [extra flags]
source /data/relcfd/chayo/physmorph_v2/repo/scripts/ops/hyde06_env.sh
export PHYSMORPH_TIMING=1
TAG=$1; GPU=$2; shift; shift
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/pipeline_run.py --arms render_full_dt_iso_nn --n 150000 --ppc 8 --loss_units density --warm_start --w_kin 5 --w_kin_var 200 --animations 6 --loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000 --bonds --archive_stride 8 --domain auto "$@" --out $OUT/timed150_$TAG > $OUT/timed150_$TAG.log 2>&1
grep "\[time\]\|min)" $OUT/timed150_$TAG.log
echo TIMED_DONE
