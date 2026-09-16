#!/bin/bash
source /data/relcfd/chayo/physmorph_v2/repo/scripts/ops/hyde06_env.sh
TAG=$1; shift
CUDA_VISIBLE_DEVICES=2 $PY -m cProfile -o $OUT/prof150_$TAG.pstats scripts/pipeline_run.py --arms render_full_dt_iso_nn --n 150000 --ppc 8 --loss_units density --warm_start --w_kin 5 --w_kin_var 200 --animations 6 --loss_res 64 --pace 0 --anneal 0.7 --mom_carry 0 --nn_far_k 1000 --bonds --archive_stride 4 --domain auto "$@" --out $OUT/prof150_$TAG > $OUT/prof150_$TAG.log 2>&1
$PY - <<PY
import pstats
p = pstats.Stats("/data/relcfd/chayo/physmorph_v2/output/prof150_$TAG.pstats")
p.sort_stats("cumulative").print_stats(40)
p.sort_stats("tottime").print_stats(25)
PY
echo PROFILE_DONE
