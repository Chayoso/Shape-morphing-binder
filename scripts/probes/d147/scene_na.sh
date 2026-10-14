#!/bin/bash
# D147 scene_na (diagnostic): the three 40k timelines without the plastic assimilation (--assim 0, M1's arm that still
# morphed): does the pair bounce apart when the impact's elastic strain is not made plastic at every commit?
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147/runs; mkdir -p $O
SP=scripts/probes/d147/specs
run() { tag=$1; shift; echo "$tag start $(date '+%F %T')" >> $O/scene_na.log; $PY scripts/scene_run.py "$@" --out $O/$tag > $O/$tag.log 2>&1; echo "$tag exit $? $(date '+%F %T')" >> $O/scene_na.log; }
run scene40k_L_na --n 40000 --seed 97 --flags="--assim 0"
run scene40k_LB_na --n 40000 --seed 97 --scene $SP/scene_long.json --flags="--assim 0"
run scene40k_LS_na --n 40000 --seed 97 --young 9.8e5 --flags="--assim 0"
echo "scene_na_end $(date '+%F %T')" >> $O/scene_na.log
