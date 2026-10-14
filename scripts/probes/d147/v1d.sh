#!/bin/bash
# D147 V1d (diagnostic): which position update lets a landed body sink through the floor. V1 at G 0.05 (40k bunny,
# seed 97) with u off, with the minimum spacing off, without the plastic assimilation, and with all three off.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147/runs; mkdir -p $O
SP=scripts/probes/d147/specs
run() { tag=$1; shift; echo "$tag start $(date '+%F %T')" >> $O/v1d.log; $PY scripts/scene_run.py "$@" --out $O/$tag > $O/$tag.log 2>&1; echo "$tag exit $? $(date '+%F %T')" >> $O/v1d.log; }
run v1d_uoff --n 40000 --seed 97 --scene $SP/v1_g005.json --flags="--u_off"
run v1d_nospace --n 40000 --seed 97 --scene $SP/v1_g005.json --flags="--min_spacing 0"
run v1d_noassim --n 40000 --seed 97 --scene $SP/v1_g005.json --flags="--assim 0"
run v1d_none --n 40000 --seed 97 --scene $SP/v1_g005.json --flags="--u_off --min_spacing 0 --assim 0"
echo "v1d_end $(date '+%F %T')" >> $O/v1d.log
