#!/bin/bash
# detproof.sh -- D145: repo_d145s (the kept frames written in place of the full archive) against repo_d145 (archive, then
# tmp/frames12.py), 40k bunny seed 97, four windows, Warp run-to-run deterministic mode (tmp/rt/det_run.py): bit for bit?
B=/data/relcfd/chayo/physmorph_v2; D=$B/output/gpu/d145_detproof; mkdir -p $D
export PHYSMORPH_OUT=$B/output/gpu
source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null
cd $B/repo_d145 && PHYSMORPH_RUN_REPO=$B/repo_d145 $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 --animations 4 --out $D/detA > $D/detA.log 2>&1
$PY $B/tmp/frames12.py $D/detA_render_full_dt_iso_nn.npz $D/detA >> $D/detA.log 2>&1 && rm -f $D/detA_render_full_dt_iso_nn.npz
cd $B/repo_d145s && PHYSMORPH_RUN_REPO=$B/repo_d145s $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --tgt assets/bunny.obj --n 40000 --seed 97 --animations 4 --out $D/detB > $D/detB.log 2>&1
$PY $B/tmp/d145/det_cmp12.py $D/detA $D/detB > $D/detproof.txt 2>&1
ls $D/detB_render_full_dt_iso_nn.npz >> $D/detproof.txt 2>&1
tail -3 $D/detproof.txt
