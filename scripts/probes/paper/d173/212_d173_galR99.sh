#!/bin/bash
# 212_d173_galR99.sh -- queue job (agent D173, for D172's seed-99 gallery row): the rule (the tag's defaults) rerun on repo_d166 at
# seed 99 on the six 40k meshes, the second baseline at seed 99 as d167 job 114's R 97 is at seed 97; then the row is re-read
# (tmp/d173/d173_gal99.py -> output/gpu/d172/auto/191_gal99) and D172_SUMMARY.md re-aggregated.
J=212_galR99; TITLE="D173 (for D172): the rule at seed 99 on the gallery"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
REPO=repo_d166; log "start"; PAR=4
for m in bunny dragon armadilo bob homer teapot; do runp gal_${m}_R_99 --tgt assets/$m.obj --n 40000 --seed 99; done
wait; log "runs done"
( PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python; export CUDA_VISIBLE_DEVICES= PYTHONPATH=$B/repo_d166; cd $B/repo_d166; $PY $W/d173_gal99.py $G/d172/auto/191_gal99 > $G/d172/auto/191_gal99/read.log 2>&1
  $PY $B/tmp/d172/d172_read_fast.py aggregate $G/D172_SUMMARY.md >> $G/d172/auto/191_gal99/read.log 2>&1 )
log "gal99: $(grep VERDICT $G/d172/auto/191_gal99/summary.md | cut -c1-300)"; log "done"
