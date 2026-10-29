#!/bin/bash
# after_tex300t.sh -- D173: Bob at 300k read with T, Rx, Rxn (job 218) and the Rx rerun with the block records (job 222), card
# ${READ_CARD:-0} -> auto/tex300t, then the summary.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; G=$B/output/gpu; Q=$B/tmp/gpu1_queue; D=$O/auto/tex300t; mkdir -p $D
log() { echo "$(date "+%m-%d %H:%M:%S") [after tex300t] $*" >> $O/status.txt; }
for i in $(seq 1 4320); do [ -e $Q/222_d173_tex300t.done ] && [ -e $Q/218_d173_tex300n.done ] && break; sleep 30; done
export PHYSMORPH_RUN_REPO=$B/repo_d173 PHYSMORPH_OUT=$G; source $B/repo_d173/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=${READ_CARD:-0} D157_THREADS=16; cd $B/repo_d173
echo "Rx 97: $($PY $B/tmp/d145/det_cmp12.py $O/runs/tex300k_Rx_97 $O/runs/tex300k_Rxt_97 2>&1 | tail -1)" > $D/rerun_cmp.txt
$PY $W/d173_texread.py $D 300000 97 T,Rx,Rxn,Rxt > $D/texread.log 2>&1
log "read: $(grep VERDICT $D/summary.md | cut -c1-300)"
$PY $W/d173_summary.py $G/D173_SUMMARY.md
