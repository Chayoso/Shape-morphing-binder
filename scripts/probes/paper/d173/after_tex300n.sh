#!/bin/bash
# after_tex300n.sh -- D173: the duck at 300k read with Rx and Rxn against the twin (d173_texread.py, card ${READ_CARD:-0}), then the summary.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; G=$B/output/gpu; Q=$B/tmp/gpu1_queue
for i in $(seq 1 2160); do [ -e $Q/218_d173_tex300n.done ] && break; sleep 30; done
export PHYSMORPH_RUN_REPO=$B/repo_d173 PHYSMORPH_OUT=$G; source $B/repo_d173/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=${READ_CARD:-0} D157_THREADS=16; cd $B/repo_d173
$PY $W/d173_texread.py $O/auto/tex300n 300000 97 T,Rx,Rxn > $O/auto/tex300n.log 2>&1
echo "$(date "+%m-%d %H:%M:%S") [after tex300n] $(grep VERDICT $O/auto/tex300n/summary.md | cut -c1-300)" >> $O/status.txt
$PY $W/d173_summary.py $G/D173_SUMMARY.md
