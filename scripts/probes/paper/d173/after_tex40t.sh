#!/bin/bash
# after_tex40t.sh -- D173: Bob at 40k with the block records (job 219): the reruns compared with the first runs bit for bit
# (det_cmp12: GPU scatter order may differ; a record), then d173_texread.py with the arms T, Rx, Rxn, Rxt, Rxnt (card
# ${READ_CARD:-0}) -> auto/tex40t, then the summary.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; G=$B/output/gpu; Q=$B/tmp/gpu1_queue; D=$O/auto/tex40t; mkdir -p $D
log() { echo "$(date "+%m-%d %H:%M:%S") [after tex40t] $*" >> $O/status.txt; }
for i in $(seq 1 2880); do [ -e $Q/219_d173_tex40t.done ] && break; sleep 30; done
export PHYSMORPH_RUN_REPO=$B/repo_d173 PHYSMORPH_OUT=$G; source $B/repo_d173/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=${READ_CARD:-0} D157_THREADS=16; cd $B/repo_d173
for s in 97 99; do for a in Rx Rxn; do echo "$a $s: $($PY $B/tmp/d145/det_cmp12.py $O/runs/tex40k_${a}_$s $O/runs/tex40k_${a}t_$s 2>&1 | tail -1)"; done; done > $D/rerun_cmp.txt
log "reruns vs the first runs: $(tr "\n" ";" < $D/rerun_cmp.txt | cut -c1-300)"
$PY $W/d173_texread.py $D 40000 97,99 T,Rx,Rxn,Rxt,Rxnt > $D/texread.log 2>&1
log "read: $(grep VERDICT $D/summary.md | cut -c1-300)"
$PY $W/d173_summary.py $G/D173_SUMMARY.md
