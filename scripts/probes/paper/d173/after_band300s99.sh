#!/bin/bash
# after_band300s99.sh -- D173: the band at 300k, seed 99 (job 223) read: colour r (D160's fast reader), the edge widths at the
# end and at the shorter run's stop (d173_edge2.py raw=), the block shares (d173_shares.py) -> auto/band300s99, the summary.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; G=$B/output/gpu; Q=$B/tmp/gpu1_queue; D=$O/auto/band300s99; mkdir -p $D
log() { echo "$(date "+%m-%d %H:%M:%S") [after band300s99] $*" >> $O/status.txt; }
for i in $(seq 1 5760); do [ -e $Q/223_d173_band300s99.done ] && break; sleep 30; done
export PHYSMORPH_RUN_REPO=$B/repo_d173 PHYSMORPH_OUT=$G; source $B/repo_d173/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=${READ_CARD:-0} D157_THREADS=16 OMP_NUM_THREADS=16; cd $B/repo_d173
rm -f $D/read_colour.jsonl $D/edge.jsonl
printf "%s\n" "band300k_T_99 band $O/runs/band300k_T_99_frames12.npz mesh=bunny" \
  "band300k_Rx_99 band $O/runs/band300k_Rx_99_frames12.npz mesh=bunny twin=band300k_T_99 timeline" \
  "band300k_Rxn_99 band $O/runs/band300k_Rxn_99_frames12.npz mesh=bunny twin=band300k_T_99 timeline" > $D/jobs_colour.txt
$PY $B/tmp/d160/d160_read_fast.py $D/read_colour.jsonl $D/jobs_colour.txt > $D/read_colour.log 2>&1; log "colour read exit $?"
st=$($PY $W/d173_shares.py stop_min $O/runs/band300k_Rx_99.json $O/runs/band300k_Rxn_99.json)
printf "%s\n" "band300k_Rx_99@$st $O/runs/band300k_Rx_99_frames12.npz raw=$st" "band300k_Rxn_99@$st $O/runs/band300k_Rxn_99_frames12.npz raw=$st" \
  "band300k_Rx_99 $O/runs/band300k_Rx_99_frames12.npz" "band300k_Rxn_99 $O/runs/band300k_Rxn_99_frames12.npz" > $D/edge_jobs.txt
$PY $W/d173_edge2.py $D/edge.jsonl $D/edge_jobs.txt > $D/edge.log 2>&1; log "edges exit $?"
$PY $W/d173_band300sum.py $D $st > $D/summary.log 2>&1; log "$(grep VERDICT $D/summary.md | cut -c1-300)"
$PY $W/d173_summary.py $G/D173_SUMMARY.md
log "done"
