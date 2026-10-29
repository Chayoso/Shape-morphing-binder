#!/bin/bash
# after_strp.sh NK SEEDS QUEUE_NAME -- D173: the stripes runs read (card ${READ_CARD:-0}): colour r (D160's fast reader), the edge
# widths (d173_edge2.py), PSNR / SSIM, kappa, block shares, rest, shape (d173_stripread.py) -> auto/strp{NK}, the summary,
# then seed 97's deliverables (deliver_colour2.sh: twin | full | target paint, front and back; grey stills; a video).
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; G=$B/output/gpu; Q=$B/tmp/gpu1_queue
NK=$1; SEEDS=$2; QN=$3; D=$O/auto/strp$NK; mkdir -p $D; N=$(( NK * 1000 ))
log() { echo "$(date "+%m-%d %H:%M:%S") [after strp$NK] $*" >> $O/status.txt; }
for i in $(seq 1 4320); do [ -e $Q/$QN.done ] && break; sleep 30; done
export PHYSMORPH_RUN_REPO=$B/repo_d173 PHYSMORPH_OUT=$G; source $B/repo_d173/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=${READ_CARD:-0} D157_THREADS=16 OMP_NUM_THREADS=16; cd $B/repo_d173
rm -f $D/jobs_colour.txt $D/edge_jobs.txt $D/read_colour.jsonl $D/edge.jsonl
for s in $(echo $SEEDS | tr , " "); do T=$O/runs/strp${NK}k_T_$s; F=$O/runs/strp${NK}k_Rxn_$s
  echo "strp${NK}k_T_$s stripes ${T}_frames12.npz mesh=bunny" >> $D/jobs_colour.txt
  echo "strp${NK}k_Rxn_$s stripes ${F}_frames12.npz mesh=bunny twin=strp${NK}k_T_$s timeline" >> $D/jobs_colour.txt
  printf "%s\n" "strp${NK}k_Rxn_$s@tgt ${F}_frames12.npz" "strp${NK}k_T_$s ${T}_frames12.npz" "strp${NK}k_Rxn_$s ${F}_frames12.npz" >> $D/edge_jobs.txt
done
$PY $B/tmp/d160/d160_read_fast.py $D/read_colour.jsonl $D/jobs_colour.txt > $D/read_colour.log 2>&1; log "colour read exit $?"
$PY $W/d173_edge2.py $D/edge.jsonl $D/edge_jobs.txt > $D/edge.log 2>&1; log "edges exit $?"
$PY $W/d173_stripread.py $D $N $SEEDS Rxn > $D/stripread.log 2>&1; log "$(grep VERDICT $D/summary.md | cut -c1-400)"
$PY $W/d173_summary.py $G/D173_SUMMARY.md
bash $W/deliver_colour2.sh strp$NK $O/runs/strp${NK}k_Rxn_97 $O/runs/strp${NK}k_T_97 3 stripes >> $D/deliver.log 2>&1
$PY $W/d173_summary.py $G/D173_SUMMARY.md
log "done"
