#!/bin/bash
# after_colour.sh -- D173: the nearest-colour runs read (after job 205): the edge widths (d173_edge.py), the band's colour r
# (D160's fast reader), the duck's PSNR / SSIM (d173_texread.py, arms T, Rx, Rxn), then the summary.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; G=$B/output/gpu; Q=$B/tmp/gpu1_queue; D=$O/auto/colour; mkdir -p $D
log() { echo "$(date "+%m-%d %H:%M:%S") [after colour] $*" >> $O/status.txt; }
for i in $(seq 1 2160); do [ -e $Q/205_d173_colour.done ] && [ -e $Q/209_d173_colour_retry.done ] && break; sleep 30; done
export PHYSMORPH_RUN_REPO=$B/repo_d173 PHYSMORPH_OUT=$G; source $B/repo_d173/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=${READ_CARD:-0} D157_THREADS=16 OMP_NUM_THREADS=16; cd $B/repo_d173
printf "%s\n" "band40n_Rxn_97 $O/runs/band40k_Rxn_97_frames12.npz" "band40n_Rxn_99 $O/runs/band40k_Rxn_99_frames12.npz" "band40_Rx_99 $G/d166/runs/band_Rx_99_frames12.npz" "band300n_Rxn_97@tgt $O/runs/band300k_Rxn_97_frames12.npz" "band300n_Rxn_97 $O/runs/band300k_Rxn_97_frames12.npz" > $D/edge_jobs.txt
$PY $W/d173_edge.py $D/edge.jsonl $D/edge_jobs.txt > $D/edge.log 2>&1
printf "%s\n" "band40k_T_97 band $G/d160/runs40/bunny40k_band_P97_frames12.npz mesh=bunny" "band40k_T_99 band $G/d160/runs40/bunny40k_band_P99_frames12.npz mesh=bunny" "band40k_Rxn_97 band $O/runs/band40k_Rxn_97_frames12.npz mesh=bunny twin=band40k_T_97 timeline" "band40k_Rxn_99 band $O/runs/band40k_Rxn_99_frames12.npz mesh=bunny twin=band40k_T_99 timeline" "band300k_T_97 band $G/d162f/runs/band300k_T_97_frames12.npz mesh=bunny" "band300k_Rxn_97 band $O/runs/band300k_Rxn_97_frames12.npz mesh=bunny twin=band300k_T_97 timeline" > $D/jobs_colour.txt
$PY $B/tmp/d160/d160_read_fast.py $D/read_colour.jsonl $D/jobs_colour.txt > $D/read_colour.log 2>&1
$PY $W/d173_texread.py $O/auto/tex40n 40000 97,99 T,Rx,Rxn > $D/texread.log 2>&1
log "reads done: $(grep VERDICT $O/auto/tex40n/summary.md | cut -c1-250)"
$PY $W/d173_summary.py $G/D173_SUMMARY.md
bash $W/deliver_colour2.sh band300n $O/runs/band300k_Rxn_97 $G/d162f/runs/band300k_T_97 3 band >> $D/deliver.log 2>&1
log "done"
