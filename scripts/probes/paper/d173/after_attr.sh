#!/bin/bash
# after_attr.sh SEED QUEUE_NAME -- D173: the attribution read for one seed (CPU; the fast detail reader), its summary, the raking
# close-up sheet (d173_close3.py: the picture target and arms 1 / 2 / 3), then D173_SUMMARY.md.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; G=$B/output/gpu; Q=$B/tmp/gpu1_queue
S=$1; QN=$2; D=$O/auto/attr$S; mkdir -p $D
log() { echo "$(date "+%m-%d %H:%M:%S") [after attr$S] $*" >> $O/status.txt; }
for i in $(seq 1 2160); do [ -e $Q/$QN.done ] && break; sleep 30; done
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=12 D157_THREADS=12 MPLCONFIGDIR=$B/cache/matplotlib PYTHONPATH=$B/repo_d173
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python; cd $B/repo_d173
SM=$G/d170/assets/d170smooth_dragon.obj; RE=$G/d172/assets/d172SE300_dragon.obj
T=$O/runs/atr300k_T_$S; [ $S = 97 ] && T=$G/d172/runs/scl300k_T_97
echo "{\"scene\": \"attr$S\", \"N\": 300000, \"tgt\": \"$SM\", \"smooth\": \"$SM\", \"detail\": \"$RE\", \"s_bg\": 3.0, \"runs\": {\"T_$S\": \"$T\", \"Hs_$S\": \"$O/runs/atr300k_Hs_$S\", \"Hm_$S\": \"$O/runs/atr300k_Hm_$S\"}}" > $D/scene.json
s=$(date +%s); $PY $B/tmp/d172/d172_detail_fast.py $D/read.json $D/scene.json > $D/read.log 2>&1; log "read exit $? ($(( $(date +%s) - s )) s)"
$PY $W/d173_attr.py $D $S > $D/attr.log 2>&1; log "$(tail -1 $D/attr.log | cut -c1-300)"
$PY $W/d173_close3.py $O/deliver/attr$S $SM 300000 $RE "(1) twin:${T}_frames12.npz" "(2) Hu, scaled pictures:$O/runs/atr300k_Hs_${S}_frames12.npz" "(3) Hu, smooth pictures:$O/runs/atr300k_Hm_${S}_frames12.npz" > $D/close.log 2>&1
log "close-up sheet exit $?"
$PY $W/d173_summary.py $G/D173_SUMMARY.md
log "done"
