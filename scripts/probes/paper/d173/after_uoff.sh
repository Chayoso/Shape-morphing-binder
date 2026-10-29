#!/bin/bash
# after_uoff.sh -- D173: the attribution at seed 97 read again with its u-off arm (job 215: Hu, --u_off) beside the twin, Hs and Hm
# (the fast detail reader: the cached context, one more run), its summary, a four-arm close-up sheet, then D173_SUMMARY.md.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; G=$B/output/gpu; Q=$B/tmp/gpu1_queue; D=$O/auto/attr97
log() { echo "$(date "+%m-%d %H:%M:%S") [after uoff97] $*" >> $O/status.txt; }
for i in $(seq 1 2880); do [ -e $Q/215_d173_uoff97.done ] && grep -qF "[after attr97] done" $O/status.txt && break; sleep 30; done
[ -s $O/runs/atr300k_Hu0_97_frames12.npz ] || { log "no u-off run: stop"; exit 1; }
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=12 D157_THREADS=12 MPLCONFIGDIR=$B/cache/matplotlib PYTHONPATH=$B/repo_d173
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python; cd $B/repo_d173
SM=$G/d170/assets/d170smooth_dragon.obj; RE=$G/d172/assets/d172SE300_dragon.obj; T=$G/d172/runs/scl300k_T_97
cp $D/summary.md $D/summary_3arm.md; cp $D/read.json $D/read_3arm.json
echo "{\"scene\": \"attr97\", \"N\": 300000, \"tgt\": \"$SM\", \"smooth\": \"$SM\", \"detail\": \"$RE\", \"s_bg\": 3.0, \"runs\": {\"T_97\": \"$T\", \"Hs_97\": \"$O/runs/atr300k_Hs_97\", \"Hm_97\": \"$O/runs/atr300k_Hm_97\", \"Hu0_97\": \"$O/runs/atr300k_Hu0_97\"}}" > $D/scene.json
s=$(date +%s); $PY $B/tmp/d172/d172_detail_fast.py $D/read.json $D/scene.json > $D/read_uoff.log 2>&1; log "read exit $? ($(( $(date +%s) - s )) s)"
$PY $W/d173_attr.py $D 97 > $D/attr_uoff.log 2>&1; log "$(tail -1 $D/attr_uoff.log | cut -c1-300)"
$PY $W/d173_close3.py $O/deliver/attr97_uoff $SM 300000 $RE "(1) twin:${T}_frames12.npz" "(2) Hu, scaled pictures:$O/runs/atr300k_Hs_97_frames12.npz" "(3) Hu, smooth pictures:$O/runs/atr300k_Hm_97_frames12.npz" "(4) Hu, u off:$O/runs/atr300k_Hu0_97_frames12.npz" > $D/close_uoff.log 2>&1
log "close-up sheet (4 arms) exit $?"
$PY $W/d173_summary.py $G/D173_SUMMARY.md
log "done"
