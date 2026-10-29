#!/bin/bash
# after_magC2.sh SEED QUEUE_NAME TWIN_PREFIX SMOOTH_ARM_QUEUE -- D173 option C read (CPU; the fast detail reader): the twin, Hu with the
# magnified-scale pictures, Hu with the smooth pictures (the stylized test's), their summary, the close-up sheet and the twin | full
# video of arm 2 (d170_render.py sbs, 4K + 1080p).
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; G=$B/output/gpu; Q=$B/tmp/gpu1_queue
S=$1; QN=$2; T=$3; QH=$4; D=$O/auto/magC$S; mkdir -p $D
log() { echo "$(date "+%m-%d %H:%M:%S") [after magC$S] $*" >> $O/status.txt; }
for i in $(seq 1 2880); do [ -e $Q/$QN.done ] && [ -e $Q/$QH.done ] && break; sleep 30; done
source $B/tmp/d146/cache_env.sh
export CUDA_VISIBLE_DEVICES= PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=12 D157_THREADS=12 MPLCONFIGDIR=$B/cache/matplotlib PYTHONPATH=$B/repo_d173
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python; cd $B/repo_d173
read KT K SB < $O/assets/magC/variant.txt; [ -n "$KT" ] || { log "no K: stop"; exit 1; }; SM=$G/d170/assets/d170smooth_dragon.obj; RE=$O/assets/magC/d173M${KT}_300_dragon.obj; sb=${SB:-$(echo "4 * $K" | bc -l)}
echo "{\"scene\": \"magC$S\", \"N\": 300000, \"tgt\": \"$SM\", \"smooth\": \"$SM\", \"detail\": \"$RE\", \"s_bg\": $sb, \"runs\": {\"T_$S\": \"$T\", \"Hs_$S\": \"$O/runs/mag300k_Hk_$S\", \"Hm_$S\": \"$O/runs/atr300k_Hm_$S\"}}" > $D/scene.json
s=$(date +%s); $PY $B/tmp/d172/d172_detail_fast.py $D/read.json $D/scene.json > $D/read.log 2>&1; log "read exit $? ($(( $(date +%s) - s )) s)"
$PY $W/d173_attr.py $D $S "(2) Hu, pictures of the dragon's scales magnified ${K}x" "the dragon's own scales magnified ${K}x (option C)" > $D/attr.log 2>&1; log "$(tail -1 $D/attr.log | cut -c1-300)"
$PY $W/d173_close3.py $O/deliver/magC$S $SM 300000 $RE "(1) twin:${T}_frames12.npz" "(2) Hu, magnified-scale pictures:$O/runs/mag300k_Hk_${S}_frames12.npz" "(3) Hu, smooth pictures:$O/runs/atr300k_Hm_${S}_frames12.npz" > $D/close.log 2>&1
stop=$($PY $B/tmp/d172/pick.py stop $O/runs/mag300k_Hk_$S.json)
out=$O/deliver/magC$S; nice -n 10 $PY $B/tmp/d170/d170_render.py sbs $out/frames $SM 300000 ${T}_frames12.npz $O/runs/mag300k_Hk_${S}_frames12.npz $stop 3840 2160 1.5708 0.12 160 24 72 > $D/sbs.log 2>&1
ffmpeg -loglevel error -y -framerate 20 -i $out/frames/f%05d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p $out/magC${S}_twin_full_4k.mp4 < /dev/null
ffmpeg -loglevel error -y -framerate 20 -i $out/frames/f%05d.jpg -vf scale=1920:1080:flags=lanczos -c:v libx264 -crf 18 -pix_fmt yuv420p $out/magC${S}_twin_full_1080p.mp4 < /dev/null
[ -s $out/magC${S}_twin_full_1080p.mp4 ] && { sz=$(du -sm $out/frames | cut -f1); rm -rf $out/frames; echo "$(date "+%Y-%m-%d %H:%M") D173: removed $out/frames ($sz MB; the magC$S video frames, both videos written)" >> $G/cleanup.log; }
$PY $W/d173_summary.py $G/D173_SUMMARY.md
log "done"
