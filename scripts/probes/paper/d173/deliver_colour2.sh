#!/bin/bash
# deliver_colour.sh NAME FULL_PREFIX TWIN_PREFIX STRIDE -- D173 (2): the same end frame grey (studio look, d170_render) and painted
# (d173_studio: twin | full | target paint, front and back), the disc colours by the kernel (as before) and by the nearest
# particle (+colour=nn: the identity, no kernel), and a +colour=nn video through the settle; card 3 (beside our probe).
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d173; O=$B/output/gpu/d173; G=$B/output/gpu
N=$1; F=$2; T=$3; S=${4:-3}; SC=${5:-band}; D=$O/deliver/$N; mkdir -p $D
export PHYSMORPH_RUN_REPO=$B/repo_d173 PHYSMORPH_OUT=$G; source $B/repo_d173/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=3; mkdir -p $TMPDIR; cd $B/repo_d173
log() { echo "$(date '+%m-%d %H:%M:%S') [deliver $N] $*" >> $O/status.txt; }
for c in kernel nn; do
  $PY $W/d173_studio.py $D/paint_$c $SC ${F}_frames12.npz ${T}_frames12.npz ${F}_frames12.npz +endonly=1 +w=1280 +h=720 +colour=$c > $D/paint_$c.log 2>&1
  log "painted end ($c) exit $?"
done
base=$($PY -c "import json; print(json.load(open('${F}.json'))['provenance']['tgt'])")
n=$($PY -c "import json; print(json.load(open('${F}.json'))['provenance']['n'])")
for side in full twin; do p=$F; [ $side = twin ] && p=$T
  CUDA_VISIBLE_DEVICES= $PY $B/tmp/d170/d170_render.py still $D/grey_${side}_end.png $base $n ${p}_frames12.npz -1 0.61 0.31 1920 1080 72 >> $D/grey.log 2>&1
done
log "grey stills done"
$PY $W/d173_studio.py $D/video_nn $SC ${F}_frames12.npz ${T}_frames12.npz ${F}_frames12.npz +stride=$S +w=960 +h=540 +colour=nn > $D/video_nn.log 2>&1
[ -s $D/video_nn/side_by_side.mp4 ] && { ffmpeg -loglevel error -y -i $D/video_nn/side_by_side.mp4 -c copy $D/${N}_twin_full_paint_nn.mp4 < /dev/null; sz=$(du -sm $D/video_nn/frames | cut -f1); rm -rf $D/video_nn/frames; echo "$(date '+%Y-%m-%d %H:%M') D173: removed $D/video_nn/frames ($sz MB; side_by_side.mp4 written)" >> $G/cleanup.log; }
log "video (nn) done"
