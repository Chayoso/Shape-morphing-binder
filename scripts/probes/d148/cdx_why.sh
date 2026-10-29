#!/bin/bash
# cdx_why.sh -- D148 CDX: once its side render's detector tracks are written, d148_why_floating.py's confirmed pieces
# (CPU), for the per-piece split of its early crevice pieces (d148_detach.py on its dump, windows 11-32).
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp/d148; D=$B/output/gpu/d148; R=$D/render
export PHYSMORPH_RUN_REPO=$B/repo_d148x; source $B/repo_d148x/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_d148x
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
F="Warp\|CUDA\|Devices\|\"cpu\"\|Kernel\|cache/warp\|Toolkit\|initialized"
until [ -f $D/img_dragon300k_CDX_side_tracks.json ] || grep -q "chain stopped at dragon300k_CDX" $D/status.txt; do sleep 120; done
nice -n 5 $PY $T/d148_why_floating.py CDX $D/dragon300k_CDX_frames12.npz $D/dragon300k_CDX.json $D/why_CDX \
  side:$D/img_dragon300k_CDX_side_tracks.json:$R/dragon300k_CDX_side/frames:90 2>&1 | grep -v "$F" > $T/why_CDX.log
echo "why CDX $(date '+%m-%d %H:%M')" >> $D/status.txt
