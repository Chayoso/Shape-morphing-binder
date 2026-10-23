#!/bin/bash
# D165 at 500k (armed 2026-10-23 on the coordinator's order): once D162's job 52 has written both 500k plaques
# (output/gpu/d162k/runs/plq500k_{HC,T}_97: H+C, the held weight with the placement at the cell, and its twin), on one
# card claimed by the lock protocol (no lock, under 2000 MiB): the stamp on each end frame (F = Fp = I, the end shape as
# the rest shape: job 52 keeps no F or Fp; no gravity), the decal (render_deliverable +fit=1 on the twin's last 8 kept
# frames, D163's relief pictures) and both base displays; the videos; then on the CPU the reading, the depth map and the
# raking-light close-ups at 74 and 84 degrees; output/gpu/d165/summary_500k.md; DONE_500k.
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d165; W=$B/tmp/d165; L=$B/gpu_locks; K=$B/output/gpu/d162k/runs
D3=$B/output/gpu/d163; S5=$O/s500; mkdir -p $S5/stamp $S5/decal $S5/video $S5/read
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%F %T') [500k] $*" >> $O/status.txt; }
log "armed: waiting for $K/plq500k_{HC,T}_97.rc"
for i in $(seq 1 2880); do [ -f $K/plq500k_HC_97.rc ] && [ -f $K/plq500k_T_97.rc ] && break; sleep 60; done
for a in HC T; do [ -f $K/plq500k_${a}_97_frames12.npz ] || { log "no $K/plq500k_${a}_97_frames12.npz (rc $(cat $K/plq500k_${a}_97.rc 2>/dev/null)): stop"; exit 1; }; done
CARD=""
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "released gpu$CARD"; CARD=""; }
trap release_card EXIT
trap "exit 143" TERM INT HUP
while [ -z "$CARD" ]; do
  for g in 3 0 2 1; do
    used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
    [ -d $L/gpu$g.lock ] || [ "$used" -gt 2000 ] && continue
    if mkdir $L/gpu$g.lock 2>/dev/null; then
      CARD=$g
      printf "agent D147 (D165)\nrun d165/500k stamp test\npid %s\nstart %s\n" "$$" "$(date '+%Y-%m-%d %H:%M:%S %Z')" > $L/gpu$g.lock/info
      break
    fi
  done
  [ -z "$CARD" ] && sleep 60
done
export CUDA_VISIBLE_DEVICES=$CARD
log "claimed gpu$CARD"
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir
( export PHYSMORPH_RUN_REPO=$B/repo_d165 PHYSMORPH_OUT=$B/output/gpu; source $B/repo_d165/scripts/ops/hyde06_env.sh > /dev/null 2>&1
  export TMPDIR=$W/tmpdir; cd $B/repo_d165
  for a in HC T; do
    $PY scripts/probes/d165/stamp.py $K/plq500k_${a}_97_frames12.npz $S5/stamp/k_$a +g=0 > $S5/stamp/k_$a.log 2>&1
    log "stamp 500k $a exit $?: $(tail -1 $S5/stamp/k_$a.log | cut -c1-160)"
    for res in 1080 2160; do
      d=$S5/video/render_${a}_$res
      $PY scripts/render_scene.py $S5/stamp/k_${a}_stamp.npz $d +stride=1 +res=$res +az=25 +el=28 +field=zbs > $d.log 2>&1
      ffmpeg -y -loglevel error -framerate 20 -pattern_type glob -i "$d/frames/*.jpg" -vf hflip -c:v libx264 -crf 18 \
        -pix_fmt yuv420p $S5/video/stamp_${a}_$res.mp4 < /dev/null && rm -rf $d/frames
    done
  done
  for res in 1080 2160; do
    ffmpeg -y -loglevel error -i $S5/video/stamp_HC_$res.mp4 -i $S5/video/stamp_T_$res.mp4 -filter_complex hstack \
      -c:v libx264 -crf 18 -pix_fmt yuv420p $S5/video/stamp500k_full_vs_twin_$res.mp4 < /dev/null
  done
  log "videos done" )
( export PHYSMORPH_RUN_REPO=$B/repo_d163 PHYSMORPH_OUT=$B/output/gpu; source $B/repo_d163/scripts/ops/hyde06_env.sh > /dev/null 2>&1
  export TMPDIR=$W/tmpdir; cd $B/repo_d163
  T=$K/plq500k_T_97_frames12.npz; H=$K/plq500k_HC_97_frames12.npz
  KT=$($PY -c "import numpy as np; print(len(np.load('$T')['raws']))"); KH=$($PY -c "import numpy as np; print(len(np.load('$H')['raws']))")
  $PY scripts/render_deliverable.py $T $O/relief_dense15.npz $S5/decal/T_decal +field=zbs +az=0 +el=8 +dist=0.55 \
    +frames=$((KT-8))-$((KT-1)) +fit=1 +steps=15 +carry=h +decay=1 > $S5/decal/T_decal.log 2>&1
  $PY scripts/render_deliverable.py $T $O/relief_dense15.npz $S5/decal/T_base +field=zbs +az=0 +el=8 +dist=0.55 \
    +frames=$((KT-1))-$((KT-1)) +fit=0 > $S5/decal/T_base.log 2>&1
  $PY scripts/render_deliverable.py $H $O/relief_dense15.npz $S5/decal/HC_base +field=zbs +az=0 +el=8 +dist=0.55 \
    +frames=$((KH-1))-$((KH-1)) +fit=0 > $S5/decal/HC_base.log 2>&1
  cd $S5/decal && for r in HC_base T_base T_decal; do f=$(ls $r/frames/*.jpg | tail -1); ffmpeg -y -loglevel error -i $f -vf hflip -q:v 2 ${r}_still.jpg; done
  ffmpeg -y -loglevel error -i HC_base_still.jpg -i T_base_still.jpg -i T_decal_still.jpg -filter_complex vstack=3,scale=1920:-2 -q:v 2 decal_compare_500k.jpg
  log "decal stills done" )
release_card
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 D157_THREADS=16 MPLCONFIGDIR=$B/cache/matplotlib PYTHONPATH=$B/repo_d163
cd $B/repo_d163
PAT=$D3/assets/d163R300plaque.obj.pattern.npz; CEIL=$D3/assets/ceiling300_frames12.npz
A=("full500 (dF_c + layer u + render loss, H+C):$S5/stamp/k_HC" "twin500 (dF_c + layer u) + post-hoc decal:$S5/stamp/k_T")
nice -n 10 $PY $W/d165_read.py $S5/read $PAT $CEIL "${A[@]}" > $S5/read.log 2>&1; log "read exit $?"
for r in 74 84; do D165_RAKE=$r nice -n 10 $PY $W/d165_render.py $S5/read/rake$r $PAT "${A[@]}" > $S5/render_rake$r.log 2>&1; log "close-ups rake $r exit $?"; done
{ echo "# D165 at 500k (auto, $(date '+%F %T %Z'))"; echo
  echo "Plaques: D162 job 52 (output/gpu/d162k/runs/plq500k_{HC,T}_97), the end frame as the rest shape (F = Fp = I), no gravity."
  echo; sed -n '3,$p' $S5/read/summary.md
  echo; echo "Files: s500/read/imprint_depth.png, s500/read/rake74|rake84/closeup_sheet.jpg (plaque face and slab print, raking light), s500/video/stamp500k_full_vs_twin_{1080,2160}.mp4, s500/decal/decal_compare_500k.jpg (top H+C, middle twin, bottom twin + decal)."; } > $O/summary_500k.md
touch $O/DONE_500k
log "done"
