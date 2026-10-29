#!/bin/bash
# d159c_closeups.sh -- D159c's close-ups, the same shots as d159_closeups.sh (studio look, render_deliverable.py +fit=0
# +field=zbs, repo_freeze1015; kept frames 20, 40, 80 and the last; the r = 2.5 display pair at 80 and the last) on
# bunny300k_C97 / dragon300k_C97, ONE card by the lock protocol (free 900 s, never GPU 2), then the three-row sheets
# (tag | D159 | D159c, d159_closeup_sheet.py). Afterwards, once the CPU readers (tmp/d159/readfix_*.pid) are done, the
# read kept frames of M97 and C97 are deleted and logged in output/gpu/cleanup.log.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d159; W=$B/tmp/d159; L=$B/gpu_locks
S=$O/status.txt
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
source $W/d159_claim.sh
claim_card "D159c close-ups"
trap 'release_card' EXIT
trap 'exit 1' INT TERM HUP
shot() {   # LABEL MESH NPZ LOOK AZ EL DIST EXTRA FRAMES  (EXTRA: render_deliverable options or -)
  local lab=$1 mesh=$2 npz=$3 look=$4 az=$5 el=$6 dist=$7 extra=$8; shift 8
  [ "$extra" = - ] && extra=""
  for k in "$@"; do
    out=$O/closeups/$lab
    ( export PHYSMORPH_RUN_REPO=$B/repo_freeze1015 PHYSMORPH_OUT=$W/tmpdir/pm_out
      source $B/repo_freeze1015/scripts/ops/hyde06_env.sh > /dev/null 2>&1; source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir
      cd $B/repo_freeze1015
      $PY scripts/render_deliverable.py $npz $G/showcase/ind/isosphere_${mesh}_dense15.npz ${out}_$k +stride=1 +video=1 +fit=0 \
        +field=zbs +az=$az +el=$el +dist=$dist +look=$look +frames=$k-$k $extra > ${out}_$k.log 2>&1 < /dev/null )
  done
}
for mesh in bunny dragon; do
  F=$O/${mesh}300k_C97_frames12.npz
  last=$($PY -c "import numpy as np; print(len(np.load('$F')['raws']) - 1)")
  if [ $mesh = bunny ]; then
    shot d159c_bunny_ears bunny $F -0.316,0.71,0.15 35 18 0.35 - 20 40 80 $last
    shot d159c_bunny_wrinkles bunny $F 0.315,-0.287,-0.103 83 0 0.3 - 20 40 80 $last
    shot d159c_bunny_wrinkles25 bunny $F 0.315,-0.287,-0.103 83 0 0.3 "+radius=2.5 +offset=0.6667" 80 $last
  else
    shot d159c_dragon_head dragon $F -0.059,0.688,0.255 35 18 0.35 - 20 40 80 $last
    shot d159c_dragon_scales dragon $F 0.29,-0.299,0.29 125 5 0.3 - 20 40 80 $last
    shot d159c_dragon_scales25 dragon $F 0.29,-0.299,0.29 125 5 0.3 "+radius=2.5 +offset=0.6667" 80 $last
  fi
  log "Y close-ups d159c_$mesh done (last kept $last)"
done
release_card
( source $B/tmp/d146/cache_env.sh; export CUDA_VISIBLE_DEVICES=""
  $PY $W/d159_closeup_sheet.py $O/closeups > $O/closeups/sheet.log 2>&1 )
log "Y close-up sheets (three rows): $(grep -c wrote $O/closeups/sheet.log) sheets"
for p in $(cat $W/readfix_bunny.pid $W/readfix_dragon.pid 2>/dev/null); do while ps -p $p > /dev/null 2>&1; do sleep 60; done; done
for run in bunny300k_M97 dragon300k_M97 bunny300k_C97 dragon300k_C97; do
  f=$O/${run}_frames12.npz
  [ -f $f ] || continue
  sz=$(du -m $f | cut -f1); rm -f $f
  echo "$(date '+%Y-%m-%d %H:%M') D159: removed $f (${sz} MB; read: R1 relief_*, R2 course_*, B1 readers, studio renders / close-ups, end1 kept)" >> $G/cleanup.log
done
log "D159c close-ups done; M97 / C97 kept frames removed"
