#!/bin/bash
# d159_closeups.sh -- D159 close-ups (studio look, render_deliverable.py +fit=0 +field=zbs, repo_freeze1015): the tag's L97 and
# D159's M97 at kept frames 20, 40, 80 and the last, four close-ups (look = the part's centre in radii, dist 0.3-0.35 x 3.6
# radii): bunny ears (az 35 el 18) and its haunch wrinkles (az 83 el 0), dragon head and horns (az 35 el 18) and belly
# scutes (az 125 el 5); and, a display reading only, the bunny wrinkles and the dragon scutes at 4 s and the end drawn
# through the r = 2.5 field (D123, offset 0.8 x 2.5 / 3); then the sheets (d159_closeup_sheet.py). Starts only after the main chain (pid in
# tmp/d159/chain_VRXG3.pid) is done, then claims ONE card by the same rule (free 900 s, never GPU 2), released at the end.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d159; W=$B/tmp/d159; L=$B/gpu_locks
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
MAIN=$(cat $W/chain_VRXG3.pid)
while ps -p $MAIN > /dev/null 2>&1; do sleep 120; done
source $W/d159_claim.sh
claim_card "close-ups"
trap 'release_card' EXIT
trap 'exit 1' INT TERM HUP
mkdir -p $O/closeups
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
  for arm in tag d159; do
    F=$G/d145t_b1/${mesh}300k_L97_frames12.npz; [ $arm = d159 ] && F=$O/${mesh}300k_M97_frames12.npz
    last=$(/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python -c "import numpy as np; print(len(np.load('$F')['raws']) - 1)")
    if [ $mesh = bunny ]; then
      shot ${arm}_bunny_ears bunny $F -0.316,0.71,0.15 35 18 0.35 - 20 40 80 $last
      shot ${arm}_bunny_wrinkles bunny $F 0.315,-0.287,-0.103 83 0 0.3 - 20 40 80 $last
      # display reading (the coordinator, 01:0x): the same frames through the r = 2.5 field (D123, offset in proportion)
      shot ${arm}_bunny_wrinkles25 bunny $F 0.315,-0.287,-0.103 83 0 0.3 "+radius=2.5 +offset=0.6667" 80 $last
    else
      shot ${arm}_dragon_head dragon $F -0.059,0.688,0.255 35 18 0.35 - 20 40 80 $last
      shot ${arm}_dragon_scales dragon $F 0.29,-0.299,0.29 125 5 0.3 - 20 40 80 $last
      shot ${arm}_dragon_scales25 dragon $F 0.29,-0.299,0.29 125 5 0.3 "+radius=2.5 +offset=0.6667" 80 $last
    fi
    log "Y close-ups ${arm}_$mesh done (last kept $last)"
  done
done
release_card
( source $B/tmp/d146/cache_env.sh; export CUDA_VISIBLE_DEVICES=""
  /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python $W/d159_closeup_sheet.py $O/closeups > $O/closeups/sheet.log 2>&1 )
log "Y close-up sheets: $(ls $O/closeups/*.jpg 2>/dev/null | wc -l) sheets"
