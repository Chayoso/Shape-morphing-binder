#!/bin/bash
# d159_chain.sh STEPS -- D159 (agent D159): the gates and the runs on ONE card by the agents' lock protocol
# (gpu_locks/gpuN.lock by mkdir with an info file, released by the trap; a card with no lock, no other user's compute
# process, < 2000 MiB, free for 15 minutes (SHOWCASE's rule: five minutes behind the queued agents' 10, so a waiting
# agent takes a freed card first), the pool's shared first-free time honoured; GPU 2 never).
#   V   the gates; any failure stops the chain before a run:
#       V1 the full suite on repo_d159 (exit status), V2 the pinned digest (test_refreeze_reference.py),
#       V3 Warp's deterministic mode, 40k bunny seed 97 four windows (tmp/rt/det_run.py): repo_d159s at the defaults
#          against rt1's S1 (output/gpu/d153/rt1/S1, det_cmp12.py): SAME,
#       V4 --render_target mesh_shell four windows: exit 0 and the "render target (D159)" line.
#   R   300k on repo_d159s (kept frames), two at a time on the card, only once the regions are fixed
#       (output/gpu/d159/regions_ok, written by the step-0 reader before any run): bunny300k_M97 / dragon300k_M97
#       (--render_target mesh_shell, seed 97), then bunny300k_P99 / dragon300k_P99 (the twin again with kept frames,
#       --render_weight_scale 0, seed 99); each followed by B1's GPU readings (read300.sh ... end 1).
#   G   the 40k gallery: MESH40k_M97 for the 19 meshes (--render_target mesh_shell, seed 97), three at a time.
#   X   the studio renders (render_deliverable.py +fit=0 +field=zbs, repo_freeze1015, 4K, az 35 el 18 dist 1) of the tag's
#       L97 and D159's M97, bunny and dragon, the 1080p side-by-side (d159_sbs.py) and its video; the 4K frames removed
#       once the video exists (cleanup.log).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d159; W=$B/tmp/d159; L=$B/gpu_locks
mkdir -p $O/gates $O/g40 $W/tmpdir
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
STEPS=${1:-"V R"}
GALLERY=${D159_MESHES:-"A armadilo beast bimba bob bunny C cheburashka cow dragon fandisk heart homer maxplanck nefertiti ogre spot teapot V"}
IDLE=${D159_IDLE:-900}
CARD=""
claim_card() {
  local what=$1; declare -A since
  local last_report=0
  while [ -z "$CARD" ]; do
    local now=$(date +%s) reasons=""
    for g in 3 1 0; do
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      local uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
      local other=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | while IFS=', ' read -r u p; do
                    o=$(ps -o user= -p $p 2>/dev/null); [ -n "$o" ] && [ "$o" != chayo ] && echo x; done | wc -l)
      local mine=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
      local why=""
      [ -d $L/gpu$g.lock ] && why="lock($(grep -m1 agent $L/gpu$g.lock/info 2>/dev/null))"
      [ "$used" -gt 2000 ] && why="$why used=${used}MiB"
      [ "$other" -gt 0 ] && why="$why other-user"
      [ "$mine" -gt 0 ] && why="$why apps=$mine"
      if [ -n "$why" ]; then since[$g]=""; reasons="$reasons gpu$g:$why;"; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      local fs=$(cat $B/tmp/d145/free_since/gpu$g 2>/dev/null); [ -n "$fs" ] && [ "$fs" -lt "${since[$g]}" ] && since[$g]=$fs
      if [ $((now - ${since[$g]})) -ge $IDLE ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        CARD=$g
        printf "agent D159\nrun d159/%s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        break
      fi
    done
    if [ -z "$CARD" ] && [ $((now - last_report)) -ge 900 ]; then
      log "waiting:$reasons free:$(for k in "${!since[@]}"; do [ -n "${since[$k]}" ] && echo -n " gpu$k since $(date -d @${since[$k]} +%H:%M:%S)"; done)"
      last_report=$now
    fi
    [ -z "$CARD" ] && sleep 60
  done
  export CUDA_VISIBLE_DEVICES=$CARD
  log "claimed gpu$CARD for [$what]"
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "released gpu$CARD"; CARD=""; }
trap 'release_card' EXIT
trap 'exit 1' INT TERM HUP
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1; export TMPDIR=$W/tmpdir; cd $B/$1; }
fail() { log "STOP: $*"; exit 1; }
log "chain start pid $$ steps [$STEPS] (a card free ${IDLE} s)"
if [[ " $STEPS " == *" V "* ]]; then
  claim_card "gates V1-V4"
  V=$O/gates
  ( envr repo_d159; $PY -m pytest -q -p no:cacheprovider tests > $V/suite.log 2>&1 < /dev/null; echo $? > $V/suite.rc )
  log "V1 suite repo_d159 exit $(cat $V/suite.rc): $(tail -1 $V/suite.log)"
  [ "$(cat $V/suite.rc)" = 0 ] || fail "V1 suite"
  ( envr repo_d159; $PY -m pytest -q -p no:cacheprovider tests/test_refreeze_reference.py > $V/digest.log 2>&1 < /dev/null
    echo $? > $V/digest.rc )
  log "V2 pinned digest exit $(cat $V/digest.rc): $(tail -1 $V/digest.log)"
  [ "$(cat $V/digest.rc)" = 0 ] || fail "V2 digest"
  det() { local name=$1; shift; ( envr repo_d159s; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 \
      --animations 4 --out $V/$name "$@" > $V/$name.log 2>&1 < /dev/null; echo $? > $V/$name.rc ); log "V det $name ($*) exit $(cat $V/$name.rc)"; }
  det S1 --tgt assets/bunny.obj
  ( envr repo_d159; $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $V/S1 > $V/cmp_rt1_S1.txt 2>&1 )
  log "V3 rt1 S1 vs repo_d159s defaults: $(tail -1 $V/cmp_rt1_S1.txt)"
  tail -1 $V/cmp_rt1_S1.txt | grep -q ": SAME$" || fail "V3 default path"
  det M1 --tgt assets/bunny.obj --render_target mesh_shell
  [ "$(cat $V/M1.rc)" = 0 ] || fail "V4 exit"
  grep -q "render target (D159)" $V/M1.log || fail "V4: no render-target line"
  log "V4: $(grep -m1 'render target (D159)' $V/M1.log | cut -c1-330)"
  log "V gates passed"
fi
run300() {   # NAME MESH SEED FLAGS...  (in the background; its rc in rc_NAME)
  local name=$1 mesh=$2 seed=$3; shift 3
  ( envr repo_d159s; export OMP_NUM_THREADS=8
    ( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CARD)"; sleep 10; done ) > $O/$name.mem &
    local mem=$! t0=$(date +%s)
    $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n 300000 --seed $seed --out $O/$name "$@" > $O/$name.log 2>&1 < /dev/null
    local rc=$?
    kill $mem
    echo "$rc $(( $(date +%s) - t0 ))" > $O/rc_$name
    log "R $name rc/wall $(cat $O/rc_$name): $(grep -m1 'ARM render_full_dt_iso_nn:' $O/$name.log | cut -c1-200)"
    [ "$rc" = 0 ] && bash $B/tmp/d145/read300.sh $CARD d159 $name $mesh end 1 > $O/read300_$name.log 2>&1
    log "R $name GPU readings done" ) &
}
if [[ " $STEPS " == *" R "* ]]; then
  [ -e $O/regions_ok ] || { release_card; log "R waits for the regions (regions_ok)"; while [ ! -e $O/regions_ok ]; do sleep 60; done; }
  [ -n "$CARD" ] || claim_card "300k runs"
  log "R pair 1 start (regions: $(head -c 200 $O/regions_ok))"
  run300 bunny300k_M97 bunny 97 --render_target mesh_shell
  sleep 30
  run300 dragon300k_M97 dragon 97 --render_target mesh_shell
  wait
  run300 bunny300k_P99 bunny 99 --render_weight_scale 0
  sleep 30
  run300 dragon300k_P99 dragon 99 --render_weight_scale 0
  wait
  log "R done"
  release_card
  for mesh in bunny dragon; do                       # the CPU readings, detached (no card)
    ( trap - EXIT INT TERM HUP; setsid nohup bash $W/d159_readcpu.sh $mesh > $O/read_$mesh.out 2>&1 < /dev/null & echo $! > $O/readcpu_$mesh.pid )
    log "R readcpu $mesh started pid $(cat $O/readcpu_$mesh.pid)"
  done
fi
if [[ " $STEPS " == *" G "* ]]; then
  claim_card "40k gallery"
  n=0
  for mesh in $GALLERY; do
    name=${mesh}40k_M97
    [ -e $O/g40/${name}_frames12.npz ] && continue
    ( envr repo_d159s; export OMP_NUM_THREADS=8
      $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n 40000 --seed 97 --render_target mesh_shell --out $O/g40/$name \
        > $O/g40/$name.log 2>&1 < /dev/null; echo $? > $O/g40/rc_$name
      log "G $name rc $(cat $O/g40/rc_$name)" ) &
    n=$((n + 1)); [ $((n % 3)) = 0 ] && wait
    sleep 10
  done
  wait
  log "G done"
  release_card
fi
if [[ " $STEPS " == *" X "* ]]; then
  claim_card "studio renders"
  mkdir -p $O/render $O/video
  for mesh in bunny dragon; do
    for arm in tag d159; do
      F=$G/d145t_b1/${mesh}300k_L97_frames12.npz; [ $arm = d159 ] && F=$O/${mesh}300k_M97_frames12.npz
      out=$O/render/${arm}_$mesh
      [ -e $out/done ] && continue
      ( envr repo_freeze1015; t0=$(date +%s)
        $PY scripts/render_deliverable.py $F $G/showcase/ind/isosphere_${mesh}_dense15.npz $out +stride=1 +video=1 +fit=0 +field=zbs \
          +az=35 +el=18 +dist=1 > $out.log 2>&1 < /dev/null && touch $out/done
        log "X render ${arm}_$mesh: $(ls $out/frames 2>/dev/null | wc -l) frames in $(( $(date +%s) - t0 )) s" )
    done
    ( envr repo_freeze1015; export CUDA_VISIBLE_DEVICES=""
      $PY $W/d159_sbs.py $O/render/tag_$mesh $O/render/d159_$mesh $O/render/sbs_$mesh "${mesh} 300k" > $O/render/sbs_$mesh.log 2>&1
      ffmpeg -hide_banner -loglevel error -y -framerate 20 -i $O/render/sbs_$mesh/sbs/%04d.jpg -c:v libx264 -preset medium -crf 18 \
        -pix_fmt yuv420p -movflags +faststart $O/video/${mesh}300k_tag_vs_d159_1080p.mp4 < /dev/null
      if [ -s $O/video/${mesh}300k_tag_vs_d159_1080p.mp4 ]; then
        for arm in tag d159; do sz=$(du -sm $O/render/${arm}_$mesh/frames | cut -f1); rm -rf $O/render/${arm}_$mesh/frames
          echo "$(date '+%Y-%m-%d %H:%M') D159: removed the 4K render frames of ${arm}_$mesh (${sz} MB; the 1080p side-by-side video and its frames kept, output/gpu/d159/video)" >> $G/cleanup.log; done
      fi )
    log "X $mesh side-by-side: $(ls -la $O/video/${mesh}300k_tag_vs_d159_1080p.mp4 2>&1 | awk '{print $5}') bytes"
  done
  release_card
fi
log "chain done"
