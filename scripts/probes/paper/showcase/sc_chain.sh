#!/bin/bash
# sc_chain.sh ID QUEUE -- SHOWCASE (2026-10-21): one of at most two chains, each on one card by the lock protocol
# (gpu_locks/gpuN.lock, agent SHOWCASE; a card with no lock, no compute process of anyone, under 2000 MiB, for 10
# minutes; at most two SHOWCASE locks at a time; the card released after HOLD items (default 3) and claimed again only
# after 10 idle minutes, and released whenever the queue has nothing left for this chain).
# The queue is read again before every item (lines may be appended while it runs); an item is taken by mkdir of its
# claim in output/gpu/showcase/claims/. Lines ('#' comments):
#   run NAME SRC TGT AZ EL [pipeline_run args]   SRC -> TGT at 300k, seed 97, on repo_t1s (freeze-2026-10-15-rt1, kept
#                                                frames only), the defaults plus the args; AZ = - : no render (a twin);
#                                                a line "NAME AZ EL" in tmp/showcase/cams.txt overrides the camera
#   render NAME AZ EL VIEW                       another camera for a run already done
# Per run, on the card: the run; the end readings of read300.sh MODE end against the independent sample of the run's
# own frame (ind/SRC_TGT_ind300k.npz): the yardstick (render_terms_probe, t3end_), the display (surface_layer_probe
# measure, whole picture, e3end_), the arrangement (a_), momentum on every kept frame (m_), the end check (endgen145t.py
# -> endgen.jsonl: silhouette IoU and chamfer against the independent and the own sample); the progress per kept frame
# (sc_mid.py, CPU); the render (render_deliverable.py +fit=0 +field=zbs, the studio look at 3840 x 2160, 20 fps): the 4K
# and 1080p videos, the stills (t = 0, 0.25, 0.5, 0.75, 1, 1.5, 2.5 s, the end and the progress-0.5 frame), the contact
# sheet; the frames folder removed once both videos exist (cleanup.log). Then, detached and without a card (nice 10,
# at most two CPU readers at a time across the chains): apart particles over the first 35 % (d120_apart.py), the
# visible pieces (visible_apart.py, windows 3-10 and the whole run every 4th kept frame), the pieces off the body at the
# end (pieces145.py); a twin's kept frames are removed after them (cleanup.log). STOP: touch showcase/STOP_ID (after the
# current item); END: touch showcase/END (the chain exits once the queue is empty).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; D=$G/showcase; T=$B/tmp/showcase; L=$B/gpu_locks
ID=$1; Q=$2; HOLD=${HOLD:-3}; PYB=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
S=$D/status.txt; R=$D/render; V=$D/video
mkdir -p $D/claims $D/ind $R $V $D/sheets $D/stills $D/mid
source $B/tmp/d146/cache_env.sh
echo $$ > $D/chain_$ID.pid
log() { echo "$(date '+%m-%d %H:%M:%S') [$ID] $*" >> $S; }
mp() { if [ -e $B/repo_t1s/assets/$1.obj ]; then echo assets/$1.obj; else echo $T/assets_extra/$1.obj; fi; }
CARD=""; NDONE=0
declare -A since
claim_card() {
  while [ -z "$CARD" ]; do
    [ -e $D/STOP_$ID ] && return 1
    local now=$(date +%s)
    local held=$(grep -l "^agent SHOWCASE" $L/gpu*.lock/info 2>/dev/null | wc -l)
    for g in 3 1 0 2; do
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      local uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
      local apps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
      if [ -d $L/gpu$g.lock ] || [ "$used" -gt 2000 ] || [ "$apps" -gt 0 ]; then since[$g]=""; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      local fs=$(cat $B/tmp/d145/free_since/gpu$g 2>/dev/null); [ -n "$fs" ] && [ "$fs" -lt "${since[$g]}" ] && since[$g]=$fs
      if [ "$held" -lt 2 ] && [ $((now - ${since[$g]})) -ge 600 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        CARD=$g; NDONE=0
        printf "agent SHOWCASE\nrun showcase/%s: 300k showcase runs, readings and renders (%s items, then released)\npid %s\nstart %s\n" \
          "$ID" "$HOLD" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        log "claimed gpu$g"
        break
      fi
    done
    [ -z "$CARD" ] && sleep 60
  done
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "released gpu$CARD"; CARD=""; }
trap 'release_card' EXIT
trap 'exit 1' INT TERM HUP
wait_file() { until [ -e "$1" ]; do sleep 60; done; }
cpu_slot() { while true; do for k in 1 2; do mkdir $T/cpu_slot_$k 2>/dev/null && { echo $k; return; }; done; sleep 60; done; }

render_one() {   # NAME AZ EL VIEW
  local name=$1 az=$2 el=$3 view=$4 F=$D/${1}_frames12.npz
  local pair=$(cat $D/pair_$name) out=$R/${1}_$4
  local dense=$D/ind/${pair}_dense15.npz
  wait_file $dense
  ( export PHYSMORPH_RUN_REPO=$B/repo_t1s; source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
    source $B/tmp/d146/cache_env.sh; rm -rf $out; t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$CARD $PY scripts/render_deliverable.py $F $dense $out +stride=1 +video=1 +fit=0 +field=zbs \
      +az=$az +el=$el > $out.log 2>&1 < /dev/null
    echo "$? $(( $(date +%s) - t0 ))" > $R/rc_${name}_$view )
  log "render ${name}_$view az $az el $el rc/wall $(cat $R/rc_${name}_$view)"
  [ "$(cut -d' ' -f1 $R/rc_${name}_$view)" = 0 ] || return
  ffmpeg -loglevel error -y -framerate 20 -i $out/frames/%04d.jpg -c:v libx264 -crf 18 -pix_fmt yuv420p \
    $V/${name}_${view}_4k.mp4 < /dev/null
  ffmpeg -loglevel error -y -framerate 20 -i $out/frames/%04d.jpg -vf scale=1920:1080:flags=lanczos -c:v libx264 \
    -crf 18 -pix_fmt yuv420p $V/${name}_${view}_1080p.mp4 < /dev/null
  local n=$(ls $out/frames | wc -l) last mid ks="" k
  last=$((n - 1)); mid=$($PYB -c "import json; print(json.load(open('$D/mid/$name.json'))['mid'])" 2>/dev/null || echo $((n / 2)))
  mkdir -p $D/stills/${name}_$view
  for k in 0 5 10 15 20 30 50 $last $mid; do
    [ $k -le $last ] && cp $out/frames/$(printf %04d $k).jpg $D/stills/${name}_$view/
  done
  for k in 0 5 10 15 20 30 50; do [ $k -lt $last ] && ks="$ks,$k"; done
  ks="${ks#,},$last"
  ( source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
    $PY $T/sc_sheet.py $D/sheets/${name}_${view}_sheet.jpg "$(cat $D/title_$name) (${view}, az $az, el $el)" \
      $out/frames $ks 4 960 > /dev/null 2>&1 )
  if [ -s $V/${name}_${view}_4k.mp4 ] && [ -s $V/${name}_${view}_1080p.mp4 ]; then
    local sz=$(du -sm $out/frames | cut -f1); rm -rf $out/frames
    echo "$(date '+%Y-%m-%d %H:%M') SHOWCASE: removed $out/frames ($sz MB; render frames of ${name}_$view, both videos written, stills in showcase/stills/${name}_$view)" >> $G/cleanup.log
  fi
}

run_one() {   # NAME SRC TGT AZ EL ARGS...
  local name=$1 src=$2 tgt=$3 az=$4 el=$5; shift 5; local args="$*"
  local pair=${src}_${tgt} F=$D/${1}_frames12.npz
  local ind=$D/ind/${pair}_ind300k.npz
  echo $pair > $D/pair_$name
  if [ "$src" = isosphere ]; then echo "sphere -> $tgt, 300k" > $D/title_$name; else echo "$src -> $tgt, 300k" > $D/title_$name; fi
  if [ ! -e $D/rc_$name ]; then
    log "run $name: $src -> $tgt, args '$args', gpu$CARD"
    ( export PHYSMORPH_RUN_REPO=$B/repo_t1s PHYSMORPH_OUT=$G
      source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1; source $B/tmp/d146/cache_env.sh; export OMP_NUM_THREADS=8
      ( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CARD)"; sleep 10; done ) > $D/$name.mem &
      local mem=$!
      local t0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$CARD $PY scripts/pipeline_run.py --src $(mp $src) --tgt $(mp $tgt) --n 300000 --seed 97 \
        --out $D/$name $args > $D/$name.log 2>&1 < /dev/null
      local rc=$?
      kill $mem
      echo "$rc $(( $(date +%s) - t0 )) $(sort -k2 -n $D/$name.mem | tail -1 | cut -d' ' -f2)" > $D/rc_$name )
    log "run $name rc/wall/peak_mib $(cat $D/rc_$name)"
  fi
  [ "$(cut -d' ' -f1 $D/rc_$name)" = 0 ] && [ -e $F ] || { log "$name FAILED or no kept frames"; return; }
  if [ ! -e $D/m_$name.log ]; then
    wait_file $ind
    ( source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null 2>&1; source $B/tmp/d146/cache_env.sh; cd $B/repo_r77
      local last=$($PY $B/tmp/lastraw.py $F)
      $PY $B/tmp/d145/endframe.py $F $D/${name}_end1.npz
      CUDA_VISIBLE_DEVICES=$CARD $PY scripts/probes/settled/render_terms_probe.py $D/${name}_end1.npz ref=$ind > $D/t3end_$name.log 2>&1 < /dev/null
      CUDA_VISIBLE_DEVICES=$CARD $PY scripts/probes/settled/surface_layer_probe.py $F $D/e3end_$name 0,0,3840,2160 measure ref=$ind target own $last > $D/e3end_$name.log 2>&1 < /dev/null
      CUDA_VISIBLE_DEVICES=$CARD $PY $B/tmp/arrangement.py $F $last > $D/a_$name.log 2>&1 < /dev/null )
    ( export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1; cd $B/repo_t
      CUDA_VISIBLE_DEVICES=$CARD $PY $T/endgen145t.py $D/endgen.jsonl "$name=$D/${name}_end1.npz,$ind" > $D/endgen_$name.log 2>&1 < /dev/null )
    ( source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1; export OMP_NUM_THREADS=4
      CUDA_VISIBLE_DEVICES="" $PY $T/sc_mid.py $D/mid/$name.json $F > $D/mid/$name.log 2>&1 < /dev/null )
    bash $B/tmp/mom.sh $CARD $D/m_$name.log "$name=$F"
    log "read gpu $name"
  fi
  local cam=$(grep "^$name " $T/cams.txt 2>/dev/null | tail -1)   # a camera chosen after the queue line was written
  [ "$az" != "-" ] && [ -n "$cam" ] && { set -- $cam; az=$2; el=$3; }
  [ "$az" != "-" ] && [ ! -e $V/${name}_main_4k.mp4 ] && render_one $name $az $el main
  # the CPU part, detached, at most two at a time
  ( trap - EXIT INT TERM HUP; k=$(cpu_slot)
    source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null 2>&1; source $B/tmp/d146/cache_env.sh
    export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
    nice -n 10 $PY $B/tmp/d120_apart.py $F 0.35 > $D/apart_$name.log 2>&1 < /dev/null
    ( export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1; cd $B/repo_t
      export CUDA_VISIBLE_DEVICES=""
      nice -n 10 $PY scripts/probes/paper/visible_apart.py $F --label $name --jsonl $D/vis_w3_10_$name.jsonl --threads 8 > $D/vis_$name.log 2>&1
      nice -n 10 $PY scripts/probes/paper/visible_apart.py $F --label $name --w0 0 --w1 100000 --every 4 --jsonl $D/vis_full_$name.jsonl --threads 8 >> $D/vis_$name.log 2>&1
      nice -n 10 $PY $B/tmp/d145r/pieces145.py $D/pieces_$name.jsonl $name=$D/${name}_end1.npz --every 1 --threads 8 > $D/pieces_$name.log 2>&1 ) < /dev/null
    if [ "$az" = "-" ] && [ -e $D/pieces_$name.jsonl ]; then
      sz=$(du -m $F | cut -f1); rm -f $F
      echo "$(date '+%Y-%m-%d %H:%M') SHOWCASE: removed $F ($sz MB; the lambda = 0 twin $name, read: its logs and end1 in showcase)" >> $G/cleanup.log
    fi
    echo "$(date '+%m-%d %H:%M:%S') [$ID] cpu done $name" >> $S
    rmdir $T/cpu_slot_$k ) > /dev/null 2>&1 &
}

next_line() {   # the first queue line not yet claimed (prints it and claims it)
  local line key
  while read -r line; do
    [ -z "$line" ] && continue; [ "${line:0:1}" = "#" ] && continue
    set -- $line
    case $1 in run) key=$2;; render) key=${2}_$5;; *) continue;; esac
    [ "$1" = render ] && [ ! -e $D/rc_$2 ] && continue          # its run not done yet
    if mkdir $D/claims/$key.claim 2>/dev/null; then echo "$line"; return 0; fi
  done < $Q
  return 1
}

log "chain start pid $$ queue $Q hold $HOLD"
while true; do
  [ -e $D/STOP_$ID ] && { log "stopped by STOP_$ID"; break; }
  line=$(next_line)
  if [ -z "$line" ]; then
    release_card
    [ -e $D/END ] && { log "queue empty and END: chain done"; break; }
    sleep 120; continue
  fi
  set -- $line; case $1 in run) key=$2;; render) key=${2}_$5;; esac
  [ -z "$CARD" ] && { claim_card || { log "stopped while waiting for a card"; rmdir $D/claims/$key.claim; break; }; }
  set -- $line
  case $1 in
    run) shift; run_one "$@";;
    render) render_one $2 $3 $4 $5;;
  esac
  NDONE=$((NDONE + 1))
  [ $NDONE -ge $HOLD ] && release_card
done
