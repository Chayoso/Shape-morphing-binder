#!/bin/bash
# ri_video_card.sh -- render influence video, pass 2 on ONE card under the lock protocol (gpu_locks/gpuN.lock by mkdir,
# info "agent RI"; a card with no lock, no other user's compute process, < 2000 MiB, free for 600 s with the pool's
# shared first-free time honoured; GPU 2 never). Waits for each mesh's pass 1 (status line "video values MESH rc 0"),
# renders with ri_video.py (repo_rgvtag's RGV renderer imported), encodes the mp4s at 20 fps (kept frames, real time),
# keeps the frames for the stills, releases the card.
B=/data/relcfd/chayo/physmorph_v2
L=$B/gpu_locks
O=$B/output/gpu/render_influence
V=$O/video
ST=$O/status.txt
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
source $B/tmp/d146/cache_env.sh
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $ST; }
IDLE=600
CARD=""
claim_card() {
  local what=$1; declare -A since
  local last_report=0
  while [ -z "$CARD" ]; do
    local now=$(date +%s) reasons=""
    for g in 1 3 0; do
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
        printf "agent RI (render influence video)\nrun render_influence/%s\npid %s\nstart %s\n" "$what" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        break
      fi
    done
    if [ -z "$CARD" ] && [ $((now - last_report)) -ge 900 ]; then log "RI waiting:$reasons"; last_report=$now; fi
    [ -z "$CARD" ] && sleep 30
  done
  export CUDA_VISIBLE_DEVICES=$CARD
  log "RI claimed gpu$CARD for [$what]"
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "RI released gpu$CARD"; CARD=""; }
trap release_card EXIT

claim_card "pass 2: bunny + dragon 300k, 1080p front/side + 2x2"
export PYTHONPATH=$B/repo_rgvtag OMP_NUM_THREADS=8
for M in ${MESHES:-bunny dragon}; do
  until grep -q "video values $M rc 0" $ST; do
    grep -q "video values $M rc [1-9]" $ST && { log "RI pass 1 $M failed; skipped"; continue 2; }
    sleep 20
  done
  # one fixed colour scale for the run: the 95th percentile over the kept frames of the per-frame 99th percentile of
  # |d| on the outer layer, rounded up to 0.25 / 0.5 / 0.75 / 1 / 1.5 / 2 / 3 pitches
  [ -f $V/dmax_$M ] || $PY -c "
import json, numpy as np
r = json.load(open('$V/vals_$M.json'))['rows']
q = float(np.percentile([x['p99_ds'] for x in r], 95))
print(next(v for v in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5) if v >= q))" > $V/dmax_$M 2>/dev/null
  DM=$(cat $V/dmax_$M)
  tries=0
  while :; do
    $PY $B/tmp/render_influence/ri_video.py $M $V $V/frames +res=1080 +views=35,90 +dmax=$DM +grid=1 > $V/render_$M.log 2>&1
    rc=$?
    log "RI render $M dmax $DM rc $rc"
    [ $rc = 0 ] && break
    tries=$((tries + 1)); [ $tries -ge 3 ] && break
    rm -f $V/retry_$M; n=0                               # hold the card while the script is fixed (retry_MESH appears)
    until [ -f $V/retry_$M ] || [ $n -ge 60 ]; do sleep 30; n=$((n + 1)); done
    [ -f $V/retry_$M ] || break
    DM=$(cat $V/dmax_$M)
  done
  [ $rc = 0 ] || continue
  for kd in map_35 map_90 grid gridhalf; do
    case $kd in map_35) nm=E_where_render_changes_shape_front_1080p ;; map_90) nm=E_where_render_changes_shape_side_1080p ;;
      grid) nm=E_2x2_front_2160p ;; gridhalf) nm=E_2x2_half_front_1080p ;; esac
    ffmpeg -loglevel error -y -framerate 20 -pattern_type glob -i "$V/frames/$M/$kd/*.jpg" -c:v libx264 -crf 18 \
      -pix_fmt yuv420p $V/${M}300k_$nm.mp4 < /dev/null
    log "RI ffmpeg $M $kd rc $? $(ls $V/frames/$M/$kd | wc -l) frames"
  done
  ffmpeg -loglevel error -y -framerate 10 -pattern_type glob -i "$V/frames/$M/gridhalf/*.jpg" -c:v libx264 -crf 18 \
    -pix_fmt yuv420p $V/${M}300k_E_2x2_halfspeed_front_1080p.mp4 < /dev/null
  log "RI ffmpeg $M gridhalf halfspeed rc $?"
done
release_card
trap - EXIT
log "RI pass 2 done"
