#!/bin/bash
# D147 chain 25 (2026-10-21, the user's demo, a showcase scene, not a gate): the drop. One body (sphere -> bunny) at
# rest released high above the floor, the stiff material (--young 9.8e5), the morph ramp ending (2.3 s) shortly before
# the landing (2.5 s), 7 s. Specs scene_drop_v15 / v20 / v25 (landing speed / sound speed 0.15 / 0.20 / 0.25 at a 2.5 s
# fall; scene_drop.json = v20). On one card claimed at once (no lock, under 2000 MiB; the demo's rule): the three 40k
# drops and the 300k v20 (render arm, seed 97) side by side; each 40k drawn at 1080p as it ends (az 0 and az 35), the
# 300k at 1080p then 4K; a contact sheet per video.
#   nohup setsid bash scripts/probes/d147/chain25.sh REPO > /dev/null 2>&1 < /dev/null &
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d147; R=$O/runs; LOG=$O/chain25.log
S=$REPO/scripts/probes/d147; SP=scripts/probes/d147/specs
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $B/tmp/d147/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null
log "chain25 start ($(tail -1 $REPO/DEPLOYED_FROM))"
CLAIM_IDLE=0 claim_card "chain25 demo drop scene (40k x3, 300k s97, renders)"
log "claimed gpu$CARD"
sheet() {                                                       # MP4 -> MP4 stem _sheet.jpg (every 10th frame, 7 x 2)
  ffmpeg -y -loglevel error -i $1 -vf "select='not(mod(n\,10))',scale=640:-1,tile=7x2" -frames:v 1 -q:v 3 \
    ${1%.mp4}_sheet.jpg && log "sheet ${1%.mp4}_sheet.jpg"
}
render() {                                                      # TAG NAME RES [+opts]
  local tag=$1 name=$2 res=$3; shift 3
  NAME=$name bash $S/render_scene_job.sh $R $tag +stride=1 +res=$res "$@" >> $LOG 2>&1
  log "render $name exit $? ($(ls $R/render_$name/frames 2>/dev/null | wc -l) frames)"
  rm -rf $R/render_$name/frames
  [ -f $R/$name.mp4 ] && sheet $R/$name.mp4
}
run() {                                                         # TAG N SPEC [args]
  local tag=$1 n=$2 spec=$3; shift 3
  log "$tag start"
  $PY scripts/scene_run.py --n $n --seed 97 --scene $SP/$spec.json --young 9.8e5 --flags="--assim 0" --save_stride 12 \
    "$@" --out $R/$tag > $R/$tag.log 2>&1
  log "$tag exit $?: $(tail -1 $R/$tag.log | cut -c1-200)"
}
( run drop300k_s97 300000 scene_drop
  render drop300k_s97 drop300k_s97_1080p 1080
  render drop300k_s97 drop300k_s97_1080p_az35 1080 +az=35
  render drop300k_s97 drop300k_s97_4k 2160
  render drop300k_s97 drop300k_s97_4k_az35 2160 +az=35 ) &
for v in v15 v20 v25; do
  ( run drop40k_${v}_s97 40000 scene_drop_$v
    render drop40k_${v}_s97 drop40k_${v}_s97_1080p 1080
    render drop40k_${v}_s97 drop40k_${v}_s97_1080p_az35 1080 +az=35 ) &
done
wait
release_card
log "chain25 end"
