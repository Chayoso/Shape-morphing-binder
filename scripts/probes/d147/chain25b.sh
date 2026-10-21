#!/bin/bash
# D147 chain 25b (2026-10-21, the demo's drop, a showcase): the 300k drop on SPEC (default scene_drop: v20 with 6 wu of
# horizontal room; chain25's 40k drops rolled forward 2.7 wu by 5.2 s, toward the 2.5 wu room's wall), render arm,
# seed 97; after chain25 ends (its card released) a card claimed at once (no lock, under 2000 MiB); 1080p and 4K at
# azimuth 0 and 35 deg, a contact sheet each.
#   nohup setsid bash scripts/probes/d147/chain25b.sh REPO [SPEC] > /dev/null 2>&1 < /dev/null &
REPO=$1; SPEC=${2:-scene_drop}
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d147; R=$O/runs; LOG=$O/chain25.log
S=$REPO/scripts/probes/d147; SP=scripts/probes/d147/specs; TAG=drop300k_s97
log() { echo "$(date '+%F %T') [25b] $*" >> $LOG; }
source $B/tmp/d147/claim.sh
trap "release_card" EXIT
trap "exit 143" TERM INT HUP
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null
log "chain25b start, spec $SPEC: $(cat $SP/$SPEC.json)"
while grep -q "run d147/chain25 demo" $L/gpu*.lock/info 2>/dev/null; do sleep 5; done
CLAIM_IDLE=0 claim_card "chain25b demo drop 300k s97 + renders"
log "claimed gpu$CARD"
sheet() {
  ffmpeg -y -loglevel error -i $1 -vf "select='not(mod(n\,10))',scale=640:-1,tile=7x2" -frames:v 1 -q:v 3 \
    ${1%.mp4}_sheet.jpg && log "sheet ${1%.mp4}_sheet.jpg"
}
render() {
  local tag=$1 name=$2 res=$3; shift 3
  NAME=$name bash $S/render_scene_job.sh $R $tag +stride=1 +res=$res "$@" >> $LOG 2>&1
  log "render $name exit $? ($(ls $R/render_$name/frames 2>/dev/null | wc -l) frames)"
  rm -rf $R/render_$name/frames
  [ -f $R/$name.mp4 ] && sheet $R/$name.mp4
}
log "$TAG start"
$PY scripts/scene_run.py --n 300000 --seed 97 --scene $SP/$SPEC.json --young 9.8e5 --flags="--assim 0" --save_stride 12 \
  --out $R/$TAG > $R/$TAG.log 2>&1
log "$TAG exit $?: $(tail -1 $R/$TAG.log | cut -c1-200)"
render $TAG ${TAG}_1080p 1080
render $TAG ${TAG}_1080p_az35 1080 +az=35
render $TAG ${TAG}_4k 2160
render $TAG ${TAG}_4k_az35 2160 +az=35
release_card
log "chain25b end"
