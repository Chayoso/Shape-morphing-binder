#!/bin/bash
# D147 chain 16 (replaces chain 10's 19:00 start; the coordinator, 15:47): the 40k throw re-rendered with the static
# main shot; the 300k throw (2 x 150k, spec scene_throw) at seed 97 now on one card, its render-off twin and seed 99
# after it or from 19:00 (one D147 card before 19:00); then the readings first (timeline, cut, progress) and, only if
# s97 passes the progress and parting rule, its 4K renders: the whole run (static), the delivered cut (cut.py's rule)
# and the contact close-up (camera=contact, 2 s before the first contact to 1 s after the parting).
#   bash scripts/probes/d147/chain16.sh REPO
REPO=$1; SPEC=scene_throw
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
log() { echo "$(date '+%F %T') $*" >> $O/chain16.log; }
claim_and_run() {                      # one D147 card before 19:00 (the coordinator), any free card after
  local tag=$1 cmd=$2
  while true; do
    if [ "$(date +%H%M)" -lt 1900 ] && grep -qs "agent D147" $L/gpu*.lock/info; then sleep 30; continue; fi
    for g in 3 0 2 1; do
      [ -d $L/gpu$g.lock ] && continue
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$used" -gt 2000 ] && continue
      if bash $S/job.sh $g "$REPO" "$tag" "$cmd"; then
        rc=$(grep "^${tag}_end " $O/jobs.log | tail -1 | awk '{print $NF}'); log "$tag gpu$g exit $rc"; return $rc
      fi
    done
    sleep 30
  done
}
R=$O/runs; SP=scripts/probes/d147/specs; B=scene300k_$SPEC
run() { echo "\$PY scripts/scene_run.py --n 150000 --seed $1 --scene $SP/$SPEC.json --flags=\"--assim 0\" --save_stride 12 $3 --out $R/$2 > $R/$2.log 2>&1; echo $2 exit \$?"; }
log "chain16 start, repo $REPO"
claim_and_run T17_scene_tests '$PY -m pytest tests/test_scene_schedule.py tests/test_scene_bodies.py tests/test_scene_assim.py tests/test_scene_frame.py -q -p no:cacheprovider'
claim_and_run RENDER_static40k "NAME=throw40k_a_static bash $S/render_scene_job.sh $R throw40k_a +stride=1"
claim_and_run P300 "$(run 97 ${B}_s97)" &
until grep -q "window 2/" $R/${B}_s97.log 2>/dev/null || grep -q "^P300_end " $O/jobs.log; do sleep 30; done
log "s97: $(grep -m1 'window 2/' $R/${B}_s97.log | cut -c1-160)"
claim_and_run P300_twin "$(run 97 ${B}_s97_off '--render_weight_scale 0')" &
sleep 120
claim_and_run P300_s99 "$(run 99 ${B}_s99)" &
wait
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
T="${B}_s97,${B}_s97_off,${B}_s99"
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
  nice $PY $S/read40k.py $R $T > $R/read_300k_$SPEC.txt 2>&1
  nice $PY $S/cut.py $R $T > $R/cut_300k_$SPEC.txt 2>&1 )
claim_and_run PROG300 "\$PY $S/progress.py $R $T > $R/progress_300k_$SPEC.txt 2>&1"
log "readings: read_300k_$SPEC.txt, progress_300k_$SPEC.txt, cut_300k_$SPEC.txt"
if grep "\"tag\": \"${B}_s97\"" $R/progress_300k_$SPEC.txt | grep -q '"in_band": true' \
   && grep "\"tag\": \"${B}_s97\"" $R/read_300k_$SPEC.txt | grep -q '"parted_in_air": true'; then
  line=$(grep "\"tag\": \"${B}_s97\"" $R/cut_300k_$SPEC.txt)
  CUT=$(echo "$line" | grep -o '"cut_step": [0-9]*' | grep -o '[0-9]*$')
  K1=$(echo "$line" | grep -o '"first_contact": [0-9]*' | grep -o '[0-9]*$')
  KP=$(echo "$line" | grep -o '"parted": [0-9]*' | grep -o '[0-9]*$')
  claim_and_run RENDER300 "bash $S/render_scene_job.sh $R ${B}_s97 +stride=1; NAME=${B}_s97_contact bash $S/render_scene_job.sh $R ${B}_s97 +stride=1 +camera=contact +frames=$((K1 > 40 ? K1 - 40 : 0))-$((KP + 20))"
  F=$R/render_${B}_s97/frames; C=$R/render_${B}_s97/cut; rm -rf $C; mkdir -p $C
  for f in $F/*.jpg; do n=$(basename $f .jpg); [ $((10#$n)) -le $CUT ] && ln -s $f $C/; done
  ( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1
    ffmpeg -y -loglevel error -framerate 20 -pattern_type glob -i "$C/*.jpg" -c:v libx264 -crf 18 -pix_fmt yuv420p \
      $R/${B}_s97_cut.mp4 ) && log "cut video ${B}_s97_cut.mp4 (to step $CUT)"
else
  log "s97 fails the progress or parting rule: no render; report first"
fi
log "chain16 end"
