#!/bin/bash
# D147 chain 19, the final 300k scene on the re-freeze (wip/d147 = tag freeze-2026-10-15 merged with the scene;
# pre-registered in docs/experiments.md, D147, 2026-10-15, before any of its runs). ONE hyde06 card, held from the suite
# to the last render: a card with no lock and under 2000 MiB for 3 minutes (D145's pool reclaims its cards within 90 s
# and keeps its priority), released by a trap.
#   1. the full suite on REPO (pytest's exit status; any failure stops the chain before a run)
#   2. final300k_s97 (render arm), final300k_s97_off (--render_weight_scale 0), final300k_s99: scene_throw, 2 x 150k,
#      --assim 0, every 12th step kept (the interim's settings); the re-freeze's defaults (D146b, D149) from run_args
#   3. per run as it ends, on the CPU (detached): the timeline (read40k.py), the amended cut (cut.py), the pieces apart
#      over the whole run (apart.py 0-16 s) and clean.py; the interim s97 re-read the same way (0-16 s) for the
#      comparison; after the three runs, on the card: progress (0.15 s), the judge, the table
#   4. the renders (studio look, render_scene.py zbs), per run: the cut (the still camera over the whole run, kept
#      frames 0..cut) and the contact close-up (camera=contact over the first contact - 0.5 s .. the parting + 0.5 s)
#      at 4K and 1080p, the whole run at 1080p; each render's frames deleted once its video is written (rows.txt kept)
#   nohup bash scripts/probes/d147/chain19.sh REPO > /dev/null 2>&1 &
REPO=$1
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/d147; R=$O/runs; L=$B/gpu_locks
S=$REPO/scripts/probes/d147; SPEC=scripts/probes/d147/specs/scene_throw.json
LOG=$O/chain19.log
T=final300k_s97,final300k_s97_off,final300k_s99
I=scene300k_scene_throw_s97                                   # the interim showcase (D135, 2026-10-14)
log() { echo "$(date '+%F %T') $*" >> $LOG; }
source $B/tmp/d146/cache_env.sh
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
source $REPO/scripts/ops/hyde06_env.sh > /dev/null            # $PY, the caches on /data, cd $REPO
gpu=""
release() { [ -n "$gpu" ] && rm -rf $L/gpu$gpu.lock && log "released gpu$gpu"; gpu=""; }
trap 'release' EXIT
trap 'exit 143' TERM INT HUP
claim() {
  declare -A since
  log "waiting for a card (no lock, under 2000 MiB, for 3 minutes)"
  while [ -z "$gpu" ]; do
    now=$(date +%s)
    for g in 3 2 1 0; do
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      if [ -d $L/gpu$g.lock ] || [ "$used" -gt 2000 ]; then since[$g]=""; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      if [ $((now - ${since[$g]})) -ge 180 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        gpu=$g
        printf "agent D147\nrun d147/chain19 final 300k scene (suite, s97, twin, s99, readings, renders; ~3 h)\npid %s\nstart %s\n" \
          "$$" "$(date '+%Y-%m-%d %H:%M:%S %Z')" > $L/gpu$g.lock/info
        break
      fi
    done
    [ -z "$gpu" ] && sleep 30
  done
  export CUDA_VISIBLE_DEVICES=$gpu
  log "claimed gpu$gpu"
}
cpu_read() {                                                    # TAG: the CPU readings of one finished run
  ( export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
    nice -n 10 $PY $S/read40k.py $R $1 > $R/read_$1.txt 2>&1
    nice -n 10 $PY $S/cut.py $R $1 > $R/cut_$1.txt 2>&1
    log "read, cut $1: $(grep -o '"t_cut": [0-9.]*' $R/cut_$1.txt)"
    nice -n 10 $PY $S/apart.py $R $1 0 16 > $R/apart_$1.txt 2>&1
    nice -n 10 $PY $S/clean.py $R $R/cut_$1.txt $1 > $R/clean_$1.txt 2>&1
    log "apart, clean $1: $(cat $R/clean_$1.txt | tr '\n' ' ')" ) &
}
log "chain19 start, repo $REPO ($(cat $REPO/DEPLOYED_FROM))"
# the interim s97 re-read over the whole run (its 2026-10-14 reading covered 8-16 s), CPU, detached
mkdir -p $R/reread_interim
( export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
  nice -n 10 $PY $S/cut.py $R $I > $R/reread_interim/cut_$I.txt 2>&1
  nice -n 10 $PY $S/apart.py $R $I 0 16 > $R/reread_interim/apart_$I.txt 2>&1
  nice -n 10 $PY $S/clean.py $R/reread_interim $R/reread_interim/cut_$I.txt $I > $R/reread_interim/clean_$I.txt 2>&1
  log "interim re-read: $(cat $R/reread_interim/clean_$I.txt | tr '\n' ' ')" ) &
claim
$PY -m pytest -q -p no:cacheprovider tests > $O/T20_suite.log 2>&1
rc=$?
log "suite exit $rc: $(tail -1 $O/T20_suite.log)"
[ $rc -eq 0 ] || { log "the suite failed: the chain stops before any run"; exit 1; }
for spec in "final300k_s97 97" "final300k_s97_off 97 --render_weight_scale 0" "final300k_s99 99"; do
  set -- $spec
  tag=$1 seed=$2; shift 2
  log "$tag start"
  $PY scripts/scene_run.py --n 150000 --seed $seed --scene $SPEC --flags="--assim 0" --save_stride 12 "$@" \
    --out $R/$tag > $R/$tag.log 2>&1
  rc=$?
  log "$tag exit $rc: $(grep -c 'window .*/96' $R/$tag.log) window lines; $(tail -1 $R/$tag.log | cut -c1-200)"
  [ $rc -eq 0 ] && [ -f $R/$tag.npz ] && cpu_read $tag
done
done_runs=$(for t in ${T//,/ }; do [ -f $R/$t.npz ] && echo $t; done | wc -l)
until [ $(grep -c "read, cut final300k" $LOG) -ge $done_runs ]; do sleep 30; done      # the timeline and the cut
for f in read cut; do : > $R/${f}_final.txt; for t in ${T//,/ }; do cat $R/${f}_$t.txt >> $R/${f}_final.txt 2>/dev/null; done; done
$PY $S/progress.py $R $T 0.15 > $R/progress_final.txt 2>&1
log "progress exit $?"
$PY $S/judge.py $R/read_final.txt $R/progress_final.txt $T > $R/judge_final.txt 2>&1
log "judge: $(tail -1 $R/judge_final.txt)"
render() {                                                      # TAG NAME K0 K1 RES [+camera=...]
  local tag=$1 name=$2 k0=$3 k1=$4 res=$5; shift 5
  NAME=$name bash $S/render_scene_job.sh $R $tag +stride=1 +frames=$k0-$k1 +res=$res "$@" >> $LOG 2>&1
  log "render $name exit $? ($(ls $R/render_$name/frames 2>/dev/null | wc -l) frames)"
  rm -rf $R/render_$name/frames
}
for t in ${T//,/ }; do
  [ -f $R/$t.npz ] || continue
  read CUT K0 K1 <<< "$($PY $S/spans.py $R $t $R/read_$t.txt $R/cut_$t.txt)"
  log "$t spans: cut kept frame $CUT, contact close-up $K0-$K1"
  [ -n "$K1" ] || continue
  render $t ${t}_4k_cut 0 $CUT 2160
  render $t ${t}_4k_contact $K0 $K1 2160 +camera=contact
  render $t ${t}_1080p_cut 0 $CUT 1080
  render $t ${t}_1080p_contact $K0 $K1 1080 +camera=contact
  render $t ${t}_1080p_full 0 100000 1080
done
release                                                         # the card goes back before the CPU tail
until [ $(grep -c "apart, clean final300k" $LOG) -ge $done_runs ]; do sleep 60; done   # apart.py and clean.py
: > $R/clean_final.txt; for t in ${T//,/ }; do cat $R/clean_$t.txt >> $R/clean_final.txt 2>/dev/null; done
( export CUDA_VISIBLE_DEVICES=""
  $PY $S/table.py $R $T $R/read_final.txt $R/progress_final.txt $R/judge_final.txt $R/cut_final.txt $R/clean_final.txt \
    > $R/table_final.txt 2>&1
  log "table exit $?" )
wait
log "chain19 end"
