# job_common.sh -- sourced by the D166 / D167 queue jobs (tmp/gpu1_queue/6x_d166_*.sh, 1xx_d167_*.sh). Set O (the job's
# output root: output/gpu/d166 or d167) and J (the job's name) before sourcing.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; W=$B/tmp/d166; A=$G/d161/assets; REPO=repo_d166
mkdir -p $O/runs $O/auto/$J $W/tmpdir
log() { echo "$(date '+%m-%d %H:%M:%S') [$J] $*" >> $O/status.txt; }
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1
         source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir; cd $B/$1; }
# the 500 GB rule: /data/relcfd/chayo held $(cat $W/disk_base.txt) GB at queueing (2026-10-24, old bulk removed); runs stop at 495
disk_ok() { local mine=$(du -s -BG $G/d166 $G/d167 2>/dev/null | awk '{s += $1} END {print s + 0}'); [ $(( $(cat $W/disk_base.txt) + mine)) -lt 495 ]; }
run1() { local name=$1; shift
  [ -e $O/runs/$name.rc ] && { log "R $name exists, skipped"; return 0; }
  disk_ok || { log "R $name SKIPPED by the disk guard (base + ours >= 495 GB)"; return 0; }
  log "R $name start ($*)"; local t0=$(date +%s)
  ( envr $REPO; $PY scripts/pipeline_run.py "$@" --out $O/runs/$name > $O/runs/$name.log 2>&1 < /dev/null; echo $? > $O/runs/$name.rc )
  log "R $name exit $(cat $O/runs/$name.rc) ($(( $(date +%s) - t0 )) s)"
  grep -q "out of memory" $O/runs/$name.log && log "R $name: CUDA out of memory"; return 0; }
PAR=${PAR:-3}                                   # 40k runs side by side on the job's card
runp() { while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 5; done; run1 "$@" & sleep 2; }
F_coin="--tgt $A/d161coin.obj --render_tgt $A/d161L40coin.obj"
F_bunny="--tgt $A/d161smooth_bunny.obj --render_tgt $A/d161L40_bunny.obj"
F_band="--tgt assets/bunny.obj --appearance band"
HS="--lambda_ema 0 --settle_windows 30"         # job 50's held + settle
TW="--render_weight_scale 0"                    # the twin: dF_c + layer u (lambda = 0)
# scenes CODE FLAGS...: coin + bunny relief + band at 40k, seeds 97 / 99, runs $O/runs/{scene}_{CODE}_{seed}
scenes() { local c=$1; shift; for s in 97 99; do for sc in coin bunny band; do eval f=\$F_$sc
  runp ${sc}_${c}_$s $f --n 40000 --seed $s "$@"; done; done; }
gal() { local c=$1; shift; for m in bunny dragon armadilo bob homer teapot; do runp gal_${m}_${c}_97 --tgt assets/$m.obj --n 40000 --seed 97 "$@"; done; }
big() { local c=$1; shift; for m in bunny dragon; do run1 ${m}300k_${c}_97 --tgt assets/$m.obj --n 300000 --seed 97 "$@"; done; }
gate_wait() { for k in $(seq 1 240); do [ -e $G/d166/gates/V3.ok ] && return 0
  [ -e $G/d166/gates/V3.fail ] && { log "V3 failed: stop"; exit 1; }; sleep 15; done; log "no V3 gate after 60 min: stop"; exit 1; }
# after MODE ARM:TWIN...: the detached CPU reader writes $O/auto/$J/summary.md (tmp/d166/after.sh, d166_read.py)
after() { local mode=$1; shift; wait
  CUDA_VISIBLE_DEVICES= setsid nohup bash $W/after.sh $mode $O/auto/$J $O/runs "$TITLE" "$@" > $O/auto/$J/after.out 2>&1 < /dev/null &
  echo $! > $O/auto/$J/after.pid; log "after-reader pid $(cat $O/auto/$J/after.pid) ($mode $*)"; }
