# d174_common.sh -- sourced by agent D174's queue jobs (tmp/gpu1_queue/2NN_d174_*.sh). Set J (the job's name) first.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d174; W=$B/tmp/d174; A1=$G/d161/assets; A2=$G/d172/assets
mkdir -p $O/runs $O/auto $W/tmpdir
log() { echo "$(date '+%F %T') [$J] $*" >> $O/status.txt; }
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1
         source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir PYTHONUNBUFFERED=1; cd $B/$1; }
# the 500 GB rule: stop launching runs at 490 GB on /data/relcfd/chayo
disk_ok() { [ $(du -s -BG $B/.. 2>/dev/null | awk '{print $1 + 0}') -lt 490 ]; }
run1() { local rp=$1 name=$2; shift 2
  [ -e $O/runs/$name.rc ] && { log "R $name exists, skipped"; return 0; }
  log "R $name start ($rp: $*)"; local t0=$(date +%s)
  ( envr $rp; $PY scripts/pipeline_run.py "$@" --out $O/runs/$name < /dev/null 2>&1 | while IFS= read -r l; do printf '%s %s\n' "$EPOCHREALTIME" "$l"; done > $O/runs/$name.log; echo ${PIPESTATUS[0]} > $O/runs/$name.rc )
  log "R $name exit $(cat $O/runs/$name.rc) ($(( $(date +%s) - t0 )) s)"
  grep -q "out of memory" $O/runs/$name.log && log "R $name: CUDA out of memory"; return 0; }
PAR=${PAR:-2}
runp() { while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 5; done; run1 "$@" & sleep 2; }
HU="--lambda_ema 0 --settle_windows 30 --no-settle_relax --render_split u"     # D172's Hu (the method in use)
F_coin="--tgt $A1/d161coin.obj --render_tgt $A1/d161L40coin.obj"
F_bunny="--tgt $A1/d161smooth_bunny.obj --render_tgt $A1/d161L40_bunny.obj"
F_band="--tgt assets/bunny.obj --appearance band"
F_scl="--tgt $G/d170/assets/d170smooth_dragon.obj --render_tgt $A2/d172SE300_dragon.obj --n 300000 --seed 97"
gate_wait() { for k in $(seq 1 720); do [ -e $O/gates/$1.ok ] && return 0; [ -e $O/gates/$1.fail ] && { log "gate $1 failed: stop"; exit 1; }
  sleep 30; done; log "no gate $1 after 6 h: stop"; exit 1; }
