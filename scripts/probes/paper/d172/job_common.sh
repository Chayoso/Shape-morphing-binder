# job_common.sh -- sourced by the D170 queue jobs (tmp/gpu1_queue/18x_d172_*.sh). Set J (the job's name) and TITLE first.
# D172 = the render gradient split by control block (repo_d172 = repo_d170 + --render_split + the render_x paint fix).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d172; W=$B/tmp/d172; A1=$G/d161/assets; A=$O/assets; REPO=repo_d172
mkdir -p $O/runs $O/auto/$J $W/tmpdir
log() { echo "$(date '+%m-%d %H:%M:%S') [$J] $*" >> $O/status.txt; }
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1
         source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir; cd $B/$1; }
# the 500 GB rule: /data/relcfd/chayo held $(cat $W/disk_base.txt) GB before D172 (after the 2026-10-28 21:20 cleanup); runs stop at 495
disk_ok() { local mine=$(du -s -BG $O 2>/dev/null | awk '{print $1 + 0}'); [ $(( $(cat $W/disk_base.txt) + mine )) -lt 495 ]; }
run1() { local name=$1; shift
  [ -e $O/runs/$name.rc ] && { log "R $name exists, skipped"; return 0; }
  disk_ok || { log "R $name SKIPPED by the disk guard (base + ours >= 495 GB)"; return 0; }
  log "R $name start ($*)"; local t0=$(date +%s)
  ( envr $REPO; $PY scripts/pipeline_run.py "$@" --out $O/runs/$name > $O/runs/$name.log 2>&1 < /dev/null; echo $? > $O/runs/$name.rc )
  log "R $name exit $(cat $O/runs/$name.rc) ($(( $(date +%s) - t0 )) s)"
  grep -q "out of memory" $O/runs/$name.log && log "R $name: CUDA out of memory"; return 0; }
PAR=${PAR:-2}
runp() { while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 5; done; run1 "$@" & sleep 2; }
RX="--lambda_ema 0 --settle_windows 30 --no-settle_relax"   # D166's Rx
TW="--render_weight_scale 0"                                 # the twin: dF_c + layer u (lambda = 0)
wait_file() { for k in $(seq 1 540); do [ -e "$1" ] && return 0; sleep 20; done; log "no $1 after 3 h: stop"; exit 1; }
HU="$RX --render_split u"                                    # D172: held on u, the rule on dFc
HF="$RX --render_split dFc"                                  # D172: held on dFc, the rule on u
gate_wait() { for k in $(seq 1 360); do [ -e $O/gates/V3.ok ] && return 0; [ -e $O/gates/V3.fail ] && { log "V3 failed: stop"; exit 1; }
  sleep 20; done; log "no V3 gate after 2 h: stop"; exit 1; }
