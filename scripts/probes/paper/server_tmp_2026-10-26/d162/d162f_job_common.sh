# job_common.sh -- sourced by the D162e / D162f gpu1 queue jobs: run NAME REPO FLAGS... into OUT (skip if archived).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; W=$B/tmp/d162
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $O/status.txt; }
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1
         source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir; cd $B/$1; }
run() { local name=$1 repo=$2; shift 2
  [ -e $O/runs/${name}_frames12.npz ] && { log "R $name exists, skipped"; return 0; }
  log "R $name start ($repo: $*)"
  local t0=$(date +%s)
  ( envr $repo; $PY scripts/pipeline_run.py "$@" --out $O/runs/$name > $O/runs/$name.log 2>&1 < /dev/null; echo $? > $O/runs/$name.rc )
  log "R $name exit $(cat $O/runs/$name.rc) ($(( $(date +%s) - t0 )) s): $(grep -m1 'ARM render_full_dt_iso_nn:' $O/runs/$name.log | cut -c1-140)"
  grep -q "out of memory" $O/runs/$name.log && log "R $name: CUDA out of memory"
  return 0; }
