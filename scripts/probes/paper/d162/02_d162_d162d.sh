#!/bin/bash
# 02_d162_d162d.sh -- gpu1 queue job (agent D162): D162d's gates (V2 the options' tests, V3 SAME at the defaults, V4 two
# four-window smokes) and, if they pass, the 8 runs of d162d_runs.txt (repo_d162ds, kept frames). The runner exports
# CUDA_VISIBLE_DEVICES=1 and holds the lock. Status in output/gpu/d162d/status.txt; the reader is d162d_after.sh (CPU).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d162d; W=$B/tmp/d162; V=$O/gates; A161=$G/d161/assets
mkdir -p $V $O/runs40 $W/tmpdir
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $O/status.txt; }
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1
         source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir; cd $B/$1; }
log "job 02 start (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, code $(cat $B/repo_d162d/VERSION))"
( envr repo_d162d; $PY -m pytest -q -p no:cacheprovider tests/test_refreeze_reference.py tests/test_d162.py \
    tests/test_render_tgt.py tests/test_proximity_berth.py tests/test_appearance.py tests/test_appearance_pipeline.py \
    tests/test_render_target_shell.py tests/test_frozen_recipe.py > $V/refs.log 2>&1 < /dev/null; echo $? > $V/refs.rc )
log "V2 references and options' tests exit $(cat $V/refs.rc): $(tail -1 $V/refs.log)"
( envr repo_d162ds; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 --tgt assets/bunny.obj \
    --out $V/S1 > $V/S1.log 2>&1 < /dev/null; echo $? > $V/S1.rc )
( envr repo_d162d; $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $V/S1 > $V/cmp_rt1_S1.txt 2>&1 )
log "V3 rt1 S1 vs repo_d162ds defaults (exit $(cat $V/S1.rc)): $(tail -1 $V/cmp_rt1_S1.txt)"
smoke() { local name=$1; shift; ( envr repo_d162ds; $PY scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 \
    --out $V/$name "$@" > $V/$name.log 2>&1 < /dev/null; echo $? > $V/$name.rc )
  log "V4 $name exit $(cat $V/$name.rc): $(grep -m1 'pitch scale (D162d)' $V/$name.log | cut -c1-200)"; }
smoke Lp_coin --tgt $A161/d161coin.obj --render_tgt $A161/d161L40coin.obj --render_weight_ref floor_pitch
smoke Lp_band --tgt assets/bunny.obj --appearance band --render_weight_ref floor_pitch
ok=1
[ "$(cat $V/refs.rc)" = 0 ] || ok=0
tail -1 $V/cmp_rt1_S1.txt | grep -q ": SAME$" || ok=0
for k in Lp_coin Lp_band; do [ "$(cat $V/$k.rc)" = 0 ] || ok=0; grep -q "pitch scale (D162d)" $V/$k.log || ok=0; done
if [ $ok = 0 ]; then log "V STOP: a gate failed (job 02 ends; the queue goes on)"; exit 1; fi
log "V gates passed"
while read -r name flags; do
  [ -z "$name" ] && continue
  case "$name" in \#*) continue;; esac
  [ -e $O/runs40/${name}_frames12.npz ] && { log "R $name exists, skipped"; continue; }
  log "R $name start"
  ( envr repo_d162ds; $PY scripts/pipeline_run.py $flags --out $O/runs40/$name > $O/runs40/$name.log 2>&1 < /dev/null
    echo $? > $O/runs40/$name.rc )
  log "R $name exit $(cat $O/runs40/$name.rc): $(grep -m1 'ARM render_full_dt_iso_nn:' $O/runs40/$name.log | cut -c1-150)"
done < $W/d162d_runs.txt
log "R done (d162d_runs.txt)"
