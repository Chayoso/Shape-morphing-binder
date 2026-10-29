#!/bin/bash
# 21_d164_v1b.sh -- gpu1 queue job (agent D164; amendment 2): V1 again on the code with the scene fix (11aab852 + the
# SceneOptimizer keeping the tag's commit), as the full suite on repo_d164b (fc730244, exit status), and the shared card's control
# for the one MPM test that asserts bit-identical CUDA rollouts at rest
# (tests/test_volume_exact.py::test_the_eval_trajectory_replays_bit_for_bit, off-cuda and carried-cuda): 10 repeats on
# repo_d164b and 10 on repo_d162tag (the tag as tagged, 6c9b0e9) in this job, on this card. Outputs output/gpu/d164/gates/.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d164; V=$O/gates; W=$B/tmp/d164
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
source $B/tmp/d146/cache_env.sh
envr() { export PHYSMORPH_RUN_REPO=$B/$1 PHYSMORPH_OUT=$G; source $B/$1/scripts/ops/hyde06_env.sh > /dev/null 2>&1; export TMPDIR=$W/tmpdir; cd $B/$1; }
log "V1b start ($(cat $B/repo_d164b/VERSION))"
( envr repo_d164b; $PY -m pytest -q -p no:cacheprovider tests > $V/suite_b.log 2>&1 < /dev/null; echo $? > $V/suite_b.rc )
log "V1b suite exit $(cat $V/suite_b.rc): $(tail -1 $V/suite_b.log); failures: $(grep '^FAILED' $V/suite_b.log | cut -c8-110 | tr '\n' ' ')"
TR="tests/test_volume_exact.py::test_the_eval_trajectory_replays_bit_for_bit"
for repo in repo_d164b repo_d162tag; do
  f=0
  for k in $(seq 1 10); do
    ( envr $repo; $PY -m pytest -q -p no:cacheprovider "$TR" -k cuda > $V/atrest_${repo}_$k.log 2>&1 < /dev/null; echo $? > $V/atrest_${repo}_$k.rc )
    [ "$(cat $V/atrest_${repo}_$k.rc)" = 0 ] || f=$((f + 1))
  done
  echo $f > $V/atrest_${repo}.fails
  log "V1b control: $TR [*-cuda] x10 on $repo: $f runs with a failure"
done
log "V1b done"
