#!/bin/bash
# D147 chain 9: the scene tests on the split's fix, LB_v2 again with the angular momentum's split (LB_v2M), and the
# same-input repeats (seed 97: LB_v2N, o75_v2b, o100_v2b) that measure how often each variant parts in the air (S3's
# LB_v2 run did not repeat S2's parting); then the readings. Two cards before 14:40, one from 14:40 to 19:00.
REPO=$1
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d147
S=$REPO/scripts/probes/d147
L=/data/relcfd/chayo/physmorph_v2/gpu_locks
log() { echo "$(date '+%F %T') $*" >> $O/chain9.log; }
claim_and_run() {
  local tag=$1 cmd=$2
  while true; do
    hm=$(date +%H%M)
    if [ "$hm" -ge 1440 ] && [ "$hm" -lt 1900 ] && grep -qs "agent D147" $L/gpu*.lock/info; then sleep 30; continue; fi
    for g in 3 2 1 0; do
      [ -d $L/gpu$g.lock ] && continue
      used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      [ "$used" -gt 30000 ] && continue
      if bash $S/job.sh $g "$REPO" "$tag" "$cmd"; then
        rc=$(grep "^${tag}_end " $O/jobs.log | tail -1 | awk '{print $NF}'); log "$tag gpu$g exit $rc"; return $rc
      fi
    done
    sleep 20
  done
}
R=$O/runs; SP=scripts/probes/d147/specs
scene() { echo "\$PY scripts/scene_run.py --n 40000 --seed 97 --scene $SP/$1.json --flags=\"--assim 0\" --out $R/$2 > $R/$2.log 2>&1; echo $2 exit \$?"; }
log "chain9 start, repo $REPO"
( claim_and_run T9 "\$PY -m pytest -q tests/test_scene_bodies.py tests/test_scene_assim.py tests/test_scene_frame.py; rc=\$?; echo pytest exit \$rc; exit \$rc"
  claim_and_run S4a "$(scene scene_long scene40k_LB_v2M); $(scene scene_long_o100 scene40k_LB_o100_v2b)" ) &
sleep 60
claim_and_run S4b "$(scene scene_long_o75 scene40k_LB_o75_v2b); $(scene scene_long scene40k_LB_v2N)"
wait
export PHYSMORPH_RUN_REPO=$REPO PHYSMORPH_OUT=$O
( source $REPO/scripts/ops/hyde06_env.sh >/dev/null 2>&1; export CUDA_VISIBLE_DEVICES=""
  nice $PY $S/read40k.py $R scene40k_LB_v2,scene40k_LB_o75_v2,scene40k_LB_o100_v2 > $R/read_s3b.txt 2>&1
  nice $PY $S/read40k.py $R scene40k_LB_v2M,scene40k_LB_v2N,scene40k_LB_v2L,scene40k_LB_o75_v2b,scene40k_LB_o100_v2b > $R/read_s4.txt 2>&1 )
log "S4 read (read_s3b.txt, read_s4.txt)"
log "chain9 end"
