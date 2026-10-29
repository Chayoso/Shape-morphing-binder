#!/bin/bash
# 200_d175_ls.sh -- queue job (agent D175): (1) the line-search trials of the column arm's segment windows split by term
# (keyframe-only ls_split record) and (2) the segment replay check on D164's measure (max of the start pair, the taped
# pairs and one ulp of E; keyframe segments only). Gate d: G1-G4 bit-SAME + the unit tests. Runs L_X97_a / L_X97_b /
# L_X99_a (= X's flags) PAR 3, then ls_read.py -> output/gpu/d175/auto/200_ls/ls.md
J=200_ls; TITLE="D175: the segment line-search split and the segment replay measure"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
D=$O/gates; RO=$O/auto/$J; mkdir -p $RO
log "start (card $CUDA_VISIBLE_DEVICES)"
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"
CMP=$B/tmp/d145/det_cmp12.py
F_coin="--tgt $G/d161/assets/d161coin.obj --render_tgt $G/d161/assets/d161L40coin.obj"
( envr $REPO
  $PY -m pytest -q -p no:cacheprovider tests/test_keyframe.py > $D/pytest_d.txt 2>&1 < /dev/null; echo $? > $D/pytest_d.rc
  $PY $DR --tgt assets/bunny.obj --out $D/E1 > $D/E1.log 2>&1 < /dev/null; $PY $CMP $G/d153/rt1/S1 $D/E1 > $D/cmpE1.txt 2>&1
  $PY $DR --tgt assets/bunny.obj $HU --out $D/E2 > $D/E2.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_H0 $D/E2 > $D/cmpE2.txt 2>&1
  $PY $DR $F_coin --lambda_ema 0 --settle_windows 2 --no-settle_relax --render_split u --out $D/E3 > $D/E3.log 2>&1 < /dev/null
  $PY $CMP $G/d172/gates/X2 $D/E3 > $D/cmpE3.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --settle_windows 3 --out $D/E4 > $D/E4.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_R0 $D/E4 > $D/cmpE4.txt 2>&1 )
ok=1; for k in 1 2 3 4; do tail -1 $D/cmpE$k.txt | grep -q ": SAME$" || ok=0; done; [ "$(cat $D/pytest_d.rc)" = 0 ] || ok=0
{ echo "# D175 gate d (repo_d175 + the segment replay measure + the ls_split record)"; for k in 1 2 3 4; do echo "- G$k: $(tail -1 $D/cmpE$k.txt)"; done
  echo "- unit tests: $(tail -1 $D/pytest_d.txt)"; echo; echo "VERDICT: $([ $ok = 1 ] && echo PASS || echo FAIL)"; } > $RO/gate.md
for k in 1 2 3 4; do rm -f $D/E${k}_frames12.npz; done
[ $ok = 1 ] || { log "gate d FAIL: stop ($RO/gate.md)"; exit 1; }
log "gate d PASS"
PAR=3
KF="--keyframe_time 1.0 --keyframe_compare column"
runp L_X97_a --tgt assets/bunny.obj --n 40000 --seed 97 $HU --keyframe $A/kf_s97.npz $KF
runp L_X97_b --tgt assets/bunny.obj --n 40000 --seed 97 $HU --keyframe $A/kf_s97.npz $KF
runp L_X99_a --tgt assets/bunny.obj --n 40000 --seed 99 $HU --keyframe $A/kf_s99.npz $KF
wait
( envr $REPO; export CUDA_VISIBLE_DEVICES=; $PY $W/ls_read.py $RO/ls.md $O/runs L_X97_a L_X97_b L_X99_a > $RO/ls.log 2>&1 < /dev/null )
log "ls read exit $?"
