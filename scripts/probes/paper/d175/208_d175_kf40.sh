#!/bin/bash
# 208_d175_kf40.sh -- queue job (agent D175): picture keyframes, the pre-registered first experiment (tmp/d175/PREREG.md).
# (0) keyframe E for seeds 97 / 99 (make_kf.py; prepare reads the card). (1) The V3 gate of repo_d175 (wip/d175): G1 the
# defaults bit-SAME as the tag (d153/rt1/S1), G2 Hu (settle 30) bit-SAME as d174/gates/ref_H0 (repo_d172), G3 Hu on the coin
# (settle 2) as d172/gates/X2, G4 the settle with its relaxation (settle 3) as d174/gates/ref_R0 (the settle loop now runs
# through the shared hold); the unit tests; the keyframe smoke K1-K3 (t_k = 0.5 s) read by kf_check.py. (2) The six runs,
# PAR 3. (3) The reader on this card -> output/gpu/d175/auto/208_kf40 and output/gpu/D175_SUMMARY.md.
J=208_kf40; TITLE="D175 keyframe E, sphere -> bunny 40k"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
D=$O/gates; rm -f $D/V3.ok $D/V3.fail
log "start (card $CUDA_VISIBLE_DEVICES)"
( envr $REPO; for s in 97 99; do [ -s $A/kf_s$s.npz ] || $PY $W/make_kf.py $A/kf_s$s.npz --seed $s > $A/kf_s$s.log 2>&1 < /dev/null; done )
[ -s $A/kf_s97.npz ] && [ -s $A/kf_s99.npz ] || { log "keyframes not built (see $A/kf_s*.log): stop"; exit 1; }
log "keyframes: $(grep '^\[kf\]' $A/kf_s97.log | cut -c1-300)"
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"
CMP=$B/tmp/d145/det_cmp12.py
F_coin="--tgt $G/d161/assets/d161coin.obj --render_tgt $G/d161/assets/d161L40coin.obj"
KS="--tgt assets/bunny.obj --n 40000 --seed 97 --animations 5 --lambda_ema 0 --settle_windows 2 --no-settle_relax --render_split u --keyframe $A/kf_s97.npz --keyframe_time 0.5"
if [ ! -e $D/V3.done ]; then
( envr $REPO
  $PY -m pytest -q -p no:cacheprovider tests/test_keyframe.py > $D/pytest.txt 2>&1 < /dev/null; echo $? > $D/pytest.rc
  $PY $DR --tgt assets/bunny.obj --out $D/G1 > $D/G1.log 2>&1 < /dev/null; $PY $CMP $G/d153/rt1/S1 $D/G1 > $D/cmp1.txt 2>&1
  $PY $DR --tgt assets/bunny.obj $HU --out $D/G2 > $D/G2.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_H0 $D/G2 > $D/cmp2.txt 2>&1
  $PY $DR $F_coin --lambda_ema 0 --settle_windows 2 --no-settle_relax --render_split u --out $D/G3 > $D/G3.log 2>&1 < /dev/null
  $PY $CMP $G/d172/gates/X2 $D/G3 > $D/cmp3.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --settle_windows 3 --out $D/G4 > $D/G4.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_R0 $D/G4 > $D/cmp4.txt 2>&1
  $PY scripts/pipeline_run.py $KS --telemetry --out $D/K1 > $D/K1.log 2>&1 < /dev/null; echo $? > $D/K1.rc
  $PY scripts/pipeline_run.py $KS $TW --out $D/K2 > $D/K2.log 2>&1 < /dev/null; echo $? > $D/K2.rc
  $PY scripts/pipeline_run.py $KS --keyframe_geometry final --out $D/K3 > $D/K3.log 2>&1 < /dev/null; echo $? > $D/K3.rc
  $PY $W/kf_check.py $D/K1 $D/K2 $D/K3 > $D/kf_check.txt 2>&1 )
touch $D/V3.done
fi
ok=1; rows=""
for k in 1 2 3 4; do r=$(tail -1 $D/cmp$k.txt 2>/dev/null); echo "$r" | grep -q ": SAME$" || ok=0; rows="$rows| G$k | $r |\n"; done
pr=$(cat $D/pytest.rc 2>/dev/null); [ "$pr" = 0 ] || ok=0
for k in 1 2 3; do [ "$(cat $D/K$k.rc 2>/dev/null)" = 0 ] || ok=0; done
kc=$(tail -1 $D/kf_check.txt 2>/dev/null); [ "$kc" = "VERDICT: PASS" ] || ok=0
v=FAIL; [ $ok = 1 ] && v=PASS
mkdir -p $O/auto/208_gate
{ echo "# D175 V3 gate (repo_d175 = repo_d172 + the storyboard; deterministic bunny 40k seed 97, 4 windows)"; echo
  echo "| check | result |"; echo "|---|---|"; printf "$rows"
  echo "| unit tests (tests/test_keyframe.py) | exit $pr: $(tail -1 $D/pytest.txt 2>/dev/null) |"
  echo "| keyframe smoke K1 / K2 / K3 (full / twin / naive, t_k = 0.5 s) | exit $(cat $D/K1.rc 2>/dev/null) / $(cat $D/K2.rc 2>/dev/null) / $(cat $D/K3.rc 2>/dev/null) |"
  echo "| kf_check (segment clock, segment merit, held lambda, epoch) | $kc |"; echo
  echo "G1 defaults vs d153/rt1/S1; G2 Hu (settle 30) vs d174 ref_H0; G3 Hu coin (settle 2) vs d172 X2; G4 settle 3 with relaxation vs d174 ref_R0"
  echo; echo '```'; cat $D/kf_check.txt 2>/dev/null; echo '```'; echo; echo "VERDICT: $v"; } > $O/auto/208_gate/summary.md
for k in 1 2 3 4; do rm -f $D/G${k}_frames12.npz; done
[ $v = PASS ] && touch $D/V3.ok || { touch $D/V3.fail; log "V3 gate FAIL: no runs (output/gpu/d175/auto/208_gate/summary.md)"; exit 1; }
log "V3 gate PASS; runs start"
KF="--keyframe_time 1.0"
for s in 97 99; do runp D_$s --tgt assets/bunny.obj --n 40000 --seed $s $HU; done
for s in 97 99; do runp S_$s --tgt assets/bunny.obj --n 40000 --seed $s $HU --keyframe $A/kf_s$s.npz $KF; done
runp T_97 --tgt assets/bunny.obj --n 40000 --seed 97 $HU --keyframe $A/kf_s97.npz $KF $TW
runp N_97 --tgt assets/bunny.obj --n 40000 --seed 97 $HU --keyframe $A/kf_s97.npz $KF --keyframe_geometry final
wait
log "runs done; reader"
( envr $REPO; $PY $W/d175_read.py $O/auto/$J $O/runs $A --summary $G/D175_SUMMARY.md > $O/auto/$J/read.log 2>&1 < /dev/null )
log "reader exit $? -> $(tail -1 $O/auto/$J/summary.md 2>/dev/null)"
