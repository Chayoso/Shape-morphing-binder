#!/bin/bash
# 208_d175_next.sh -- queue job (agent D175): the three next measurements of tmp/d175/MEAS_a.md (stated before the runs).
# (0) the V3 gate of the keyframe-only replay record: G1-G4 bit-SAME as in job 208, the unit tests. (a) pull_meas.py on
# S97's dumps and at the source state. (b) reruns R_S99_1 / R_S99_2 / R_S97_1 with the 208 flags (the replay split
# recorded). (c) P_dFc_y13: dF_c alone (lambda 0, u off), sphere -> the stretched sphere as a 3D target. PAR 4 on this card.
# Readers: output/gpu/d175/auto/208_next/{pull.md, next.md}.
J=208_next; TITLE="D175 next measurements (a) (b) (c)"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
D=$O/gates; RO=$O/auto/$J; mkdir -p $RO
log "start (card $CUDA_VISIBLE_DEVICES)"
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"
CMP=$B/tmp/d145/det_cmp12.py
F_coin="--tgt $G/d161/assets/d161coin.obj --render_tgt $G/d161/assets/d161L40coin.obj"
( envr $REPO
  $PY -m pytest -q -p no:cacheprovider tests/test_keyframe.py > $D/pytest_b.txt 2>&1 < /dev/null; echo $? > $D/pytest_b.rc
  $PY $DR --tgt assets/bunny.obj --out $D/H1 > $D/H1.log 2>&1 < /dev/null; $PY $CMP $G/d153/rt1/S1 $D/H1 > $D/cmpH1.txt 2>&1
  $PY $DR --tgt assets/bunny.obj $HU --out $D/H2 > $D/H2.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_H0 $D/H2 > $D/cmpH2.txt 2>&1
  $PY $DR $F_coin --lambda_ema 0 --settle_windows 2 --no-settle_relax --render_split u --out $D/H3 > $D/H3.log 2>&1 < /dev/null
  $PY $CMP $G/d172/gates/X2 $D/H3 > $D/cmpH3.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --settle_windows 3 --out $D/H4 > $D/H4.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_R0 $D/H4 > $D/cmpH4.txt 2>&1 )
ok=1; for k in 1 2 3 4; do tail -1 $D/cmpH$k.txt | grep -q ": SAME$" || ok=0; done; [ "$(cat $D/pytest_b.rc)" = 0 ] || ok=0
{ echo "# D175 gate b (repo_d175 + the keyframe-only replay record)"; for k in 1 2 3 4; do echo "- G$k: $(tail -1 $D/cmpH$k.txt)"; done
  echo "- unit tests: $(tail -1 $D/pytest_b.txt)"; echo; echo "VERDICT: $([ $ok = 1 ] && echo PASS || echo FAIL)"; } > $RO/gate.md
for k in 1 2 3 4; do rm -f $D/H${k}_frames12.npz; done
[ $ok = 1 ] || { log "gate b FAIL: stop ($RO/gate.md)"; exit 1; }
log "gate b PASS"
awk '/^v /{printf "v %.7f %.7f %.7f\n", $2 / sqrt(1.3), $3 * 1.3, $4 / sqrt(1.3); next} /^(mtllib|usemtl)/{next} {print}' \
  $B/$REPO/assets/isosphere.obj > $A/isosphere_y13.obj
meas() { ( envr $REPO; $PY $W/pull_meas.py $RO/pull.md $O/forensic/S97_dump $A/kf_s97.npz > $RO/pull.log 2>&1 < /dev/null )
         log "(a) pull_meas exit $? -> $(grep '^VERDICT' $RO/pull.md 2>/dev/null)"; }
PAR=4
meas & sleep 2
KF="--keyframe_time 1.0"
runp R_S99_1 --tgt assets/bunny.obj --n 40000 --seed 99 $HU --keyframe $A/kf_s99.npz $KF
runp R_S99_2 --tgt assets/bunny.obj --n 40000 --seed 99 $HU --keyframe $A/kf_s99.npz $KF
runp R_S97_1 --tgt assets/bunny.obj --n 40000 --seed 97 $HU --keyframe $A/kf_s97.npz $KF
runp P_dFc_y13 --src assets/isosphere.obj --tgt $A/isosphere_y13.obj --n 40000 --seed 97 $HU $TW --u_off
wait
( envr $REPO; export CUDA_VISIBLE_DEVICES=; $PY $W/next_read.py $RO/next.md $O/runs > $RO/next.log 2>&1 < /dev/null )
log "(b) (c) read exit $? -> $(grep '^VERDICT' $RO/next.md 2>/dev/null)"
