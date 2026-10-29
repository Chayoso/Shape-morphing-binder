#!/bin/bash
# 200_d175_col.sh -- queue job (agent D175): D175b (tmp/d175/PREREG_b.md, written before this run): the keyframe compared as
# the particles' projected count picture (--keyframe_compare column). Gate c: G1-G4 bit-SAME (defaults, Hu, Hu coin, settle
# with relaxation) + the unit tests. Then X97 / X99 (208's S flags + --keyframe_compare column), read against 208's D97 /
# D99 / T97 by the same reader and bars -> output/gpu/d175/auto/200_col/summary.md
J=200_col; TITLE="D175b: the column keyframe comparison"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
D=$O/gates; RO=$O/auto/$J; RB=$O/runs_b; mkdir -p $RO $RB
log "start (card $CUDA_VISIBLE_DEVICES)"
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"
CMP=$B/tmp/d145/det_cmp12.py
F_coin="--tgt $G/d161/assets/d161coin.obj --render_tgt $G/d161/assets/d161L40coin.obj"
( envr $REPO
  $PY -m pytest -q -p no:cacheprovider tests/test_keyframe.py > $D/pytest_c.txt 2>&1 < /dev/null; echo $? > $D/pytest_c.rc
  $PY $DR --tgt assets/bunny.obj --out $D/C1 > $D/C1.log 2>&1 < /dev/null; $PY $CMP $G/d153/rt1/S1 $D/C1 > $D/cmpC1.txt 2>&1
  $PY $DR --tgt assets/bunny.obj $HU --out $D/C2 > $D/C2.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_H0 $D/C2 > $D/cmpC2.txt 2>&1
  $PY $DR $F_coin --lambda_ema 0 --settle_windows 2 --no-settle_relax --render_split u --out $D/C3 > $D/C3.log 2>&1 < /dev/null
  $PY $CMP $G/d172/gates/X2 $D/C3 > $D/cmpC3.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --settle_windows 3 --out $D/C4 > $D/C4.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_R0 $D/C4 > $D/cmpC4.txt 2>&1 )
ok=1; for k in 1 2 3 4; do tail -1 $D/cmpC$k.txt | grep -q ": SAME$" || ok=0; done; [ "$(cat $D/pytest_c.rc)" = 0 ] || ok=0
{ echo "# D175 gate c (repo_d175 + --keyframe_compare column)"; for k in 1 2 3 4; do echo "- G$k: $(tail -1 $D/cmpC$k.txt)"; done
  echo "- unit tests: $(tail -1 $D/pytest_c.txt)"; echo; echo "VERDICT: $([ $ok = 1 ] && echo PASS || echo FAIL)"; } > $RO/gate.md
for k in 1 2 3 4; do rm -f $D/C${k}_frames12.npz; done
[ $ok = 1 ] || { log "gate c FAIL: stop ($RO/gate.md)"; exit 1; }
log "gate c PASS"
for r in D_97 D_99 T_97; do for x in .json _frames12.npz .rc .log; do ln -sf ../runs/$r$x $RB/$r$x; done; done
PAR=2
KF="--keyframe_time 1.0 --keyframe_compare column"
O_RUNS=$O/runs; O=$O; for s in 97 99; do
  runp X_$s --tgt assets/bunny.obj --n 40000 --seed $s $HU --keyframe $A/kf_s$s.npz $KF; done
wait
for s in 97 99; do for x in .json _frames12.npz .rc .log; do ln -sf ../runs/X_$s$x $RB/S_$s$x; done; done
( envr $REPO; $PY $W/d175_read.py $RO $RB $A --runs D97,D99,S97,S99,T97 \
    --title "D175b: keyframe E compared as projected counts (rows S = the X arm, --keyframe_compare column), sphere -> bunny 40k" \
    > $RO/read.log 2>&1 < /dev/null )
log "reader exit $? -> $(grep '^VERDICT' $RO/summary.md 2>/dev/null)"
