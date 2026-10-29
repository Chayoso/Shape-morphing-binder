#!/bin/bash
# 200_d175_snd.sh -- queue job (agent D175): D175c (tmp/d175/PREREG_c.md, written before this run): the segment owned by
# its keyframe comparison alone (--keyframe_soundness final, on D175b's column comparison). Gate e: G1-G4 bit-SAME (defaults, Hu, Hu coin, settle
# with relaxation) + the unit tests. Then Z97 / Z99 (X's flags + --keyframe_soundness final), read against 208's D97 /
# D99 / T97 by the same reader and bars -> output/gpu/d175/auto/200_snd/summary.md
J=200_snd; TITLE="D175c: the segment owned by its keyframe comparison alone"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
D=$O/gates; RO=$O/auto/$J; RB=$O/runs_c; mkdir -p $RO $RB
log "start (card $CUDA_VISIBLE_DEVICES)"
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"
CMP=$B/tmp/d145/det_cmp12.py
F_coin="--tgt $G/d161/assets/d161coin.obj --render_tgt $G/d161/assets/d161L40coin.obj"
( envr $REPO
  $PY -m pytest -q -p no:cacheprovider tests/test_keyframe.py > $D/pytest_e.txt 2>&1 < /dev/null; echo $? > $D/pytest_e.rc
  $PY $DR --tgt assets/bunny.obj --out $D/F1 > $D/F1.log 2>&1 < /dev/null; $PY $CMP $G/d153/rt1/S1 $D/F1 > $D/cmpE5_1.txt 2>&1
  $PY $DR --tgt assets/bunny.obj $HU --out $D/F2 > $D/F2.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_H0 $D/F2 > $D/cmpE5_2.txt 2>&1
  $PY $DR $F_coin --lambda_ema 0 --settle_windows 2 --no-settle_relax --render_split u --out $D/F3 > $D/F3.log 2>&1 < /dev/null
  $PY $CMP $G/d172/gates/X2 $D/F3 > $D/cmpE5_3.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --settle_windows 3 --out $D/F4 > $D/F4.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_R0 $D/F4 > $D/cmpE5_4.txt 2>&1 )
ok=1; for k in 1 2 3 4; do tail -1 $D/cmpE5_$k.txt | grep -q ": SAME$" || ok=0; done; [ "$(cat $D/pytest_e.rc)" = 0 ] || ok=0
{ echo "# D175 gate e (repo_d175 + --keyframe_soundness)"; for k in 1 2 3 4; do echo "- G$k: $(tail -1 $D/cmpE5_$k.txt)"; done
  echo "- unit tests: $(tail -1 $D/pytest_e.txt)"; echo; echo "VERDICT: $([ $ok = 1 ] && echo PASS || echo FAIL)"; } > $RO/gate.md
for k in 1 2 3 4; do rm -f $D/F${k}_frames12.npz; done
[ $ok = 1 ] || { log "gate e FAIL: stop ($RO/gate.md)"; exit 1; }
log "gate e PASS"
for r in D_97 D_99 T_97; do for x in .json _frames12.npz .rc .log; do ln -sf ../runs/$r$x $RB/$r$x; done; done
PAR=2
KF="--keyframe_time 1.0 --keyframe_compare column --keyframe_soundness final"
O_RUNS=$O/runs; O=$O; for s in 97 99; do
  runp Z_$s --tgt assets/bunny.obj --n 40000 --seed $s $HU --keyframe $A/kf_s$s.npz $KF; done
wait
for s in 97 99; do for x in .json _frames12.npz .rc .log; do ln -sf ../runs/Z_$s$x $RB/S_$s$x; done; done
( envr $REPO; $PY $W/d175_read.py $RO $RB $A --runs D97,D99,S97,S99,T97 \
    --title "D175c: the segment owned by the count comparison alone (rows S = the Z arm, --keyframe_compare column --keyframe_soundness final), sphere -> bunny 40k" \
    > $RO/read.log 2>&1 < /dev/null )
log "reader exit $? -> $(grep '^VERDICT' $RO/summary.md 2>/dev/null)"
