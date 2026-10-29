#!/bin/bash
# 200_d175_w.sh -- queue job (agent D175): D175d (tmp/d175/PREREG_d.md, written before this run): D175b's count
# comparison with the segment replay measure (no new definition; the per-block render share recorded). Gate f: G1-G4 bit-SAME (defaults, Hu, Hu coin, settle
# with relaxation) + the unit tests. Then W97 / W99 (X's flags), read against 208's D97 /
# D99 / T97 by the same reader and bars -> output/gpu/d175/auto/200_w/summary.md; on PASS it queues the 300k showcase (209_d175_show300)
J=200_w; TITLE="D175d: the count comparison with the segment replay measure"
source /data/relcfd/chayo/physmorph_v2/tmp/d175/job_common.sh
D=$O/gates; RO=$O/auto/$J; RB=$O/runs_d; mkdir -p $RO $RB
log "start (card $CUDA_VISIBLE_DEVICES)"
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"
CMP=$B/tmp/d145/det_cmp12.py
F_coin="--tgt $G/d161/assets/d161coin.obj --render_tgt $G/d161/assets/d161L40coin.obj"
( envr $REPO
  $PY -m pytest -q -p no:cacheprovider tests/test_keyframe.py > $D/pytest_f.txt 2>&1 < /dev/null; echo $? > $D/pytest_f.rc
  $PY $DR --tgt assets/bunny.obj --out $D/H61 > $D/H61.log 2>&1 < /dev/null; $PY $CMP $G/d153/rt1/S1 $D/H61 > $D/cmpF6_1.txt 2>&1
  $PY $DR --tgt assets/bunny.obj $HU --out $D/H62 > $D/H62.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_H0 $D/H62 > $D/cmpF6_2.txt 2>&1
  $PY $DR $F_coin --lambda_ema 0 --settle_windows 2 --no-settle_relax --render_split u --out $D/H63 > $D/H63.log 2>&1 < /dev/null
  $PY $CMP $G/d172/gates/X2 $D/H63 > $D/cmpF6_3.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --settle_windows 3 --out $D/H64 > $D/H64.log 2>&1 < /dev/null; $PY $CMP $G/d174/gates/ref_R0 $D/H64 > $D/cmpF6_4.txt 2>&1 )
ok=1; for k in 1 2 3 4; do tail -1 $D/cmpF6_$k.txt | grep -q ": SAME$" || ok=0; done; [ "$(cat $D/pytest_f.rc)" = 0 ] || ok=0
{ echo "# D175 gate f (repo_d175 + the per-block render share record)"; for k in 1 2 3 4; do echo "- G$k: $(tail -1 $D/cmpF6_$k.txt)"; done
  echo "- unit tests: $(tail -1 $D/pytest_f.txt)"; echo; echo "VERDICT: $([ $ok = 1 ] && echo PASS || echo FAIL)"; } > $RO/gate.md
for k in 1 2 3 4; do rm -f $D/H6${k}_frames12.npz; done
[ $ok = 1 ] || { log "gate f FAIL: stop ($RO/gate.md)"; exit 1; }
log "gate f PASS"
for r in D_97 D_99 T_97; do for x in .json _frames12.npz .rc .log; do ln -sf ../runs/$r$x $RB/$r$x; done; done
PAR=2
KF="--keyframe_time 1.0 --keyframe_compare column"
O_RUNS=$O/runs; O=$O; for s in 97 99; do
  runp W_$s --tgt assets/bunny.obj --n 40000 --seed $s $HU --keyframe $A/kf_s$s.npz $KF; done
wait
for s in 97 99; do for x in .json _frames12.npz .rc .log; do ln -sf ../runs/W_$s$x $RB/S_$s$x; done; done
( envr $REPO; $PY $W/d175_read.py $RO $RB $A --runs D97,D99,S97,S99,T97 \
    --title "D175d: the count comparison with the segment replay measure (rows S = the W arm, --keyframe_compare column), sphere -> bunny 40k" \
    > $RO/read.log 2>&1 < /dev/null )
log "reader exit $? -> $(grep '^VERDICT' $RO/summary.md 2>/dev/null)"
if grep -q "^VERDICT: PASS" $RO/summary.md; then
  Q=$B/tmp/gpu1_queue; cp $W/209_d175_show300.sh $Q/209_d175_show300.sh.tmp && mv $Q/209_d175_show300.sh.tmp $Q/209_d175_show300.sh
  log "W PASS: the 300k showcase queued (209_d175_show300)"
fi
