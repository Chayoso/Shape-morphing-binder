#!/bin/bash
# 200_d173_gate_tex40.sh -- queue job (agent D173): the V3 gate of repo_d173 (wip/d173 98303b79 = repo_d172 + the texture
# appearance): (1) defaults bit-SAME as the tag (d153/rt1/S1); (2) the band bit-SAME as repo_d172 (job 184's B1); (3) the
# texture scene runs (4 windows). Then the textured duck at 40k: the twin and Rx, seeds 97 / 99.
J=200_gate_tex40; TITLE="D173 gate + texture 40k"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
D=$O/gates; mkdir -p $D; rm -f $D/V3.ok $D/V3.fail
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"; CMP=$B/tmp/d145/det_cmp12.py
TX="--appearance texture --texture $A/d173bob_tex.obj,$A/d173bob_diffuse.png"
log "gate start"
( envr repo_d173; $PY $DR --tgt assets/bunny.obj --out $D/S1 > $D/S1.log 2>&1 < /dev/null; $PY $CMP $G/d153/rt1/S1 $D/S1 > $D/cmp.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --appearance band --out $D/B1 > $D/B1.log 2>&1 < /dev/null; $PY $CMP $G/d172/gates/B1 $D/B1 > $D/cmp_band.txt 2>&1
  $PY $DR --tgt assets/bob.obj $TX --out $D/X1 > $D/X1.log 2>&1 < /dev/null; echo $? > $D/X1.rc )
a=$(tail -1 $D/cmp.txt); b=$(tail -1 $D/cmp_band.txt); c=$(cat $D/X1.rc)
log "V3 defaults: $a | band vs repo_d172: $b | texture smoke exit $c"
if echo "$a" | grep -q ": SAME$" && echo "$b" | grep -q ": SAME$" && [ "$c" = 0 ] && [ -s $D/X1_frames12.npz ]; then touch $D/V3.ok; v=PASS; else touch $D/V3.fail; v=FAIL; fi
mkdir -p $O/auto/200_gate
{ echo "# D173 V3 gate (repo_d173 = repo_d172 + the texture appearance, wip/d173 98303b79)"; echo; echo "| check | result |"; echo "|---|---|"
  echo "| defaults vs the tag (d153/rt1/S1; bunny 40k, 4 windows) | $a |"; echo "| --appearance band, repo_d173 vs repo_d172 (job 184 B1) | $b |"
  echo "| --appearance texture (bob 40k, 4 windows) | exit $c |"; echo; echo "VERDICT: $v (defaults and the band bit-SAME; the texture scene runs)"; } > $O/auto/200_gate/summary.md
[ $v = PASS ] || { log "gate FAIL"; exit 1; }
log "start"; PAR=4
for s in 97 99; do
  runp tex40k_T_$s --tgt assets/bob.obj $TX --n 40000 --seed $s $TW
  runp tex40k_Rx_$s --tgt assets/bob.obj $TX --n 40000 --seed $s $RX
done
wait; log "done"
