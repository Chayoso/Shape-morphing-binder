#!/bin/bash
# 194_d172_hur.sh -- queue job (agent D172): Hu_r = Hu's step (held weight on u, the rule's on dF_c) with the RULE in the merit,
# the line search and the stop rules (--render_split u_rule, no --lambda_ema 0), Rx's settle. First the gate of e8dc5941:
# (1) defaults bit-SAME as the tag (d153/rt1/S1); (2) Hu (--render_split u) bit-SAME as job 184's X2 (the Hu runs already
# going); (3) u_rule DIFFERS from the rule with the same settle. Then coin / bunny relief / band 40k x 97 / 99, the 6-mesh
# gallery seed 97 and the 300k plaque.
J=194_hur; TITLE="D172: Hu_r (Hu's step, the rule's merit and stops)"
source /data/relcfd/chayo/physmorph_v2/tmp/d172/job_common.sh
D=$O/gates; rm -f $D/V3b.ok $D/V3b.fail
F_coin="--tgt $A1/d161coin.obj --render_tgt $A1/d161L40coin.obj"
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"
CMP=$B/tmp/d145/det_cmp12.py
log "gate b start"
( envr repo_d172; $PY $DR --tgt assets/bunny.obj --out $D/S1b > $D/S1b.log 2>&1 < /dev/null; $PY $CMP $G/d153/rt1/S1 $D/S1b > $D/cmpb.txt 2>&1
  $PY $DR $F_coin --lambda_ema 0 --settle_windows 2 --no-settle_relax --render_split u --out $D/X2b > $D/X2b.log 2>&1 < /dev/null
  $PY $CMP $D/X2 $D/X2b > $D/cmpb_hu.txt 2>&1
  $PY $DR $F_coin --settle_windows 2 --no-settle_relax --out $D/X3 > $D/X3.log 2>&1 < /dev/null
  $PY $DR $F_coin --settle_windows 2 --no-settle_relax --render_split u_rule --out $D/X4 > $D/X4.log 2>&1 < /dev/null
  $PY $CMP $D/X3 $D/X4 > $D/cmpb_hur.txt 2>&1 )
a=$(tail -1 $D/cmpb.txt); b=$(tail -1 $D/cmpb_hu.txt); c=$(tail -1 $D/cmpb_hur.txt)
log "V3b defaults: $a | Hu vs job 184's X2: $b | u_rule vs the rule: $c"
if echo "$a" | grep -q ": SAME$" && echo "$b" | grep -q ": SAME$" && [ -s $D/X4_frames12.npz ] && ! echo "$c" | grep -q ": SAME$"; then
  touch $D/V3b.ok; v=PASS; else touch $D/V3b.fail; v=FAIL; fi
mkdir -p $O/auto/194_gate_b
{ echo "# D172 V3 gate b (repo_d172 with --render_split u_rule, wip/d172 e8dc5941)"; echo; echo "| check | result |"; echo "|---|---|"
  echo "| defaults vs the tag (d153/rt1/S1; bunny 40k, 4 windows) | $a |"
  echo "| Hu (--render_split u), now vs job 184's run | $b |"
  echo "| u_rule vs the rule with the same settle (must differ) | $c |"; echo
  echo "VERDICT: $v (defaults and Hu bit-SAME; u_rule acts)"; } > $O/auto/194_gate_b/summary.md
[ $v = PASS ] || { log "gate b FAIL: no Hu_r runs"; exit 1; }
HUR="--settle_windows 30 --no-settle_relax --render_split u_rule"
log "start"; PAR=4
F_bunny="--tgt $A1/d161smooth_bunny.obj --render_tgt $A1/d161L40_bunny.obj"
F_band="--tgt assets/bunny.obj --appearance band"
runp plq300k_Hur_97 --tgt $G/d163/assets/d163plaque.obj --render_tgt $G/d163/assets/d163R300plaque.obj --n 300000 --seed 97 $HUR
for s in 97 99; do for sc in coin bunny band; do eval f=\$F_$sc; runp ${sc}_Hur_$s $f --n 40000 --seed $s $HUR; done; done
for m in dragon armadilo homer bunny bob teapot; do runp gal_${m}_Hur_97 --tgt assets/$m.obj --n 40000 --seed 97 $HUR; done
wait; log "done"
