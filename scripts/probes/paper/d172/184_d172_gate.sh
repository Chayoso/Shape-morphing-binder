#!/bin/bash
# 184_d172_gate.sh -- queue job (agent D172): the V3 gate of repo_d172 (wip/d172 4a0b2537 = repo_d170 + --render_split +
# the render_x paint fix): (1) defaults bit-SAME as the tag (d153/rt1/S1); (2) Rx's flags bit-SAME as repo_d166's (D170's
# gate run X0); (3) --render_split u must DIFFER from (2); (4) the band (--appearance band, no --render_tgt) bit-SAME as on
# repo_d166 (the paint fix touches only --render_tgt + --appearance). Writes gates/V3.ok or V3.fail.
J=184_gate; TITLE="D172 V3 gate"
source /data/relcfd/chayo/physmorph_v2/tmp/d172/job_common.sh
D=$O/gates; mkdir -p $D; rm -f $D/V3.ok $D/V3.fail
F_coin="--tgt $A1/d161coin.obj --render_tgt $A1/d161L40coin.obj"
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"
log "gate start"
( envr repo_d172; $PY $DR --tgt assets/bunny.obj --out $D/S1 > $D/S1.log 2>&1 < /dev/null; $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $D/S1 > $D/cmp.txt 2>&1
  $PY $DR $F_coin --lambda_ema 0 --settle_windows 2 --no-settle_relax --out $D/X1 > $D/X1.log 2>&1 < /dev/null
  $PY $B/tmp/d145/det_cmp12.py $G/d170/gates/X0 $D/X1 > $D/cmp_rx.txt 2>&1
  $PY $DR $F_coin --lambda_ema 0 --settle_windows 2 --no-settle_relax --render_split u --out $D/X2 > $D/X2.log 2>&1 < /dev/null
  $PY $B/tmp/d145/det_cmp12.py $D/X1 $D/X2 > $D/cmp_split.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --appearance band --out $D/B1 > $D/B1.log 2>&1 < /dev/null )
( envr repo_d166; $PY $DR --tgt assets/bunny.obj --appearance band --out $D/B0 > $D/B0.log 2>&1 < /dev/null
  $PY $B/tmp/d145/det_cmp12.py $D/B0 $D/B1 > $D/cmp_band.txt 2>&1 )
a=$(tail -1 $D/cmp.txt); b=$(tail -1 $D/cmp_rx.txt); c=$(tail -1 $D/cmp_split.txt); d=$(tail -1 $D/cmp_band.txt)
log "V3 defaults: $a | Rx flags vs repo_d166: $b | --render_split u vs Rx: $c | band vs repo_d166: $d"
if echo "$a" | grep -q ": SAME$" && echo "$b" | grep -q ": SAME$" && echo "$d" | grep -q ": SAME$" && [ -s $D/X2_frames12.npz ] && ! echo "$c" | grep -q ": SAME$"; then
  touch $D/V3.ok; v=PASS; else touch $D/V3.fail; v=FAIL; fi
mkdir -p $O/auto/184_gate
{ echo "# D172 V3 gate (repo_d172 = repo_d170 + --render_split + the render_x paint fix, wip/d172 4a0b2537)"; echo
  echo "| check | result |"; echo "|---|---|"
  echo "| defaults vs the tag (d153/rt1/S1; bunny 40k, 4 windows) | $a |"
  echo "| Rx's flags, repo_d172 vs repo_d166 (coin, held, settle 2, --no-settle_relax) | $b |"
  echo "| --render_split u vs Rx (must differ) | $c |"
  echo "| --appearance band (no --render_tgt), repo_d172 vs repo_d166 | $d |"; echo
  echo "VERDICT: $v (defaults and the band bit-SAME; the off-default split acts)"; } > $O/auto/184_gate/summary.md
[ $v = PASS ]
