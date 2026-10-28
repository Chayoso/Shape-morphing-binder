#!/bin/bash
# 168_d170_gate_dragon.sh -- queue job (agent D170): (1) the V3 gate of repo_d170 (= repo_d166 + --layer_relax, wip/d170
# 2664ca56): defaults bit-SAME as the tag (d153/rt1/S1); Rx's flags bit-SAME as on repo_d166; --no-layer_relax must DIFFER.
# (2) if it passes, the dragon render-only diagnostics at 40k, seeds 97 / 99 (the relaxation off throughout):
# H0 / H0m = held weight + --no-layer_relax (pictures from samples / from the mesh shell), R0m = the rule + --no-layer_relax
# (mesh shell); each with settle 30 and --no-settle_relax like Rx.
J=168_gate_dragon; TITLE="D170 V3 gate (repo_d170) + dragon relaxation-off diagnostics"
source /data/relcfd/chayo/physmorph_v2/tmp/d170/job_common.sh
D=$O/gates; mkdir -p $D; rm -f $D/V3.ok $D/V3.fail
F_coin="--tgt $A1/d161coin.obj --render_tgt $A1/d161L40coin.obj"
log "gate start"
( envr repo_d170; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 --tgt assets/bunny.obj \
    --out $D/S1 > $D/S1.log 2>&1 < /dev/null; $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $D/S1 > $D/cmp.txt 2>&1 )
( envr repo_d166; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 $F_coin --lambda_ema 0 \
    --settle_windows 2 --no-settle_relax --out $D/X0 > $D/X0.log 2>&1 < /dev/null )
( envr repo_d170; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 $F_coin --lambda_ema 0 \
    --settle_windows 2 --no-settle_relax --out $D/X1 > $D/X1.log 2>&1 < /dev/null; $PY $B/tmp/d145/det_cmp12.py $D/X0 $D/X1 > $D/cmp_rx.txt 2>&1
  $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 $F_coin --lambda_ema 0 \
    --settle_windows 2 --no-settle_relax --no-layer_relax --out $D/X2 > $D/X2.log 2>&1 < /dev/null; $PY $B/tmp/d145/det_cmp12.py $D/X1 $D/X2 > $D/cmp_off.txt 2>&1 )
a=$(tail -1 $D/cmp.txt); b=$(tail -1 $D/cmp_rx.txt); c=$(tail -1 $D/cmp_off.txt)
log "V3 defaults: $a | Rx flags repo_d170 vs repo_d166: $b | --no-layer_relax vs Rx: $c"
if echo "$a" | grep -q ": SAME$" && echo "$b" | grep -q ": SAME$" && [ -s $D/X2_frames12.npz ] && ! echo "$c" | grep -q ": SAME$"; then
  touch $D/V3.ok; v=PASS; else touch $D/V3.fail; v=FAIL; fi
mkdir -p $O/auto/168_gate
{ echo "# D170 V3 gate (repo_d170 = repo_d166 + --layer_relax, wip/d170 2664ca56)"; echo; echo "| check | result |"; echo "|---|---|"
  echo "| defaults vs the tag (d153/rt1/S1; bunny 40k, 4 windows) | $a |"
  echo "| Rx's flags, repo_d170 vs repo_d166 (coin, held, settle 2, --no-settle_relax) | $b |"
  echo "| --no-layer_relax vs Rx (must differ) | $c |"; echo
  echo "VERDICT: $v (defaults bit-SAME; the off-default switch acts)"; } > $O/auto/168_gate/summary.md
[ $v = PASS ] || { log "gate FAIL: no diagnostics"; exit 1; }
REPO=repo_d170; PAR=4
DR="--tgt $A/d170smooth_dragon.obj --render_tgt assets/dragon.obj --n 40000"
OFF="--settle_windows 30 --no-settle_relax --no-layer_relax"
for s in 97 99; do
  runp dsm40k_H0m_$s $DR --seed $s --lambda_ema 0 $OFF --render_target mesh_shell
  runp dsm40k_R0m_$s $DR --seed $s $OFF --render_target mesh_shell
  runp dsm40k_H0_$s $DR --seed $s --lambda_ema 0 $OFF
done
wait; log "done"
