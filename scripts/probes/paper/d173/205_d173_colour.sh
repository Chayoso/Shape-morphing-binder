#!/bin/bash
# 205_d173_colour.sh -- queue job (agent D173): the gate of wip/d173 0ac73e72 (--colour_kernel nearest, default kernel): (1) defaults
# bit-SAME as the tag; (2) the band bit-SAME as repo_d172 (job 184 B1); (3) the texture bit-SAME as job 200's X1; (4) nearest DIFFERS
# on the band. Then Rx with the nearest colour (Rxn): the band 40k x 97 / 99, the duck texture 40k x 97 / 99, the band 300k (97).
J=205_colour; TITLE="D173 colour kernel nearest"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
D=$O/gates; rm -f $D/V3c.ok $D/V3c.fail
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"; CMP=$B/tmp/d145/det_cmp12.py
TX="--appearance texture --texture $A/d173bob_tex.obj,$A/d173bob_diffuse.png"
log "gate c start"
( envr repo_d173; $PY $DR --tgt assets/bunny.obj --out $D/S1c > $D/S1c.log 2>&1 < /dev/null; $PY $CMP $G/d153/rt1/S1 $D/S1c > $D/cmpc.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --appearance band --out $D/B1c > $D/B1c.log 2>&1 < /dev/null; $PY $CMP $G/d172/gates/B1 $D/B1c > $D/cmpc_band.txt 2>&1
  $PY $DR --tgt assets/bob.obj $TX --out $D/X1c > $D/X1c.log 2>&1 < /dev/null; $PY $CMP $D/X1 $D/X1c > $D/cmpc_tex.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --appearance band --colour_kernel nearest --out $D/B2c > $D/B2c.log 2>&1 < /dev/null; $PY $CMP $D/B1c $D/B2c > $D/cmpc_near.txt 2>&1 )
a=$(tail -1 $D/cmpc.txt); b=$(tail -1 $D/cmpc_band.txt); c=$(tail -1 $D/cmpc_tex.txt); d=$(tail -1 $D/cmpc_near.txt)
log "V3c defaults: $a | band: $b | texture: $c | nearest vs kernel: $d"
if echo "$a" | grep -q ": SAME$" && echo "$b" | grep -q ": SAME$" && echo "$c" | grep -q ": SAME$" && [ -s $D/B2c_frames12.npz ] && ! echo "$d" | grep -q ": SAME$"; then touch $D/V3c.ok; v=PASS; else touch $D/V3c.fail; v=FAIL; fi
mkdir -p $O/auto/205_gate_c
{ echo "# D173 V3 gate c (--colour_kernel, wip/d173 0ac73e72)"; echo; echo "| check | result |"; echo "|---|---|"; echo "| defaults vs the tag | $a |"; echo "| band vs repo_d172 | $b |"
  echo "| texture vs job 200 X1 | $c |"; echo "| nearest vs kernel (band, must differ) | $d |"; echo; echo "VERDICT: $v"; } > $O/auto/205_gate_c/summary.md
[ $v = PASS ] || { log "gate c FAIL"; exit 1; }
log "start"; PAR=4; NC="--colour_kernel nearest"
runp band300k_Rxn_97 --tgt assets/bunny.obj --appearance band --n 300000 --seed 97 $RX $NC
for s in 97 99; do
  runp band40k_Rxn_$s --tgt assets/bunny.obj --appearance band --n 40000 --seed $s $RX $NC
  runp tex40k_Rxn_$s --tgt assets/bob.obj $TX --n 40000 --seed $s $RX $NC
done
wait; log "done"
