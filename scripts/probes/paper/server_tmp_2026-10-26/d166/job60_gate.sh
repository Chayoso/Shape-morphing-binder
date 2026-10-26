#!/bin/bash
# 60_d166_gate.sh -- queue job (agent D166, rerun 2026-10-24): the V3 gate as in job 50 on repo_d166 with the capture fix
# (runner.py 315-316). (1) defaults bit-SAME as the tag (d153/rt1/S1); (2) the settle path at its defaults bit-SAME as
# repo_d162hs (job 50's code); (3) --no-settle_relax must now DIFFER from (2) (the fix acts). Writes gates/V3.ok or V3.fail.
O=/data/relcfd/chayo/physmorph_v2/output/gpu/d166; J=60_gate; TITLE="D166 V3 gate"
source /data/relcfd/chayo/physmorph_v2/tmp/d166/job_common.sh
D=$O/gates; mkdir -p $D; rm -f $D/V3.ok $D/V3.fail
log "start (code $(cat $B/$REPO/VERSION); runner.py capture fix)"
( envr $REPO; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 --tgt assets/bunny.obj \
    --out $D/S1 > $D/S1.log 2>&1 < /dev/null; $PY $B/tmp/d145/det_cmp12.py $G/d153/rt1/S1 $D/S1 > $D/cmp.txt 2>&1 )
( envr repo_d162hs; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 $F_coin --lambda_ema 0 \
    --settle_windows 2 --out $D/H0 > $D/H0.log 2>&1 < /dev/null )
( envr $REPO; $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 $F_coin --lambda_ema 0 \
    --settle_windows 2 --out $D/H1 > $D/H1.log 2>&1 < /dev/null; $PY $B/tmp/d145/det_cmp12.py $D/H0 $D/H1 > $D/cmp_settle.txt 2>&1
  $PY $B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4 $F_coin --lambda_ema 0 \
    --settle_windows 2 --no-settle_relax --out $D/H2 > $D/H2.log 2>&1 < /dev/null; $PY $B/tmp/d145/det_cmp12.py $D/H1 $D/H2 > $D/cmp_relax.txt 2>&1 )
a=$(tail -1 $D/cmp.txt); b=$(tail -1 $D/cmp_settle.txt); c=$(tail -1 $D/cmp_relax.txt)
log "V3 defaults: $a | settle defaults vs repo_d162hs: $b | --no-settle_relax vs default settle: $c"
if echo "$a" | grep -q ": SAME$" && echo "$b" | grep -q ": SAME$" && [ -s $D/H2_frames12.npz ] && ! echo "$c" | grep -q ": SAME$"; then
  touch $D/V3.ok; v=PASS; else touch $D/V3.fail; v=FAIL; fi
{ echo "# D166 V3 gate (repo_d166 with the settle_relax capture fix)"; echo; echo "| check | result |"; echo "|---|---|"
  echo "| defaults vs the tag (d153/rt1/S1; bunny 40k, 4 windows) | $a |"
  echo "| settle defaults vs repo_d162hs (coin, held, settle 2) | $b |"
  echo "| --no-settle_relax vs the default settle (must differ) | $c |"; echo
  echo "VERDICT: $v (defaults bit-SAME; the off-default relaxation switch acts)"; } > $O/auto/$J/summary.md
[ $v = PASS ]
