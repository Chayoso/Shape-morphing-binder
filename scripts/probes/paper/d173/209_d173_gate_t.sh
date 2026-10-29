#!/bin/bash
# 209_d173_gate_t.sh -- queue job (agent D173): the gate of wip/d173's per-block render shares on every run (window/solve.py,
# run/runner.py: records only) and the stripes scene (appearance.py, --appearance stripes): defaults bit-SAME as the tag (bunny
# 40k, 4 windows), the band scene bit-SAME as repo_d172's B1, Hu (coin, 4 windows) bit-SAME as repo_d172's X2b (det_cmp12 reads
# the fields both runs carry: the new records do not count), the stripes scene runs, and the new records are present in the
# optimisation windows of the default run, the Hu run and the stripes run. Writes gates/V3t.ok or V3t.fail; jobs 213-222 wait
# for it.
J=209_gate_t; TITLE="D173 gate t (block shares on every run, the stripes scene)"
source /data/relcfd/chayo/physmorph_v2/tmp/d173/job_common.sh
D=$O/gates; mkdir -p $D; rm -f $D/V3t.ok $D/V3t.fail
F_coin="--tgt $A1/d161coin.obj --render_tgt $A1/d161L40coin.obj"
DR="$B/tmp/rt/det_run.py scripts/pipeline_run.py --n 40000 --seed 97 --animations 4"; CMP=$B/tmp/d145/det_cmp12.py
log "gate t start"
( envr repo_d173
  $PY $DR --tgt assets/bunny.obj --out $D/S1t > $D/S1t.log 2>&1 < /dev/null; $PY $CMP $G/d153/rt1/S1 $D/S1t > $D/cmpt.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --appearance band --out $D/B1t > $D/B1t.log 2>&1 < /dev/null; $PY $CMP $G/d172/gates/B1 $D/B1t > $D/cmpt_band.txt 2>&1
  $PY $DR $F_coin --lambda_ema 0 --settle_windows 2 --no-settle_relax --render_split u --out $D/X2t > $D/X2t.log 2>&1 < /dev/null
  $PY $CMP $G/d172/gates/X2b $D/X2t > $D/cmpt_hu.txt 2>&1
  $PY $DR --tgt assets/bunny.obj --appearance stripes $RX --colour_kernel nearest --out $D/P1t > $D/P1t.log 2>&1 < /dev/null; echo $? > $D/P1t.rc
  $PY - > $D/keys_t.txt 2>&1 << EOP
import json
ok = True
for p in ("$D/S1t", "$D/X2t", "$D/P1t"):
    a = next(iter(json.load(open(p + ".json"))["arms"].values()))
    opt = [h for h in a["history"] if h.get("frame_end") and not h.get("settle") and not h.get("null_commit")]
    got = [(h.get("g_share_u"), h.get("g_share_dfc"), h.get("lam_block_u"), h.get("lam_block_dfc")) for h in opt]
    good = bool(opt) and all(all(v is not None for v in g) for g in got)
    ok &= good
    print(p.rsplit("/", 1)[-1], "windows", len(opt), "records", "present" if good else "MISSING", got[:2])
print("KEYS", "OK" if ok else "MISSING")
EOP
)
a=$(tail -1 $D/cmpt.txt); b=$(tail -1 $D/cmpt_band.txt); c=$(tail -1 $D/cmpt_hu.txt); d=$(cat $D/P1t.rc); k=$(tail -1 $D/keys_t.txt)
log "V3t defaults: $a | band: $b | Hu vs X2b: $c | stripes exit $d | $k"
if echo "$a" | grep -q ": SAME$" && echo "$b" | grep -q ": SAME$" && echo "$c" | grep -q ": SAME$" && [ "$d" = 0 ] && [ -s $D/P1t_frames12.npz ] \
   && [ "$k" = "KEYS OK" ]; then touch $D/V3t.ok; v=PASS; else touch $D/V3t.fail; v=FAIL; fi
mkdir -p $O/auto/209_gate_t
{ echo "# D173 V3 gate t (the per-block render shares recorded on every run; --appearance stripes)"; echo; echo "| check | result |"; echo "|---|---|"
  echo "| defaults vs the tag (d153/rt1/S1; bunny 40k, 4 windows) | $a |"; echo "| --appearance band vs repo_d172 (B1) | $b |"
  echo "| Hu (coin, 4 windows) vs repo_d172 (X2b) | $c |"; echo "| --appearance stripes, Rx + nearest colour (bunny 40k, 4 windows) | exit $d |"
  echo "| the new records in every optimisation window (default, Hu, stripes) | $k |"; echo; echo "VERDICT: $v"; } > $O/auto/209_gate_t/summary.md
log "gate t $v"
