#!/bin/bash
# read_t1016.sh -- D145 post-tag reading (CPU only): tab145.py rows of the tag's B1 (d145t_b1) and amendment 6's 300k arms
# (d145t_a6), tab145t.py's post-tag measures on them and on the pre-tag B1 rows (tab_1015.jsonl, relabelled OL / OP),
# and band145t.py's comparisons: B1 L vs P, A6 NRD / NUC / NSS vs L (seeds 97 / 99 / 101), tag vs pre-tag (L vs OL,
# P vs OP). Outputs in output/gpu/d145t_evidence.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; T=$B/tmp/d145; R=$B/tmp/d145r; E=$G/d145t_evidence; mkdir -p $E
export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES=
specs=""
for d in d145t_b1 d145t_a6; do
  for f in $G/$d/*300k_*.json; do n=$(basename $f .json); case $n in jv_*|tab*|ind*) continue;; esac; specs="$specs $d/$n"; done
done
rm -f $E/tab_t.jsonl
$PY $T/tab145.py $E/tab_t.jsonl $specs > $E/tab_t.log 2>&1
grep '"dir": "d145_b1"' $G/d145_evidence/tab_1015.jsonl > $E/tab_pre.jsonl
$PY $R/tab145t.py $E/tab_t.jsonl $E/tab_tx.jsonl
$PY $R/tab145t.py $E/tab_pre.jsonl $E/tab_prex.jsonl --relabel d145_b1=O
cat $E/tab_tx.jsonl $E/tab_prex.jsonl > $E/tab_allx.jsonl
M="bunny dragon teapot bob armadilo homer"
{ echo "=== B1 on the tag: L vs P, all seeds (bunny / dragon 97-105, the others 97-101)"; $PY $R/band145t.py $E/tab_tx.jsonl L P $M; } > $E/b1_L_vs_P.txt 2>&1
{ echo "=== B1 on the tag: L vs P, seeds 97 / 99 / 101"; $PY $R/band145t.py $E/tab_tx.jsonl L P $M --seeds 97,99,101; } > $E/b1_L_vs_P_s3.txt 2>&1
for a in NRD NUC NSS; do
  { echo "=== A6 on the tag: $a vs L, seeds 97 / 99 / 101"; $PY $R/band145t.py $E/tab_tx.jsonl $a L bunny dragon --seeds 97,99,101; } > $E/a6_${a}_vs_L.txt 2>&1
done
{ echo "=== tag L vs pre-tag L (D135, deliverable-2026-10-13), all seeds"; $PY $R/band145t.py $E/tab_allx.jsonl L OL $M
  echo "=== tag P vs pre-tag P, all seeds"; $PY $R/band145t.py $E/tab_allx.jsonl P OP $M
  echo "=== pre-tag L vs pre-tag P (the same measures, for the lead's comparison)"; $PY $R/band145t.py $E/tab_allx.jsonl OL OP $M; } > $E/pre_vs_tag.txt 2>&1
echo done
