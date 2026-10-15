#!/bin/bash
# read_1015.sh -- D145: the overnight 300k runs (B1 seeds 97-105 and the four meshes, B2 arms, B8 materials, B9 gravity /
# floor / drag) into tab145 rows and band145 comparisons (CPU only).
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; T=$B/tmp/d145; E=$G/d145_evidence; mkdir -p $E
export PHYSMORPH_RUN_REPO=$B/repo_d145; source $B/repo_d145/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES=
specs=""
for d in d145_b1 d145_b2 d145_b8 d145_b9; do
  for f in $G/$d/*300k_*.json; do n=$(basename $f .json); case $n in jv_*|tab*|ind*) continue;; esac; specs="$specs $d/$n"; done
done
rm -f $E/tab_1015.jsonl
$PY $T/tab145.py $E/tab_1015.jsonl $specs > $E/tab_1015.log 2>&1
grep -E '300k_[A-Za-z0-9]*[A-Za-z](97|99|101)"' $E/tab_1015.jsonl > $E/tab_1015_s3.jsonl
{
echo "=== B1 L vs P, all seeds (bunny/dragon 97-105, others 97-101)"; $PY $T/band145.py $E/tab_1015.jsonl L P bunny dragon teapot bob armadilo homer
for arm in C U R V X; do echo "=== B2 $arm vs L, seeds 97/99/101"; $PY $T/band145.py $E/tab_1015_s3.jsonl $arm L bunny dragon teapot bob armadilo homer; done
for arm in ESL EHL G2L; do echo "=== $arm vs L, seeds 97/99/101"; $PY $T/band145.py $E/tab_1015_s3.jsonl $arm L bunny dragon; done
} > $E/band_1015.txt 2>&1
echo done
