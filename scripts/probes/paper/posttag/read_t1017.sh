#!/bin/bash
# read_t1017.sh -- D145 post-tag reading 2 (CPU only): A6 at 40k, B3 (gallery and its ablations), B2, B8, B9, B10 and
# B1's every-frame readings on the tag, each beside the pre-tag reading where one exists. Outputs in
# output/gpu/d145t_evidence/r2/. Needs endgen_t1017.sh (end checks) and pieces_t1016.sh DIRS="d145t_b2 d145t_b8
# d145t_b9" (pieces) first.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; T=$B/tmp/d145; R=$B/tmp/d145r; E=$G/d145t_evidence; O=$E/r2
mkdir -p $O
export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES=
P="nice -n 10 $PY"
# --- 40k: amendment 6 and B3's ablations against V; the gallery
$P $R/a6_40k.py $O/a6_40k.md > /dev/null
$P $R/a6_40k.py $O/b3_abl.md --arms U,R,X,V0 --adir d145t_b3 --title B3 > /dev/null
$P $R/a6_40k.py $O/b3_abl_pre.md --arms U,R,X,V0 --adir d145_b3 --vdir d145_b3 --title "B3 pre-tag" > /dev/null
$P $R/gal145t.py $O/b3_gallery.md > /dev/null
# --- 300k rows: B2, B8, B9 on the tag and before it (the volume-off arm V0 renamed VO, so that its seed reads 97 / 99 / 101)
specs=""
for d in d145t_b2 d145t_b8 d145t_b9; do
  for f in $G/$d/*300k_*.json; do n=$(basename $f .json); case $n in jv_*|tab*|ind*) continue;; esac; specs="$specs $d/$n"; done
done
rm -f $O/tab_t2.jsonl
$P $T/tab145.py $O/tab_t2.jsonl $specs > $O/tab_t2.log 2>&1
grep -E '"dir": "d145_b(2|8|9)"' $G/d145_evidence/tab_1015.jsonl > $O/tab_pre2.jsonl
$P $R/tab145t.py $O/tab_t2.jsonl $O/tab_t2x.jsonl
$P $R/tab145t.py $O/tab_pre2.jsonl $O/tab_pre2x.jsonl --relabel d145_b2=O --relabel d145_b8=O --relabel d145_b9=O
sed -E 's/"name": "([A-Za-z]+300k)_V0([0-9]+)"/"name": "\1_VO\2"/' $O/tab_t2x.jsonl > $O/tab_t2x_r.jsonl
sed -E 's/"name": "([A-Za-z]+300k)_OV0([0-9]+)"/"name": "\1_OVO\2"/' $O/tab_pre2x.jsonl > $O/tab_pre2x_r.jsonl
$P $R/tab145t.py $E/tab_t.jsonl $O/tab_b1x.jsonl                       # B1 and A6 300k again, with the new fields
$P $R/tab145t.py $E/tab_pre.jsonl $O/tab_b1prex.jsonl --relabel d145_b1=O
cat $O/tab_b1x.jsonl $O/tab_b1prex.jsonl $O/tab_t2x_r.jsonl $O/tab_pre2x_r.jsonl > $O/tab_all.jsonl
M6="bunny dragon teapot bob armadilo homer"
BD="bunny dragon"
S3="--seeds 97,99,101"
{ for a in X U R VO VJ VD VA E M C; do echo "=== B2 tag: $a vs L, seeds 97 / 99 / 101"
    [ $a = C ] && m="$M6" || m="$BD"; $P $R/band145t.py $O/tab_all.jsonl $a L $m $S3; done
  for a in X U R VO C; do echo "=== B2 pre-tag: O$a vs OL, seeds 97 / 99 / 101"
    [ $a = C ] && m="$M6" || m="$BD"; $P $R/band145t.py $O/tab_all.jsonl O$a OL $m $S3; done
} > $O/b2.txt 2>&1
{ for a in ESL EHL; do echo "=== B8 tag: $a vs L, seeds 97 / 99 / 101"; $P $R/band145t.py $O/tab_all.jsonl $a L $BD $S3
    echo "=== B8 pre-tag: O$a vs OL"; $P $R/band145t.py $O/tab_all.jsonl O$a OL $BD $S3; done
  echo "=== B8 tag: NUL vs L, seed 97 against L's three"; $P $R/band145t.py $O/tab_all.jsonl NUL L $BD $S3
  for a in ES EH NU; do echo "=== B8 tag, within the material: ${a}L vs ${a}P, seed 97"
    $P $R/band145t.py $O/tab_all.jsonl ${a}L ${a}P $BD --seeds 97
    echo "=== B8 pre-tag, within the material: O${a}L vs O${a}P, seed 97"
    $P $R/band145t.py $O/tab_all.jsonl O${a}L O${a}P $BD --seeds 97; done
} > $O/b8.txt 2>&1
{ echo "=== B9 tag: G2L vs L, seeds 97 / 99 / 101"; $P $R/band145t.py $O/tab_all.jsonl G2L L $BD $S3
  echo "=== B9 tag: G2L vs G2P, seed 97"; $P $R/band145t.py $O/tab_all.jsonl G2L G2P $BD --seeds 97
  echo "=== B9 tag: GFL vs G2L, seed 97"; $P $R/band145t.py $O/tab_all.jsonl GFL G2L $BD --seeds 97
  echo "=== B9 tag: D0L vs L (D0L seed 97 against L's three)"; $P $R/band145t.py $O/tab_all.jsonl D0L L $BD $S3
  echo "=== B9 pre-tag: OG2L vs OL"; $P $R/band145t.py $O/tab_all.jsonl OG2L OL $BD $S3
  echo "=== B9 pre-tag: OG2L vs OG2P, seed 97"; $P $R/band145t.py $O/tab_all.jsonl OG2L OG2P $BD --seeds 97
  echo "=== B9 tag vs pre-tag: G2L vs OG2L"; $P $R/band145t.py $O/tab_all.jsonl G2L OG2L $BD $S3
} > $O/b9.txt 2>&1
# --- B10: the loss terms per window from the records (B1's runs on the tag; the four telemetry runs)
rm -f $O/b10.jsonl; specs=""
for f in $G/d145t_b1/*300k_[LP][0-9]*.json $G/d145t_b10/*300k_T[LP]97.json; do
  specs="$specs $(basename $(dirname $f))/$(basename $f .json)"; done
$P $T/b10_curves.py $O/b10.jsonl $specs > $O/b10.log 2>&1
$P $R/b10sum.py $O/b10.jsonl L P $M6 --md $O/b10_LP.md > /dev/null
$P $R/b10sum.py $O/b10.jsonl TL TP $BD --md $O/b10_tel.md > /dev/null
# --- B1 every kept frame: the lead over the trajectory
$P $R/lead145.py $G/d145t_b1 L P $BD --md $O/b1_lead.md > /dev/null
echo done
