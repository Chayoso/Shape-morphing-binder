#!/bin/bash
# r4_prep.sh -- D145 reading 4: the external methods' end frames as end1 archives (r4_end1.py; CPU, nice), every seed of
# every method of the methods table on the six meshes -> output/gpu/d145_ext/r4/end1/.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; X=$G/d145_ext/r4; R=$B/tmp/d145r
mkdir -p $X/end1
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
specs=""
for m in bunny dragon teapot bob armadilo homer; do
  specs="$specs ${m}300k_XU97=$G/d145_b4/${m}300k_XU97_frames12.npz ${m}300k_XU99=$G/d145_b4/${m}300k_XU99_end1.npz ${m}300k_XU101=$G/d145_b4/${m}300k_XU101_end1.npz"
  for f in $G/d145_b6/${m}300k_OT_frames12.npz $G/d145_b11/${m}300k_ARAP_frames12.npz; do [ -e $f ] && specs="$specs $(basename $f _frames12.npz)=$f"; done
  for t in BUO BUO2 BUO3 NISE; do f=$G/d145_b5/${m}300k_${t}_frames12.npz; [ -e $f ] && specs="$specs ${m}300k_$t=$f"; done
  for t in ISDn1 ISDn2 ISDn3 ISDr1 FDn1 FDn2 FDn3 FDr1; do f=$G/d145_b14/${m}300k_${t}_frames12.npz; [ -e $f ] && specs="$specs ${m}300k_$t=$f"; done
  for a in LSa LSb FCa FCb; do for s in 97 99 101; do f=$G/d145_b12/${m}300k_${a}${s}_frames12.npz; [ -e $f ] && specs="$specs ${m}300k_${a}${s}=$f"; done; done
done
nice -n 10 $PY $R/r4_end1.py $X/end1 $specs > $X/end1.log 2>&1
echo "prep done $(ls $X/end1 | wc -l) end frames $(date '+%m-%d %H:%M')" >> $X/status.txt
