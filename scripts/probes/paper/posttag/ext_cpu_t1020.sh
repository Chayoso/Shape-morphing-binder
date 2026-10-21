#!/bin/bash
# ext_cpu_t1020.sh WHAT -- D145's methods table (2026-10-20), the CPU readings (nice 10, 8 threads, one process each).
#   vol     vol145n.py where D145's one volume routine had not read a run yet: ours on the tag (L97 on the six meshes,
#           L99 / L101 on bunny and dragon), Buonomo et al.'s seeds 2-3, and the two B14 runs with an unfillable frame
#           (bob ISDn2, dragon ISDn3) -> d145_ext/vol_ext.jsonl
#   pieces  pieces145.py on every external method's first seed, every frame (the end and the far pieces over the run,
#           the same reader as ours in reading 1) -> d145_ext/pieces_ext.jsonl; MESHES="m1 m2 ..." limits the meshes and
#           writes d145_ext/pieces_ext_m1.jsonl (two such processes in parallel on disjoint meshes)
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; X=$G/d145_ext; R=$B/tmp/d145r; mkdir -p $X
source $B/tmp/d145/env_d145.sh
source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
P="nice -n 10 $PY"
M6="bunny dragon teapot bob armadilo homer"
echo "start $1 $(date '+%m-%d %H:%M')" >> $X/status.txt
if [ "$1" = vol ]; then
  rm -f $X/vol_ext.jsonl
  for m in $M6; do
    specs="ours_L97=$G/d145t_b1/${m}300k_L97_frames12.npz"
    case $m in bunny|dragon) specs="$specs ours_L99=$G/d145t_b1/${m}300k_L99_frames12.npz ours_L101=$G/d145t_b1/${m}300k_L101_frames12.npz";; esac
    specs="$specs buo2=$G/d145_b5/${m}300k_BUO2_frames12.npz buo3=$G/d145_b5/${m}300k_BUO3_frames12.npz"
    case $m in bob) specs="$specs isd_ISDn2=$G/d145_b14/bob300k_ISDn2_frames12.npz";; dragon) specs="$specs isd_ISDn3=$G/d145_b14/dragon300k_ISDn3_frames12.npz";; esac
    $P $R/vol145n.py $X/vol_ext.jsonl $m $G/d90/${m}_ind300k.npz $specs >> $X/vol_ext.log 2>&1
  done
else
  [ -n "$MESHES" ] && M6="$MESHES"; PO=$X/pieces_ext${MESHES:+_${MESHES%% *}}.jsonl; rm -f $PO
  for m in $M6; do
    specs="$m:XU97=$G/d145_b4/${m}300k_XU97_frames12.npz"
    for f in $G/d145_b6/${m}300k_OT_frames12.npz; do [ -e $f ] && specs="$specs $m:OT=$f"; done
    for f in $G/d145_b11/${m}300k_ARAP_frames12.npz; do [ -e $f ] && specs="$specs $m:ARAP=$f"; done
    specs="$specs $m:BUO=$G/d145_b5/${m}300k_BUO_frames12.npz $m:NISE=$G/d145_b5/${m}300k_NISE_frames12.npz"
    specs="$specs $m:ISDn1=$G/d145_b14/${m}300k_ISDn1_frames12.npz $m:FDn1=$G/d145_b14/${m}300k_FDn1_frames12.npz"
    for a in LSa97 LSb97 FCa97 FCb97; do specs="$specs $m:$a=$G/d145_b12/${m}300k_${a}_frames12.npz"; done
    $P $R/pieces145.py $PO $specs --every 1 >> $X/pieces_ext.log 2>&1
  done
fi
echo "done $1 $(date '+%m-%d %H:%M')" >> $X/status.txt
