#!/bin/bash
# r4_gpu.sh WHAT -- D145 reading 4, the card part, one card under the agents' lock protocol (no lock dir, < 2000 MiB,
# idle in eleven looks a minute apart; mkdir lock, info inside, released by a trap). Outputs in output/gpu/d145_ext/r4.
#   yard   (1) the yardstick (render_terms_probe.py) and the display (surface_layer_probe.py), repo_r77 as read300.sh,
#          against the independent sample, on every r4/end1/*_end1.npz (the external methods' end frames)
#   disp   (3) ext_end145.py on our display meshes' 300k samples (r4/display/*_sample.npz)
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; X=$G/d145_ext/r4; L=$B/gpu_locks; R=$B/tmp/d145r
mkdir -p $X/yard; what=$1
declare -A idle; g=""; looks=0
while [ -z "$g" ] && [ $looks -lt 360 ]; do
  for c in 0 1 2 3; do
    if [ ! -e $L/gpu$c.lock ] && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c)" -lt 2000 ]; then
      idle[$c]=$(( ${idle[$c]:-0} + 1 )); else idle[$c]=0; fi
  done
  for c in 0 1 2 3; do [ ${idle[$c]:-0} -ge 11 ] && mkdir $L/gpu$c.lock 2>/dev/null && { g=$c; break; }; done
  [ -z "$g" ] && { looks=$((looks + 1)); sleep 60; }
done
[ -z "$g" ] && { echo "$what: no card idle 10 min within 6 h $(date '+%m-%d %H:%M')" >> $X/status.txt; exit 1; }
trap "rm -rf $L/gpu$g.lock" EXIT
printf "agent D145r\nrun d145_ext/r4 $what\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
echo "start $what gpu $g $(date '+%m-%d %H:%M')" >> $X/status.txt
export CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8
if [ "$what" = yard ]; then
  unset PHYSMORPH_RUN_REPO; source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null 2>&1; cd $B/repo_r77
  for f in $X/end1/*_end1.npz; do
    name=$(basename $f _end1.npz); mesh=${name%%300k_*}; IND=$G/d90/${mesh}_ind300k.npz
    case $mesh in bunny) box=1900,100,2800,700;; dragon) box=1900,60,2500,560;; *) box=0,0,3840,2160;; esac
    raw=$($PY -c "import numpy as np; print(int(np.load('$f')['raws'][-1]))")
    $PY scripts/probes/settled/render_terms_probe.py $f ref=$IND > $X/yard/t3end_$name.log 2>&1 < /dev/null
    $PY scripts/probes/settled/surface_layer_probe.py $f $X/yard/e3end_$name $box measure ref=$IND target own $raw > $X/yard/e3end_$name.log 2>&1 < /dev/null
    echo "yard $name rc $? $(date '+%m-%d %H:%M')" >> $X/status.txt
  done
else
  export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1; cd $B/repo_t
  specs=""
  for f in $X/display/*_sample.npz; do
    n=$(basename $f _sample.npz); mesh=${n%%_*}; lab=${n#*_}; specs="$specs $mesh:ours_disp:$lab=$f"
  done
  rm -f $X/disp_end.jsonl
  $PY $R/ext_end145.py $X/disp_end.jsonl $specs > $X/disp_end.log 2>&1
fi
echo "done $what gpu $g $(date '+%m-%d %H:%M')" >> $X/status.txt
