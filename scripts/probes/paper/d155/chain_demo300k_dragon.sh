#!/bin/bash
# chain_demo300k_dragon.sh -- the dragon half of chain_demo300k.sh, re-run after the 17:00 demo (the coordinator,
# 2026-10-21): the first try ran the tag render beside the sim on one card and the sim died of memory (rc 1);
# the tag render r_tag_dragon is kept, so only the ramp sim, its render and the side-by-side remain, one after another.
# The original header follows.
#
# (A3D: --morph_ramp elastic --morph_u_gate final --morph_assim final; render arm, seed 97; the run's own elastic period)
# on repo_d155ds (rt1 + wip/d155 42c8f81, kept frames), one card by the lock protocol (agent D155: a card idle 10 min,
# never GPU 2, no other user's process); the studio-look renders (render_deliverable.py +fit=0 +field=zbs, az 35 el 18,
# frames 0-120 and the last kept frame) of it and of the tag's bunny300k_L97 (d145t_b1), the 1080p side-by-side and the
# contact sheet (d155_sbs.py). Then, if before 15:40 and the card still ours, dragon 300k the same way.
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; T=$B/tmp; G=$B/output/gpu; W=$T/d155; O=$G/d155/demo300k
source $T/d146/cache_env.sh
export TMPDIR=$W/tmpdir; mkdir -p $TMPDIR $O
S=$O/status.txt
PY=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
declare -A idle
GPU=""
echo "demo300k_dragon ($$) waiting for a card idle 10 min $(date '+%m-%d %H:%M:%S')" >> $S
while [ -z "$GPU" ]; do
  for g in 0 1 3; do
    if [ -e $L/gpu$g.lock ]; then idle[$g]=0; continue; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g)
    napps=0; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $g); do [ "$(ps -o user= -p $p 2>/dev/null | tr -d " ")" = chayo ] || napps=$((napps + 1)); done
    if [ "$used" -lt 2000 ] && [ "$util" -lt 5 ] && [ "$napps" -eq 0 ]; then idle[$g]=$(( ${idle[$g]:-0} + 1 )); else idle[$g]=0; fi
    if [ "${idle[$g]}" -ge 10 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
      printf "agent D155\nrun d155/demo300k_dragon: dragon 300k with D155d's ramp (~30 min) and its studio render beside the tag's (coordinator re-run)\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
      GPU=$g; break
    fi
  done
  [ -z "$GPU" ] && sleep 60
done
trap "rm -rf $L/gpu$GPU.lock" EXIT
trap "exit 1" INT TERM HUP
echo "demo300k_dragon claimed gpu $GPU $(date '+%m-%d %H:%M:%S')" >> $S
export CUDA_VISIBLE_DEVICES=$GPU
sim() {   # sim MESH
  local name=$1300k_A3DL97
  (
    export PHYSMORPH_RUN_REPO=$B/repo_d155ds PHYSMORPH_OUT=$G
    source $B/repo_d155ds/scripts/ops/hyde06_env.sh > /dev/null
    export TMPDIR=$W/tmpdir
    cd $B/repo_d155ds
    t0=$(date +%s)
    $PY scripts/pipeline_run.py --tgt assets/$1.obj --n 300000 --seed 97 --morph_ramp elastic --morph_u_gate final \
      --morph_assim final --record_elastic --out $O/$name > $O/$name.log 2>&1 < /dev/null
    echo "$name sim rc $? wall $(( $(date +%s) - t0 )) s $(date '+%m-%d %H:%M:%S')" >> $S
  )
}
render() {   # render LABEL FRAMES12 DENSE OUTDIR AZ
  (
    export PHYSMORPH_RUN_REPO=$B/repo_freeze1015 PHYSMORPH_OUT=$W/tmpdir/pm_out
    source $B/repo_freeze1015/scripts/ops/hyde06_env.sh > /dev/null 2>&1
    source $T/d146/cache_env.sh
    export TMPDIR=$W/tmpdir
    cd $B/repo_freeze1015
    t0=$(date +%s)
    last=$($PY -c "import numpy as np; print(len(np.load('$2')['raws']) - 1)")
    rm -rf $4 ${4}_end
    $PY scripts/render_deliverable.py $2 $3 $4 +stride=1 +video=1 +fit=0 +field=zbs +az=$5 +el=18 +dist=1 +frames=0-120 \
      > $4.log 2>&1 < /dev/null
    $PY scripts/render_deliverable.py $2 $3 ${4}_end +stride=1 +video=1 +fit=0 +field=zbs +az=$5 +el=18 +dist=1 \
      +frames=$last-$last > ${4}_end.log 2>&1 < /dev/null
    cp ${4}_end/frames/*.jpg $4/frames/ 2>/dev/null
    echo "render $1 frames $(ls $4/frames 2>/dev/null | wc -l) (last kept $last) in $(( $(date +%s) - t0 )) s $(date '+%m-%d %H:%M:%S')" >> $S
  )
}
compose() {   # compose NAME LEFT RIGHT
  (
    export TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES="" MPLCONFIGDIR=$W/tmpdir/mpl
    rm -rf $O/$1
    nice -n 10 $PY $W/d155_sbs.py $2 $3 $O/$1 > $O/$1.log 2>&1
    ffmpeg -hide_banner -loglevel error -y -framerate 20 -i $O/$1/sbs/%04d.jpg -c:v libx264 -preset medium -crf 18 \
      -pix_fmt yuv420p -movflags +faststart $O/${1}_1080p.mp4 < /dev/null
    echo "compose $1: $(ls -la $O/${1}_1080p.mp4 2>&1 | awk '{print $5}') bytes, contact sheet $(ls $O/$1/contact_sheet.jpg 2>&1) $(date '+%m-%d %H:%M:%S')" >> $S
  )
}
DD=$G/showcase/ind/isosphere_dragon_dense15.npz
sim dragon
render ramp_dragon $O/dragon300k_A3DL97_frames12.npz $DD $O/r_ramp_dragon 35
compose dragon300k_frozen_vs_ramp $O/r_tag_dragon $O/r_ramp_dragon
echo "dragon re-run done $(date '+%m-%d %H:%M:%S')" >> $S
