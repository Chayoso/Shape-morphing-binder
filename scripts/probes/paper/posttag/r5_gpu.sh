#!/bin/bash
# r5_gpu.sh -- D145 reading 5, the card part: one card held for the chain under the agents' lock protocol (no lock dir,
# < 2000 MiB, idle in eleven looks a minute apart; mkdir lock, info inside, released by a trap).
#   (1) six evidence runs at 300k with --cell_follows_n on repo_t1s (freeze-2026-10-15-rt1, kept frames only), seed 97:
#       teapot, homer, dragon x render arm LC97 and twin PC97 (--render_weight_scale 0) -> output/gpu/d145t_cell/; each
#       followed on the same card by B1's GPU readings (read300.sh's GPU part: the end frame, the yardstick t3end, the
#       display e3end, the arrangement, momentum on every kept frame) and ext_end145.py on its end; its CPU readings
#       (r5_runcpu.sh) appended to r5's CPU queue. The dragon starts only when no other dragon 300k run is on the server.
#   (3) B1's L97 kept frames of teapot, bob, armadilo, homer: the yardstick on every kept frame (t3_) and the display on
#       every second kept frame and the last (e3_), into output/gpu/d145t_b1 beside bunny's and dragon's.
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp; G=$B/output/gpu; L=$B/gpu_locks; R=$B/tmp/d145r
D=$G/d145t_cell; X=$G/d145_ext/r5; S=$D/status.txt
mkdir -p $D $X
echo $$ > $X/gpu.pid
source $T/d145/env_d145.sh
declare -A idle; g=""; looks=0
echo "chain waits for a card $(date '+%m-%d %H:%M')" >> $S
while [ -z "$g" ] && [ $looks -lt 720 ]; do
  for c in 0 1 2 3; do
    if [ ! -e $L/gpu$c.lock ] && [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $c)" -lt 2000 ]; then
      idle[$c]=$(( ${idle[$c]:-0} + 1 )); else idle[$c]=0; fi
  done
  for c in 0 1 2 3; do [ ${idle[$c]:-0} -ge 11 ] && mkdir $L/gpu$c.lock 2>/dev/null && { g=$c; break; }; done
  [ -z "$g" ] && { looks=$((looks + 1)); sleep 60; }
done
[ -z "$g" ] && { echo "chain: no card idle 10 min within 12 h $(date '+%m-%d %H:%M')" >> $S; exit 1; }
trap "rm -rf $L/gpu$g.lock" EXIT
printf "agent D145r\nrun d145t_cell: 6 x 300k --cell_follows_n (rt1) + B1 L97 every-frame readings (4 meshes)\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
echo "chain gpu $g $(date '+%m-%d %H:%M')" >> $S
box_of() { case $1 in bunny) echo 1900,100,2800,700;; dragon) echo 1900,60,2500,560;; *) echo 0,0,3840,2160;; esac; }
for spec in "teapot LC97 --seed 97 --cell_follows_n" "teapot PC97 --seed 97 --render_weight_scale 0 --cell_follows_n" \
            "homer LC97 --seed 97 --cell_follows_n" "homer PC97 --seed 97 --render_weight_scale 0 --cell_follows_n" \
            "dragon LC97 --seed 97 --cell_follows_n" "dragon PC97 --seed 97 --render_weight_scale 0 --cell_follows_n"; do
  set -- $spec; mesh=$1; arm=$2; shift 2; args="$*"; name=${mesh}300k_$arm
  if [ -e $D/rc_$name ] && [ "$(cat $D/rc_$name)" = 0 ]; then              # done before (a restarted chain): only
    if [ ! -s $X/ext_cell_$name.log ]; then                                 # the end reading it may have missed
      ( export PHYSMORPH_RUN_REPO=$B/repo_t; source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1; cd $B/repo_t
        CUDA_VISIBLE_DEVICES=$g $PY $R/ext_end145.py $X/ext_cell.jsonl $mesh:cell:$name=$D/${name}_end1.npz > $X/ext_cell_$name.log 2>&1 < /dev/null )
      grep -q "r5_runcpu.sh $name " $X/tasks.txt || echo "bash $R/r5_runcpu.sh $name $mesh" >> $X/tasks.txt
    fi
    continue
  fi
  # a pause (the coordinator, e.g. cards for a demo): touch PAUSE beside the tasks; the chain releases its card and stops
  # before its next run, and a restart resumes there
  [ -e $X/PAUSE ] && { echo "paused before $name $(date '+%m-%d %H:%M')" >> $S; exit 0; }
  if [ "$mesh" = dragon ]; then
    while [ "$(ps -eo args | grep '[p]ipeline_run.py' | grep 'assets/dragon.obj' | grep -c -- '--n 300000')" != 0 ]; do sleep 60; done
  fi
  echo "start $name gpu $g repo_t1s $args $(date '+%m-%d %H:%M')" >> $S
  (
    export PHYSMORPH_RUN_REPO=$B/repo_t1s PHYSMORPH_OUT=$G
    source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null
    cd $B/repo_t1s
    ( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)"; sleep 10; done ) > $D/$name.mem &
    mem=$!
    t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$g $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n 300000 --out $D/$name $args > $D/$name.log 2>&1 < /dev/null
    rc=$?
    kill $mem
    echo $rc > $D/rc_$name
    echo "$name sim $rc wall $(( $(date +%s) - t0 )) peak_mib $(sort -k2 -n $D/$name.mem | tail -1 | cut -d' ' -f2) $(date '+%m-%d %H:%M')" >> $S
  )
  [ "$(cat $D/rc_$name)" != 0 ] && { echo "$name FAILED $(date '+%m-%d %H:%M')" >> $S; continue; }
  F=$D/${name}_frames12.npz; IND=$G/d90/${mesh}_ind300k.npz
  (
    unset PHYSMORPH_RUN_REPO
    source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null
    cd $B/repo_r77
    last=$($PY $T/lastraw.py $F)
    $PY $T/d145/endframe.py $F $D/${name}_end1.npz
    CUDA_VISIBLE_DEVICES=$g $PY scripts/probes/settled/render_terms_probe.py $D/${name}_end1.npz ref=$IND > $D/t3end_$name.log 2>&1 < /dev/null
    CUDA_VISIBLE_DEVICES=$g $PY scripts/probes/settled/surface_layer_probe.py $F $D/e3end_$name $(box_of $mesh) measure ref=$IND target own $last > $D/e3end_$name.log 2>&1 < /dev/null
    CUDA_VISIBLE_DEVICES=$g $PY $T/arrangement.py $F $last > $D/a_$name.log 2>&1 < /dev/null
    $T/mom.sh $g $D/m_$name.log "$mesh ${name#${mesh}300k_}=$F"
  )
  (
    export PHYSMORPH_RUN_REPO=$B/repo_t
    source $B/repo_t/scripts/ops/hyde06_env.sh > /dev/null 2>&1
    cd $B/repo_t
    CUDA_VISIBLE_DEVICES=$g $PY $R/ext_end145.py $X/ext_cell.jsonl $mesh:cell:$name=$D/${name}_end1.npz > $X/ext_cell_$name.log 2>&1 < /dev/null
  )
  echo "read gpu end $name $(date '+%m-%d %H:%M')" >> $S
  echo "bash $R/r5_runcpu.sh $name $mesh" >> $X/tasks.txt
done
# (3) the every-frame readings of B1's L97 on the four other meshes
for mesh in teapot bob armadilo homer; do
  name=${mesh}300k_L97; F=$G/d145t_b1/${name}_frames12.npz; IND=$G/d90/${mesh}_ind300k.npz; O=$G/d145t_b1
  grep -q "^read gpu full $name " $S && continue
  [ -e $X/PAUSE ] && { echo "paused before full $name $(date '+%m-%d %H:%M')" >> $S; exit 0; }
  echo "start full $name gpu $g $(date '+%m-%d %H:%M')" >> $S
  (
    unset PHYSMORPH_RUN_REPO
    source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null
    cd $B/repo_r77
    raws=$($PY -c "import numpy as np; r = [int(v) for v in np.load('$F', mmap_mode='r')['raws']]; print(' '.join(str(v) for v in sorted(set(r[::2] + [r[-1]]))))")
    CUDA_VISIBLE_DEVICES=$g $PY scripts/probes/settled/render_terms_probe.py $F ref=$IND > $O/t3_$name.log 2>&1 < /dev/null
    CUDA_VISIBLE_DEVICES=$g $PY scripts/probes/settled/surface_layer_probe.py $F $O/e3_$name 0,0,3840,2160 measure ref=$IND target own $raws > $O/e3_$name.log 2>&1 < /dev/null
  )
  echo "read gpu full $name $(date '+%m-%d %H:%M')" >> $S
done
echo "chain done gpu $g $(date '+%m-%d %H:%M')" >> $S
