#!/bin/bash
# pool.sh QUEUE WID -- D145's lock-aware worker (start several; each runs one item at a time).
# QUEUE lines: "NAME MESH N REPO DIR EVAL KEEP [pipeline_run args]" ('#' = comment), in priority order.
#   EVAL: full300 | end300 | g40 (see read300.sh / the g40 branch below); KEEP: 1 keeps the kept frames, 0 removes them
#   after the readings (logged in output/gpu/cleanup.log).
# A card (the agents' shared protocol, 2026-10-14): no lock dir gpu_locks/gpuN.lock and < 2000 MiB used; claimed by mkdir,
# info written inside (agent, run, pid, start), released by a trap when the item's process exits. At most
# tmp/d145/max_cards cards held by D145 at once (default 3: one left for D146/D147).
# Disk (the user's rule): a 300k item starts only when du(output) + the estimated end-of-run archives of every 300k
# pipeline_run on the server (ps, read-only; dragon 18 GB, other meshes 12, Xu 4) + its own <= 48 GB, and a dragon only
# when no other dragon 300k run is on the server (and our dragon lock is free). 40k items backfill while 300k items wait.
# STOP: touch tmp/d145/STOP ends every worker after its current item.
source /data/relcfd/chayo/physmorph_v2/tmp/d145/env_d145.sh
B=/data/relcfd/chayo/physmorph_v2; T=$B/tmp; G=$B/output/gpu; L=$B/gpu_locks; mkdir -p $L
Q=$1; wid=$2; LOG=$G/d145_pool.log
est() {  # MESH N REPO ARGS... -> estimated end-of-run disk GB (a run copy writing the kept frames only: 2)
  local mesh=$1 n=$2 repo=$3; shift 3
  [ "$n" -lt 300000 ] && { echo 1; return; }
  case " $repo " in *repo_d145s*|*repo_d146h*) echo 2; return;; esac
  case " $* " in *" --baseline xu"*) echo 4; return;; esac
  [ "$mesh" = dragon ] && echo 18 || echo 12
}
running_est() {  # sum of the estimates of every 300k pipeline_run now on the server (its repo from /proc/PID/cwd)
  ps -eo pid=,args= | grep "[p]ipeline_run.py" | while read -r pid a; do
    m=$(echo "$a" | sed -n 's#.*--tgt assets/\([A-Za-z0-9_]*\)\.obj.*#\1#p'); n=$(echo "$a" | sed -n 's#.*--n \([0-9]*\).*#\1#p')
    est "${m:-dragon}" "${n:-300000}" "$(readlink /proc/$pid/cwd 2>/dev/null || echo unknown)" $a
  done | awk '{s += $1} END {print s + 0}'
}
full_dragons() {  # 300k dragon runs on the server that write the full archive
  ps -eo pid=,args= | grep "[p]ipeline_run.py" | grep "assets/dragon.obj" | grep -- "--n 300000" | while read -r pid a; do
    [ "$(est dragon 300000 "$(readlink /proc/$pid/cwd 2>/dev/null || echo unknown)" $a)" -gt 2 ] && echo x
  done | wc -l
}
startable() {  # MESH N REPO ARGS... -> 0 if the disk and dragon rules allow it now
  local mesh=$1 n=$2 repo=$3; shift 3
  [ "$mesh" = job ] && return 0
  [ -e $T/d145/no_dragon ] && [ "$mesh" = dragon ] && [ "$n" -ge 300000 ] && return 1   # D148 window (coordinator)
  [ "$n" -lt 300000 ] && return 0
  local used=$USED_GB
  local mine=$(est $mesh $n $repo "$@")
  local tot=$(( used + RUN_EST + mine ))
  [ $tot -le 490 ] || return 1          # the user's limit 500 GB (2026-10-14 ~09:15), headroom kept
  # (the one-dragon-at-a-time rule was lifted with the 200 GB limit)
  return 0
}
claim_card() {  # -> card id, or nothing
  local maxc=$(cat $T/d145/max_cards 2>/dev/null || echo 3)
  local mine=$(grep -l "^agent D145" $L/gpu*.lock/info 2>/dev/null | wc -l)
  [ $mine -lt $maxc ] || return 1
  for g in 0 1 2 3; do
    [ -e $L/gpu$g.lock ] && continue
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)" -lt 2000 ] || continue
    if mkdir $L/gpu$g.lock 2>/dev/null; then
      printf "agent D145\nrun pending\npid $$\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
      echo $g; return 0
    fi
  done
  return 1
}
while true; do
  [ -e $T/d145/STOP ] && { echo "$(date '+%m-%d %H:%M') worker $wid stops (STOP)" >> $LOG; exit 0; }
  g=$(claim_card) || { sleep 60; continue; }
  picked=""; USED_GB=$(( $(du -sm $B/output | cut -f1) / 1024 )); RUN_EST=$(running_est)
  while read -r name mesh n repo dir ev keep args; do
    [ -z "$name" ] && continue; [ "${name:0:1}" = "#" ] && continue
    D=$G/$dir; mkdir -p $D
    [ -e $D/$name.claim ] && continue
    startable $mesh $n $repo $args || continue
    dl=0
    if mkdir $D/$name.claim 2>/dev/null; then picked="$name $mesh $n $repo $dir $ev $keep $args"; break; fi
    [ "$dl" = 1 ] && rmdir $G/d145_dragon.lock
  done < $Q
  if [ -z "$picked" ]; then rm -rf $L/gpu$g.lock; sleep 60; continue; fi
  set -- $picked; name=$1; mesh=$2; n=$3; repo=$4; dir=$5; ev=$6; keep=$7; shift 7; args="$*"
  D=$G/$dir; S=$D/status.txt
  (
    trap "rm -rf $L/gpu$g.lock" EXIT
    printf "agent D145\nrun $dir/$name\npid $BASHPID\nstart $(date '+%Y-%m-%d %H:%M:%S %Z')\n" > $L/gpu$g.lock/info
    echo "start $name gpu $g w$wid $repo $args $(date '+%m-%d %H:%M')" >> $S
    if [ "$mesh" = job ]; then
      ( export OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$g; bash -c "$args" > $D/$name.log 2>&1 < /dev/null; echo $? > $D/rc_$name )
      echo "done $name job rc $(cat $D/rc_$name) $(date '+%m-%d %H:%M')" >> $S
      exit 0
    fi
    (
      export PHYSMORPH_RUN_REPO=$B/$repo PHYSMORPH_OUT=$G OMP_NUM_THREADS=8
      source $B/$repo/scripts/ops/hyde06_env.sh > /dev/null
      cd $B/$repo
      ( while true; do echo "$(date +%s) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)"; sleep 10; done ) > $D/$name.mem &
      mem=$!
      t0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$g $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n $n --out $D/$name $args > $D/$name.log 2>&1 < /dev/null
      rc=$?
      kill $mem
      echo $rc > $D/rc_$name
      echo "$name sim $rc wall $(( $(date +%s) - t0 )) peak_mib $(sort -k2 -n $D/$name.mem | tail -1 | cut -d' ' -f2) $(date '+%m-%d %H:%M')" >> $S
      A=$D/${name}_render_full_dt_iso_nn.npz
      [ $rc = 0 ] && [ -e $A ] && $PY $T/frames12.py $A $D/$name > $D/$name.frames12.log 2>&1
      rm -f $A
    )
    [ "$dl" = 1 ] && rmdir $G/d145_dragon.lock
    if [ "$(cat $D/rc_$name 2>/dev/null)" != 0 ]; then echo "$name FAILED $(date '+%m-%d %H:%M')" >> $S; exit 0; fi
    F=$D/${name}_frames12.npz
    if [ "${ev%f}" = g40 ]; then             # GPU: the independent-sample reading; then the card is released
      IND40=$G/d92/ind_${mesh}.npz
      if [ "$ev" = g40f ]; then ( source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; $PY $T/d145/ind_shift.py $IND40 $F $D/ind_floor_$name.npz > $D/ind_shift_$name.log 2>&1 ); IND40=$D/ind_floor_$name.npz; fi
      ( source $B/repo_r77/scripts/ops/hyde06_env.sh > /dev/null; cd $B/repo_r77
        CUDA_VISIBLE_DEVICES=$g $PY $T/gallery_ind.py $D/ind.log "$name=$F,$IND40" > /dev/null 2>&1 < /dev/null )
      rm -rf $L/gpu$g.lock; trap - EXIT
      setsid nohup bash $T/d145/read300_cpu.sh $dir $name $mesh $keep > /dev/null 2>&1 < /dev/null &
    elif [ "$ev" = rec ]; then
      rm -rf $L/gpu$g.lock; trap - EXIT
      if [ "$keep" = 0 ]; then sz=$(du -m $F | cut -f1); rm -f $F
        echo "$(date '+%Y-%m-%d %H:%M') $dir: removed $F ($sz MB; D145 run $name, records only)" >> $G/cleanup.log; fi
      echo "done $name $(date '+%m-%d %H:%M')" >> $S
    else                                     # GPU readings on the card, the CPU part detached (read300.sh)
      bash $T/d145/read300.sh $g $dir $name $mesh $ev $keep >> $D/read_$name.out 2>&1 < /dev/null
      rm -rf $L/gpu$g.lock; trap - EXIT
    fi
  )
  sleep 90   # fairness: other agents polling the locks get a window before this worker claims again
done
