#!/bin/bash
# b13_chain300k.sh REPO_S QUEUE [BATCH] -- B13's 300k runs on the re-freeze tag: each QUEUE line "NAME MESH SEED KEEP
# FLAGS..." (NAME = MESH300k_ARM, e.g. bunny300k_F97; KEEP 1 keeps the kept frames after the readings, 0 removes them,
# logged in output/gpu/cleanup.log by read300_cpu.sh) runs scripts/pipeline_run.py from REPO_S (the tag's kept-frames
# copy, repo_ts) into output/gpu/b13, then on the same card:
#   - the end against the independent sample with the floor (tmp/d145/endgen145.py -> endgen_b13.jsonl),
#   - D145's readings (tmp/d145/read300.sh MODE end: the end frame, the yardstick and the display at the end, the
#     arrangement, momentum on every kept frame; its CPU part detached: apart particles, the tail pulse),
#   - extras(): the default arm's (B1 L on the tag, output/gpu/d145t_b1) end against the independent sample as its runs
#     land (endgen_L.jsonl; the four meshes whose L99 / L101 frames D145 removes are read from the hard links that
#     b13_hold.sh keeps in output/gpu/b13/hold, removed after the reading, cleanup.log), and the harness at matched
#     progress per mesh on seed 97 (tmp/d145/harness.sh: L97, F97, FM97 -> output/gpu/b13/harness).
# One card at a time, claimed through the lock directory only when it has had no lock, under 2 GB and under 10 % use for
# 10 consecutive minutes; held for BATCH items (default 3), then released and claimed again, so that the agents before
# B13 can take it. D147's final scene goes first: no claim while a D147 process runs without a D147 lock (it is waiting
# for a card), and the card is released after the current item when that happens. After the queue, the chain waits for
# the default arm's remaining runs and claims a card only when one of them is ready for extras(). A line already marked
# "done NAME" in the status file is skipped. Disk: an item starts only while du(output) stays under 480 GB.
source /data/relcfd/chayo/physmorph_v2/tmp/d146/cache_env.sh
B=/data/relcfd/chayo/physmorph_v2; L=$B/gpu_locks; G=$B/output/gpu; RS=$1; Q=$2; BATCH=${3:-3}
D=$G/b13; S=$D/status300.txt; LD=$G/d145t_b1; H=$D/hold
MESHES="bunny dragon teapot bob armadilo homer"
mkdir -p $D $H $D/harness
gpu=""
d147_waiting() {  # a D147 process exists (its scripts: output/scratch/d147_*, repo_d147*) and no lock says agent D147
  ps -eo args= | grep -q "[d]147" || return 1
  grep -qs "^agent D147" $L/gpu*.lock/info && return 1
  return 0
}
info() {  # the lock's info file: agent / run / pid / start
  printf "agent B13\nrun b13/%s (force baseline 300k, %s items a claim)\npid %s\nstart %s\n" "$1" "$BATCH" "$$" \
    "$(date '+%Y-%m-%d %H:%M:%S %Z')" > $L/gpu$gpu.lock/info
}
claim() {
  declare -A since
  local said=""
  echo "waiting for a card idle 10+ min $(date '+%m-%d %H:%M:%S')" >> $S
  while [ -z "$gpu" ]; do
    now=$(date +%s)
    if d147_waiting; then
      [ -z "$said" ] && echo "D147 waiting for a card: no claim until it holds one ($(ps -eo pid=,args= | grep "[d]147" | head -1 | cut -c1-120)) $(date '+%m-%d %H:%M:%S')" >> $S
      said=1; since=(); sleep 60; continue
    fi
    for g in 0 1 2 3; do
      q=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i $g | tr -d ' ')
      used=${q%,*}; util=${q#*,}
      if [ ! -d $L/gpu$g.lock ] && [ "$used" -lt 2000 ] && [ "$util" -lt 10 ]; then
        [ -z "${since[$g]}" ] && since[$g]=$now
        if [ $(( now - ${since[$g]} )) -ge 600 ] && mkdir $L/gpu$g.lock 2>/dev/null; then gpu=$g; break; fi
      else
        since[$g]=""
      fi
    done
    [ -z "$gpu" ] && sleep 30
  done
  info "claimed"
  echo "claimed gpu $gpu $(date '+%m-%d %H:%M:%S')" >> $S
}
release() {
  [ -n "$gpu" ] && rm -rf $L/gpu$gpu.lock && echo "released gpu $gpu $(date '+%m-%d %H:%M:%S')" >> $S
  gpu=""
}
endgen() {  # OUT_JSONL MESH FRAMES -- on the held card
  ( export PHYSMORPH_RUN_REPO=$RS; source $RS/scripts/ops/hyde06_env.sh > /dev/null 2>&1; cd $RS
    CUDA_VISIBLE_DEVICES=$gpu $PY $B/tmp/d145/endgen145.py $1 $2=$3 >> $D/endgen.log 2>&1 < /dev/null )
}
l_ready() {  # NAME -> the frames path of a finished L run (d145t_b1, or the hold link), or nothing
  local name=$1
  grep -q "^$name sim 0 " $LD/status.txt 2>/dev/null || return 1
  if [ -f $LD/${name}_frames12.npz ]; then echo $LD/${name}_frames12.npz; return 0; fi
  if [ -f $H/${name}_frames12.npz ]; then echo $H/${name}_frames12.npz; return 0; fi
  return 1
}
pending() {  # count of extras still to do; with "ready", only those whose inputs are there now
  local c=0 m s f
  for m in $MESHES; do
    for s in 97 99 101; do
      grep -q "^L-endgen ${m}300k_L$s " $S && continue
      if [ "$1" = ready ]; then l_ready ${m}300k_L$s > /dev/null && c=$((c + 1)); else c=$((c + 1)); fi
    done
    grep -q "^harness $m " $S && continue
    if [ "$1" = ready ]; then
      f=1; l_ready ${m}300k_L97 > /dev/null || f=0
      for a in F97 FM97; do grep -q "^done ${m}300k_$a rc 0" $S && [ -f $D/${m}300k_${a}_frames12.npz ] || f=0; done
      c=$((c + f))
    else c=$((c + 1)); fi
  done
  echo $c
}
extras() {  # on the held card: whatever of the default arm's readings is ready now
  local m s name f
  for m in $MESHES; do
    for s in 97 99 101; do
      name=${m}300k_L$s
      grep -q "^L-endgen $name " $S && continue
      f=$(l_ready $name) || continue
      info "endgen $name"
      endgen $D/endgen_L.jsonl $m $f
      echo "L-endgen $name rc $? from $f $(date '+%m-%d %H:%M')" >> $S
      if [ "${f#$H/}" != "$f" ]; then
        sz=$(du -m $f | cut -f1); rm -f $f
        echo "$(date '+%Y-%m-%d %H:%M') b13: removed $f ($sz MB; hard link to D145's $name kept for B13's end reading, read: endgen_L.jsonl)" >> $G/cleanup.log
      fi
    done
    grep -q "^harness $m " $S && continue
    f=$(l_ready ${m}300k_L97) || continue
    [ "${f#$H/}" = "$f" ] || continue
    grep -q "^done ${m}300k_F97 rc 0" $S && grep -q "^done ${m}300k_FM97 rc 0" $S || continue
    [ -f $D/${m}300k_F97_frames12.npz ] && [ -f $D/${m}300k_FM97_frames12.npz ] || continue
    info "harness $m"
    ( export CUDA_VISIBLE_DEVICES=$gpu; bash $B/tmp/d145/harness.sh $D/harness $m $G/d90/${m}_ind300k.npz L97=$f \
        F97=$D/${m}300k_F97_frames12.npz FM97=$D/${m}300k_FM97_frames12.npz > /dev/null 2>&1 < /dev/null )
    echo "harness $m rc $? $(date '+%m-%d %H:%M')" >> $S
  done
}
trap 'release' EXIT
trap 'exit 1' INT TERM HUP
echo "chain300k start: repo $RS ($(cat $RS/TAG 2>/dev/null)), queue $Q, batch $BATCH, pid $$ $(date '+%m-%d %H:%M:%S')" >> $S
n=0
while read -r name mesh seed keep flags; do
  [ -z "$name" ] && continue; [ "${name:0:1}" = "#" ] && continue
  grep -q "^done $name " $S 2>/dev/null && continue
  while [ "$(du -sm $B/output | cut -f1)" -gt 480000 ]; do
    echo "disk: output over 480 GB, waiting $(date '+%m-%d %H:%M')" >> $S; release; sleep 600
  done
  [ -z "$gpu" ] && claim
  info "$name"
  export PHYSMORPH_RUN_REPO=$RS PHYSMORPH_OUT=$D
  source $RS/scripts/ops/hyde06_env.sh > /dev/null
  cd $RS
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n 300000 --seed $seed $flags \
    --out $D/$name > $D/$name.log 2>&1 < /dev/null
  rc=$?
  echo "run $name rc $rc wall $(( $(date +%s) - t0 )) s gpu $gpu $(date '+%m-%d %H:%M')" >> $S
  if [ -f $D/${name}_frames12.npz ]; then
    info "$name end against the independent sample"
    endgen $D/endgen_b13.jsonl $mesh $D/${name}_frames12.npz
    echo "endgen $name rc $? $(date '+%m-%d %H:%M')" >> $S
    info "$name readings"
    bash $B/tmp/d145/read300.sh $gpu b13 $name $mesh end $keep >> $D/read_$name.out 2>&1 < /dev/null
    echo "read gpu $name $(date '+%m-%d %H:%M')" >> $S
  fi
  echo "done $name rc $rc $(date '+%m-%d %H:%M')" >> $S
  extras
  n=$((n + 1))
  if [ $(( n % BATCH )) = 0 ]; then release; elif d147_waiting; then echo "D147 waiting: card yielded" >> $S; release; fi
done < $Q
release
echo "chain300k queue done $(date '+%m-%d %H:%M'); extras left $(pending all)" >> $S
# the default arm's runs that land later: claim only when one is ready (up to 72 h)
t_end=$(( $(date +%s) + 259200 ))
while [ "$(pending all)" -gt 0 ] && [ $(date +%s) -lt $t_end ]; do
  if [ "$(pending ready)" -gt 0 ]; then claim; extras; release; else sleep 600; fi
done
echo "chain300k done $(date '+%m-%d %H:%M'); extras left $(pending all)" >> $S
