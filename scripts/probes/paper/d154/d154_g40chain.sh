#!/bin/bash
# d154_g40chain.sh -- D154 follow-up (3b): the other 14 gallery meshes at 40k x V / W x 97 / 99 / 101 (B3's lines:
# --tgt assets/MESH.obj --n 40000 --seed S, W with --render_weight_scale 0) on repo_t1s (rt1's kept-frames copy), in
# three seed batches. Each batch claims ONE card by the lock protocol (no lock, no other user's compute process, none of
# mine, < 2000 MiB, idle 10 min by the pool's shared first-free time; GPU 2 never) and releases it at the batch's end, so
# other agents can take turns. Each batch is then read on the CPU (mesh_frames.py at N = 40000, d154_mesh_read.py with
# D154_REF=own, 2 readers at nice 10) while the next batch waits for a card; one reading at a time. Frames are kept.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d154; D=$O/g40runs; W=$B/tmp/d154; L=$B/gpu_locks
mkdir -p $D $O/read $W/jobs40b
source $B/tmp/d146/cache_env.sh
PYB=/home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python
S=$O/status.txt
MESHES="A beast bimba bob C cheburashka cow fandisk heart maxplanck nefertiti ogre spot V"
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
CARD=""
claim_card() {
  declare -A since
  while [ -z "$CARD" ]; do
    local now=$(date +%s)
    for g in 3 1 0; do
      local used=$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)
      local uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader)
      local other=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | while IFS=', ' read -r u p; do
                    o=$(ps -o user= -p $p 2>/dev/null); [ -n "$o" ] && [ "$o" != chayo ] && echo x; done | wc -l)
      local mine=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")
      if [ -d $L/gpu$g.lock ] || [ "$used" -gt 2000 ] || [ "$other" -gt 0 ] || [ "$mine" -gt 0 ]; then since[$g]=""; continue; fi
      [ -n "${since[$g]}" ] || since[$g]=$now
      local fs=$(cat $B/tmp/d145/free_since/gpu$g 2>/dev/null); [ -n "$fs" ] && [ "$fs" -lt "${since[$g]}" ] && since[$g]=$fs
      if [ $((now - ${since[$g]})) -ge 600 ] && mkdir $L/gpu$g.lock 2>/dev/null; then
        CARD=$g
        printf "agent D154\nrun d154/g40 batch %s (14 meshes x V/W at 40k, about 55 min; released at the batch end)\npid %s\nstart %s\n" \
          "$1" "$$" "$(date "+%Y-%m-%d %H:%M:%S %Z")" > $L/gpu$g.lock/info
        break
      fi
    done
    [ -z "$CARD" ] && sleep 60
  done
}
release_card() { [ -n "$CARD" ] && rm -rf $L/gpu$CARD.lock && log "g40: released gpu$CARD"; CARD=""; }
trap 'release_card' EXIT
trap 'exit 1' INT TERM HUP
read_batch() {   # SEED: the batch's frames against the exact mesh (CPU)
  local s=$1 specs="" k=0
  for m in $MESHES; do specs="$specs $m:$s:40000:$D/${m}40k_V${s}_frames12.npz"; done
  ( export PHYSMORPH_CACHE=$B/cache/samples OMP_NUM_THREADS=4 TMPDIR=$W/tmpdir CUDA_VISIBLE_DEVICES=""; cd $B/repo_ts
    nice -n 10 $PYB $W/mesh_frames.py $O/meshes40 $specs 2>/dev/null | grep '^{' >> $O/mesh_frames40.jsonl )
  rm -f $W/jobs40b/j${s}_*.txt
  for m in $MESHES; do
    grep "\"mesh\": \"$m\", \"seed\": $s, \"n\": 40000, \"same_as_run_target\": true" $O/mesh_frames40.jsonl > /dev/null || { log "g40 read $s: $m frame not reproduced, skipped"; continue; }
    f=$W/jobs40b/j${s}_$((k % 2)).txt
    printf "%s %s V %s end\n%s %s W %s end\n%s %s own %s tgt\n" $m $s $D/${m}40k_V${s}_frames12.npz $m $s $D/${m}40k_W${s}_frames12.npz \
      $m $s $D/${m}40k_V${s}_frames12.npz >> $f
    k=$((k + 1))
  done
  cd $W
  for j in 0 1; do
    CUDA_VISIBLE_DEVICES="" D154_REF=own D154_MESHES=$O/meshes40 D154_THREADS=6 OMP_NUM_THREADS=6 TMPDIR=$W/tmpdir \
      nice -n 10 $PYB d154_mesh_read.py $O/read/g40b_${s}_$j.jsonl $W/jobs40b/j${s}_$j.txt > $O/read/g40b_${s}_$j.log 2>&1 < /dev/null &
  done
  wait
  log "g40 read $s done: $(cat $O/read/g40b_${s}_*.jsonl | wc -l) states"
}
log "g40 chain start pid $$ (3b: 84 runs in three seed batches; waiting for a card idle 10 min)"
rpid=""
for seed in 97 99 101; do
  claim_card "seed $seed"
  log "g40 batch $seed: claimed gpu$CARD"
  for mesh in $MESHES; do for arm in V W; do
    name=${mesh}40k_${arm}$seed
    [ -e $D/${name}_frames12.npz ] && { log "g40 $name exists, skipped"; continue; }
    flag=""; [ $arm = W ] && flag="--render_weight_scale 0"
    ( export PHYSMORPH_RUN_REPO=$B/repo_t1s PHYSMORPH_OUT=$G OMP_NUM_THREADS=8
      source $B/repo_t1s/scripts/ops/hyde06_env.sh > /dev/null 2>&1; export TMPDIR=$W/tmpdir; cd $B/repo_t1s
      t0=$(date +%s)
      CUDA_VISIBLE_DEVICES=$CARD $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n 40000 --seed $seed --out $D/$name $flag \
        > $D/$name.log 2>&1 < /dev/null
      echo "$? $(( $(date +%s) - t0 ))" > $D/rc_$name )
    log "g40 $name rc/wall $(cat $D/rc_$name)"
  done; done
  release_card
  [ -n "$rpid" ] && wait $rpid
  read_batch $seed &
  rpid=$!
done
wait $rpid
log "g40 chain done"
