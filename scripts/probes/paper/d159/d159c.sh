#!/bin/bash
# d159c.sh -- D159c (pre-registered in docs/experiments.md, D159 entry): D159's pictures (--render_target mesh_shell) with
# the render weight held from the first window (--lambda_ema 0, the existing pre-D81 option), 300k bunny and dragon, seed
# 97, --term_dump; on repo_d159s (kept frames), ONE card by the lock protocol (d159_claim.sh: free 600 s, never GPU 2),
# both runs on it, then B1's GPU readings (read300.sh ... end 1), the card released, then the CPU readings
# (d159c_read.sh). LAUNCH ONLY when D160c's after-arrival timeline shows the colour still moving with lambda held.
B=/data/relcfd/chayo/physmorph_v2; G=$B/output/gpu; O=$G/d159; W=$B/tmp/d159; L=$B/gpu_locks
S=$O/status.txt
log() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $S; }
source $B/tmp/d146/cache_env.sh
export TMPDIR=$W/tmpdir
source $W/d159_claim.sh
trap 'release_card' EXIT
trap 'exit 1' INT TERM HUP
log "D159c start pid $$"
claim_card "D159c: 300k bunny/dragon, mesh_shell + lambda_ema 0"
mkdir -p $O/terms
run() {   # NAME MESH
  local name=$1 mesh=$2
  ( export PHYSMORPH_RUN_REPO=$B/repo_d159s PHYSMORPH_OUT=$G; source $B/repo_d159s/scripts/ops/hyde06_env.sh > /dev/null 2>&1
    source $B/tmp/d146/cache_env.sh; export TMPDIR=$W/tmpdir OMP_NUM_THREADS=8; cd $B/repo_d159s
    t0=$(date +%s)
    $PY scripts/pipeline_run.py --tgt assets/$mesh.obj --n 300000 --seed 97 --render_target mesh_shell --lambda_ema 0 \
      --term_dump $O/terms/$name --out $O/$name > $O/$name.log 2>&1 < /dev/null
    rc=$?
    echo "$rc $(( $(date +%s) - t0 ))" > $O/rc_$name
    log "D159c $name rc/wall $(cat $O/rc_$name): $(grep -m1 'ARM render_full_dt_iso_nn:' $O/$name.log | cut -c1-200)"
    [ "$rc" = 0 ] && bash $B/tmp/d145/read300.sh $CARD d159 $name $mesh end 1 > $O/read300_$name.log 2>&1
    log "D159c $name GPU readings done" ) &
}
run bunny300k_C97 bunny
sleep 30
run dragon300k_C97 dragon
wait
release_card
for mesh in bunny dragon; do
  ( trap - EXIT INT TERM HUP; setsid nohup bash $W/d159c_read.sh $mesh > $O/read_c_$mesh.out 2>&1 < /dev/null & )
done
log "D159c runs done; CPU readings started"
