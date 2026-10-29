#!/bin/bash
# hyde01 render node, the server side (installed as ~/physmorph_render/bin/render_job.sh; driven by the local relay
# scripts/ops/hyde01_render.sh). Everything stays under ~/physmorph_render: hyde01's /data is another, full filesystem.
#   render_job.sh submit JOB B64ENV   stdin = a tar (the frames npz, dense samples, optionally physmorph/ + scripts/):
#                                     unpack into queue/JOB, write job.env, start the worker, wait for it, print status
#   render_job.sh wait JOB            wait for a started job (after a dropped connection) and print its status
#   render_job.sh fetch JOB           the job's outputs (mp4, logs, rows, frame tars, status) as a tar on stdout
#   render_job.sh run JOB             the worker: claim a free GPU, render each view, ffmpeg, delete the frames
# A GPU is used only while nvidia-smi lists no compute process on it (other users' jobs, or another job of ours: one
# render per GPU) and its lock (locks/gpuN.lock, mkdir) is free. The frames npz and the jpg frames are deleted after the
# job; the mp4s, logs and rows stay in queue/JOB until it is 3 days old.
source $HOME/physmorph_render/env.sh
MODE=$1
JOB=$2
J=$R/queue/$JOB
LOCKS=$R/locks
st() { echo "$* $(date '+%m-%d %H:%M:%S')" >> $J/status; }
busy() {                                       # any compute process on GPU $1
  local uuid
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1") || return 0
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$uuid"
}
wait_done() {
  while [ ! -e $J/DONE ]; do
    if [ -s $J/worker.pid ] && ! kill -0 "$(cat $J/worker.pid)" 2>/dev/null && [ ! -e $J/DONE ]; then
      st "worker died"; echo 1 > $J/DONE
    fi
    sleep 20
  done
  cat $J/status
  exit "$(cat $J/DONE)"
}

case $MODE in
submit)
  find $R/queue -mindepth 1 -maxdepth 1 -type d -mtime +3 -exec rm -rf {} \; 2>/dev/null
  avail=$(df --output=avail -BG $HOME | tail -1 | tr -dc 0-9)
  [ "$avail" -ge 40 ] || { echo "hyde01 home has only ${avail} GB free"; exit 4; }
  mkdir -p $J
  echo "$3" | base64 -d > $J/job.env
  source $J/job.env
  t0=$(date +%s%N)
  tar xf - -C $J || { echo "upload failed"; exit 3; }
  t1=$(date +%s%N)
  size=$(stat -c %s $J/$NPZ_BASE 2>/dev/null || echo 0)
  [ "$size" = "$NPZ_SIZE" ] || { echo "upload truncated: $size of $NPZ_SIZE bytes"; exit 3; }
  awk -v a=$t0 -v b=$t1 -v s=$size 'BEGIN{t=(b-a)/1e9; printf "upload %.0f MB in %.0f s (%.1f MB/s)\n", s/1e6, t, s/1e6/t}' >> $J/status
  dense=""
  for b in ${DENSE_BASES//,/ }; do
    [ -e $J/$b ] && mv $J/$b $R/cache/dense/$b
    [ -e $R/cache/dense/$b ] || { echo "missing dense sample $b"; exit 5; }
    dense="$dense${dense:+,}$R/cache/dense/$b"
  done
  { echo "NPZ=$J/$NPZ_BASE"; echo "DENSE=$dense"
    if [ "$SHIP_REPO" = 1 ]; then echo "REPO_DIR=$J"; else echo "REPO_DIR=$R/repo"; fi; } >> $J/job.env
  setsid nohup bash $R/bin/render_job.sh run $JOB > $J/worker.log 2>&1 < /dev/null &
  echo $! > $J/worker.pid
  echo "LAUNCHED $JOB"
  wait_done
  ;;
wait)
  [ -d $J ] || { echo "no job $JOB"; exit 6; }
  wait_done
  ;;
fetch)
  source $J/job.env
  cd $J && cp status $NAME.hyde01_status && tar cf - $(ls | grep -E '\.(mp4|log|rows\.txt|sha256|tar)$' | grep -v '^worker\.log$') $NAME.hyde01_status
  ;;
run)
  source $J/job.env
  export PYTHONPATH=$REPO_DIR
  mkdir -p $LOCKS
  st "queued for gpus [$GPUS]"
  GPU=
  while [ -z "$GPU" ]; do
    for g in $GPUS; do
      L=$LOCKS/gpu$g.lock
      if [ -s $L/pid ] && ! kill -0 "$(cat $L/pid)" 2>/dev/null; then rm -rf $L; fi    # a dead holder's lock
      busy $g && continue
      mkdir $L 2>/dev/null || continue
      echo $$ > $L/pid
      echo $JOB > $L/job
      GPU=$g
      break
    done
    [ -z "$GPU" ] && sleep 60
  done
  trap 'rm -rf $LOCKS/gpu$GPU.lock' EXIT
  st "claimed gpu $GPU"
  fail=0
  for v in $VIEWS; do
    vn=${v%%:*}
    vo=""
    [ "$vn" != "$v" ] && vo=${v#*:}
    vo=${vo//,/ }
    out=$J/${NAME}_$vn
    rm -rf $out
    while busy $GPU; do st "gpu $GPU lists another compute process, waiting"; sleep 60; done
    t0=$(date +%s)
    if [ "$KIND" = scene ]; then
      CUDA_VISIBLE_DEVICES=$GPU $PY $REPO_DIR/scripts/render_scene.py $NPZ $out $OPTS $vo > $out.log 2>&1 < /dev/null
    else
      CUDA_VISIBLE_DEVICES=$GPU $PY $REPO_DIR/scripts/render_deliverable.py $NPZ $DENSE $out $OPTS $vo > $out.log 2>&1 < /dev/null
    fi
    rc=$?
    frc=-
    if [ $rc -eq 0 ]; then
      ffmpeg -loglevel error -y -framerate $FPS -pattern_type glob -i "$out/frames/*.jpg" -c:v libx264 -crf 18 \
        -pix_fmt yuv420p $J/${NAME}_$vn.mp4 < /dev/null
      frc=$?
    fi
    nf=$(ls $out/frames 2>/dev/null | wc -l)
    [ "$nf" -gt 0 ] && (cd $out/frames && sha256sum *.jpg) > $J/${NAME}_$vn.sha256   # to check frames against hyde06's
    st "render ${NAME}_$vn rc=$rc ffmpeg=$frc frames=$nf $(( $(date +%s) - t0 )) s hyde01 gpu $GPU"
    [ "$rc" = 0 ] && [ "$frc" = 0 ] || fail=1
    [ -e $out/rows.txt ] && cp $out/rows.txt $J/${NAME}_$vn.rows.txt
    [ "$KEEP_FRAMES" = 1 ] && [ -d $out/frames ] && tar cf $J/${NAME}_${vn}_frames.tar -C $out frames
    rm -rf $out
  done
  rm -f $NPZ
  [ "$SHIP_REPO" = 1 ] && rm -rf $J/physmorph $J/scripts
  st "DONE fail=$fail"
  echo $fail > $J/DONE
  ;;
*)
  echo "usage: render_job.sh submit|wait|fetch|run JOB [B64ENV]"; exit 2
  ;;
esac
