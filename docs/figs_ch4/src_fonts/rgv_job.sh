#!/bin/bash
# rgv_job.sh -- FIGS_V2 (2026-10-22), called by the gpu1 queue job 34_figs_v2_rgv_tnr.sh (foreground; CUDA_VISIBLE_DEVICES=1
# from the runner): the dissertation's RGV figure (fig_render_gradient_early.jpg / _late.jpg) again with every text in
# Times New Roman (the user) and the body flipped back left-right before the text (the studio camera mirrors; RI's
# render_gradient_viz_flip.py, copied to tmp/figs_v2/). Only the two kept frames the figure uses, 24 (t = 1.2 s) and 100
# (t = 5.0 s); otherwise exactly the tag's render2 call (tmp/rgv_tag/rgv_render_tag.sh: RG97's values, A, B2, C, D,
# side view, 4K panels, +interp=1 +c_avg=5 +d_avg=5, the half-size 2 x 2). Then the figure as before
# (docs/figs_ch4/src_render_gradient/make_render_gradient.sh): the top 1050 rows of each 1920 x 1080 quadrant of the
# half-size 2 x 2, 2 x 2 again, scaled to 2400 x 1312. Needs 14000 MiB free on gpu1, else waits 5 min and queues itself
# again as 99_figs_v2_rgv_tnr_rNN.sh (at most 96 tries).
B=/data/relcfd/chayo/physmorph_v2; O=$B/output/gpu/figs_v2/rgv; T=$B/tmp/figs_v2; Q=$B/tmp/gpu1_queue
mkdir -p $O $T/tmpdir
log() { echo "$(date '+%m-%d %H:%M:%S') rgv: $*" >> $B/output/gpu/figs_v2/status.txt; }
free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
if [ "$free" -lt 14000 ]; then
  n=$(cat $O/tries 2>/dev/null || echo 0); n=$((n + 1)); echo $n > $O/tries
  log "waits: gpu1 has $free MiB free (< 14000), try $n"
  [ $n -ge 96 ] && { log "GAVE UP after $n tries"; exit 3; }
  sleep 300
  nm=$(printf "99_figs_v2_rgv_tnr_r%02d.sh" $n)
  printf '#!/bin/bash\n# FIGS_V2 retry %s of the RGV figure in Times New Roman (gpu1 lacked memory)\nexec bash %s\n' $n $T/rgv_job.sh > $Q/$nm.tmp
  mv $Q/$nm.tmp $Q/$nm
  exit 3
fi
source $B/repo_rgvtag/scripts/ops/hyde06_env.sh > /dev/null 2>&1
source $B/tmp/d146/cache_env.sh
export PYTHONPATH=$GPU_DEPS:$B/repo_rgvtag OMP_NUM_THREADS=8 TMPDIR=$T/tmpdir
cd $B/repo_rgvtag
log "start, gpu1 free $free MiB"
for k in 24 100; do
  rm -rf $O/f$k
  t0=$(date +%s)
  $PY $T/rgv_tnr.py $T/render_gradient_viz_flip.py $B/output/gpu/rgv_tag/bunny300k_RG97_viz.npz \
    $B/output/gpu/d111/bunny_dense15.npz $O/f$k +viz=A,B2,C,D +views=90 +res=2160 +interp=1 +grid=1 +gridfull=0 \
    +c_avg=5 +d_avg=5 +code=freeze-2026-10-15 +frames=$k-$k > $O/f$k.log 2>&1 < /dev/null
  rc=$?
  log "frame $k rc $rc $(( $(date +%s) - t0 )) s"
  [ $rc = 0 ] || exit 1
done
for pair in "early 24" "late 100"; do
  set -- $pair
  g=$(ls $O/f$2/gridB2half_90_2160/*.jpg | head -1)
  [ -s "$g" ] || { log "no half-size 2 x 2 for frame $2"; exit 1; }
  ffmpeg -loglevel error -y -i $g -filter_complex \
    "[0]split=4[a][b][c][d];[a]crop=1920:1050:0:0[A];[b]crop=1920:1050:1920:0[Bq];[c]crop=1920:1050:0:1080[C];[d]crop=1920:1050:1920:1080[D];[A][Bq]hstack=2[t];[C][D]hstack=2[u];[t][u]vstack=2,scale=2400:1312" \
    -frames:v 1 -q:v 3 $O/fig_render_gradient_$1.jpg < /dev/null
  log "fig_render_gradient_$1.jpg rc $?"
done
exit 0
