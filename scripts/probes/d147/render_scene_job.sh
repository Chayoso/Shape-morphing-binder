#!/bin/bash
# D147: a scene archive drawn at 4K in the studio look with the floor (scripts/render_scene.py), every 12th simulated
# frame (20 fps, real time), joined into an mp4. Run inside job.sh's environment ($PY, cwd = the repo).
#   [NAME=out_name] bash scripts/probes/d147/render_scene_job.sh RUNS_DIR TAG [extra +options]
# The frames go to RUNS_DIR/render_<NAME or TAG>/frames, the video to RUNS_DIR/<NAME or TAG>.mp4.
R=$1; TAG=$2; shift 2
OUT=${NAME:-$TAG}
OUTD=$R/render_$OUT
rm -rf "$OUTD/frames"                                   # this script's own frames: a new render replaces them whole
$PY scripts/render_scene.py $R/$TAG.npz $OUTD +stride=12 +field=zbs "$@" > $OUTD.log 2>&1
rc=$?; echo "render exit $rc"; [ $rc -eq 0 ] || exit $rc
ffmpeg -y -loglevel error -framerate 20 -pattern_type glob -i "$OUTD/frames/*.jpg" -c:v libx264 -crf 18 \
  -pix_fmt yuv420p $R/$OUT.mp4 && echo "video $R/$OUT.mp4"
