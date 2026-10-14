#!/bin/bash
# D147: a scene archive drawn at 4K in the studio look with the floor (scripts/render_scene.py), every 12th simulated
# frame (20 fps, real time), joined into an mp4. Run inside job.sh's environment ($PY, cwd = the repo).
#   bash scripts/probes/d147/render_scene_job.sh RUNS_DIR TAG [extra +options]
R=$1; TAG=$2; shift 2
OUTD=$R/render_$TAG
$PY scripts/render_scene.py $R/$TAG.npz $OUTD +stride=12 +field=zbs "$@" > $OUTD.log 2>&1
echo "render exit $?"
ffmpeg -y -loglevel error -framerate 20 -pattern_type glob -i "$OUTD/frames/*.jpg" -c:v libx264 -crf 18 \
  -pix_fmt yuv420p $R/$TAG.mp4 && echo "video $R/$TAG.mp4"
