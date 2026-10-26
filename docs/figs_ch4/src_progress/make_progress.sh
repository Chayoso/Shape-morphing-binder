#!/usr/bin/env bash
# fig_progress.png: how the physics core and the render term fall over a 300k morph (bunny top, dragon bottom; the full
# method, seed 97 thick, other seeds thin; left the linear share of each term's drop, right the share of its decades),
# with the share of the particle movement done. Made from the original plot (progress_original.png, 2160 x 1216) by
# cutting away its title line, which carried the run label.
# The original was plotted on the GPU server by d155_figs_original.py (a copy of scripts/probes/paper/d155/d155_figs.py)
# from output/gpu/d155/records.json and frames_summary.json, which exist on hyde06 only: to change the panel titles
# (e.g. "render arm" -> "full method"), edit that script and regenerate on hyde06, then rerun this crop.
# Usage:  bash make_progress.sh
set -euo pipefail
FF=${FFMPEG:-/c/ffmpeg/bin/ffmpeg}
HERE=$(cd "$(dirname "$0")" && pwd)
"$FF" -v error -y -i "$HERE/progress_original.png" -vf "crop=2160:1156:0:60" -frames:v 1 -update 1 "$HERE/../fig_progress.png"
echo "wrote $HERE/../fig_progress.png"
