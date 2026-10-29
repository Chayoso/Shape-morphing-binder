#!/usr/bin/env bash
# fig_render_gradient_early.jpg (t = 1.2 s, window 8 of 79) and fig_render_gradient_late.jpg (t = 5.0 s, window 30):
# the render gradient on the deforming 300k bunny (render arm, seed 97, frozen configuration, side view), four panels:
# A where the render acts on dF_c (its share of the pull), B its pull against its typical value now, C where the render and
# the physics pull the same way or against each other, D what the render changed in dF_c (a lambda = 0 counterfactual of
# every window from the same start). The bottom line of each panel's small print (the run label) is cropped away.
# Usage:  bash make_render_gradient.sh            # compose from the frames in this folder
#         bash make_render_gradient.sh extract    # re-extract the panels from the 4K source video first
set -euo pipefail
FF=${FFMPEG:-/c/ffmpeg/bin/ffmpeg}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
SRC=$ROOT/output/results_2026-10-16/render_gradient_viz_tag/4K/bunny300k_AB2CD_2x2_half_side_4K.mp4
if [ "${1:-}" = "extract" ]; then
  for pair in "early 24" "late 100"; do
    set -- $pair
    # the 3840 x 2160 frame holds four 1920 x 1080 panels; keep the top 1050 rows of each (the run label sits below)
    for p in "A 0 0" "B 1920 0" "C 0 1080" "D 1920 1080"; do
      set -- $pair $p
      "$FF" -v error -y -i "$SRC" -vf "select=eq(n\,$2),crop=1920:1050:$4:$5" -frames:v 1 -q:v 2 "$HERE/${1}_$3.jpg"
    done
  done
fi
for when in early late; do
  "$FF" -v error -y -i "$HERE/${when}_A.jpg" -i "$HERE/${when}_B.jpg" -i "$HERE/${when}_C.jpg" -i "$HERE/${when}_D.jpg" \
    -filter_complex "[0][1]hstack=2[t];[2][3]hstack=2[b];[t][b]vstack=2,scale=2400:1312" \
    -frames:v 1 -q:v 3 "$HERE/../fig_render_gradient_${when}.jpg"
  echo "wrote $HERE/../fig_render_gradient_${when}.jpg"
done
