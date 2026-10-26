#!/usr/bin/env bash
# fig_throw_scene.jpg: the two-body throw scene (2 x 150k particles, stiff material E = 9.8e5, render arm, seed 97),
# 4K still camera, studio look with the floor. Panels (left to right, top to bottom):
# t = 0 (launch), 2.0 s, 3.85 s (first contact), 5.7 s (parted), 8.0 s, 9.2 s (body 1 has landed at 8.95 s),
# 11.2 s (body 0 has landed at 10.95 s), 16.0 s (end).
# Usage:  bash make_throw_scene.sh            # compose from the frames in this folder
#         bash make_throw_scene.sh extract    # re-extract the frames from the source video first
# One video frame = one kept simulation frame (0.05 s).
set -euo pipefail
FF=${FFMPEG:-/c/ffmpeg/bin/ffmpeg}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
SRC=$ROOT/output/results_2026-10-17/d147_throw_fixed/stiff300k_s97_4k_cut.mp4
CROP="crop=1680:1860:1080:180"
if [ "${1:-}" = "extract" ]; then
  i=0
  for n in 0 40 77 114 160 184 224 320; do
    "$FF" -v error -y -i "$SRC" -vf "select=eq(n\,$n),$CROP" -frames:v 1 -q:v 2 "$HERE/throw_$i.jpg"
    i=$((i + 1))
  done
fi
# compose: each frame scaled to 512 x 566 and framed in white to 520 x 574 (even sizes), four a row, two rows
"$FF" -v error -y \
  -i "$HERE/throw_0.jpg" -i "$HERE/throw_1.jpg" -i "$HERE/throw_2.jpg" -i "$HERE/throw_3.jpg" \
  -i "$HERE/throw_4.jpg" -i "$HERE/throw_5.jpg" -i "$HERE/throw_6.jpg" -i "$HERE/throw_7.jpg" \
  -filter_complex "[0]scale=512:566,pad=520:574:4:4:white[a0];[1]scale=512:566,pad=520:574:4:4:white[a1];[2]scale=512:566,pad=520:574:4:4:white[a2];[3]scale=512:566,pad=520:574:4:4:white[a3];\
[4]scale=512:566,pad=520:574:4:4:white[b0];[5]scale=512:566,pad=520:574:4:4:white[b1];[6]scale=512:566,pad=520:574:4:4:white[b2];[7]scale=512:566,pad=520:574:4:4:white[b3];\
[a0][a1][a2][a3]hstack=4[ra];[b0][b1][b2][b3]hstack=4[rb];[ra][rb]vstack=2" \
  -frames:v 1 -q:v 3 "$HERE/../fig_throw_scene.jpg"
echo "wrote $HERE/../fig_throw_scene.jpg"
