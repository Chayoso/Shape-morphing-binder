#!/usr/bin/env bash
# fig_morph_sequence.jpg: 300k bunny (top) and dragon (bottom), side view, the frozen configuration (render arm, seed 97),
# the base display (surface discs of the corrected Zhu-Bridson field), columns t = 0, 0.5, 1, 2, 4 s and the delivered end.
# Usage:  bash make_morph_sequence.sh            # compose the figure from the frames in this folder
#         bash make_morph_sequence.sh extract    # re-extract the frames from the source videos first
# One video frame = one kept simulation frame (every 12th step, 0.05 s), so frame n is t = 0.05 n s.
set -euo pipefail
FF=${FFMPEG:-/c/ffmpeg/bin/ffmpeg}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
SRC=$ROOT/output/results_2026-10-15/d151_gate
# the left half (1920 x 1080) of each side-by-side video is the frozen configuration; the crop leaves out its label
CROP="crop=1240:872:340:58"
if [ "${1:-}" = "extract" ]; then
  for mesh in bunny dragon; do
    if [ $mesh = bunny ]; then END=257; else END=604; fi        # the last kept frame (bunny 12.85 s, dragon 30.2 s)
    i=0
    for n in 0 10 20 40 80 $END; do
      "$FF" -v error -y -i "$SRC/${mesh}300k_side_SL_vs_NL.mp4" -vf "select=eq(n\,$n),$CROP" -frames:v 1 -q:v 2 \
        "$HERE/${mesh}_$i.jpg"
      i=$((i + 1))
    done
  done
fi
# compose: each frame scaled to 612 x 428 and framed in white to 620 x 436 (even sizes: the JPEG encoder needs them), six a row, bunny over dragon
"$FF" -v error -y \
  -i "$HERE/bunny_0.jpg" -i "$HERE/bunny_1.jpg" -i "$HERE/bunny_2.jpg" -i "$HERE/bunny_3.jpg" -i "$HERE/bunny_4.jpg" -i "$HERE/bunny_5.jpg" \
  -i "$HERE/dragon_0.jpg" -i "$HERE/dragon_1.jpg" -i "$HERE/dragon_2.jpg" -i "$HERE/dragon_3.jpg" -i "$HERE/dragon_4.jpg" -i "$HERE/dragon_5.jpg" \
  -filter_complex "[0]scale=612:428,pad=620:436:4:4:white[a0];[1]scale=612:428,pad=620:436:4:4:white[a1];[2]scale=612:428,pad=620:436:4:4:white[a2];[3]scale=612:428,pad=620:436:4:4:white[a3];[4]scale=612:428,pad=620:436:4:4:white[a4];[5]scale=612:428,pad=620:436:4:4:white[a5];\
[6]scale=612:428,pad=620:436:4:4:white[b0];[7]scale=612:428,pad=620:436:4:4:white[b1];[8]scale=612:428,pad=620:436:4:4:white[b2];[9]scale=612:428,pad=620:436:4:4:white[b3];[10]scale=612:428,pad=620:436:4:4:white[b4];[11]scale=612:428,pad=620:436:4:4:white[b5];\
[a0][a1][a2][a3][a4][a5]hstack=6[ra];[b0][b1][b2][b3][b4][b5]hstack=6[rb];[ra][rb]vstack=2" \
  -frames:v 1 -q:v 3 "$HERE/../fig_morph_sequence.jpg"
echo "wrote $HERE/../fig_morph_sequence.jpg"
