#!/usr/bin/env bash
# fig_gallery40k_mesh.png: the 40k gallery (19 meshes, seeds 97 / 99 / 101) read against the exact target mesh, the full
# method over its lambda = 0 twin; three of the original plot's five panels: exact binary 1 - IoU at 1024 px, outline shape
# error (offset spread) and the normal angle at the nearest mesh point (dot: median over seeds, bar: range, grey tick: a
# perfect particle sample, dashed: the twin). "arm" in the panel titles is the full method.
# Made from gallery40k_original.png (plotted on the GPU server by scripts/probes/paper/d154/d154_g40_fig.py from
# data/g40_table.json) by cutting away its title, its bottom label and the two panels whose titles use internal names.
# To change labels, regenerate the plot on hyde06 from data/g40_table.json (the plot needs the server's matplotlib).
# Usage:  bash make_gallery40k_mesh.sh
set -euo pipefail
FF=${FFMPEG:-/c/ffmpeg/bin/ffmpeg}
HERE=$(cd "$(dirname "$0")" && pwd)
# panel x ranges in the 2080 x 1056 original: the first panel with the mesh names 0-530, the outline panel 1300-1680,
# the normal panel 1700-2080; rows 55-1000 (below the title, above the bottom label)
"$FF" -v error -y -i "$HERE/gallery40k_original.png" \
  -filter_complex "[0]split=3[a][b][c];[a]crop=530:945:0:55[p1];[b]crop=380:945:1300:55[p4];[c]crop=380:945:1700:55[p5];\
[p1][p4][p5]hstack=3" -frames:v 1 -update 1 "$HERE/../fig_gallery40k_mesh.png"
echo "wrote $HERE/../fig_gallery40k_mesh.png"
