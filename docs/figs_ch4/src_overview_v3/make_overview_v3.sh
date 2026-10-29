#!/usr/bin/env bash
# make_overview_v3.sh -- the overview figure v3 from overview_v3.svg with a local headless Chrome (no Python):
# ../fig_overview_v3.pdf (vector, the strip PNGs embedded) and ../fig_overview_v3.png (preview, 2x).
# The strip images (overview_strip/*.png) come from hyde06 output/gpu/figs_v2/final/overview_strip/ (gpu1 job 22_figs_v2_overview_strip).
set -euo pipefail
CHROME=${CHROME:-"/c/Program Files/Google/Chrome/Application/chrome.exe"}
HERE=$(cd "$(dirname "$0")" && pwd); W=$(cd "$HERE" && pwd -W)
cat > "$HERE/overview_v3.html" <<HTML
<!doctype html><html><head><meta charset="utf-8"><style>@page{size:1600px 690px;margin:0}html,body{margin:0;padding:0}
img{display:block;width:1600px;height:690px}</style></head><body><img src="overview_v3.svg"></body></html>
HTML
"$CHROME" --headless=new --disable-gpu --allow-file-access-from-files --no-pdf-header-footer \
  --print-to-pdf="$W/../fig_overview_v3.pdf" "file:///$W/overview_v3.html" 2>/dev/null
"$CHROME" --headless=new --disable-gpu --allow-file-access-from-files --hide-scrollbars --force-device-scale-factor=2 \
  --window-size=1600,690 --screenshot="$W/../fig_overview_v3.png" "file:///$W/overview_v3.svg" 2>/dev/null
ls -la "$HERE/../fig_overview_v3.pdf" "$HERE/../fig_overview_v3.png"
