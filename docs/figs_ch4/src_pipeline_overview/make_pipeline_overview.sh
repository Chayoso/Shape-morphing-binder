#!/usr/bin/env bash
# pipeline_overview: the chapter draws the diagram from ../pipeline_overview.tikz (TikZ, compiled with the chapter).
# ../pipeline_overview.svg is the same diagram, hand-written, for viewing and editing without LaTeX: keep the two in sync.
# This script only refreshes ../pipeline_overview_preview.png from the SVG with a local headless Chrome (no Python).
# Usage:  bash make_pipeline_overview.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CHROME=${CHROME:-"/c/Program Files/Google/Chrome/Application/chrome.exe"}
DIR=$(cd "$HERE/.." && pwd -W)            # a Windows path (C:/...) for Chrome; Git Bash's pwd -W prints it
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size=1520,800 \
  --screenshot="$DIR/pipeline_overview_preview.png" "file:///$DIR/pipeline_overview.svg"
echo "wrote $DIR/pipeline_overview_preview.png"
