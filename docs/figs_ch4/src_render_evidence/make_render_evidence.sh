#!/usr/bin/env bash
# fig_render_effect_end.png, fig_iou_sensitivity.png, fig_render_switch.png: REGENERATE ON HYDE06. The plots are drawn on
# the GPU server (its Python has matplotlib; no Python runs locally), this script only copies the script and the data up
# and the PNGs back over ssh / scp. Edit plot_render_evidence.py (labels, rows, colours) and rerun.
# Data (copied unchanged): data/table1.json and data/switch.json from output/results_2026-10-17/d154_render_evidence/data/,
# data/h1_table.json and data/iou_calib.json from output/results_2026-10-16/d152_render_effect/data/.
# Usage:  bash make_render_evidence.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
HOST=${HOST:-chayo@hyde06.dabh.io}
RDIR=/data/relcfd/chayo/physmorph_v2/tmp/ch4/render_evidence
ssh -o BatchMode=yes "$HOST" "mkdir -p $RDIR/data $RDIR/out"
scp -q -o BatchMode=yes "$HERE/plot_render_evidence.py" "$HOST:$RDIR/"
scp -q -o BatchMode=yes "$HERE"/data/*.json "$HOST:$RDIR/data/"
ssh -o BatchMode=yes "$HOST" "source /data/relcfd/chayo/physmorph_v2/repo_t1/scripts/ops/hyde06_env.sh >/dev/null 2>&1; \
  cd $RDIR && nice -n 10 \$PY plot_render_evidence.py data out"
for f in fig_render_effect_end.png fig_iou_sensitivity.png fig_render_switch.png; do
  scp -q -o BatchMode=yes "$HOST:$RDIR/out/$f" "$HERE/../$f"
  echo "wrote $HERE/../$f"
done
