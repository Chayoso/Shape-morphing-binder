#!/bin/bash
# Fetch a BUILT report page (index.html + the media the page references) from hyde06 into
# output/report_<prefix>_page/, skipping the raw GIFs and full-size PBR PNGs the builder has already
# converted (mp4 / jpg). Usage: bash scripts/ops/fetch_page.sh <prefix>     e.g. h150v7, n150v7
# (build first on hyde06: build_report150.py $OUT/report_<prefix> "<title>" $OUT/<md>)
cd "$(dirname "$0")/../.."
PFX=$1
L=output/report_${PFX}_page
mkdir -p $L
ssh -o BatchMode=yes -J chayo@hyde01.dabh.io chayo@hyde06.dabh.io \
  "cd /data/relcfd/chayo/physmorph_v2/output && tar czf - --exclude='*.gif' --exclude='*_pbr_az*.png' report_$PFX" \
  | tar xzf - -C $L --strip-components=1
echo "fetched $L: $(find $L -type f | wc -l) files, $(du -sm $L | cut -f1) MB"
