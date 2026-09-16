#!/bin/bash
# Fetch post-processed report folders from hyde06 into output/report_<prefix>/ (report150 for h150).
# Usage: bash scripts/ops/fetch_report.sh <prefix> <targets...>
cd "$(dirname "$0")/../.."
PFX=$1; shift
case $PFX in h150) L=output/report150; R=report150 ;; *) L=output/report_$PFX; R=report_$PFX ;; esac
mkdir -p $L
for T in "$@"; do
  scp -q -r -o ProxyJump=chayo@hyde01.dabh.io "chayo@hyde06.dabh.io:/data/relcfd/chayo/physmorph_v2/output/$R/$T" $L/ && echo "fetched $L/$T ($(ls $L/$T | wc -l) files)"
done
