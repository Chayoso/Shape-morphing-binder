#!/bin/bash
# 21_d164_pilot_gates.sh -- gpu1 queue job (agent D164; resubmitted 20:3x after 20_ stopped in 52 s before any run, the probe
# having imported torch before physmorph; the coordinator's order 2026-10-22 19:30 / 19:4x): D164's step P
# (the pilot, its pre-registered stop rules read mechanically; a STOP / AMEND or a failed pilot run ends the job with
# exit 1 before the gates) and step V (V1 the full suite, V2 the contracts test 20x on repo_d164 plus the reports on
# repo_d164_start, V3 the pinned references and the switch's tests, V4 Warp-deterministic 40k bunny four windows twice:
# SAME run to run and SAME against rt1's S1). tmp/d164/d164_chain.sh with QUEUE=1: no claim and no lock (the runner holds
# gpu1); CUDA_VISIBLE_DEVICES=1 from the runner. Outputs: output/gpu/d164/{pilot,gates}/, status in output/gpu/d164/status.txt.
# Peak memory: one 40k run at a time (~4.5 GB), the suite's small clouds, the deterministic 40k eager run (< 10 GB).
B=/data/relcfd/chayo/physmorph_v2
cd $B
QUEUE=1 bash $B/tmp/d164/d164_chain.sh "P V"
