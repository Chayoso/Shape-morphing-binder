#!/bin/bash
# 21_d164_p2_gates.sh -- gpu1 queue job (agent D164; amendment 1, approved by the coordinator 2026-10-22 22:0x): step P2
# (the pilot on D164's defaults, the commit reading the accepted candidate's potentials and lists; the pre-registered stop
# rules read mechanically, a STOP / AMEND or a failed run ends the job with exit 1 before the gates) and step V (V1 the
# full suite, V2 the three formerly flaky tests 20x each on repo_d164 (gates) and on repo_d164_start (reported), V3 the
# pinned references, the digest and the switch's tests, V4 Warp-deterministic 40k bunny four windows twice: SAME run to
# run and SAME against rt1's S1). tmp/d164/d164_chain.sh with QUEUE=1: no claim, no lock (the runner holds gpu1).
# Outputs output/gpu/d164/{pilot2,gates}/, status output/gpu/d164/status.txt. Peak memory: one 40k run (~4.5 GB), the
# suite's small clouds, the deterministic 40k eager run (< 10 GB).
B=/data/relcfd/chayo/physmorph_v2
cd $B
QUEUE=1 bash $B/tmp/d164/d164_chain.sh "P2 V"
