#!/bin/bash
# d162_beta.sh -- D162, card beta: waits for this launch's gate verdict (a "V gates passed" / "V STOP" line in the status
# after the wrapper's start; gates/PASSED or FAILED of an earlier try do not count), then list B (the gallery's other half)
# from a private copy of d162_chain.sh.
B=/data/relcfd/chayo/physmorph_v2; W=$B/tmp/d162; O=$B/output/gpu/d162; S=$O/status.txt
n0=$(wc -l < $S)
until tail -n +$((n0 + 1)) $S | grep -q "V gates passed\|V STOP"; do sleep 60; done
tail -n +$((n0 + 1)) $S | grep -q "V gates passed" || exit 1
C=$W/run_beta_chain.sh
cp $W/d162_chain.sh $C
RUNS_FILE=d162_runs_B.txt bash $C R
