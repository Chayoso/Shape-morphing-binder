#!/bin/bash
# startS40.sh -- D163 arm S at 40k: once D162b's gates pass (output/gpu/d162b/gates/GO), a byte copy of repo_d162bs as
# repo_d163bs, then plq40k_S_97 on the card D163 holds (gpu0), then its read against T97 / T99.
B=/data/relcfd/chayo/physmorph_v2; S=$B/output/gpu/d163/status.txt; L=$B/gpu_locks; G2=$B/output/gpu/d162b
until [ -e $G2/gates/GO ] || grep -q "STOP" $G2/status.txt 2>/dev/null; do sleep 60; done
if [ ! -e $G2/gates/GO ]; then echo "$(date '+%m-%d %H:%M:%S') S: D162b's gates did not pass; no S run" >> $S; exit 0; fi
rm -rf $B/repo_d163bs && cp -a $B/repo_d162bs $B/repo_d163bs && echo "repo_d163bs: byte copy of repo_d162bs ($(cat $B/repo_d162bs/VERSION)) after D162b's GO $(date)" > $B/repo_d163bs/D163_NOTE
echo "$(date '+%m-%d %H:%M:%S') S: D162b GO; repo_d163bs = repo_d162bs $(cat $B/repo_d163bs/VERSION)" >> $S
if [ ! -d $L/gpu0.lock ] && mkdir $L/gpu0.lock 2>/dev/null; then
  printf "agent D163\nrun d163/d163_runs40_S.txt\npid %s\n" "$$" > $L/gpu0.lock/info
fi
grep -q "^agent D163" $L/gpu0.lock/info 2>/dev/null || { echo "$(date '+%m-%d %H:%M:%S') S: gpu0 not held by D163; waiting by the protocol" >> $S; }
cd $B/tmp/d163
USE_CARD=0 REPO=repo_d163bs OUTD=$B/output/gpu/d163/runs ./d163_chain3.sh $B/tmp/d163/d163_runs40_S.txt 1
./d163_after.sh p40S "R plq40k_S_97 exit" $B/tmp/d163/scene40_S.json
