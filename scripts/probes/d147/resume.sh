#!/bin/bash
# D147: the one command for when hyde06 accepts the key again (run locally from the wip/d147 worktree):
#   bash scripts/probes/d147/resume.sh
# deploys this worktree's HEAD to repo_d147r on hyde06 and starts chain.sh there (tests -> suite -> prototype re-runs
# -> 40k batch -> readings), each GPU stage through the lock directory. Progress: output/gpu/d147/chain.log.
set -e
R=/data/relcfd/chayo/physmorph_v2/repo_d147r
cd "$(git rev-parse --show-toplevel)"
git diff --quiet HEAD || { echo "uncommitted changes: commit first (the deploy is HEAD)"; exit 1; }
git archive HEAD | ssh hyde06j "rm -rf $R.new && mkdir -p $R.new && tar -x -C $R.new && rm -rf $R && mv $R.new $R"
ssh hyde06j "mkdir -p /data/relcfd/chayo/physmorph_v2/output/gpu/d147 && cd $R && \
  (nohup bash scripts/probes/d147/chain.sh $R > /data/relcfd/chayo/physmorph_v2/output/gpu/d147/chain.out 2>&1 &) && \
  echo chain started from $R at \$(date)"
