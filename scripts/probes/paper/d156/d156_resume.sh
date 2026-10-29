#!/bin/bash
# d156_resume.sh -- D156: from local Git Bash, once `ssh -o BatchMode=yes hyde06j true` works again: copy the scripts to
# tmp/d156 and start the chain (gates, then the 36 40k runs) detached on hyde06. It waits there for a card idle 10 min
# (the lock protocol). After its status line "R done": on hyde06, from tmp/d156,
#   $PY d156_jobs.py jobs40.txt && bash d156_cpu.sh jobs40.txt $D156_OUT/read40 3
#   cat $D156_OUT/read40_s*.jsonl > $D156_OUT/read40.jsonl && $PY d156_table.py $D156_OUT/read40.jsonl \
#       $D156_OUT/runs40 $D156_OUT/table40.json
set -e
D=$(cd "$(dirname "$0")" && pwd)
H=hyde06j
ssh -o BatchMode=yes $H true || { echo "ssh to hyde06 fails: stop (the user re-adds the key)"; exit 1; }
# redeploy repo_d156 / repo_d156s (agent D156's own copies, made 2026-10-21) from wip/d156's head: git archive, as before
SHA=$(git -C /c/dev/smb_d156 rev-parse --short wip/d156)
git -C /c/dev/smb_d156 archive --format=tar.gz -o "$TMP/repo_d156.tgz" wip/d156
scp -q -o BatchMode=yes "$TMP/repo_d156.tgz" $H:/data/relcfd/chayo/physmorph_v2/tmp/repo_d156.tgz
ssh -o BatchMode=yes $H "set -e; B=/data/relcfd/chayo/physmorph_v2; cd \$B; rm -rf repo_d156 repo_d156s; mkdir repo_d156;
  tar -xzf tmp/repo_d156.tgz -C repo_d156; echo $SHA > repo_d156/VERSION;
  echo '$SHA (wip/d156: freeze-2026-10-15-rt1 + the D156 render target, default off), git archive' > repo_d156/DEPLOYED_FROM;
  cp -a repo_d156 repo_d156s; source tmp/d146/cache_env.sh;
  /home/chayo/miniforge3/envs/diffmpm_v2.3.0/bin/python tmp/d145/patch_s.py repo_d156s;
  echo 'repo_d156 + tmp/d145/patch_s.py (kept frames in place of the full archive)' >> repo_d156s/DEPLOYED_FROM; rm tmp/repo_d156.tgz"
scp -q -o BatchMode=yes $D/d156_chain.sh $D/d156_read.py $D/d157_lib.py $D/d156_table.py $D/d156_jobs.py $D/d156_cpu.sh \
    $D/hull_carve.py $H:/data/relcfd/chayo/physmorph_v2/tmp/d156/
# launch once: the chain logs its own pid ("chain start pid") in output/gpu/d156/status.txt
ssh -o BatchMode=yes $H 'cd /data/relcfd/chayo/physmorph_v2/tmp/d156 && (setsid nohup bash d156_chain.sh "V R" > chain.out 2>&1 < /dev/null &);
  sleep 3; tail -2 /data/relcfd/chayo/physmorph_v2/output/gpu/d156/status.txt'
