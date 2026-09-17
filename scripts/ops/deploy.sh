#!/bin/bash
# Deploy the tracked repo (minus assets/legacy/output) to hyde06 under /data and print the version.
# Files are staged and rsynced (each file replaced atomically): a run importing during a deploy never
# sees a missing module (2026-09-16: a tar-in-place deploy raced a batch launch and crashed n150 maxplanck).
# Usage: bash scripts/ops/deploy.sh [assets...]   (from the repo root; needs ssh -J access)
#   extra arguments are asset names (without .obj) to copy into the server repo's assets/
set -e
cd "$(dirname "$0")/../.."
if [ $# -gt 0 ]; then
  for A in "$@"; do scp -q -o ProxyJump=chayo@hyde01.dabh.io assets/$A.obj chayo@hyde06.dabh.io:/data/relcfd/chayo/physmorph_v2/repo/assets/ && echo "asset $A.obj copied"; done
fi
git rev-parse HEAD > VERSION
git ls-files | grep -v -E "^(assets|legacy|output)/" > /tmp/physmorph_deploy_files.txt
echo VERSION >> /tmp/physmorph_deploy_files.txt
tar --force-local -czf /tmp/physmorph_deploy.tgz -T /tmp/physmorph_deploy_files.txt
scp -q -o ProxyJump=chayo@hyde01.dabh.io /tmp/physmorph_deploy.tgz chayo@hyde06.dabh.io:/data/relcfd/chayo/physmorph_v2/deploy.tgz
ssh -o BatchMode=yes -J chayo@hyde01.dabh.io chayo@hyde06.dabh.io \
  'cd /data/relcfd/chayo/physmorph_v2 && rm -rf repo.stage && mkdir repo.stage && tar xzf deploy.tgz -C repo.stage && rsync -a repo.stage/ repo/ && rm -rf repo.stage && chmod +x repo/scripts/ops/*.sh && echo "deployed $(cut -c1-7 repo/VERSION) to $(pwd)/repo"'
