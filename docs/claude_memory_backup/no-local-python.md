---
name: no-local-python
description: User rule 2026-09-17 — never run Python (tests, scripts, experiments) on the local Windows machine; everything runs on hyde06 under /data
metadata:
  type: feedback
---

Do not run Python programs on the local Windows machine — no `python`, no `pytest`, no ad-hoc
experiment scripts. The local machine is for editing, git, and docs only. All execution
(test suite, synthetic checks, pipeline runs, report builds that need the env) happens on
hyde06 in the diffmpm env from `/data/relcfd/chayo/physmorph_v2/repo` (see
[[hyde06-ops-etiquette]]).

**Why:** the user said so explicitly ("local에서는 python 프로그램 돌리면 안 되는 거 알지?")
after a local pytest + synthetic script; the local box is not a compute host.

**How to apply:** for a quick script, scp it to `/data/relcfd/chayo/physmorph_v2/output/scratch/`
and run it there with `PYTHONPATH=$REPO` (no deploy needed for scratch scripts); for tests,
deploy first (`scripts/ops/deploy.sh`) then run pytest over ssh with the thread caps (8) and
a free GPU (0 or 2) or CPU. Never gate a commit on a local test run.
