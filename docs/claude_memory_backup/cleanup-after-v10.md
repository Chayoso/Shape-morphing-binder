---
name: cleanup-after-v10
description: "User instruction 2026-09-17 23:55: once the gallery passes v10, sweep out the useless earlier results (server + local)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-09-18T04:56:22.654Z
---

**Instruction (user, 2026-09-17 late evening, verbatim):** "v10이 넘어가면 이전 결과들 중 쓸데 없는 것들 한 번 싹 지워 줘" —
once the 150k gallery version passes v10, do a full sweep of the earlier results that are no longer useful.

**Why:** the server output dir (/data/relcfd/chayo/physmorph_v2/output) and the local output/ accumulate
falsified-ladder runs, superseded gallery versions and scratch probes; the user wants one decisive cleanup at
that milestone rather than continuous pruning (the same request was made at v6 on 2026-09-17 15:30 and done).

**How to apply:** when a v11+ gallery is published: (1) keep only the runs the docs cite as evidence (the current
gallery version, the render-proof twins + controls, the material study, the no-net proofs) and the two latest gallery
versions' report folders; (2) delete falsified-ladder outputs (logs/json/npz), superseded report_* folders on the
server, stale qa_frames/scratch scripts, and local output/ page folders older than the last two versions; (3) archive
logs first (tgz, as output/logs_archive_<date>.tgz) so numbers stay recoverable; (4) never touch other users' data
([[hyde06-ops-etiquette]]); (5) list what was deleted in docs/experiments.md. Related: [[highres-150k-phase]].
