---
name: cleanup-over-100gb
description: "User rule — keep the server results (hyde06, /data/relcfd/chayo) under a limit — 200 GB since 2026-10-14 (was 50 GB from 2026-09-29, 100 GB before); clean up superseded runs, always (archive logs first)"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-14T14:17:22.904Z
---

The user (2026-09-26, after the 472 → 98 GB cleanup): "그리고 결과가 100GB 이상이면 한 번 정리 하고 항상" — whenever the
results pass 100 GB, do a cleanup, every time.

**Why:** the 300k runs are 4–16 GB each and gallery sweeps 8–10 GB; the output directory had reached 472 GB unnoticed.

**How to apply:** check `du -sh $OUT` (hyde06, /data/relcfd/chayo/physmorph_v2/output) whenever runs finish or at each
status check; above 100 GB, run the cleanup pattern of `scratch/cleanup_20260925.sh`: keep the deliverables (g41, report_*),
the adopted-form gallery, the references the current work cites, and anything running; archive every deleted run's
logs/json/verdicts to `logs_archive_<date>.tgz` with a deleted-list file first; dry-run with guards; never touch other users'
data; record what was deleted in docs/experiments.md.

**2026-09-29 tightened to 50 GB** ("이제 결과는 50GB 이상 넘어가지 않도록 해 줘"): the limit is now 50 GB, not 100.
**2026-10-14 (later) RAISED to 500 GB.** The user said: "이제부터 모든 실험은 relcfd 아래 있는 내 폴더에서 하자. 대규모로 진행하고 … 500GB까지 풀어서 빠르게 실험해도". Every experiment and all its I/O goes under /data/relcfd/chayo; run at scale. Guard at ~490 GB.
**2026-10-14 RAISED to 200 GB:** /data has 12 TB free, and the 50 GB guard was throttling the D145 evidence runs to one dragon at a time. The user chose "200 GB로". Bulk files of read runs are still deleted with a log entry; the limit is headroom, not a target. Also, the user said "miniforge3는 내비둬라 혹시 모르니까": never move or delete ~/miniforge3. neuma_eval/data_download and Shape-morphing-binder/output were moved to /data/relcfd/chayo/home_moved with symlinks left in place.
Under the earlier 50 GB rule: keep it below 50 GB at all times: runs that only feed a JSON result or a census delete their archive right after
(census inside the run script), and big 300k archives are kept only while a render or census still needs them. See [[cleanup-after-v10]], [[hyde06-ops-etiquette]].
