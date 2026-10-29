---
name: runtime-findings-2026-10-02
description: "2026-10-02 D17–D19 — how the 300k dragon gets to ~16 min: records off, 96 px render from the start, a 40-window budget; what is adopted, what is pending"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-03T09:12:54.931Z
---

Runtime phase, formulation untouched (docs/experiments.md D17–D19; server code repo_r35 = HEAD, repo_r35b = the same with `render_res` 96).

- **Records off (D17, committed 2026-10-02):** the per-window diagnostic records and the steering telemetry sit behind the existing `work_telemetry` flag, default off, CLI `--telemetry`. 19.8 s vs 23.6 s a window at 300k (−16 %). Forensics that need per-window thin/active-set records must pass `--telemetry` (and `--term_dump` still works on its own).
- **96 px from the start (D18):** the 64-px coarse stage held the 300k dragon's world-thin at ~1.5 % for 30 min; with 96 px from window 0 it is 0.5 % by window 30–40. My prediction (the gain is the epoch reset, not the resolution) was wrong. In both schedules world-thin is lowest about window 40 and then RISES while the merit falls (the tail buys merit, not thin): a stopping/selection observation, parked.
- **D19 (2026-10-02 23:32):** 300k dragon, 96 px from start, records off, `--animations 40`, alone on a GPU: 16.4 min, sil 0.9841, world-thin 0.53 % (as good as the 45–54 min run). The first 40 windows cost 24.6 s each (Sinkhorn sweeps peak in windows 11–35), so 15 min = about 35–36 windows, or the Sinkhorn warm start. 40k gallery ×19: silhouette median +0.0007, no mesh lower by > 0.003, windows 48 → 33, time 90 → 67 min; 300k bunny 0.9856 vs 0.9874, 8.7 vs 14.0 min. The pre-registered rule says: 96 px becomes the default and the coarse stage + c2f event go. NOT yet changed in the code (only the diagnostic copy repo_r35b): the user decides.
- **Update 2026-10-03 (D22–D31, see [[floaters-and-15min-until-solved]]):** `render_res` 96 is now the committed default (d8846ae); the c2f code is inert, not yet removed. The Sinkhorn warm start (seeded at the window start) was built, is 2–15x cheaper late in a run and was DROPPED: it under-reports change. What worked instead: the cross problem with alternating sweeps and the self problem at the blur (e13e77a): a window 10.7 s instead of 24.6 s late in the run, the 300k dragon in 11 minutes (three runs), values nearer the converged solve.
- Still open: higher terminal kinetic under the single epoch; removing the inert c2f code and its test.

**How to apply:** when asked about runtime, start from these numbers; do not re-run D17–D19. `--animations N` only bounds the loop (checked). See [[formulation-stop-rule]], [[surface-resolution-findings-2026-10-02]].
