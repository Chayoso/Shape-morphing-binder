---
name: runtime-10min-goal
description: "User goal 2026-10-09: after the algorithm freeze (freeze-2026-10-09), optimise runtime hard so a 300k run finishes in about 10 minutes (now bunny ~27 min, dragon ~41 min on one RTX 6000 Ada)"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-11T22:41:48.361Z
---

User, 2026-10-09: "일단 알고리즘 fix 한 후에 최적화를 해서 시간 줄이면 되니까. 뭐 물론 최적화 빡세게 해서 10분 내외로 끝내게 해야지".

**Order:** freeze first (40k gallery gate → defaults → tag), then the runtime phase. Target ≈ 10 min per 300k run (bunny and dragon), i.e. ~3–4× faster than the frozen version.

**Approach:** profile one 300k window by stage (forward rollout, adjoint per term, render terms incl. the exterior disc search, Sinkhorn/transport, line-search trials, commit/eval rollouts, records) and the window count; exact speed-ups first (results bit-for-bit or within replay noise: graphs, fusion, cached searches, fewer host syncs, precision where provably safe); result-changing ones (iterations per window, window count/stopping, picture resolution schedule) only as pre-registered A/B with the frozen quality gate (independent-sample yardstick/display, at rest, floating pieces, render lead, momentum; 40k gallery). Prior findings: [[runtime-findings-2026-10-02]] (records off −16 %/window, 96 px, alternating sweeps 11 min on the 10-02 code; "15 min not reachable by exact speed-ups" then), D96 (exterior searches 1.77 → 1.16 s exact).

**Started 2026-10-09 20:10 CDT** on the frozen tag freeze-2026-10-09.

- **Profile, 3-run baseline at seed 97:** bunny 25–26 min (83–88 attempts, ~18 s each); dragon 42 min (107–110 attempts, ~23 s each).
- **Where the time goes:** Sinkhorn is 36 % on dragon (87³ grid); the MPM adjoint is 29–36 % (G2P adjoint 57 % of a sweep); the exterior search 9–14 %; layer_relief setup 15 % on bunny (fixed exactly: layer rows only).
- **The merit still falls at the stop**, so fewer windows means a worse result.
- **Exact levers alone** are estimated at ~2× (bunny 14, dragon 21 min). 10 min will need gated result-changing A/Bs.
- **User approval 2026-10-09:** port ONLY D47, D50 and D53 (Sinkhorn potentials reuse, replay warm-start reuse, record energy from the commit) from tag settled-2026-10-03-d53. D49 (proximity k=1) and everything else there are still not approved.
- **hyde06's root filesystem is full:** keep caches and temp under /data.
- **2026-10-10 ~02:00, after batch 4 (a4bc26d; contention-free P2G / G2P adjoint), exact levers only, shape cell:** bunny 9.2–10.8 min, dragon 17.1–18.8 min (frozen code: 22–27 / 38–44). Table: output/results_2026-10-10/runtime/runtime_table.md.
- **The dragon early stop** (4 batch runs vs 5 frozen; p 0.04 / 0.016; ranges overlap; no step reproduces it) is recommended to be accepted as recipe spread.
- **Result-changing levers held until D140 is decided:** (a) Sinkhorn warm start across trials, −10–18 % dragon; (b) cleanup merged into the physics channel, −8 %; (c) D49 proximity, −6 %; (d) 8 → 6 iterations per window.
- **USER 2026-10-10:**
  - The dragon early stop is accepted as recipe spread.
  - Result-changing A/Bs approved: ONLY (a) the Sinkhorn warm start and (b) cleanup into the physics channel, run after D140 is decided.
  - (c) D49 and (d) 8 → 6 iterations are NOT approved.
- **2026-10-10 ~19:30, batch 7 (2e85df4; node-owned P2G):** bunny 7.9–8.9 min, dragon 17.2–17.8 min; all pass.
  - The exact path ends here (~16 min dragon at best).
  - The 3-cotangent adjoint is exact but not faster (wip/rt-3cot).
  - (a)+(b) are projected at 12.5–13.5 min for the dragon. A 10-minute dragon needs (d), which is not approved, so ask the user after (a)/(b).
  - The shape-cell band for (a)/(b) is in tmp/rt/rt_reband.py.
- **2026-10-11, both A/Bs FAILED; both switches stay off.**
  - D142 (a), the warm start: not at rest; the runs stop early with the body still moving.
  - D143 (b), cleanup in the physics channel (64b62b9): −10 % per attempt, but the dragon needs 151–159 attempts (slower overall, 21–22 min). Its windows 8–9 trajectory moves on the dragon, and bunny B1's thin reaches 5.47 %.
  - Lesson: the bunny window-6 band (two runs 0.01 pitch apart) is too narrow, since the default code also misses there. Future trajectory bands need ≥3 baseline runs.
  - The runtime stands at the exact path: bunny 7.9–8.9 min, dragon 17.2–17.8 min.
- **Deterministic mode** (Warp RUN_TO_RUN) costs 9–14×, so it is for proofs and forensics only.
- **Code cleanup** landed as 63b5a88, bit-for-bit against 09b.

Related: [[pipeline-freeze-2026-10-07]], [[no-parameter-only-fixes]].
