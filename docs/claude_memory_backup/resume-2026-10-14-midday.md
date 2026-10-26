---
name: resume-2026-10-14-midday
description: "Handoff 2026-10-14 ~12:30 CDT before a local restart — four agents' state (D145 evidence, D146b wobble, D147 throw scene, D148 gap split), what runs detached on hyde06, the decisions queued for the user, and how to resume"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-14T20:24:59.672Z
---

The user restarted the local computer around 12:30 CDT on 2026-10-14. Everything below runs detached on hyde06; after the restart, resume the agents by SendMessage (their IDs are in the session) or spawn fresh agents with these notes.

**Server rules now in force:**
- Results live under /data/relcfd/chayo with a 500 GB limit.
- All caches and temp files go to /data.
- Never touch ~/miniforge3.
- GPU locks in physmorph_v2/gpu_locks; one run per card.
- ssh: batch calls, ≥10 min between polls, one retry after 2 min on an abort.
- Today's card priority: D148 holds up to 3 cards until 19:00; D145's cap is 1 until 19:00, then 3.

**D145 evidence (plan in docs/experiments.md D145):**
- Done:
  - B4 Xu: behind on every measure, never at rest.
  - B3 40k gallery: render ahead of twin 19/19, p 3.8e-6.
  - 40k ablations: relief helps 17/19; the volume body cuts early floaters 4–180×; u_off mixed; cloud-read render ahead on particle silhouette 13/19.
  - 300k bunny exterior vs cloud (seed 97): exterior ahead 15–27 % on display/yardstick; cloud ahead on cloud-silhouette. Not a frozen-algorithm issue unless the seeds or dragon flip it.
  - B5 Buonomo: bunny 0.960, dragon 0.902, teapot static failure 0.916.
  - B11 ARAP: done on genus-0 meshes; dragon (genus 15), teapot and bob N/A.
  - B6 OT: done. Harness: done. Kept-frames copy proof: SAME.
- Running: B5 bob/armadillo/homer, dragon X97, then B1/B2/B8/B9 300k (most of them after 19:00).
- First tables: output/results_2026-10-14/d145_evidence/d145_first_tables.md.

**D146b late wobble** (5cee211):
- relax_driven + u_adam_carry: wobble ×0.4, reversal gone.
- Fails by the letter: (b) ratio 1.58 / 1.55, (2) bunny window-8 p90 +0.02, (6) a new dragon bead at 19.75–22.3 s.
- The dragon now runs 165 windows / 27.5 s / 29.7 min.
- The repeat CDL/CBL plus ABL run after 19:00 on the pre-leak copy (attribution rule).
- Defaults unchanged.

**D147 throw scene:**
- Code on wip/d147 f556ac6, with the leak fix and the weakref fix. Scene tests and suite passed at 12:00 / 12:05.
- The controls-only assimilation (load-free rollout) is implemented.
- Chain 7 on hyde06: V1e, S2, A1 (the 40k pick by the rule), V0 seed 3, the memory probe.
- The 300k scene (2×150k) comes after 19:00.

**D148 gap split** (`--gap_split on`, 380 tests pass; leak fix c38951c; traj.py back under 500 lines via arrays.py):
- Dragon 300k s97, item by item:
  - the 3–6 s crevice is clean (pass);
  - NEW detached beads: crevice opens early with a bead at 1.5–2.25 s, loop piece at ~2.4 s, upper-hole bead at 10–14 s (item 1 fail);
  - items 2 and 3 pass;
  - cost +43 % (fail).
- BA, BT, GA99 and the relief readings are running. A bead forensic is in progress (labels / uncovered gap / early split). Any fix is pre-registered as D148b and not run.
- The user wants the D148 verdict TODAY.

**USER, 2026-10-14 ~15:40:** "D146 + Gaussian 떠 있는 거 없애기 + 최적화 하면 끝나겠다". The final algorithm is:
- D135 plus D146b (`--relax_driven` + `--u_adam_carry`, ADOPTED; it fixes the late wobble ×0.41);
- the floating-piece removal (D148 track: walls/D148c gate running, "최대한 D148로");
- then runtime optimisation on the combined code;
- then the re-freeze (new tag), followed by a redo of the algorithm-dependent evidence (B1–B3, B7–B10) before 11-07.

**Decisions queued for the user** (frozen-algorithm level):
1. D146b adoption, plus the 10 s longer dragon run.
2. D148 adoption, or D148b, given the new beads and the +43 % cost.
3. D147: the "assimilate only the control's share" rule (scene only), already in use for the scene.
4. Re-freeze in the 11-02–06 slot if any is adopted.

**Memory leak note:** settled-base from 545f968 to c38951c leaked ~7.5 GB per window at 300k. Run copies at the deliverable tag 592e97a, and the D146 copies, were unaffected.
