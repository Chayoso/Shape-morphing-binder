---
name: resume-2026-10-14-autonomy
description: "State 2026-10-14 ~03:00 CDT during the user's 6-hour absence — hyde06 rejects the ssh key since 02:28 (user must re-add); D145 evidence pool running detached; D146 late wobble root cause + D146a/b; D147 scene prototype passed; resume commands"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-14T17:17:22.920Z
---

**BLOCKER:** since 02:28 CDT on 2026-10-14 hyde06 answers "Permission denied (publickey,password)" (hyde01 hostname does not resolve locally).
- The user must re-append ~/.ssh/id_ed25519.pub (ed25519 …ZCyZV ab4295@Chayo, SHA256:KdgXwOO…) to hyde06's ~/.ssh/authorized_keys.
- Make no ssh attempts until then. Detached server jobs keep running.

**On access return, message each agent:**
- **D146 (late wobble):** run `bash <scratchpad>/d146/d146b_resume.sh`. It deploys wip/d146b cc2293a, runs the tests plus up to 3 suites (accepts exit 0 only; known flake test_the_eval_trajectory_replays_bit_for_bit on a shared card), then queue2 arms CDL/CBL/CBP/CDP/ABL/ADL. Read with d146_gate_read.py + d146_alt.py.
- **D147 (throw scene):** its server waiter (pid 3155285) may already have run the scene tests. Then the full suite (bit-for-bit), then the 40k batch: V0, V1, scene arm + twin (pre-registered dd82f97).
- **D145 (evidence):** pool v3 workers run detached. First item after 02:58 is the kept-frames copy repo_d145s bit-for-bit proof. Then B1–B3 on the deliverable tag, plus B4 Xu, B5 Buonomo (its own sub-agent), B7 reach, B8 materials, B9 floor + G 0.02, B10 loss curves.

**D145 resume order** (from the agent's final note, pushed through e503ff3):
1. Fix the repo_d145g disk estimate (it writes a ~12 GB full archive) and check du history for overshoot.
2. Read the repo_d145s kept-frames proof. If SAME, move B1–B3 / B8–B10 to it and patch repo_d145g the same way.
3. Check the harness dry run, OT and the Xu readings (tab145.py).
4. B5 (Buonomo SGP 2025, MIT, commit 9767ed9): run the other 5 pairs with `scripts/run_pair2.sh MESH ADADIV BUO`, one at a time, from /data/relcfd/chayo/physmorph_v2/baselines/buonomo.
   - Upload summarize.py from the B5 agent's local scratchpad first.
   - Each pair needs ~1 h of card time (the SDFs train on the CPU).
   - Input: 100k surface points per shape, the paper's default config.
   - The teapot uses the outer body only; its cavities are filled in our sampler, which is a disclosure point.
   - LF-INSD variant: not started (6–11 h per pair).
5. Slower monitoring and milestone reports.

**D146 findings** (eb42749, fd0a75e, 792f45b; pictures in output/results_2026-10-14/d146_late_wobble/):
- The late wobble is physical. The outer layer moves along the normal, the twin wobbles the same way, and it is not a display artefact.
- Cause 1: the layer relaxation runs in the released half too (code ≠ setup.py doc since 835af64) and undoes u. `--relax_driven` (109b8c2) halves it, but the pulse stays.
- Cause 2: u re-solved from zero with fresh Adam each window, a period-2 dither (u slope −0.53). D146b `--u_adam_carry` (per-coordinate bias age) is on wip/d146b, untested by suite.
- Neither changes defaults until the user decides (frozen-algorithm change).

**D147:**
- Contact prototype passed: e 0.68–0.74, momentum 4e-6, angular momentum 8e-5.
- Single field doesn't stick in a fast hit, but leaves a one-cell air cushion and crushes the faces.
- Code ~1500 lines on wip/d147. Plan: 40k scene by 10-21, 300k by 10-24, video by 10-28.

**D147 resume:** run `bash scripts/probes/d147/resume.sh` from C:/dev/smb_d147 (wip/d147).
- My decision, received and pre-registered by the agent (c91ab47; read40k.py applies the rule): the 40k batch runs THREE timelines.
  - the registered scene;
  - long flight: G 0.02, floor 5.2 wu down, 10 s;
  - stiff: E×7, G 0.05.
- The 300k scene is picked by a rule fixed before reading: separates in the air, lands and tumbles within the scene, lower end shape error; a tie goes to the shorter contact.
- F-smoothing 0.955 stays (frozen).

**2026-10-14 midday results:**
- **D146b verdict** (5cee211), relax_driven + u_adam_carry: the late wobble falls to ×0.4 and the reversal is gone. It fails by the letter:
  - (b) driven/other ratio 1.58 / 1.55 (still below the deliverable's 1.81);
  - (2) bunny window-8 p90 0.02 outside;
  - (6) a new dragon hind-leg bead at 19.75–22.3 s and a neck bead.
- The dragon now runs 165 windows / 27.5 s / 29.7 min, because the merit keeps improving.
- The repeat CDL/CBL runs after 19:00 to attribute the misses.
- **D148 dragon** (gap split): the 3–6 s crevice is clean, BUT there are new detached beads (early crevice opening at 1.5–2.25 s, loop at 2.4 s, upper-hole bead at 10–14 s) and cost +43 %. Items 2 and 3 pass. Bunny, GA99 and a bead forensic are pending; tonight's verdict.
- **Memory leak:** settled-base from 545f968 (D147 step split) leaked ~7.5 GB per window at 300k (`_st` → tape cycle). Fixed in c38951c, plus D147's weakref fix.

**Queued for the user (serious decisions):**
- adopt relax_driven and/or u_adam_carry as defaults (a re-freeze; evidence redo in the 11-02–06 slot);
- re-add the ssh key.
